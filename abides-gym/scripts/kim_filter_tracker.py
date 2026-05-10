"""
kim_filter_tracker.py — Regime-switching Kalman filter (Kim 1994)
=================================================================

Runs K=3 parallel Kalman filters, one per market regime, and maintains
a discrete posterior over which regime is active.

Regimes
-------
  0  VALUE     — dominant value agents, standard OU mean-reversion.
                 alpha = exp(-kappa*dt) ≈ 1,  Q = Q_base,  R = R_base.

  1  MOMENTUM  — dominant momentum agents, price trends away from r_bar.
                 alpha = 1 (random walk),  Q = 20*Q_base (fast-drifting
                 fundamental),  R = 0.5*R_base (tighter observations).

  2  NOISE     — dominant noise traders, no predictable structure.
                 alpha = alpha_ou,  Q = 0.3*Q_base (slow-moving fund),
                 R = 15*R_base (very noisy observations).

Algorithm (Kim 1994 collapse approximation)
-------------------------------------------
  Each step:
  1. For every (prior_regime i, current_regime j) pair, run a Kalman
     predict+update using regime-j dynamics and regime-i posterior.
     Compute the Gaussian log-likelihood of the mid-price observation.
  2. Update joint p(S_{t-1}=i, S_t=j | y_{1:t}) via Bayes.
  3. Marginalise over i to get p(S_t=j | y_{1:t}).
  4. Kim collapse: merge K² posteriors back into K Gaussians by
     computing the conditional mixture mean and variance.

Output
------
  mu_t       — mixture-mean fundamental estimate (cents)
  sigma_t    — mixture-std (cents)
  probs      — [p_value, p_momentum, p_noise]
  belief_dev — (mu_t - mid_t) / r_bar  (same sign contract as Kalman)

Regime-aware agent
------------------
  VALUE regime   → fade the deviation (buy cheap, sell expensive)
  MOMENTUM regime → follow the trend (buy rising, sell falling)
  NOISE regime   → hold
  Mixed          → blend signals by regime probability weights
"""

import numpy as np
import gym
from abides_core.utils import str_to_ns


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────

class KimFilterTracker:
    """
    Regime-switching Kalman filter with K=3 regimes.

    Parameters
    ----------
    r_bar, kappa_oracle, fund_vol, dt_ns
        Same OU calibration as KalmanBeliefTracker.
    obs_noise_frac
        Baseline observation noise as fraction of r_bar.
    regime_stay_prob
        Diagonal of the regime transition matrix (same for all regimes).
        Off-diagonal = (1 - regime_stay_prob) / (K-1).
    prior
        Initial regime probability vector [p_value, p_momentum, p_noise].
    """

    K = 3   # VALUE, MOMENTUM, NOISE

    def __init__(
        self,
        r_bar: int            = 100_000,
        kappa_oracle: float   = 1.67e-16,
        fund_vol: float       = 5e-5,
        dt_ns: int            = 60_000_000_000,
        obs_noise_frac: float = 0.005,
        regime_stay_prob: float = 0.97,
        prior: list           = None,
    ):
        dt_s = dt_ns / 1e9
        self.r_bar = float(r_bar)
        self.K     = KimFilterTracker.K

        # ── OU base parameters ────────────────────────────────────────────────
        alpha_ou = float(np.exp(-kappa_oracle * dt_s))
        if kappa_oracle > 1e-30:
            Q_base = float((r_bar * fund_vol)**2 / (2 * kappa_oracle)
                          * (1 - alpha_ou**2))
        else:
            Q_base = float((r_bar * fund_vol)**2 * dt_s)
        R_base = float((r_bar * obs_noise_frac)**2)

        # ── Per-regime dynamics ───────────────────────────────────────────────
        # [VALUE, MOMENTUM, NOISE]
        # VALUE:    standard OU mean-reversion, standard Q and R
        # MOMENTUM: random walk (alpha=1) with large Q (trending), larger R
        #           (momentum bursts are noisier, not quieter — R must be > VALUE
        #            so VALUE dominates when observations are small-innovation)
        # NOISE:    slow-moving fundamental (small Q), very wide observation noise
        self.alphas  = np.array([alpha_ou, 1.0,          alpha_ou])
        self.r_bars  = np.array([r_bar,    r_bar,         r_bar   ], dtype=float)
        self.Qs      = np.array([Q_base,   Q_base * 20.0, Q_base * 0.3])
        self.Rs      = np.array([R_base,   R_base * 2.0,  R_base * 15.0])

        # ── Regime transition matrix ──────────────────────────────────────────
        off = (1.0 - regime_stay_prob) / (self.K - 1)
        self.Pi = np.full((self.K, self.K), off)
        np.fill_diagonal(self.Pi, regime_stay_prob)

        # ── Prior over regimes ────────────────────────────────────────────────
        self._prior = np.array(prior if prior is not None else [0.6, 0.2, 0.2],
                                dtype=float)
        self._prior /= self._prior.sum()

        # ── Posterior state (reset on every episode) ──────────────────────────
        self.probs: np.ndarray = self._prior.copy()
        self.mus:   np.ndarray = np.full(self.K, float(r_bar))
        self.s2s:   np.ndarray = np.array([self._stat_var(k) for k in range(self.K)])

    # ── helpers ───────────────────────────────────────────────────────────────

    def _stat_var(self, k: int) -> float:
        d = 1.0 - self.alphas[k]**2
        return float(self.Qs[k] / d) if d > 1e-12 else float(self.Qs[k] * 1e12)

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.probs = self._prior.copy()
        self.mus   = np.full(self.K, self.r_bar)
        self.s2s   = np.array([self._stat_var(k) for k in range(self.K)])

    def step(self, mid_price: float):
        """
        One Kim predict-collapse-update cycle.

        Returns
        -------
        mu_mix    : float  — mixture-mean fundamental estimate
        sigma_mix : float  — mixture std
        probs     : ndarray shape (3,)  — [p_value, p_momentum, p_noise]
        """
        K = self.K
        mu_post  = np.zeros((K, K))   # mu_post[i,j]  = posterior mean | S_{t-1}=i, S_t=j
        s2_post  = np.zeros((K, K))
        log_l    = np.zeros((K, K))   # Gaussian log-likelihood of observation

        for i in range(K):
            for j in range(K):
                # Predict using regime-j dynamics, starting from regime-i posterior
                mu_p = self.alphas[j] * self.mus[i] + (1 - self.alphas[j]) * self.r_bars[j]
                s2_p = self.alphas[j]**2 * self.s2s[i] + self.Qs[j]

                # Innovation variance and log-likelihood
                S = s2_p + self.Rs[j]
                innov = mid_price - mu_p
                log_l[i, j] = -0.5 * (np.log(2 * np.pi * S) + innov**2 / S)

                # Kalman update
                Kg = s2_p / S
                mu_post[i, j] = mu_p + Kg * innov
                s2_post[i, j] = (1.0 - Kg) * s2_p

        # Joint log-weight: log p(S_{t-1}=i) + log π_{ij} + log l(y_t|i,j)
        log_joint = (np.log(self.probs[:, None] + 1e-300)
                     + np.log(self.Pi + 1e-300)
                     + log_l)                                    # (K, K)

        # Normalise in log-space for numerical stability
        log_Z    = np.logaddexp.reduce(log_joint.ravel())
        joint    = np.exp(log_joint - log_Z)                     # (K, K)

        # Marginal: p(S_t=j | y_{1:t})
        new_probs = joint.sum(axis=0)                            # (K,)
        new_probs = np.clip(new_probs, 1e-300, None)
        new_probs /= new_probs.sum()

        # Kim collapse: merge K² posteriors → K Gaussians
        new_mus = np.zeros(K)
        new_s2s = np.zeros(K)
        for j in range(K):
            if new_probs[j] < 1e-12:
                new_mus[j] = self.r_bars[j]
                new_s2s[j] = self._stat_var(j)
                continue
            cond_j  = joint[:, j] / new_probs[j]           # p(S_{t-1}=i|S_t=j, y_{1:t})
            new_mus[j] = float(cond_j @ mu_post[:, j])
            new_s2s[j] = float(cond_j @ (s2_post[:, j]
                                          + (mu_post[:, j] - new_mus[j])**2))

        self.probs = new_probs
        self.mus   = new_mus
        self.s2s   = new_s2s

        # Mixture statistics
        mu_mix  = float(new_probs @ new_mus)
        s2_mix  = float(new_probs @ (new_s2s + (new_mus - mu_mix)**2))
        return mu_mix, float(np.sqrt(max(s2_mix, 0.0))), new_probs.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Gym wrapper
# ─────────────────────────────────────────────────────────────────────────────

class KimBeliefWrapper(gym.Wrapper):
    """
    Augments the observation with 5 Kim filter features (same last-2 contract
    as BeliefAugmentedWrapper plus regime probabilities):

      obs[-5] = belief_dev   = (mu_mix - mid) / r_bar
      obs[-4] = p_value
      obs[-3] = p_momentum
      obs[-2] = p_noise
      obs[-1] = belief_std   = sigma_mix / r_bar

    The wrapper stores `tracker` as a public attribute so regime-aware
    agents can read `wrapper.tracker.probs` directly.
    """

    def __init__(
        self,
        env: gym.Env,
        r_bar: int            = 100_000,
        kappa_oracle: float   = 1.67e-16,
        fund_vol: float       = 5e-5,
        timestep_duration: str = "60s",
        obs_noise_frac: float  = 0.005,
        regime_stay_prob: float = 0.97,
    ):
        super().__init__(env)
        dt_ns = int(str_to_ns(timestep_duration))
        self.tracker = KimFilterTracker(
            r_bar=r_bar, kappa_oracle=kappa_oracle, fund_vol=fund_vol,
            dt_ns=dt_ns, obs_noise_frac=obs_noise_frac,
            regime_stay_prob=regime_stay_prob,
        )
        self.r_bar      = float(r_bar)
        self._mid_price = float(r_bar)

        n_extra = 5
        self.observation_space = gym.spaces.Box(
            low  = np.vstack([env.observation_space.low,
                              np.full((n_extra, 1), -20.0, dtype=np.float32)]),
            high = np.vstack([env.observation_space.high,
                              np.full((n_extra, 1),  20.0, dtype=np.float32)]),
            dtype=np.float32,
        )

    def reset(self):
        obs = self.env.reset()
        self.env.previous_marked_to_market = self.env.starting_cash
        self.tracker.reset()
        self._mid_price = self.r_bar
        mu, sigma, probs = self.tracker.step(self._mid_price)
        return self._build_obs(obs, mu, sigma, probs)

    def step(self, action: int):
        obs, reward, done, info = self.env.step(action)
        self._update_mid(info)
        mu, sigma, probs = self.tracker.step(self._mid_price)
        return self._build_obs(obs, mu, sigma, probs), reward, done, info

    def _update_mid(self, info: dict) -> None:
        bid = info.get("best_bid")
        ask = info.get("best_ask")
        if bid is not None and ask is not None and ask > bid:
            self._mid_price = (float(bid) + float(ask)) / 2.0

    def _build_obs(self, obs, mu, sigma, probs) -> np.ndarray:
        belief_dev = np.float32((mu - self._mid_price) / self.r_bar)
        belief_std = np.float32(sigma / self.r_bar)
        extra = np.array([[belief_dev],
                          [np.float32(probs[0])],   # p_value
                          [np.float32(probs[1])],   # p_momentum
                          [np.float32(probs[2])],   # p_noise
                          [belief_std]], dtype=np.float32)
        return np.vstack([obs, extra])
