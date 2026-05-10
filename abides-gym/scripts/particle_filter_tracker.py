"""
particle_filter_tracker.py — Bootstrap particle filter for OU fundamental
=========================================================================

Maintains N weighted particles {F_t^(i), w_t^(i)} to approximate the full
posterior p(F_t | y_{1:t}) over the latent fundamental.

Advantages over Kalman
----------------------
  • Non-Gaussian: handles fat-tailed microstructure noise and
    multi-modal posteriors that arise after megashocks.
  • Megashock support: occasional large jumps in F_t (modelled in RMSC04
    via megashock_lambda_a) are correctly propagated through the particle
    cloud, whereas the Kalman filter just treats them as large innovations
    and recovers slowly.
  • No linearity assumption: if the observation model is actually
    non-linear (e.g. mid-price is the median of the LOB, not the mean),
    the particle filter remains exact.

Algorithm (bootstrap / SIR)
----------------------------
  Each step:
  1. Systematic resample: draw N particles from current weighted distribution.
  2. Propagate: each particle evolves under the OU dynamics + optional megashock.
  3. Weight: w^(i) ∝ p(y_t | F_t^(i)) = N(y_t; F_t^(i), R).
  4. Normalise weights.
  5. Compute posterior mean and std from weighted cloud.

Systematic resampling (vs multinomial) reduces variance and is O(N).

Parameters
----------
n_particles : int
    Number of particles (default 300). Accurate from ~100 for 1-D state.
megashock_prob : float
    Per-step probability of a megashock (matches RMSC04 megashock_lambda_a
    if set to megashock_lambda_a * dt_s). Default 0 (disabled).
megashock_mean, megashock_std : float
    Mean and std of the shock magnitude in cents.
    RMSC04 defaults: mean=1000, var=50000 → std≈224.
"""

import numpy as np
import gym
from abides_core.utils import str_to_ns


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────

class ParticleFilterTracker:
    """
    Bootstrap (SIR) particle filter for the OU fundamental process.

    Parameters
    ----------
    r_bar, kappa_oracle, fund_vol, dt_ns
        Same OU calibration used by KalmanBeliefTracker.
    n_particles : int
        Particle count. 200–500 is accurate and fast for 1-D state.
    obs_noise_frac : float
        Observation noise as a fraction of r_bar (same default as Kalman).
    megashock_prob, megashock_mean, megashock_std : float
        Optional megashock model matching RMSC04 parameters.
    """

    def __init__(
        self,
        r_bar: int            = 100_000,
        kappa_oracle: float   = 1.67e-16,
        fund_vol: float       = 5e-5,
        dt_ns: int            = 60_000_000_000,
        n_particles: int      = 300,
        obs_noise_frac: float = 0.005,
        megashock_prob: float = 0.0,
        megashock_mean: float = 1_000.0,
        megashock_std: float  = 224.0,
    ):
        dt_s = dt_ns / 1e9
        self.r_bar  = float(r_bar)
        self.alpha  = float(np.exp(-kappa_oracle * dt_s))
        self.Q_std  = float(np.sqrt((r_bar * fund_vol)**2 * dt_s))
        self.R_std  = float(r_bar * obs_noise_frac)
        self.N      = n_particles

        # Megashock
        self.shock_prob = megashock_prob
        self.shock_mean = megashock_mean
        self.shock_std  = megashock_std

        # Particle cloud (initialised in reset)
        self.particles: np.ndarray = np.full(n_particles, float(r_bar))
        self.weights:   np.ndarray = np.ones(n_particles) / n_particles

    # ── helpers ───────────────────────────────────────────────────────────────

    def _stationary_std(self) -> float:
        """Stationary std of the OU process (approximate for tiny kappa)."""
        var_ou = self.Q_std**2
        denom  = 1.0 - self.alpha**2
        return float(np.sqrt(var_ou / denom)) if denom > 1e-12 else float(self.Q_std * 1e6)

    def _systematic_resample(self) -> np.ndarray:
        """O(N) systematic resampling — lower variance than multinomial."""
        N   = self.N
        pos = (np.arange(N) + np.random.uniform()) / N
        cum = np.cumsum(self.weights)
        idx = np.searchsorted(cum, pos)
        return self.particles[np.clip(idx, 0, N - 1)].copy()

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Initialise particle cloud near r_bar with spread = R_std.

        We intentionally avoid using the OU stationary std here: for near-
        random-walk kappa (kappa_oracle = 1.67e-16), the stationary variance
        is ~2.7e8 cents — far larger than any sensible prior at episode start.
        R_std (~500 cents) is a tight but reasonable initial uncertainty that
        lets the filter converge in a few steps rather than suffering immediate
        weight degeneracy.
        """
        self.particles = np.random.normal(self.r_bar, self.R_std, self.N)
        self.weights   = np.ones(self.N) / self.N

    def step(self, mid_price: float):
        """
        One SIR step: resample → propagate → weight → normalise.

        Returns
        -------
        mu    : float — posterior mean
        sigma : float — posterior std
        """
        # 1. Systematic resample
        self.particles = self._systematic_resample()

        # 2. Propagate each particle through OU dynamics
        noise = np.random.normal(0.0, self.Q_std, self.N)
        self.particles = (self.alpha * self.particles
                          + (1.0 - self.alpha) * self.r_bar
                          + noise)

        # Optional megashocks: signed impulse applied to each particle
        # independently with probability shock_prob per step.
        if self.shock_prob > 0.0:
            mask   = np.random.random(self.N) < self.shock_prob
            shocks = np.random.normal(self.shock_mean, self.shock_std, self.N)
            signs  = np.random.choice([-1.0, 1.0], self.N)
            self.particles += mask * shocks * signs

        # 3. Importance weights: p(y_t | F_t^(i)) = N(y_t; F_t^(i), R)
        log_w  = -0.5 * ((mid_price - self.particles) / self.R_std) ** 2
        log_w -= log_w.max()                  # shift for numerical stability
        w      = np.exp(log_w)
        w_sum  = w.sum()
        if w_sum < 1e-300:                    # catastrophic degeneracy — reinit
            self.reset()
            return self.r_bar, self._stationary_std()
        self.weights = w / w_sum

        # 4. Posterior statistics
        mu    = float(self.weights @ self.particles)
        var   = float(self.weights @ (self.particles - mu) ** 2)
        sigma = float(np.sqrt(max(var, 0.0)))
        return mu, sigma

    @property
    def effective_sample_size(self) -> float:
        """ESS = 1 / sum(w^2). Drops toward 1 at severe degeneracy."""
        return float(1.0 / (self.weights ** 2).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Gym wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ParticleBeliefWrapper(gym.Wrapper):
    """
    Augments the observation with 2 particle-filter belief features:

      obs[-2] = belief_dev = (mu_t - mid_t) / r_bar
      obs[-1] = belief_std = sigma_t / r_bar

    Same obs contract as BeliefAugmentedWrapper — drop-in replacement.
    """

    def __init__(
        self,
        env: gym.Env,
        r_bar: int            = 100_000,
        kappa_oracle: float   = 1.67e-16,
        fund_vol: float       = 5e-5,
        timestep_duration: str = "60s",
        n_particles: int       = 300,
        obs_noise_frac: float  = 0.005,
        megashock_prob: float  = 0.0,
        megashock_mean: float  = 1_000.0,
        megashock_std: float   = 224.0,
    ):
        super().__init__(env)
        dt_ns = int(str_to_ns(timestep_duration))
        self.tracker = ParticleFilterTracker(
            r_bar=r_bar, kappa_oracle=kappa_oracle, fund_vol=fund_vol,
            dt_ns=dt_ns, n_particles=n_particles,
            obs_noise_frac=obs_noise_frac,
            megashock_prob=megashock_prob,
            megashock_mean=megashock_mean,
            megashock_std=megashock_std,
        )
        self.r_bar      = float(r_bar)
        self._mid_price = float(r_bar)

        self.observation_space = gym.spaces.Box(
            low  = np.vstack([env.observation_space.low,
                              np.full((2, 1), -20.0, dtype=np.float32)]),
            high = np.vstack([env.observation_space.high,
                              np.full((2, 1),  20.0, dtype=np.float32)]),
            dtype=np.float32,
        )

    def reset(self):
        obs = self.env.reset()
        self.env.previous_marked_to_market = self.env.starting_cash
        self.tracker.reset()
        self._mid_price = self.r_bar
        mu, sigma = self.tracker.step(self._mid_price)
        return self._build_obs(obs, mu, sigma)

    def step(self, action: int):
        obs, reward, done, info = self.env.step(action)
        self._update_mid(info)
        mu, sigma = self.tracker.step(self._mid_price)
        return self._build_obs(obs, mu, sigma), reward, done, info

    def _update_mid(self, info: dict) -> None:
        bid = info.get("best_bid")
        ask = info.get("best_ask")
        if bid is not None and ask is not None and ask > bid:
            self._mid_price = (float(bid) + float(ask)) / 2.0

    def _build_obs(self, obs: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        belief_dev = np.float32((mu - self._mid_price) / self.r_bar)
        belief_std = np.float32(sigma / self.r_bar)
        extra = np.array([[belief_dev], [belief_std]], dtype=np.float32)
        return np.vstack([obs, extra])
