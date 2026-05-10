"""
belief_tracker.py — Belief state estimation for ABIDES daily investor env
==========================================================================

Two tools:

  KalmanBeliefTracker
    Optimal Bayesian filter for the OU fundamental process.
    Discretizes the continuous OU model, then runs predict/update each step.

  BeliefAugmentedWrapper
    gym.Wrapper that appends two belief features to the observation:
      belief_dev  = (mu_t - mid_price_t) / r_bar   — signed deviation of
                    estimated fundamental from current market price.
                    Positive → fundamental above market → bullish signal.
      belief_std  = sqrt(sigma2_t) / r_bar           — posterior uncertainty.

  Why not LSTM instead?
    An LSTM *can* learn an implicit belief representation end-to-end and is
    especially useful when kappa/fund_vol are unknown (as in the sweep).
    But the Kalman filter is the exact closed-form solution for the linear-
    Gaussian OU model — zero training, interpretable, and a strong baseline.
    The recommended sequence:
      1. Add Kalman belief features → re-run baselines / train MLP policy.
      2. If the sweep shows regime-sensitivity, switch policy to LSTM so it
         can adapt kappa/fund_vol from in-episode price history.

Usage
-----
  # Standalone tracker
  tracker = KalmanBeliefTracker(r_bar=100_000, kappa=1.67e-16,
                                 fund_vol=5e-5, dt_ns=60_000_000_000)
  tracker.reset()
  mu, sigma = tracker.step(mid_price)

  # Wrapper (requires debug_mode=True for mid-price recovery via info dict)
  import gym, abides_gym
  env = gym.make("markets-daily_investor-v0",
                 background_config="rmsc04",
                 debug_mode=True)
  env = BeliefAugmentedWrapper(env, r_bar=100_000, kappa_oracle=1.67e-16,
                               fund_vol=5e-5, timestep_duration="60s")

  obs = env.reset()           # shape (num_features + 2, 1)
  obs, r, done, info = env.step(1)
"""

import numpy as np
import gym
from abides_core.utils import str_to_ns


# ────────────────────────────────────────────────────────────────────────────
# Kalman filter for the OU fundamental
# ────────────────────────────────────────────────────────────────────────────

class KalmanBeliefTracker:
    """
    Discrete-time Kalman filter for the Ornstein-Uhlenbeck fundamental.

    State model  (per timestep of dt_s seconds):
        F_{t+1} = alpha * F_t + (1 - alpha) * r_bar + w_t,   w_t ~ N(0, Q)

    Observation model:
        y_t = F_t + v_t,   v_t ~ N(0, R)

    where y_t is the observed mid-price and R absorbs market microstructure
    noise (half-spread squared is a reasonable proxy).

    Parameters
    ----------
    r_bar : int
        Long-run fundamental mean (cents). Default RMSC04: 100_000.
    kappa : float
        OU mean-reversion rate of the *oracle* process (kappa_oracle).
        RMSC04 default: 1.67e-16. Very small → near random walk at 60s steps.
    fund_vol : float
        Oracle fundamental volatility (σ in the SDE). RMSC04: 5e-5.
        Process noise in price levels: Q ≈ (r_bar * fund_vol)^2 * dt_s.
    dt_ns : int
        Timestep duration in nanoseconds (e.g. 60_000_000_000 for 60s).
    obs_noise_frac : float
        Observation noise as a fraction of r_bar. Default 0.005 (0.5%).
        Set to (half_spread / r_bar) if you know the typical spread.
    """

    def __init__(
        self,
        r_bar: int,
        kappa: float,
        fund_vol: float,
        dt_ns: int,
        obs_noise_frac: float = 0.005,
    ):
        dt_s = dt_ns / 1e9
        self.r_bar = float(r_bar)

        # OU discretization: F_{t+1} = alpha*F_t + (1-alpha)*r_bar + noise
        self.alpha = float(np.exp(-kappa * dt_s))

        # Process noise variance Q
        # Continuous SDE: dF = κ(r̄-F)dt + σ*F dW  (multiplicative vol)
        # In levels, per-step noise ≈ (r_bar * fund_vol)^2 * dt_s
        # For tiny kappa (≈ random walk), this matches the exact formula.
        if kappa > 1e-30:
            # Exact formula for OU: Q = σ²_level / (2κ) * (1 - α²)
            sigma2_level = (r_bar * fund_vol) ** 2
            self.Q = float(sigma2_level / (2 * kappa) * (1 - self.alpha ** 2))
        else:
            self.Q = float((r_bar * fund_vol) ** 2 * dt_s)

        # Observation noise variance R
        self.R = float((r_bar * obs_noise_frac) ** 2)

        # Posterior state (reset sets these to prior)
        self.mu: float = float(r_bar)
        self.sigma2: float = self._stationary_var()

    def _stationary_var(self) -> float:
        denom = 1.0 - self.alpha ** 2
        return self.Q / denom if denom > 1e-30 else self.Q * 1e30

    def reset(self) -> None:
        """Call at the start of each episode."""
        self.mu = self.r_bar
        self.sigma2 = self._stationary_var()

    def step(self, mid_price: float):
        """
        One Kalman predict + update cycle.

        Parameters
        ----------
        mid_price : float
            Observed mid-price (cents) at this timestep.

        Returns
        -------
        mu : float
            Posterior mean of the fundamental.
        sigma : float
            Posterior standard deviation.
        """
        # Predict
        mu_pred = self.alpha * self.mu + (1.0 - self.alpha) * self.r_bar
        s2_pred = self.alpha ** 2 * self.sigma2 + self.Q

        # Update (Kalman gain)
        K = s2_pred / (s2_pred + self.R)
        self.mu = mu_pred + K * (mid_price - mu_pred)
        self.sigma2 = (1.0 - K) * s2_pred

        return self.mu, float(np.sqrt(self.sigma2))

    @property
    def belief(self):
        return self.mu, float(np.sqrt(self.sigma2))


# ────────────────────────────────────────────────────────────────────────────
# Gym wrapper
# ────────────────────────────────────────────────────────────────────────────

class BeliefAugmentedWrapper(gym.Wrapper):
    """
    Augments the daily-investor observation with two Kalman belief features.

    New obs (shape: (original_features + 2, 1)):
        [...original obs...,
         belief_dev,   # (mu_t - mid_price_t) / r_bar
         belief_std]   # sqrt(sigma2_t) / r_bar

    belief_dev > 0 means the tracker thinks the fundamental is ABOVE the
    current market price — a bullish signal for a value investor.

    Requirements
    ------------
    - env must be created with debug_mode=True (needed to get best_bid /
      best_ask from info dict to reconstruct mid-price accurately).
    - If debug_mode=False the wrapper falls back to integrating the return
      from obs[4], anchored at r_bar — less accurate but functional.

    Parameters
    ----------
    env : gym.Env
        An unwrapped markets-daily_investor-v0 instance.
    r_bar, kappa_oracle, fund_vol : float
        OU parameters matching the background config.
    timestep_duration : str
        Timestep string, e.g. "60s". Must match the env's timestep.
    obs_noise_frac : float
        Passed to KalmanBeliefTracker. Tune to match typical half-spread.
    """

    def __init__(
        self,
        env: gym.Env,
        r_bar: int = 100_000,
        kappa_oracle: float = 1.67e-16,
        fund_vol: float = 5e-5,
        timestep_duration: str = "60s",
        obs_noise_frac: float = 0.005,
    ):
        super().__init__(env)
        dt_ns = int(str_to_ns(timestep_duration))
        self.tracker = KalmanBeliefTracker(
            r_bar=r_bar,
            kappa=kappa_oracle,
            fund_vol=fund_vol,
            dt_ns=dt_ns,
            obs_noise_frac=obs_noise_frac,
        )
        self.r_bar = float(r_bar)
        self._mid_price = self.r_bar  # running mid-price level

        # Extend observation space: add 2 rows to (n, 1) array
        old_low = env.observation_space.low    # (n, 1)
        old_high = env.observation_space.high  # (n, 1)
        extra_low = np.full((2, 1), -20.0, dtype=np.float32)
        extra_high = np.full((2, 1), 20.0, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.vstack([old_low, extra_low]),
            high=np.vstack([old_high, extra_high]),
            dtype=np.float32,
        )

    def reset(self):
        obs = self.env.reset()
        # Bug fix: must set previous_marked_to_market after every reset
        self.env.previous_marked_to_market = self.env.starting_cash
        self.tracker.reset()
        self._mid_price = self.r_bar
        # No info on reset — use r_bar as the initial mid-price observation
        mu, sigma = self.tracker.step(self._mid_price)
        return self._build_obs(obs, mu, sigma)

    def step(self, action: int):
        obs, reward, done, info = self.env.step(action)
        self._update_mid_price(obs, info)
        mu, sigma = self.tracker.step(self._mid_price)
        return self._build_obs(obs, mu, sigma), reward, done, info

    # ── helpers ──────────────────────────────────────────────────────────────

    def _update_mid_price(self, obs: np.ndarray, info: dict) -> None:
        """
        Recover mid-price from info (preferred) or from cumulative returns.
        """
        bid = info.get("best_bid")
        ask = info.get("best_ask")
        if bid is not None and ask is not None and ask > bid:
            self._mid_price = (float(bid) + float(ask)) / 2.0
        else:
            # Fallback: add the most-recent return (obs[4, 0] if available)
            if obs.shape[0] > 4:
                self._mid_price += float(obs[4, 0])

    def _build_obs(self, obs: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        belief_dev = np.float32((mu - self._mid_price) / self.r_bar)
        belief_std = np.float32(sigma / self.r_bar)
        extra = np.array([[belief_dev], [belief_std]], dtype=np.float32)
        return np.vstack([obs, extra])


# ────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ────────────────────────────────────────────────────────────────────────────

def _smoke_test():
    """Run one short episode and print belief feature trajectory."""
    import gym
    import abides_gym  # noqa: F401

    base_env = gym.make(
        "markets-daily_investor-v0",
        background_config="rmsc04",
        debug_mode=True,
        timestep_duration="60s",
    )
    env = BeliefAugmentedWrapper(
        base_env,
        r_bar=100_000,
        kappa_oracle=1.67e-16,
        fund_vol=5e-5,
        timestep_duration="60s",
    )

    env.seed(42)
    obs = env.reset()
    print(f"obs shape: {obs.shape}")
    print(f"step 0  belief_dev={obs[-2, 0]:+.6f}  belief_std={obs[-1, 0]:.6f}")

    for step in range(5):
        action = 1  # hold
        obs, reward, done, info = env.step(action)
        mid = info.get("best_bid", 0) / 2 + info.get("best_ask", 0) / 2
        print(
            f"step {step+1}  mid={mid:.0f}  "
            f"mu={env.tracker.mu:.1f}  "
            f"belief_dev={obs[-2, 0]:+.6f}  "
            f"belief_std={obs[-1, 0]:.6f}"
        )
        if done:
            break

    env.close()
    print("smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
