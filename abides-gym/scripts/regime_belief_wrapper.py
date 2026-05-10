"""
regime_belief_wrapper.py — Regime belief wrappers for v5 oracle experiments
============================================================================

Four drop-in components for the PPO training pipeline:

  OracleRegimeWrapper (E1: belief in observation)
    Appends the hard one-hot oracle label b* = (p_val, p_mom) to every
    observation in the episode.  The label is set at reset() from the
    environment's background parameters and is constant for the episode.

  KimRegimeBeliefWrapper (E3 attempt — NOTE: does not distinguish regimes)
    Appends a live soft 2-vector (p_val_t, p_mom_t) produced by the Kim
    regime-switching Kalman filter at every timestep.  Found to produce
    identical output in both Val and Mom regimes; kept for reference only.

  TrainedRegimeBeliefWrapper (E3: learned inferred belief)
    Appends (p_val_t, p_mom_t) from a trained MLP or LSTM classifier.
    The classifier learns regime signatures from autocorrelation patterns
    and order-flow imbalance.  MLP+Kalman achieves 100% episode accuracy.
    Model is loaded from a checkpoint saved by regime_classifier.py.

  RegimeRandomizedGymnasiumEnv
    Gymnasium-compatible adapter that randomises the market regime (Val or
    Mom) independently at every reset().  Combines regime randomisation,
    belief augmentation, and the Gymnasium API conversion needed by RLlib.
    Accepts belief_mode in {"oracle", "kim", "trained", "none"}.

Regime definitions (from v5 sweep data)
----------------------------------------
  Val  (value-dominated):  num_momentum_agents=0,  num_value_agents=102
  Mom  (default AAPL):     num_momentum_agents=12, num_value_agents=102

Observation layout after augmentation
--------------------------------------
  belief_mode="none"    : obs shape (7,) — unchanged
  belief_mode="oracle"  : obs shape (9,) — [...o_t, p_val, p_mom]
  belief_mode="kim"     : obs shape (9,) — [...o_t, p_val_t, p_mom_t]
  belief_mode="trained" : obs shape (9,) — [...o_t, p_val_t, p_mom_t]

Usage (standalone)
------------------
  env = RegimeRandomizedGymnasiumEnv({
      "belief_mode": "trained",
      "trained_model_path": "models/mlp_kalman_regime.pt",
  })
  obs, info = env.reset()   # obs shape (9,)
  obs, r, term, trunc, info = env.step(1)

Usage with RLlib (train_ppo_daily_investor.py)
----------------------------------------------
  Replace GymnasiumDailyInvestorAdapter with RegimeRandomizedGymnasiumEnv
  in the PPOConfig .environment() call and pass belief_mode in env_config.
"""

from __future__ import annotations

import sys
import os
# Ensure scripts dir is on path for relative imports (kim_filter_tracker).
sys.path.insert(0, os.path.dirname(__file__))

from typing import Any, Dict, Optional, Tuple

import gym
import gymnasium
import numpy as np

import abides_gym  # noqa: F401  — registers markets-daily_investor-v0

from abides_gym.envs.markets_daily_investor_environment_v0 import (
    SubGymMarketsDailyInvestorEnv_v0,
)
from kim_filter_tracker import KimFilterTracker
from regime_classifier import TrainedRegimeBeliefWrapper, load_classifier
from abides_core.utils import str_to_ns


# ─────────────────────────────────────────────────────────────────────────────
# Regime definitions (v5 Table 3)
# ─────────────────────────────────────────────────────────────────────────────

REGIME_VAL = {"num_momentum_agents": 0,  "num_value_agents": 102}
REGIME_MOM = {"num_momentum_agents": 12, "num_value_agents": 102}

# Hard one-hot oracle belief for each regime
ORACLE_BELIEF: Dict[str, np.ndarray] = {
    "val": np.array([1.0, 0.0], dtype=np.float32),
    "mom": np.array([0.0, 1.0], dtype=np.float32),
}

N_BELIEF_FEATURES = 2   # (p_val, p_mom)


# ─────────────────────────────────────────────────────────────────────────────
# OracleRegimeWrapper  (E1: belief in observation)
# ─────────────────────────────────────────────────────────────────────────────

class OracleRegimeWrapper(gym.Wrapper):
    """
    Appends the hard oracle belief b* = (p_val, p_mom) to every observation.

    The label is derived from the environment's background parameters at
    reset() time and is fixed for the whole episode — the agent sees a
    constant 2-vector that tells it exactly which regime it is in.

    Parameters
    ----------
    env : gym.Env
        A SubGymMarketsDailyInvestorEnv_v0 instance (or wrapper around one).
    regime : str
        "val" or "mom".  Set externally by RegimeRandomizedGymnasiumEnv.
    """

    def __init__(self, env: gym.Env, regime: str = "mom"):
        super().__init__(env)
        self._regime = regime
        self._belief = ORACLE_BELIEF[regime].copy()

        old_low  = env.observation_space.low
        old_high = env.observation_space.high
        extra_low  = np.full((N_BELIEF_FEATURES, 1), 0.0,  dtype=np.float32)
        extra_high = np.full((N_BELIEF_FEATURES, 1), 1.0,  dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.vstack([old_low,  extra_low]),
            high=np.vstack([old_high, extra_high]),
            dtype=np.float32,
        )

    def set_regime(self, regime: str) -> None:
        self._regime = regime
        self._belief = ORACLE_BELIEF[regime].copy()

    def reset(self):
        obs = self.env.reset()
        self.env.previous_marked_to_market = self.env.starting_cash
        return self._augment(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._augment(obs), reward, done, info

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        extra = self._belief.reshape(N_BELIEF_FEATURES, 1).astype(np.float32)
        return np.vstack([obs, extra])


# ─────────────────────────────────────────────────────────────────────────────
# KimRegimeBeliefWrapper  (E3: inferred belief)
# ─────────────────────────────────────────────────────────────────────────────

class KimRegimeBeliefWrapper(gym.Wrapper):
    """
    Appends a live soft 2-vector (p_val_t, p_mom_t) from the Kim filter.

    The Kim filter runs K=3 regimes (VALUE / MOMENTUM / NOISE).  We project
    to the 2D v5 belief by:

        p_val_t = p_value_kim          (Kim VALUE ↔ v5 Val)
        p_mom_t = p_momentum_kim       (Kim MOMENTUM ↔ v5 Mom)

    These do not sum to 1 when p_noise > 0, but that is informative:
    a high noise-regime probability means the tracker is uncertain about
    which market structure is active.  The policy can learn to hold when
    (p_val + p_mom) is small.

    To get a proper simplex (useful if you want to enforce sum=1), pass
    normalize=True — this sets p_val = p_value / (p_value + p_momentum)
    and p_mom = p_momentum / (p_value + p_momentum), setting both to 0.5
    when Kim is in pure noise.

    Parameters
    ----------
    env : gym.Env
    r_bar, kappa_oracle, fund_vol, timestep_duration, obs_noise_frac,
    regime_stay_prob : floats
        Passed to KimFilterTracker.  Match the background config.
    normalize : bool
        If True, renormalize to drop the noise regime (sum to 1).
    """

    def __init__(
        self,
        env: gym.Env,
        r_bar: int             = 100_000,
        kappa_oracle: float    = 1.67e-16,
        fund_vol: float        = 5e-5,
        timestep_duration: str = "60s",
        obs_noise_frac: float  = 0.005,
        regime_stay_prob: float = 0.97,
        normalize: bool        = False,
    ):
        super().__init__(env)
        dt_ns = int(str_to_ns(timestep_duration))
        self.tracker = KimFilterTracker(
            r_bar=r_bar,
            kappa_oracle=kappa_oracle,
            fund_vol=fund_vol,
            dt_ns=dt_ns,
            obs_noise_frac=obs_noise_frac,
            regime_stay_prob=regime_stay_prob,
        )
        self.r_bar      = float(r_bar)
        self._mid_price = float(r_bar)
        self._normalize = normalize

        old_low  = env.observation_space.low
        old_high = env.observation_space.high
        extra_low  = np.full((N_BELIEF_FEATURES, 1), 0.0, dtype=np.float32)
        extra_high = np.full((N_BELIEF_FEATURES, 1), 1.0, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.vstack([old_low,  extra_low]),
            high=np.vstack([old_high, extra_high]),
            dtype=np.float32,
        )

    def reset(self):
        obs = self.env.reset()
        self.env.previous_marked_to_market = self.env.starting_cash
        self.tracker.reset()
        self._mid_price = self.r_bar
        _, _, probs = self.tracker.step(self._mid_price)
        return self._augment(obs, probs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._update_mid(info)
        _, _, probs = self.tracker.step(self._mid_price)
        return self._augment(obs, probs), reward, done, info

    def _update_mid(self, info: dict) -> None:
        bid = info.get("best_bid")
        ask = info.get("best_ask")
        if bid is not None and ask is not None and ask > bid:
            self._mid_price = (float(bid) + float(ask)) / 2.0

    def _augment(self, obs: np.ndarray, probs: np.ndarray) -> np.ndarray:
        p_val = float(probs[0])   # Kim VALUE   → v5 Val
        p_mom = float(probs[1])   # Kim MOMENTUM → v5 Mom
        if self._normalize:
            denom = p_val + p_mom
            if denom > 1e-8:
                p_val /= denom
                p_mom /= denom
            else:
                p_val = p_mom = 0.5
        extra = np.array([[np.float32(p_val)],
                          [np.float32(p_mom)]], dtype=np.float32)
        return np.vstack([obs, extra])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared by the Gymnasium adapter
# ─────────────────────────────────────────────────────────────────────────────

def _to_gymnasium_space(space):
    if isinstance(space, gym.spaces.Discrete):
        return gymnasium.spaces.Discrete(space.n)
    if isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(
            low=space.low, high=space.high,
            shape=space.shape, dtype=space.dtype,
        )
    raise TypeError(f"Unsupported space type: {type(space)!r}")


# ─────────────────────────────────────────────────────────────────────────────
# RegimeRandomizedGymnasiumEnv  (RLlib-compatible)
# ─────────────────────────────────────────────────────────────────────────────

class RegimeRandomizedGymnasiumEnv(gymnasium.Env):
    """
    Gymnasium-compatible env that:
      1. Randomly picks Val or Mom regime on every reset().
      2. Applies a belief wrapper (oracle / kim / trained / none).
      3. Converts the legacy gym API to gymnasium's (reset/step return values).

    Required env_config keys
    ------------------------
    belief_mode : str
        "oracle"  — hard one-hot per episode (E1).
        "kim"     — soft Kim filter per step (reference; does not work well).
        "trained" — MLP/LSTM classifier loaded from trained_model_path (E3).
        "none"    — no belief augmentation (baseline).

    Optional env_config keys
    ------------------------
    val_prob : float (default 0.5)
        Probability of drawing a Val episode.  0.5 = balanced training.
    trained_model_path : str
        Path to a .pt checkpoint saved by regime_classifier.py.
        Required when belief_mode="trained".
    r_bar, kappa_oracle, fund_vol, timestep_duration, obs_noise_frac,
    regime_stay_prob, kim_normalize : forwarded to KimRegimeBeliefWrapper.
    starting_cash, order_fixed_size, state_history_length,
    market_data_buffer_length, first_interval, reward_mode, done_ratio,
    debug_mode, mkt_close : forwarded to SubGymMarketsDailyInvestorEnv_v0.

    Info dict extras
    ----------------
    Every step() info dict gains two keys:
      "regime"       : "val" or "mom"
      "belief"       : np.ndarray shape (2,) — the appended (p_val, p_mom)
    """

    metadata = {"render_modes": []}

    # Default env kwargs (match v4 / v5 protocol)
    _ENV_DEFAULTS: Dict[str, Any] = dict(
        background_config          = "rmsc04",
        mkt_close                  = "16:00:00",
        timestep_duration          = "60s",
        starting_cash              = 1_000_000,
        order_fixed_size           = 10,
        state_history_length       = 4,
        market_data_buffer_length  = 5,
        first_interval             = "00:05:00",
        reward_mode                = "dense",
        done_ratio                 = 0.3,
        debug_mode                 = True,
    )

    # Default RMSC04 background params (everything except n_mom / n_val)
    _BG_DEFAULTS: Dict[str, Any] = dict(
        r_bar                  = 100_000,
        kappa                  = 1.67e-15,
        lambda_a               = 5.7e-12,
        kappa_oracle           = 1.67e-16,
        sigma_s                = 0,
        fund_vol               = 5e-5,
        megashock_lambda_a     = 2.77778e-18,
        megashock_mean         = 1000,
        megashock_var          = 50_000,
        mm_window_size         = "adaptive",
        mm_pov                 = 0.025,
        mm_num_ticks           = 10,
        mm_wake_up_freq        = "60S",
        mm_min_order_size      = 1,
        mm_skew_beta           = 0,
        mm_price_skew          = 4,
        mm_level_spacing       = 5,
        mm_spread_alpha        = 0.75,
        mm_backstop_quantity   = 0,
        mm_cancel_limit_delay  = 50,
        num_noise_agents       = 1000,
    )

    def __init__(self, env_config: Dict[str, Any]):
        cfg = dict(env_config)

        self._belief_mode  = str(cfg.pop("belief_mode", "none")).lower()
        self._val_prob     = float(cfg.pop("val_prob", 0.5))
        self._info_mode    = str(cfg.pop("adapter_info_mode", "full")).lower()

        # Kim-specific overrides
        self._r_bar             = float(cfg.pop("r_bar",             100_000))
        self._kappa_oracle      = float(cfg.pop("kappa_oracle",      1.67e-16))
        self._fund_vol          = float(cfg.pop("fund_vol",          5e-5))
        self._timestep_duration = str(cfg.pop("timestep_duration",   "60s"))
        self._obs_noise_frac    = float(cfg.pop("obs_noise_frac",    0.005))
        self._regime_stay_prob  = float(cfg.pop("regime_stay_prob",  0.97))
        self._kim_normalize     = bool(cfg.pop("kim_normalize",      False))

        # Env construction kwargs
        self._env_kwargs = dict(self._ENV_DEFAULTS)
        self._env_kwargs["timestep_duration"] = self._timestep_duration
        # Absorb any remaining keys that the env constructor accepts
        for k in list(cfg.keys()):
            if k in self._env_kwargs or k in (
                "debug_mode", "reward_mode", "done_ratio",
                "starting_cash", "order_fixed_size",
            ):
                self._env_kwargs[k] = cfg.pop(k)

        # BG extra kwargs (regime params are added at reset time)
        self._bg_kwargs = dict(self._BG_DEFAULTS)
        self._bg_kwargs["r_bar"]        = int(self._r_bar)
        self._bg_kwargs["kappa_oracle"] = self._kappa_oracle
        self._bg_kwargs["fund_vol"]     = self._fund_vol
        for k in list(cfg.keys()):
            if k in self._bg_kwargs:
                self._bg_kwargs[k] = cfg.pop(k)

        # Trained classifier (loaded once, shared across all envs)
        trained_model_path = str(cfg.pop("trained_model_path", ""))
        self._trained_model      = None
        self._trained_model_type = "mlp"
        self._trained_use_kalman = True
        self._trained_seq_len    = 20
        if self._belief_mode == "trained":
            if not trained_model_path:
                raise ValueError("belief_mode='trained' requires trained_model_path in env_config")
            (self._trained_model,
             self._trained_model_type,
             self._trained_use_kalman,
             self._trained_seq_len) = load_classifier(trained_model_path)

        self._regime: str           = "mom"
        self._belief: np.ndarray    = ORACLE_BELIEF["mom"].copy()
        self._wrapped_env: Optional[gym.Env] = None

        # Probe the BASE env (no belief wrapper) to get raw obs dimension, then add belief features.
        # Using _make_env() here would double-count because wrappers expand the obs space.
        _probe_kw = dict(self._env_kwargs)
        _probe_kw["background_config_extra_kvargs"] = {**self._bg_kwargs, **REGIME_MOM}
        _probe_base = SubGymMarketsDailyInvestorEnv_v0(**_probe_kw)
        raw_obs_dim = int(np.prod(_probe_base.observation_space.shape))
        _probe_base.close()

        obs_dim = raw_obs_dim
        if self._belief_mode in ("oracle", "kim", "trained"):
            obs_dim += N_BELIEF_FEATURES

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Discrete(3)

    # ── env construction ──────────────────────────────────────────────────────

    def _make_env(self, regime: str) -> gym.Env:
        """Build a fresh base env for the given regime."""
        bg = dict(self._bg_kwargs)
        bg.update(REGIME_VAL if regime == "val" else REGIME_MOM)

        kw = dict(self._env_kwargs)
        kw["background_config_extra_kvargs"] = bg

        base = SubGymMarketsDailyInvestorEnv_v0(**kw)

        if self._belief_mode == "oracle":
            return OracleRegimeWrapper(base, regime=regime)
        elif self._belief_mode == "kim":
            return KimRegimeBeliefWrapper(
                base,
                r_bar              = int(self._r_bar),
                kappa_oracle       = self._kappa_oracle,
                fund_vol           = self._fund_vol,
                timestep_duration  = self._timestep_duration,
                obs_noise_frac     = self._obs_noise_frac,
                regime_stay_prob   = self._regime_stay_prob,
                normalize          = self._kim_normalize,
            )
        elif self._belief_mode == "trained":
            return TrainedRegimeBeliefWrapper(
                base,
                model       = self._trained_model,
                model_type  = self._trained_model_type,
                seq_len     = self._trained_seq_len,
                use_kalman  = self._trained_use_kalman,
            )
        else:
            return base   # belief_mode="none"

    # ── gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options=None):
        # Close previous env
        if self._wrapped_env is not None:
            try:
                self._wrapped_env.close()
            except Exception:
                pass

        # Sample regime
        rng = np.random.default_rng(seed)
        self._regime = "val" if rng.random() < self._val_prob else "mom"

        self._wrapped_env = self._make_env(self._regime)
        if seed is not None:
            self._wrapped_env.seed(seed)

        raw_obs = self._wrapped_env.reset()
        self._belief = self._extract_belief(raw_obs)

        return self._flatten(raw_obs), {}

    def step(self, action):
        raw_obs, reward, done, info = self._wrapped_env.step(int(action))
        self._belief = self._extract_belief(raw_obs)

        terminated = bool(done)
        truncated  = False

        if self._info_mode == "full":
            info = self._sanitize(info)
        else:
            info = {}

        info["regime"] = self._regime
        info["belief"] = self._belief.copy()

        return self._flatten(raw_obs), float(reward), terminated, truncated, info

    def close(self):
        if self._wrapped_env is not None:
            try:
                self._wrapped_env.close()
            except Exception:
                pass
            self._wrapped_env = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _flatten(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _extract_belief(self, obs: np.ndarray) -> np.ndarray:
        """Pull the last 2 rows from obs as the current belief vector."""
        if self._belief_mode in ("oracle", "kim", "trained"):
            flat = np.asarray(obs, dtype=np.float32).reshape(-1)
            return flat[-N_BELIEF_FEATURES:].copy()
        return np.zeros(N_BELIEF_FEATURES, dtype=np.float32)

    def _sanitize(self, info):
        if isinstance(info, dict):
            return {str(k): self._sanitize(v) for k, v in info.items()}
        if isinstance(info, (list, tuple)):
            return type(info)(self._sanitize(v) for v in info)
        return info


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test(belief_mode: str = "oracle", n_steps: int = 5,
                trained_model_path: str = ""):
    print(f"\n── smoke test: belief_mode={belief_mode!r} ──")
    cfg: Dict[str, Any] = {"belief_mode": belief_mode, "val_prob": 0.5, "debug_mode": True}
    if belief_mode == "trained":
        cfg["trained_model_path"] = trained_model_path
    env = RegimeRandomizedGymnasiumEnv(cfg)
    obs, _ = env.reset(seed=42)
    print(f"  regime={env._regime}  obs shape={obs.shape}")
    if belief_mode in ("oracle", "kim", "trained"):
        print(f"  initial belief=(p_val={obs[-2]:.3f}, p_mom={obs[-1]:.3f})")
    for i in range(n_steps):
        obs, r, term, trunc, info = env.step(1)  # hold
        if belief_mode in ("oracle", "kim", "trained"):
            b = info["belief"]
            print(f"  step {i+1}  regime={info['regime']}  "
                  f"p_val={b[0]:.3f}  p_mom={b[1]:.3f}  reward={r:.4f}")
        if term or trunc:
            break
    env.close()
    print("  passed.\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained-model", default="models/mlp_kalman_regime.pt",
                    help="Path to trained classifier checkpoint")
    ns = ap.parse_args()

    _smoke_test("none")
    _smoke_test("oracle")
    _smoke_test("trained", trained_model_path=ns.trained_model)
