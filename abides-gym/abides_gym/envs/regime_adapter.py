"""
RegimeAdapter — gymnasium.Env for oracle belief experiments (E1 / E2).

Holds two SubGymMarketsDailyInvestorEnv_v0 instances (one per regime).
On each reset(), samples a regime and resets the matching inner env.
Appends oracle belief b* = (1,0) or (0,1) to the flattened 7-dim obs → 9-dim.

Regime definition (ρ = num_momentum_agents / num_value_agents, ρ* = 0.06):
  Val: ρ < 0.06  → num_momentum_agents=0,  b* = (1, 0)
  Mom: ρ ≥ 0.06  → num_momentum_agents=12, b* = (0, 1)

env_config keys consumed here (not passed to base env):
  regime_mode          : "val" | "mom" | "random" (default "random")
  adapter_info_mode    : "minimal" | "full"        (default "full")
  per_episode_seed_base: int                        (default 0)

All remaining keys are forwarded to SubGymMarketsDailyInvestorEnv_v0.
background_config_extra_kvargs is deep-copied per regime so each inner env
gets the correct num_momentum_agents without aliasing.
"""

import random
from copy import deepcopy
from typing import Any, Dict, Optional

import gym
import gymnasium
import numpy as np

from abides_gym.envs.markets_daily_investor_environment_v0 import (
    SubGymMarketsDailyInvestorEnv_v0,
)

_SEED_MOD = (1 << 31) - 1

RHO_STAR = 0.06

_REGIME_PARAMS = {
    "val": {"num_momentum_agents": 0,  "num_value_agents": 102},
    "mom": {"num_momentum_agents": 12, "num_value_agents": 102},
}

_REGIME_BELIEFS = {
    "val": np.array([1.0, 0.0], dtype=np.float32),
    "mom": np.array([0.0, 1.0], dtype=np.float32),
}

_ADAPTER_KEYS = {"regime_mode", "adapter_info_mode", "per_episode_seed_base"}


class ResetMarkedToMarketWrapper(gym.Wrapper):
    """Applies BUG 1 fix: resets previous_marked_to_market after every env.reset()."""

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        base = self.env.unwrapped
        if hasattr(base, "starting_cash"):
            base.previous_marked_to_market = base.starting_cash
        return out


def _to_gymnasium_space(space):
    if isinstance(space, gym.spaces.Discrete):
        return gymnasium.spaces.Discrete(space.n)
    if isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(
            low=space.low, high=space.high, shape=space.shape, dtype=space.dtype
        )
    raise TypeError(f"Unsupported space type: {type(space)!r}")


class RegimeAdapter(gymnasium.Env):
    """
    Standalone gymnasium.Env that presents a 9-dim observation:
      obs[:7] — base daily-investor features (flattened)
      obs[7:] — oracle regime belief b* (2-dim float32)

    Two inner envs are created at construction (one per regime); only the
    selected regime's env is reset/stepped each episode. This avoids any
    in-place env_config patching.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: Dict[str, Any]):
        cfg = dict(env_config)

        self._regime_mode: str = str(cfg.pop("regime_mode", "random"))
        self._info_mode: str = str(cfg.pop("adapter_info_mode", "full")).lower()
        self._episode_seed_base: int = int(cfg.pop("per_episode_seed_base", 0)) % _SEED_MOD
        self._episode_index: int = 0
        self._stream_offset: int = (id(self) * 1_000_003) % _SEED_MOD

        if self._regime_mode not in {"val", "mom", "random"}:
            raise ValueError(f"regime_mode must be 'val', 'mom', or 'random', got {self._regime_mode!r}")
        if self._info_mode not in {"minimal", "full"}:
            raise ValueError(f"adapter_info_mode must be 'minimal' or 'full', got {self._info_mode!r}")

        # Build one inner env per regime.
        self._envs: Dict[str, ResetMarkedToMarketWrapper] = {}
        for regime, regime_params in _REGIME_PARAMS.items():
            regime_cfg = dict(cfg)
            bkg = deepcopy(regime_cfg.get("background_config_extra_kvargs", {}))
            bkg.update(regime_params)
            regime_cfg["background_config_extra_kvargs"] = bkg
            # Remove any leftover adapter keys before passing to base env.
            for k in _ADAPTER_KEYS:
                regime_cfg.pop(k, None)
            self._envs[regime] = ResetMarkedToMarketWrapper(
                SubGymMarketsDailyInvestorEnv_v0(**regime_cfg)
            )

        self._current_regime: str = "val"
        self._belief: np.ndarray = _REGIME_BELIEFS["val"].copy()
        self._current_env: ResetMarkedToMarketWrapper = self._envs["val"]

        # Observation space: 7-dim base + 2-dim belief.
        base_obs = self._envs["val"].observation_space
        flat_low  = np.asarray(base_obs.low,  dtype=np.float32).reshape(-1)
        flat_high = np.asarray(base_obs.high, dtype=np.float32).reshape(-1)
        self.observation_space = gymnasium.spaces.Box(
            low=np.concatenate([flat_low,  [-np.inf, -np.inf]]),
            high=np.concatenate([flat_high, [ np.inf,  np.inf]]),
            dtype=np.float32,
        )
        self.action_space = _to_gymnasium_space(self._envs["val"].action_space)

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        if self._regime_mode == "random":
            self._current_regime = random.choice(["val", "mom"])
        else:
            self._current_regime = self._regime_mode

        self._belief = _REGIME_BELIEFS[self._current_regime].copy()
        self._current_env = self._envs[self._current_regime]

        if seed is not None:
            ep_seed = int(seed) % _SEED_MOD
        else:
            ep_seed = (
                self._episode_seed_base + self._stream_offset + self._episode_index
            ) % _SEED_MOD
            self._episode_index += 1

        self._current_env.seed(ep_seed)
        obs = self._current_env.reset()
        flat_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        return np.concatenate([flat_obs, self._belief]), {"regime": self._current_regime}

    def step(self, action):
        obs, reward, done, info = self._current_env.step(action)
        terminated = bool(done)
        flat_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        out_info = self._sanitize_info(info)
        out_info["regime"] = self._current_regime
        out_info["belief"] = self._belief.tolist()
        return (
            np.concatenate([flat_obs, self._belief]),
            float(reward),
            terminated,
            False,
            out_info,
        )

    def close(self):
        for env in self._envs.values():
            try:
                env.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sanitize_info(self, info):
        if self._info_mode == "minimal":
            return {}
        if isinstance(info, dict):
            return {str(k): self._sanitize_info(v) for k, v in info.items()}
        if isinstance(info, (list, tuple)):
            return type(info)(self._sanitize_info(v) for v in info)
        return info

    @property
    def unwrapped_base(self):
        """Access the underlying SubGymMarketsDailyInvestorEnv_v0 for the current regime."""
        return self._current_env.env.unwrapped
