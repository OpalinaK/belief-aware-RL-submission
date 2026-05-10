"""
BeliefEstimatorEnv — E3 standalone gymnasium.Env with learned regime belief.

Wraps SubGymMarketsDailyInvestorEnv_v0 (no oracle regime) and appends a 2-dim
estimated belief ĥ from a trained MLP, producing a 9-dim observation:
  obs[:7] — standard daily-investor features (flattened)
  obs[7:] — ĥ = softmax output of regime MLP over last W obs steps

The estimator is loaded from estimator_model_path in env_config.

env_config keys consumed here (not forwarded to base env):
  estimator_model_path : str   path to estimator.pt (required)
  estimator_window     : int   sliding window size (default 20)

All other keys are forwarded to SubGymMarketsDailyInvestorEnv_v0.
"""

from collections import deque
from typing import Any, Dict, Optional

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from abides_gym.envs.regime_adapter import ResetMarkedToMarketWrapper, _to_gymnasium_space
from abides_gym.envs.markets_daily_investor_environment_v0 import (
    SubGymMarketsDailyInvestorEnv_v0,
)

_ADAPTER_KEYS = {"estimator_model_path", "estimator_window",
                 "regime_mode", "adapter_info_mode", "per_episode_seed_base", "reward_alpha"}


class _RegimeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden, n_classes: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_estimator(model_path: str):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = _RegimeMLP(
        input_dim=checkpoint["input_dim"],
        hidden=checkpoint["hidden"],
        n_classes=checkpoint["n_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, int(checkpoint["window"])


class BeliefEstimatorEnv(gymnasium.Env):
    """
    Standalone gymnasium.Env that presents a 9-dim observation:
      obs[:7] — base daily-investor features (flattened)
      obs[7:] — estimated belief ĥ from sliding-window MLP (no oracle)

    Uses default rmsc04 config — no regime switching. The agent must infer
    the current market regime from historical observations via the MLP.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: Dict[str, Any]):
        cfg = dict(env_config)

        model_path = cfg.pop("estimator_model_path")
        window_override = cfg.pop("estimator_window", None)
        for k in _ADAPTER_KEYS:
            cfg.pop(k, None)

        self._model, self._window = _load_estimator(model_path)
        if window_override is not None:
            self._window = int(window_override)

        self._obs_buffer: deque = deque(maxlen=self._window)
        self._h_hat = np.array([0.5, 0.5], dtype=np.float32)

        self._env = ResetMarkedToMarketWrapper(SubGymMarketsDailyInvestorEnv_v0(**cfg))

        base_obs = self._env.observation_space
        flat_low  = np.asarray(base_obs.low,  dtype=np.float32).reshape(-1)
        flat_high = np.asarray(base_obs.high, dtype=np.float32).reshape(-1)
        self.observation_space = gymnasium.spaces.Box(
            low=np.concatenate([flat_low,  [-np.inf, -np.inf]]),
            high=np.concatenate([flat_high, [ np.inf,  np.inf]]),
            dtype=np.float32,
        )
        self.action_space = _to_gymnasium_space(self._env.action_space)

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._env.seed(int(seed))
        obs = self._env.reset()
        self._obs_buffer.clear()
        flat7 = np.asarray(obs, dtype=np.float32).reshape(-1)[:7]
        self._obs_buffer.append(flat7)
        self._h_hat = self._estimate()
        return np.concatenate([flat7, self._h_hat]), {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        flat7 = np.asarray(obs, dtype=np.float32).reshape(-1)[:7]
        self._obs_buffer.append(flat7)
        self._h_hat = self._estimate()
        return np.concatenate([flat7, self._h_hat]), float(reward), bool(done), False, info

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def _estimate(self) -> np.ndarray:
        buf = np.array(self._obs_buffer, dtype=np.float32)  # (≤W, 7)
        if len(buf) < self._window:
            pad = np.zeros((self._window - len(buf), 7), dtype=np.float32)
            buf = np.vstack([pad, buf])
        x = torch.as_tensor(buf.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            logits = self._model(x)
            h_hat = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        return h_hat.astype(np.float32)
