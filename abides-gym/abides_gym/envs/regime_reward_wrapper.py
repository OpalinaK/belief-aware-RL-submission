"""
RegimeRewardWrapper — E2 reward shaping wrapper for oracle belief experiments.

Wraps RegimeAdapter and adds an alignment bonus to the reward during training:
  shaped_reward = env_reward + alpha * align(action, regime)

Alignment:
  Val  (obs[-2] > 0.5): +1 if action == 0 (BUY),  else 0
  Mom  (obs[-1] > 0.5): +1 if action == 0 (BUY)  when obs[4] > 0  (uptrend)
                         +1 if action == 2 (SELL) when obs[4] <= 0 (downtrend)
                         else 0

obs[4] is the oldest padded return in the flattened 7-dim base observation.
Action encoding (source of truth): 0=BUY, 1=HOLD, 2=SELL.

Strip this wrapper at eval time — eval uses RegimeAdapter only.

Usage (E2 training):
  from abides_gym.envs.regime_adapter import RegimeAdapter
  from abides_gym.envs.regime_reward_wrapper import RegimeRewardWrapper

  base = RegimeAdapter(env_config)
  env  = RegimeRewardWrapper(base, alpha=0.001)
"""

import numpy as np
import gymnasium

_RETURN_IDX = 4  # obs[4] in the flattened 7-dim base obs — oldest padded return


class RegimeRewardWrapper(gymnasium.Wrapper):
    """
    Adds alpha * align(action, regime_belief) to the reward.

    Reads regime belief from the last two dimensions of the 9-dim observation
    produced by RegimeAdapter.  Observation space is unchanged (still 9-dim).
    """

    def __init__(self, env: gymnasium.Env, alpha: float = 0.001):
        super().__init__(env)
        self.alpha = float(alpha)
        self._belief = np.zeros(2, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._belief = np.asarray(obs[-2:], dtype=np.float32)
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._belief = np.asarray(obs[-2:], dtype=np.float32)
        shaped = float(reward) + self.alpha * self._alignment(int(action), obs)
        return obs, shaped, terminated, truncated, info

    def _alignment(self, action: int, obs) -> float:
        if self._belief[0] > 0.5:
            # Val regime: reward BUY (action=0)
            return 1.0 if action == 0 else 0.0
        else:
            # Mom regime: reward following trend
            ret = float(obs[_RETURN_IDX])
            preferred = 0 if ret > 0.0 else 2
            return 1.0 if action == preferred else 0.0
