"""
Smoke tests for the parameter sweep.

1. BUG 1 invariant — HOLD P&L must be exactly 0 regardless of market params.
   Any non-zero value means previous_marked_to_market was not reset after env.reset().

2. run_param shape — correct number of result rows produced.
"""
import importlib
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pytest

# locate sweep.py and test.py relative to this file
_scripts = os.path.join(os.path.dirname(__file__), "../../abides-gym/scripts")
sys.path.insert(0, os.path.abspath(_scripts))


@pytest.fixture(scope="module")
def env_factory():
    import gym
    import abides_gym  # noqa: F401 — registers envs

    def make(bg_extra=None, timestep="300s"):
        return gym.make(
            "markets-daily_investor-v0",
            background_config              = "rmsc04",
            timestep_duration              = timestep,
            starting_cash                  = 1_000_000,
            order_fixed_size               = 10,
            state_history_length           = 4,
            market_data_buffer_length      = 5,
            first_interval                 = "00:05:00",
            reward_mode                    = "dense",
            done_ratio                     = 0.3,
            debug_mode                     = True,
            background_config_extra_kvargs = bg_extra or {},
        )

    return make


def _run_hold(env, seed):
    _test = importlib.import_module("test")
    ep = _test.run_episode(env, _test.HoldBaseline(), seed=seed)
    return ep["final_pnl"]


# ── BUG 1 invariant ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("bg_extra", [
    {},                                              # default params
    {"mm_wake_up_freq": "10S"},                      # fast MM
    {"num_value_agents": 10},                        # thin value anchoring
])
def test_hold_pnl_is_zero(env_factory, bg_extra):
    """HOLD never trades — M2M must equal starting_cash at every episode end."""
    env = env_factory(bg_extra=bg_extra, timestep="300s")
    for seed in range(3):
        pnl = _run_hold(env, seed)
        assert pnl == 0, (
            f"HOLD pnl={pnl} != 0 for bg_extra={bg_extra} seed={seed}. "
            "BUG 1: previous_marked_to_market not reset after env.reset()?"
        )
    env.close()


# ── run_param shape ───────────────────────────────────────────────────────────

def test_run_param_row_count(env_factory):
    """run_param should produce exactly n_episodes rows per strategy."""
    sweep = importlib.import_module("sweep")
    _test  = importlib.import_module("test")

    n_episodes = 2
    strategies = {"HOLD": _test.HoldBaseline, "MR": _test.MeanReversionBaseline}
    results = []

    env_kw = dict(
        background_config         = "rmsc04",
        timestep_duration         = "300s",
        starting_cash             = 1_000_000,
        order_fixed_size          = 10,
        state_history_length      = 4,
        market_data_buffer_length = 5,
        first_interval            = "00:05:00",
        reward_mode               = "dense",
        done_ratio                = 0.3,
        debug_mode                = True,
    )

    sweep.run_param("mm_wake_up_freq", "60S", env_kw, {}, strategies, n_episodes, results)

    assert len(results) == n_episodes * len(strategies), (
        f"Expected {n_episodes * len(strategies)} rows, got {len(results)}"
    )
    for row in results:
        assert "final_pnl" in row
        assert "strategy"  in row
        assert "param"     in row
