"""
sweep_alpha_e2.py — E2a: alpha sweep for reward shaping.

Runs short PPO training for each alpha value (default: 1e-4, 1e-3, 1e-2) using
RegimeRewardEnv (RegimeRewardWrapper around RegimeAdapter).  Reports final
episode_reward_mean per alpha and picks the best for full E2 training.

Each alpha runs --timesteps steps (default 50_000) on seed 0 only.

Output: results/e2_alpha_sweep/alpha_sweep.json

Usage:
  .venv/bin/python abides-gym/scripts/sweep_alpha_e2.py
  .venv/bin/python abides-gym/scripts/sweep_alpha_e2.py --alphas 1e-4 1e-3 1e-2 --timesteps 50000
  .venv/bin/python abides-gym/scripts/sweep_alpha_e2.py --alphas 1e-3 --timesteps 4000  # smoke
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig

import abides_gym  # noqa: F401
from abides_gym.envs.regime_adapter import RegimeAdapter
from abides_gym.envs.regime_reward_wrapper import RegimeRewardEnv

RMSC04_PARAMS = dict(
    starting_cash=10_000_000,
    num_noise_agents=1000,
    num_value_agents=102,
    r_bar=100_000,
    kappa=1.67e-15,
    lambda_a=5.7e-12,
    kappa_oracle=1.67e-16,
    sigma_s=0,
    fund_vol=5e-5,
    megashock_lambda_a=2.77778e-18,
    megashock_mean=1000,
    megashock_var=50_000,
    mm_window_size="adaptive",
    mm_pov=0.025,
    mm_num_ticks=10,
    mm_wake_up_freq="60S",
    mm_min_order_size=1,
    mm_skew_beta=0,
    mm_price_skew=4,
    mm_level_spacing=5,
    mm_spread_alpha=0.75,
    mm_backstop_quantity=0,
    mm_cancel_limit_delay=50,
    num_momentum_agents=12,
)

ENV_DEFAULTS = dict(
    background_config="rmsc04",
    mkt_close="16:00:00",
    timestep_duration="60s",
    starting_cash=1_000_000,
    order_fixed_size=10,
    state_history_length=4,
    market_data_buffer_length=5,
    first_interval="00:05:00",
    reward_mode="dense",
    done_ratio=0.3,
)


def _ts_log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _extract_timesteps_total(result: Dict[str, Any]) -> int:
    for key in ("timesteps_total", "num_env_steps_sampled_lifetime", "agent_timesteps_total"):
        if key in result:
            return _safe_int(result[key], 0)
    return 0


def _extract_reward_mean(result: Dict[str, Any]) -> float:
    env_runners = result.get("env_runners", {})
    if isinstance(env_runners, dict):
        for k in ("episode_return_mean", "episode_reward_mean"):
            if k in env_runners:
                return _safe_float(env_runners[k], float("nan"))
    return _safe_float(result.get("episode_reward_mean", float("nan")), float("nan"))


def train_alpha(alpha: float, args, seed: int = 0) -> Dict[str, Any]:
    """Run a short PPO training for one alpha. Returns per-iteration reward history."""
    _ts_log(f"  alpha={alpha:.1e}  seed={seed}  timesteps={args.timesteps}  workers={args.num_workers}")

    train_env_config = dict(ENV_DEFAULTS)
    train_env_config["debug_mode"] = False
    train_env_config["adapter_info_mode"] = "minimal"
    train_env_config["regime_mode"] = "random"
    train_env_config["per_episode_seed_base"] = seed
    train_env_config["background_config_extra_kvargs"] = dict(RMSC04_PARAMS)
    train_env_config["reward_alpha"] = float(alpha)

    config = (
        PPOConfig()
        .environment(env=RegimeRewardEnv, env_config=train_env_config, disable_env_checking=True)
        .framework("torch")
        .resources(num_gpus=args.num_gpus)
        .env_runners(num_env_runners=args.num_workers, observation_filter="MeanStdFilter")
        .training(
            gamma=1.0,
            lr=1e-4,
            train_batch_size=2000,
            minibatch_size=128,
            num_epochs=5,
            entropy_coeff=0.01,
        )
        .rl_module(model_config={"fcnet_hiddens": [64, 64], "fcnet_activation": "tanh"})
        .debugging(seed=seed)
    )
    algo = config.build_algo()

    iterations = []
    timesteps_total = 0
    iteration = 0
    t_start = time.time()

    while timesteps_total < args.timesteps:
        iteration += 1
        result = algo.train()
        timesteps_total = _extract_timesteps_total(result)
        reward_mean = _extract_reward_mean(result)
        iterations.append({"iteration": iteration, "timesteps": timesteps_total, "reward_mean": reward_mean})
        _ts_log(
            f"    iter={iteration:>3}  ts={timesteps_total:>7}  "
            f"reward_mean={reward_mean:+.5f}"
        )

    algo.stop()
    wall_time = time.time() - t_start

    valid_rewards = [r["reward_mean"] for r in iterations if not np.isnan(r["reward_mean"])]
    final_reward = valid_rewards[-1] if valid_rewards else float("nan")
    # smoothed over last 20% of iterations for stability
    tail_n = max(1, len(valid_rewards) // 5)
    tail_reward = float(np.mean(valid_rewards[-tail_n:])) if valid_rewards else float("nan")

    return {
        "alpha": float(alpha),
        "seed": seed,
        "timesteps": timesteps_total,
        "iterations": iteration,
        "wall_time_sec": wall_time,
        "final_reward_mean": final_reward,
        "tail_reward_mean": tail_reward,
        "history": iterations,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="E2a: alpha sweep for reward shaping")
    parser.add_argument("--alphas",      nargs="+", type=float, default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--timesteps",   type=int,  default=50_000)
    parser.add_argument("--seed",        type=int,  default=0)
    parser.add_argument("--out-dir",     type=str,  default="results/e2_alpha_sweep")
    parser.add_argument("--num-workers", type=int,  default=0)
    parser.add_argument("--num-gpus",    type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _ts_log(f"E2a alpha sweep: alphas={args.alphas} timesteps={args.timesteps} seed={args.seed}")

    ray.shutdown()
    _ray_init: Dict[str, Any] = {"ignore_reinit_error": True, "include_dashboard": False, "log_to_driver": False}
    _ray_tmp = os.environ.get("RAY_TMPDIR") or (
        str(Path(os.environ["TMPDIR"]) / "ray") if os.environ.get("TMPDIR") else None
    )
    if _ray_tmp:
        Path(_ray_tmp).mkdir(parents=True, exist_ok=True)
        _ray_init["_temp_dir"] = _ray_tmp
    ray.init(**_ray_init)

    results = []
    for alpha in args.alphas:
        _ts_log(f"\n{'='*50}")
        _ts_log(f"Alpha = {alpha:.1e}")
        _ts_log(f"{'='*50}")
        r = train_alpha(alpha, args, seed=args.seed)
        results.append(r)
        _ts_log(
            f"  alpha={alpha:.1e}  final={r['final_reward_mean']:+.5f}  "
            f"tail={r['tail_reward_mean']:+.5f}  time={r['wall_time_sec']:.0f}s"
        )

    ray.shutdown()

    # Pick best alpha by tail reward mean.
    valid = [r for r in results if not np.isnan(r["tail_reward_mean"])]
    best = max(valid, key=lambda r: r["tail_reward_mean"]) if valid else results[0]

    summary = {
        "best_alpha": best["alpha"],
        "best_tail_reward_mean": best["tail_reward_mean"],
        "protocol": {"timesteps_per_alpha": args.timesteps, "seed": args.seed},
        "alphas": results,
    }

    out_path = out_dir / "alpha_sweep.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _ts_log(f"\nWrote {out_path}")

    _ts_log(f"\n{'='*50}")
    _ts_log("ALPHA SWEEP RESULTS")
    _ts_log(f"{'='*50}")
    for r in results:
        marker = " <-- best" if r["alpha"] == best["alpha"] else ""
        _ts_log(
            f"  alpha={r['alpha']:.1e}  tail_reward={r['tail_reward_mean']:+.5f}  "
            f"final_reward={r['final_reward_mean']:+.5f}{marker}"
        )
    _ts_log(f"\nBest alpha for E2: {best['alpha']:.1e}")


if __name__ == "__main__":
    main()
