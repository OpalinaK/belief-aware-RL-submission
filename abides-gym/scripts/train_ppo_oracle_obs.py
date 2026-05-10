"""
train_ppo_oracle_obs.py — E1: PPO with oracle regime belief in observation.

Trains PPO on a 9-dim observation:
  obs[:7] — standard daily-investor features
  obs[7:] — oracle belief b* = (1,0) for Val regime, (0,1) for Mom regime

The env samples a regime uniformly at random each episode. The agent sees the
true regime label — this establishes an upper bound on what a belief-aware
policy can achieve.

Outputs mirror ppo_baseline:
  results/e1_belief_obs/seed_{seed}/eval_metrics.json
  results/e1_belief_obs/seed_{seed}/eval_episodes.json
  results/e1_belief_obs/seed_{seed}/train_metrics.json
  results/e1_belief_obs/seed_{seed}/timing.json
  results/e1_belief_obs/seed_{seed}/checkpoints/

Usage:
  .venv/bin/python abides-gym/scripts/train_ppo_oracle_obs.py --seed 0
  .venv/bin/python abides-gym/scripts/train_ppo_oracle_obs.py --seed 0 --timesteps 200000
"""

import argparse
import csv
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import gym
import gymnasium
import numpy as np
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns

import abides_gym  # noqa: F401 — registers gym envs
from abides_gym.envs.regime_adapter import RegimeAdapter

ENV_ID = "markets-daily_investor-v0"
_SEED_MOD = (1 << 31) - 1

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
    debug_mode=True,
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
        if "episode_return_mean" in env_runners:
            return _safe_float(env_runners["episode_return_mean"], 0.0)
        if "episode_reward_mean" in env_runners:
            return _safe_float(env_runners["episode_reward_mean"], 0.0)
    return _safe_float(result.get("episode_reward_mean", 0.0), 0.0)


def _to_export_scalar(value: Any) -> Optional[Union[int, float, bool, str]]:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return value
    if isinstance(value, np.generic):
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
    return None


def _flatten_train_result(result: Dict[str, Any], max_depth: int = 14) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}

    def walk(obj, prefix, depth):
        if depth > max_depth:
            return
        if isinstance(obj, dict):
            for key, val in obj.items():
                ks = str(key)
                if ks in ("config", "timers"):
                    continue
                path = f"{prefix}/{ks}" if prefix else ks
                if isinstance(val, dict):
                    walk(val, path, depth + 1)
                elif isinstance(val, (list, tuple)):
                    if len(val) == 1:
                        walk(val[0], path, depth + 1)
                else:
                    s = _to_export_scalar(val)
                    if s is not None:
                        flat[path] = s
        else:
            s = _to_export_scalar(obj)
            if s is not None and prefix:
                flat[prefix] = s

    walk(result, "", 0)
    return flat


def _write_training_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    preferred = ["iteration", "timesteps_total", "episode_reward_mean",
                 "training_iteration", "iter_wall_time_sec"]
    keys: Set[str] = set()
    for r in rows:
        keys.update(r.keys())
    fieldnames = [c for c in preferred if c in keys] + sorted(keys - set(preferred))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _summarize_array(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _compute_max_drawdown(m2m_trace: List[float], starting_cash: float) -> float:
    if not m2m_trace:
        return 0.0
    m2m = np.asarray([starting_cash] + m2m_trace, dtype=np.float64)
    peaks = np.maximum.accumulate(m2m)
    drawdowns = np.where(peaks > 0.0, (peaks - m2m) / peaks, 0.0)
    return float(np.max(drawdowns))


def _tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _inference_action(module, obs: np.ndarray) -> int:
    batch = {Columns.OBS: torch.as_tensor(obs[None, :], dtype=torch.float32)}
    out = module.forward_inference(batch)
    if Columns.ACTIONS in out:
        return int(np.asarray(_tensor_to_numpy(out[Columns.ACTIONS])).reshape(-1)[0])
    if "actions" in out:
        return int(np.asarray(_tensor_to_numpy(out["actions"])).reshape(-1)[0])
    if Columns.ACTION_DIST_INPUTS in out:
        logits = _tensor_to_numpy(out[Columns.ACTION_DIST_INPUTS])
        return int(np.argmax(np.asarray(logits)[0]))
    if "action_dist_inputs" in out:
        logits = _tensor_to_numpy(out["action_dist_inputs"])
        return int(np.argmax(np.asarray(logits)[0]))
    raise RuntimeError(f"Unsupported inference output keys: {list(out.keys())}")


def run_evaluation(algo, env_config: Dict[str, Any], seeds: List[int]) -> Dict[str, Any]:
    t_eval = time.time()
    episodes: List[Dict[str, Any]] = []
    module = algo.get_module()

    for seed in seeds:
        env = RegimeAdapter(env_config)
        obs, info = env.reset(seed=seed)
        regime = info.get("regime", "unknown")

        terminated = truncated = False
        total_reward = 0.0
        actions: List[int] = []
        m2m_trace: List[float] = []
        regimes_seen: List[str] = [regime]

        while not (terminated or truncated):
            action_i = _inference_action(module, obs)
            obs, reward, terminated, truncated, info = env.step(action_i)
            total_reward += float(reward)
            actions.append(action_i)
            m2m = info.get("marked_to_market")
            if m2m is not None:
                m2m_trace.append(float(m2m))

        env.close()

        starting_cash = float(env.unwrapped_base.starting_cash)
        final_m2m = m2m_trace[-1] if m2m_trace else starting_cash
        final_pnl = final_m2m - starting_cash
        max_drawdown = _compute_max_drawdown(m2m_trace, starting_cash)

        episodes.append({
            "seed": int(seed),
            "regime": regime,
            "total_reward": float(total_reward),
            "final_m2m": float(final_m2m),
            "final_pnl": float(final_pnl),
            "episode_length": int(len(actions)),
            "won": bool(final_pnl >= 0.0),
            "max_drawdown": float(max_drawdown),
            "action_counts": {
                "BUY": int(actions.count(0)),
                "HOLD": int(actions.count(1)),
                "SELL": int(actions.count(2)),
            },
        })

    final_pnls = [ep["final_pnl"] for ep in episodes]
    wins = [ep["won"] for ep in episodes]
    lengths = [ep["episode_length"] for ep in episodes]
    max_dds = [ep["max_drawdown"] for ep in episodes]
    total_rewards = [ep["total_reward"] for ep in episodes]

    buy_total  = sum(ep["action_counts"]["BUY"]  for ep in episodes)
    hold_total = sum(ep["action_counts"]["HOLD"] for ep in episodes)
    sell_total = sum(ep["action_counts"]["SELL"] for ep in episodes)
    action_total = buy_total + hold_total + sell_total
    action_pct = (
        {"BUY": buy_total/action_total, "HOLD": hold_total/action_total, "SELL": sell_total/action_total}
        if action_total > 0 else {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0}
    )

    regime_counts = {}
    for ep in episodes:
        r = ep["regime"]
        regime_counts[r] = regime_counts.get(r, 0) + 1

    pnl_stats = _summarize_array(final_pnls)
    std = pnl_stats["std"]
    sharpe = float(pnl_stats["mean"] / std) if std > 1e-12 else 0.0

    return {
        "episodes": episodes,
        "metrics": {
            "n_episodes": len(episodes),
            "final_pnl_cents": pnl_stats,
            "total_reward": _summarize_array(total_rewards),
            "episode_length": _summarize_array(lengths),
            "max_drawdown": _summarize_array(max_dds),
            "win_rate": float(np.mean(wins)) if wins else 0.0,
            "sharpe": sharpe,
            "action_pct": action_pct,
            "regime_counts": regime_counts,
            "eval_wall_time_sec": float(time.time() - t_eval),
            "protocol": {"timestep_duration": "60s", "obs_dim": 9},
        },
    }


def build_env_config(args, debug_mode: bool, info_mode: str) -> Dict[str, Any]:
    env_config = dict(ENV_DEFAULTS)
    env_config["timestep_duration"] = args.timestep
    env_config["debug_mode"] = bool(debug_mode)
    env_config["adapter_info_mode"] = str(info_mode)
    base = args.seed if args.per_episode_seed_base is None else args.per_episode_seed_base
    env_config["per_episode_seed_base"] = int(base)
    env_config["background_config_extra_kvargs"] = dict(RMSC04_PARAMS)
    env_config["regime_mode"] = "random"
    return env_config


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        s = v.strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

    parser = argparse.ArgumentParser(description="E1: Train PPO with oracle regime belief in obs")
    parser.add_argument("--timesteps",      type=int,   default=200_000)
    parser.add_argument("--timestep",       type=str,   default="60s")
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--out-dir",        type=str,   default=None)
    parser.add_argument("--checkpoint-freq",type=int,   default=50)
    parser.add_argument("--eval-seeds",     nargs="+",  type=int, default=[100, 101, 102, 103, 104])
    parser.add_argument("--num-gpus",       type=float, default=0.0)
    parser.add_argument("--num-workers",    type=int,   default=0)
    parser.add_argument("--train-batch-size",   type=int,   default=2000)
    parser.add_argument("--minibatch-size",     type=int,   default=128)
    parser.add_argument("--num-epochs",         type=int,   default=5)
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--entropy-coeff",      type=float, default=0.01)
    parser.add_argument("--train-debug-mode",   type=str2bool, default=False)
    parser.add_argument("--eval-debug-mode",    type=str2bool, default=True)
    parser.add_argument("--train-info-mode",    choices=["minimal", "full"], default="minimal")
    parser.add_argument("--eval-info-mode",     choices=["minimal", "full"], default="full")
    parser.add_argument("--per-episode-seed-base", type=int, default=None)
    parser.add_argument("--profile-phases",     type=str2bool, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir or f"results/e1_belief_obs/seed_{args.seed}")
    checkpoints_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    _ts_log(
        f"E1 starting: out_dir={out_dir} timesteps={args.timesteps} "
        f"timestep={args.timestep} seed={args.seed} workers={args.num_workers}"
    )

    train_env_config = build_env_config(args, debug_mode=args.train_debug_mode, info_mode=args.train_info_mode)
    eval_env_config  = build_env_config(args, debug_mode=args.eval_debug_mode,  info_mode=args.eval_info_mode)

    ray.shutdown()
    _ray_init: Dict[str, Any] = {"ignore_reinit_error": True, "include_dashboard": False, "log_to_driver": False}
    _ray_tmp = os.environ.get("RAY_TMPDIR") or (
        str(Path(os.environ["TMPDIR"]) / "ray") if os.environ.get("TMPDIR") else None
    )
    if _ray_tmp:
        Path(_ray_tmp).mkdir(parents=True, exist_ok=True)
        _ray_init["_temp_dir"] = _ray_tmp
    ray.init(**_ray_init)

    t_setup = time.time()
    config = (
        PPOConfig()
        .environment(env=RegimeAdapter, env_config=train_env_config, disable_env_checking=True)
        .framework("torch")
        .resources(num_gpus=args.num_gpus)
        .env_runners(num_env_runners=args.num_workers, observation_filter="MeanStdFilter")
        .training(
            gamma=1.0,
            lr=args.lr,
            train_batch_size=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
            entropy_coeff=args.entropy_coeff,
        )
        .rl_module(model_config={"fcnet_hiddens": [64, 64], "fcnet_activation": "tanh"})
        .debugging(seed=args.seed)
    )
    algo = config.build_algo()
    train_setup_sec = time.time() - t_setup

    train_rows: List[Dict[str, Any]] = []
    training_jsonl_path = out_dir / "training_iterations.jsonl"
    t_start = time.time()
    timesteps_total = 0
    iteration = 0

    _ts_log(
        f"Ray+PPO setup done in {train_setup_sec:.1f}s. "
        f"Starting training (batch={args.train_batch_size}, workers={args.num_workers})."
    )

    with training_jsonl_path.open("w", encoding="utf-8") as jsonl_f:
        while timesteps_total < args.timesteps:
            iteration += 1
            t_iter = time.time()
            result = algo.train()
            iter_wall_sec = time.time() - t_iter

            timesteps_total = _extract_timesteps_total(result)
            episode_reward_mean = _extract_reward_mean(result)

            row = {
                "iteration": iteration,
                "timesteps_total": timesteps_total,
                "episode_reward_mean": episode_reward_mean,
                "training_iteration": _safe_int(result.get("training_iteration"), iteration),
                "iter_wall_time_sec": iter_wall_sec,
            }
            row.update(_flatten_train_result(result))
            train_rows.append(row)
            jsonl_f.write(json.dumps(row, default=str) + "\n")
            jsonl_f.flush()

            _ts_log(
                f"iter={iteration:>3}  timesteps={timesteps_total:>8}  "
                f"reward_mean={episode_reward_mean:+.5f}  iter_time={iter_wall_sec:.2f}s"
            )

            if args.checkpoint_freq > 0 and iteration % args.checkpoint_freq == 0:
                ckpt = algo.save_to_path(str(checkpoints_dir.resolve()))
                _ts_log(f"checkpoint -> {ckpt}")

    final_ckpt = algo.save_to_path(str(checkpoints_dir.resolve()))
    _ts_log(f"final checkpoint -> {final_ckpt}")

    train_loop_sec = time.time() - t_start
    env_steps_per_sec = timesteps_total / train_loop_sec if train_loop_sec > 0 else 0.0

    train_metrics = {
        "summary": {
            "timesteps_total": timesteps_total,
            "train_iterations": iteration,
            "wall_time_sec": train_loop_sec,
            "env_steps_per_sec": env_steps_per_sec,
        },
        "iterations": train_rows,
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(train_metrics, indent=2), encoding="utf-8")
    _write_training_csv(out_dir / "training_curve.csv", train_rows)

    _ts_log(f"Starting evaluation on {len(args.eval_seeds)} seed(s)...")
    eval_payload = run_evaluation(algo, eval_env_config, args.eval_seeds)

    (out_dir / "eval_episodes.json").write_text(
        json.dumps(eval_payload["episodes"], indent=2), encoding="utf-8"
    )
    (out_dir / "eval_metrics.json").write_text(
        json.dumps(eval_payload["metrics"], indent=2), encoding="utf-8"
    )

    timing = {
        "timesteps": args.timesteps,
        "timestep_duration": args.timestep,
        "wall_time_sec": train_loop_sec,
        "env_steps_per_sec": env_steps_per_sec,
        "train_iterations": iteration,
        "eval_wall_time_sec": eval_payload["metrics"]["eval_wall_time_sec"],
        "obs_dim": 9,
        "experiment": "e1_belief_obs",
    }
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")

    algo.stop()
    ray.shutdown()
    _ts_log(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
