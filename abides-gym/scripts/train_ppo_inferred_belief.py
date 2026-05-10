"""
train_ppo_inferred_belief.py — E3: PPO with inferred regime belief in observation.

Uses RegimeRandomizedGymnasiumEnv with belief_mode="trained" (from regime_belief_wrapper.py).
The trained MLP+Kalman classifier (mlp_kalman_regime.pt) outputs (p_val, p_mom) per step
from observable market features — no oracle access.

Observation: 9-dim  [7 base features | p_val_t | p_mom_t]
Training: 50/50 Val/Mom regime episodes (regime_randomized at reset).
Evaluation: fixed seeds, same env.

Usage:
  .venv/bin/python abides-gym/scripts/train_ppo_inferred_belief.py --seed 0 \
      --model-path abides-gym/scripts/models/mlp_kalman_regime.pt
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns

# Add scripts dir to path so regime_belief_wrapper can import kim_filter_tracker etc.
_SCRIPTS_DIR = str(Path(__file__).parent.resolve())
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import abides_gym  # noqa: F401
from regime_belief_wrapper import RegimeRandomizedGymnasiumEnv


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
    preferred = ["iteration", "timesteps_total", "episode_reward_mean", "iter_wall_time_sec"]
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
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "min": float(np.min(arr)), "max": float(np.max(arr))}


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
    for key in (Columns.ACTIONS, "actions"):
        if key in out:
            return int(np.asarray(_tensor_to_numpy(out[key])).reshape(-1)[0])
    for key in (Columns.ACTION_DIST_INPUTS, "action_dist_inputs"):
        if key in out:
            logits = _tensor_to_numpy(out[key])
            return int(np.argmax(np.asarray(logits)[0]))
    raise RuntimeError(f"Unsupported inference output keys: {list(out.keys())}")


def run_evaluation(algo, env_config: Dict[str, Any], seeds: List[int]) -> Dict[str, Any]:
    t_eval = time.time()
    episodes: List[Dict[str, Any]] = []
    module = algo.get_module()

    for seed in seeds:
        env = RegimeRandomizedGymnasiumEnv(env_config)
        obs, info = env.reset(seed=seed)
        regime = info.get("regime", "unknown")

        terminated = truncated = False
        total_reward = 0.0
        actions: List[int] = []
        m2m_trace: List[float] = []

        while not (terminated or truncated):
            action_i = _inference_action(module, obs)
            obs, reward, terminated, truncated, info = env.step(action_i)
            total_reward += float(reward)
            actions.append(action_i)
            m2m = info.get("marked_to_market")
            if m2m is not None:
                m2m_trace.append(float(m2m))

        env.close()

        starting_cash = float(env_config.get("starting_cash", 1_000_000))
        final_m2m = m2m_trace[-1] if m2m_trace else starting_cash
        final_pnl = final_m2m - starting_cash
        max_drawdown = _compute_max_drawdown(m2m_trace, starting_cash)

        episodes.append({
            "seed": int(seed),
            "regime": regime,
            "total_reward": float(total_reward),
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
    max_dds = [ep["max_drawdown"] for ep in episodes]
    total_rewards = [ep["total_reward"] for ep in episodes]
    regime_counts: Dict[str, int] = {}
    for ep in episodes:
        r = ep["regime"]
        regime_counts[r] = regime_counts.get(r, 0) + 1

    buy_total  = sum(ep["action_counts"]["BUY"]  for ep in episodes)
    hold_total = sum(ep["action_counts"]["HOLD"] for ep in episodes)
    sell_total = sum(ep["action_counts"]["SELL"] for ep in episodes)
    action_total = buy_total + hold_total + sell_total
    action_pct = (
        {"BUY": buy_total/action_total, "HOLD": hold_total/action_total, "SELL": sell_total/action_total}
        if action_total > 0 else {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0}
    )

    pnl_stats = _summarize_array(final_pnls)
    std = pnl_stats["std"]
    sharpe = float(pnl_stats["mean"] / std) if std > 1e-12 else 0.0

    return {
        "episodes": episodes,
        "metrics": {
            "n_episodes": len(episodes),
            "final_pnl_cents": pnl_stats,
            "total_reward": _summarize_array(total_rewards),
            "max_drawdown": _summarize_array(max_dds),
            "win_rate": float(np.mean(wins)) if wins else 0.0,
            "sharpe": sharpe,
            "action_pct": action_pct,
            "regime_counts": regime_counts,
            "eval_wall_time_sec": float(time.time() - t_eval),
            "protocol": {"timestep_duration": "60s", "obs_dim": 9, "belief_mode": "trained"},
        },
    }


def build_env_config(args, debug_mode: bool) -> Dict[str, Any]:
    model_path = str(Path(args.model_path).resolve())
    return {
        "belief_mode": "trained",
        "trained_model_path": model_path,
        "val_prob": 0.5,
        "debug_mode": bool(debug_mode),
        "timestep_duration": args.timestep,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="E3: PPO with inferred regime belief")
    parser.add_argument("--model-path", type=str,
                        default="abides-gym/scripts/models/mlp_kalman_regime.pt",
                        help="Path to trained regime classifier .pt file")
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
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir or f"results/e3_inferred_belief/seed_{args.seed}")
    checkpoints_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model_path = str(Path(args.model_path).resolve())
    _ts_log(
        f"E3 starting: out_dir={out_dir} timesteps={args.timesteps} "
        f"seed={args.seed} workers={args.num_workers} model={model_path}"
    )

    train_env_config = build_env_config(args, debug_mode=False)
    eval_env_config  = build_env_config(args, debug_mode=True)

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
        .environment(env=RegimeRandomizedGymnasiumEnv, env_config=train_env_config, disable_env_checking=True)
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
    _ts_log(f"Setup done in {time.time()-t_setup:.1f}s. Training (batch={args.train_batch_size}, workers={args.num_workers}).")

    train_rows: List[Dict[str, Any]] = []
    training_jsonl_path = out_dir / "training_iterations.jsonl"
    t_start = time.time()
    timesteps_total = 0
    iteration = 0

    with training_jsonl_path.open("w", encoding="utf-8") as jsonl_f:
        while timesteps_total < args.timesteps:
            iteration += 1
            t_iter = time.time()
            result = algo.train()
            iter_wall_sec = time.time() - t_iter

            timesteps_total = _extract_timesteps_total(result)
            reward_mean = _extract_reward_mean(result)

            row = {
                "iteration": iteration,
                "timesteps_total": timesteps_total,
                "episode_reward_mean": reward_mean,
                "iter_wall_time_sec": iter_wall_sec,
            }
            row.update(_flatten_train_result(result))
            train_rows.append(row)
            jsonl_f.write(json.dumps(row, default=str) + "\n")
            jsonl_f.flush()

            _ts_log(
                f"iter={iteration:>3}  timesteps={timesteps_total:>8}  "
                f"reward_mean={reward_mean:+.5f}  iter_time={iter_wall_sec:.2f}s"
            )

            if args.checkpoint_freq > 0 and iteration % args.checkpoint_freq == 0:
                ckpt = algo.save_to_path(str(checkpoints_dir.resolve()))
                _ts_log(f"checkpoint -> {ckpt}")

    final_ckpt = algo.save_to_path(str(checkpoints_dir.resolve()))
    _ts_log(f"final checkpoint -> {final_ckpt}")

    train_loop_sec = time.time() - t_start
    env_steps_per_sec = timesteps_total / train_loop_sec if train_loop_sec > 0 else 0.0

    (out_dir / "train_metrics.json").write_text(json.dumps({
        "summary": {
            "timesteps_total": timesteps_total,
            "train_iterations": iteration,
            "wall_time_sec": train_loop_sec,
            "env_steps_per_sec": env_steps_per_sec,
        },
        "iterations": train_rows,
    }, indent=2), encoding="utf-8")
    _write_training_csv(out_dir / "training_curve.csv", train_rows)

    _ts_log(f"Starting evaluation on {len(args.eval_seeds)} seed(s)...")
    eval_payload = run_evaluation(algo, eval_env_config, args.eval_seeds)

    (out_dir / "eval_episodes.json").write_text(json.dumps(eval_payload["episodes"], indent=2), encoding="utf-8")
    (out_dir / "eval_metrics.json").write_text(json.dumps(eval_payload["metrics"], indent=2), encoding="utf-8")
    (out_dir / "timing.json").write_text(json.dumps({
        "timesteps": args.timesteps,
        "wall_time_sec": train_loop_sec,
        "env_steps_per_sec": env_steps_per_sec,
        "train_iterations": iteration,
        "eval_wall_time_sec": eval_payload["metrics"]["eval_wall_time_sec"],
        "obs_dim": 9,
        "experiment": "e3_inferred_belief",
        "model_path": model_path,
    }, indent=2), encoding="utf-8")

    algo.stop()
    ray.shutdown()
    _ts_log(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
