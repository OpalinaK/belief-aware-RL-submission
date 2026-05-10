"""
Baseline PPO trainer for markets-daily_investor-v0.

This script trains a belief-blind PPO policy on standard daily-investor observations only.
It includes a reset-safe wrapper for BUG 1 (`previous_marked_to_market` reset), checkpointing,
and timing metrics for smoke/full run planning.

Training metrics are written to ``train_metrics.json`` (full per-iter rows),
``training_iterations.jsonl`` (one JSON object per line, flushed each iter),
and ``training_curve.csv`` (wide table for spreadsheets / pandas).
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import gym
import gymnasium as gymnasium
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
import torch

# Registers markets-daily_investor-v0 for local `gym.make` callers; workers use direct ctor below.
import abides_gym  # noqa: F401

from abides_gym.envs.markets_daily_investor_environment_v0 import (
    SubGymMarketsDailyInvestorEnv_v0,
)

ENV_ID = "markets-daily_investor-v0"
_SEED_MOD = (1 << 31) - 1  # keep gym seeds in signed 32-bit range


def _ts_log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


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
    timestep_duration="300s",
    starting_cash=1_000_000,
    order_fixed_size=10,
    state_history_length=4,
    market_data_buffer_length=5,
    first_interval="00:05:00",
    reward_mode="dense",
    done_ratio=0.3,
    debug_mode=True,
)


class ResetMarkedToMarketWrapper(gym.Wrapper):
    """Ensure BUG 1 invariant after every reset."""

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        base_env = self.env.unwrapped
        if hasattr(base_env, "starting_cash"):
            base_env.previous_marked_to_market = base_env.starting_cash
        return out


def _to_gymnasium_space(space):
    """Convert old gym spaces to gymnasium spaces."""
    if isinstance(space, gym.spaces.Discrete):
        return gymnasium.spaces.Discrete(space.n)
    if isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(
            low=space.low,
            high=space.high,
            shape=space.shape,
            dtype=space.dtype,
        )
    raise TypeError(f"Unsupported space type for Gymnasium adapter: {type(space)!r}")


class GymnasiumDailyInvestorAdapter(gymnasium.Env):
    """
    Gymnasium-compatible adapter around the legacy Gym daily investor env.
    Converts reset/step API and applies BUG 1 reset invariant.

    Per-episode seeding (octopus-style): when ``reset(seed=None)``, uses
    ``per_episode_seed_base + stream_offset + episode_index`` so successive episodes
    see different ABIDES RNG draws. ``stream_offset`` separates parallel env instances.
    When ``reset(seed=...)`` is passed (e.g. eval), that seed is used exactly.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: Dict[str, Any]):
        cfg = dict(env_config)
        self._info_mode = str(cfg.pop("adapter_info_mode", "full")).lower()
        if self._info_mode not in {"minimal", "full"}:
            raise ValueError(f"Unsupported adapter_info_mode={self._info_mode!r}")

        self._episode_seed_base = int(cfg.pop("per_episode_seed_base", 0)) % _SEED_MOD
        self._episode_index = 0
        # Parallel Ray runners each construct their own adapter; offset by object id
        # so workers do not all replay the same day sequence.
        self._stream_offset = (id(self) * 1_000_003) % _SEED_MOD

        # Direct class avoids gym.registry on Ray workers (registry may be empty in worker procs).
        base_env = SubGymMarketsDailyInvestorEnv_v0(**cfg)
        self._env = ResetMarkedToMarketWrapper(base_env)
        self.action_space = _to_gymnasium_space(self._env.action_space)
        base_obs = self._env.observation_space
        if not isinstance(base_obs, gym.spaces.Box):
            raise TypeError(f"Expected Box observation space, got {type(base_obs)!r}")
        self.observation_space = gymnasium.spaces.Box(
            low=np.asarray(base_obs.low, dtype=np.float32).reshape(-1),
            high=np.asarray(base_obs.high, dtype=np.float32).reshape(-1),
            shape=(int(np.prod(base_obs.shape)),),
            dtype=np.float32,
        )

    def _flatten_obs(self, obs):
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _sanitize_info(self, info):
        if self._info_mode == "minimal":
            return {}
        if isinstance(info, dict):
            return {str(k): self._sanitize_info(v) for k, v in info.items()}
        if isinstance(info, list):
            return [self._sanitize_info(v) for v in info]
        if isinstance(info, tuple):
            return tuple(self._sanitize_info(v) for v in info)
        return info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            ep_seed = int(seed) % _SEED_MOD
        else:
            ep_seed = (self._episode_seed_base + self._stream_offset + self._episode_index) % _SEED_MOD
            self._episode_index += 1
        self._env.seed(ep_seed)
        obs = self._env.reset()
        return self._flatten_obs(obs), {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        terminated = bool(done)
        truncated = False
        return (
            self._flatten_obs(obs),
            float(reward),
            terminated,
            truncated,
            self._sanitize_info(info),
        )

    def close(self):
        self._env.close()


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
    for key in (
        "timesteps_total",
        "num_env_steps_sampled_lifetime",
        "agent_timesteps_total",
    ):
        if key in result:
            return _safe_int(result[key], 0)
    return 0


def _extract_reward_mean(result: Dict[str, Any]) -> float:
    # Ray 2.x tends to report under env_runners.
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
    """
    Flatten RLlib `algo.train()` metrics for CSV/JSONL export.
    Drops `config` and `timers` subtrees to keep files readable.
    """

    flat: Dict[str, Any] = {}

    def walk(obj: Any, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(obj, dict):
            for key, val in obj.items():
                ks = str(key)
                if ks == "config":
                    continue
                if ks == "timers":
                    continue
                path = f"{prefix}/{ks}" if prefix else ks
                if isinstance(val, dict):
                    walk(val, path, depth + 1)
                elif isinstance(val, (list, tuple)):
                    if len(val) == 1:
                        walk(val[0], path, depth + 1)
                else:
                    scalar = _to_export_scalar(val)
                    if scalar is not None:
                        flat[path] = scalar
        else:
            scalar = _to_export_scalar(obj)
            if scalar is not None and prefix:
                flat[prefix] = scalar

    walk(result, "", 0)
    return flat


def _write_training_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    preferred = [
        "iteration",
        "timesteps_total",
        "episode_reward_mean",
        "training_iteration",
        "iter_wall_time_sec",
    ]
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

    # New API may return actions directly or only distribution inputs.
    if Columns.ACTIONS in out:
        actions = _tensor_to_numpy(out[Columns.ACTIONS])
        return int(np.asarray(actions).reshape(-1)[0])
    if "actions" in out:
        actions = _tensor_to_numpy(out["actions"])
        return int(np.asarray(actions).reshape(-1)[0])
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
        env = GymnasiumDailyInvestorAdapter(env_config)
        obs, _ = env.reset(seed=seed)

        terminated = False
        truncated = False
        total_reward = 0.0
        actions: List[int] = []
        m2m_trace: List[float] = []

        while not (terminated or truncated):
            action_i = _inference_action(module, obs)
            obs, reward, terminated, truncated, info = env.step(action_i)

            total_reward += float(reward)
            actions.append(action_i)
            m2m_trace.append(_safe_float(info.get("marked_to_market"), env._env.unwrapped.starting_cash))

        env.close()

        starting_cash = float(env._env.unwrapped.starting_cash)
        final_m2m = m2m_trace[-1] if m2m_trace else starting_cash
        final_pnl = final_m2m - starting_cash
        max_drawdown = _compute_max_drawdown(m2m_trace, starting_cash)

        episodes.append(
            {
                "seed": int(seed),
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
            }
        )

    final_pnls = [ep["final_pnl"] for ep in episodes]
    total_rewards = [ep["total_reward"] for ep in episodes]
    lengths = [ep["episode_length"] for ep in episodes]
    max_dds = [ep["max_drawdown"] for ep in episodes]
    wins = [ep["won"] for ep in episodes]

    buy_total = sum(ep["action_counts"]["BUY"] for ep in episodes)
    hold_total = sum(ep["action_counts"]["HOLD"] for ep in episodes)
    sell_total = sum(ep["action_counts"]["SELL"] for ep in episodes)
    action_total = buy_total + hold_total + sell_total
    if action_total > 0:
        action_pct = {
            "BUY": float(buy_total / action_total),
            "HOLD": float(hold_total / action_total),
            "SELL": float(sell_total / action_total),
        }
    else:
        action_pct = {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0}

    pnl_stats = _summarize_array(final_pnls)
    std = pnl_stats["std"]
    sharpe = float(pnl_stats["mean"] / std) if std > 1e-12 else 0.0

    eval_metrics = {
        "n_episodes": int(len(episodes)),
        "final_pnl_cents": pnl_stats,
        "total_reward": _summarize_array(total_rewards),
        "episode_length": _summarize_array(lengths),
        "max_drawdown": _summarize_array(max_dds),
        "win_rate": float(np.mean(wins)) if wins else 0.0,
        "sharpe": sharpe,
        "action_pct": action_pct,
        "eval_wall_time_sec": float(time.time() - t_eval),
    }

    return {"episodes": episodes, "metrics": eval_metrics}


def parse_args():
    def str2bool(v: str) -> bool:
        if isinstance(v, bool):
            return v
        s = v.strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

    parser = argparse.ArgumentParser(description="Train baseline PPO on markets-daily_investor-v0")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--timestep", type=str, default="300s")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="results/ppo_baseline_smoke")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50,
        help="Save a checkpoint every N training iterations. "
        "0 disables mid-training checkpoints (final save still runs). "
        "Use 1 for smoke/debug; default 50 reduces I/O on longer runs.",
    )
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[100, 101, 102, 103, 104])
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=2000)
    parser.add_argument("--minibatch-size", "--sgd-minibatch-size", dest="minibatch_size", type=int, default=128)
    parser.add_argument("--num-epochs", "--num-sgd-iter", dest="num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--train-debug-mode", type=str2bool, default=False)
    parser.add_argument("--eval-debug-mode", type=str2bool, default=True)
    parser.add_argument("--train-info-mode", choices=["minimal", "full"], default="minimal")
    parser.add_argument("--eval-info-mode", choices=["minimal", "full"], default="full")
    parser.add_argument("--profile-phases", type=str2bool, default=False)
    parser.add_argument(
        "--per-episode-seed-base",
        type=int,
        default=None,
        help="When reset(seed=None), env RNG seed is base+offset+episode_idx. "
        "Defaults to --seed. Ignored when the caller passes an explicit reset seed.",
    )
    return parser.parse_args()


def _env_reset_step_probe(env_config: Dict[str, Any], n_steps: int = 5, seed: int = 0) -> Dict[str, float]:
    """Local single-process timing of adapter reset/step (not RLlib workers)."""
    env = GymnasiumDailyInvestorAdapter(env_config)
    t0 = time.time()
    env.reset(seed=seed)
    reset_sec = time.time() - t0
    t1 = time.time()
    for _ in range(n_steps):
        env.step(1)  # HOLD
    step_total_sec = time.time() - t1
    env.close()
    n = max(1, n_steps)
    return {
        "n_steps": int(n_steps),
        "reset_sec": float(reset_sec),
        "total_step_sec": float(step_total_sec),
        "mean_step_sec": float(step_total_sec / n),
    }


def build_env_config(args, debug_mode: bool, info_mode: str) -> Dict[str, Any]:
    env_config = dict(ENV_DEFAULTS)
    env_config["timestep_duration"] = args.timestep
    env_config["debug_mode"] = bool(debug_mode)
    env_config["adapter_info_mode"] = str(info_mode)
    base = args.seed if args.per_episode_seed_base is None else args.per_episode_seed_base
    env_config["per_episode_seed_base"] = int(base)
    env_config["background_config_extra_kvargs"] = dict(RMSC04_PARAMS)
    return env_config


def main():
    args = parse_args()
    if args.checkpoint_freq < 0:
        raise SystemExit("--checkpoint-freq must be >= 0 (0 = only final checkpoint)")
    out_dir = Path(args.out_dir)
    _ts_log(
        f"Starting run out_dir={out_dir} timesteps={args.timesteps} timestep={args.timestep} "
        f"train_seed={args.seed} workers={args.num_workers} num_gpus={args.num_gpus}"
    )
    checkpoints_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_path = str(checkpoints_dir.resolve())

    train_env_config = build_env_config(
        args,
        debug_mode=args.train_debug_mode,
        info_mode=args.train_info_mode,
    )
    eval_env_config = build_env_config(
        args,
        debug_mode=args.eval_debug_mode,
        info_mode=args.eval_info_mode,
    )

    run_config = {
        "timesteps_target": args.timesteps,
        "seed": args.seed,
        "per_episode_seed_base": args.per_episode_seed_base
        if args.per_episode_seed_base is not None
        else args.seed,
        "eval_seeds": args.eval_seeds,
        "env_id": ENV_ID,
        "env_config": {
            "train": train_env_config,
            "eval": eval_env_config,
        },
        "ppo": {
            "gamma": 1.0,
            "lr": args.lr,
            "train_batch_size": args.train_batch_size,
            "minibatch_size": args.minibatch_size,
            "num_epochs": args.num_epochs,
            "entropy_coeff": args.entropy_coeff,
            "framework": "torch",
            "num_gpus": args.num_gpus,
            "num_env_runners": args.num_workers,
            "env_runners": {
                "observation_filter": "MeanStdFilter",
            },
            "model_config": {
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "tanh",
            },
        },
        "profile_phases": bool(args.profile_phases),
        "checkpoint_freq": int(args.checkpoint_freq),
        "eval_after_train_only": True,
    }
    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    ray.shutdown()
    t_setup = time.time()
    _ray_init: Dict[str, Any] = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "log_to_driver": False,
    }
    _ray_tmp = os.environ.get("RAY_TMPDIR")
    if not _ray_tmp and os.environ.get("TMPDIR"):
        _ray_tmp = str(Path(os.environ["TMPDIR"]) / "ray")
    if _ray_tmp:
        Path(_ray_tmp).mkdir(parents=True, exist_ok=True)
        _ray_init["_temp_dir"] = _ray_tmp
    ray.init(**_ray_init)

    config = (
        PPOConfig()
        .environment(
            env=GymnasiumDailyInvestorAdapter,
            env_config=train_env_config,
            disable_env_checking=True,
        )
        .framework("torch")
        .resources(num_gpus=args.num_gpus)
        .env_runners(
            num_env_runners=args.num_workers,
            observation_filter="MeanStdFilter",
        )
        .training(
            gamma=1.0,
            lr=args.lr,
            train_batch_size=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
            entropy_coeff=args.entropy_coeff,
        )
        .rl_module(
            model_config={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "tanh",
            }
        )
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
        f"Ray+PPO setup done in {train_setup_sec:.1f}s. Starting training loop "
        f"(first algo.train() can take a long time: batch={args.train_batch_size}, "
        f"workers={args.num_workers}, per_episode_seed_base="
        f"{args.per_episode_seed_base if args.per_episode_seed_base is not None else args.seed})."
    )

    with training_jsonl_path.open("w", encoding="utf-8") as jsonl_f:
        while timesteps_total < args.timesteps:
            iteration += 1
            t_iter = time.time()
            _ts_log(
                f"PPO iteration {iteration} starting (target batch ~{args.train_batch_size} env steps)..."
            )
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
            flat_metrics = _flatten_train_result(result)
            row.update(flat_metrics)
            train_rows.append(row)
            jsonl_f.write(json.dumps(row, default=str) + "\n")
            jsonl_f.flush()

            _ts_log(
                f"iter={iteration:>3}  timesteps={timesteps_total:>8}  "
                f"reward_mean={episode_reward_mean:+.5f}  iter_time={iter_wall_sec:.2f}s"
            )

            if args.checkpoint_freq > 0 and iteration % args.checkpoint_freq == 0:
                ckpt_dir = algo.save_to_path(checkpoints_path)
                _ts_log(f"checkpoint -> {ckpt_dir}")

    # Always save a final checkpoint for reproducible eval.
    final_ckpt = algo.save_to_path(checkpoints_path)
    _ts_log(f"final checkpoint -> {final_ckpt}")

    train_loop_sec = time.time() - t_start
    wall_time_sec = train_loop_sec
    env_steps_per_sec = (timesteps_total / wall_time_sec) if wall_time_sec > 0 else 0.0
    train_iterations_per_hour = (iteration / wall_time_sec * 3600.0) if wall_time_sec > 0 else 0.0

    train_metrics = {
        "summary": {
            "timesteps_total": timesteps_total,
            "train_iterations": iteration,
            "wall_time_sec": wall_time_sec,
            "env_steps_per_sec": env_steps_per_sec,
            "train_iterations_per_hour": train_iterations_per_hour,
        },
        "artifacts": {
            "training_iterations_jsonl": str(training_jsonl_path.resolve()),
            "training_curve_csv": str((out_dir / "training_curve.csv").resolve()),
        },
        "iterations": train_rows,
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(train_metrics, indent=2), encoding="utf-8")
    _write_training_csv(out_dir / "training_curve.csv", train_rows)
    _ts_log(
        f"Wrote training data -> {training_jsonl_path.name}, training_curve.csv, train_metrics.json"
    )

    _ts_log(f"Starting evaluation on {len(args.eval_seeds)} seed(s)...")
    eval_payload = run_evaluation(algo, eval_env_config, args.eval_seeds)
    _ts_log("Evaluation finished; writing JSON artifacts.")
    (out_dir / "eval_episodes.json").write_text(
        json.dumps(eval_payload["episodes"], indent=2),
        encoding="utf-8",
    )
    (out_dir / "eval_metrics.json").write_text(
        json.dumps(eval_payload["metrics"], indent=2),
        encoding="utf-8",
    )

    timing: Dict[str, Any] = {
        "timesteps": args.timesteps,
        "timestep_duration": args.timestep,
        "wall_time_sec": wall_time_sec,
        "env_steps_per_sec": env_steps_per_sec,
        "train_iterations": iteration,
        "train_iterations_per_hour": train_iterations_per_hour,
        "eval_wall_time_sec": eval_payload["metrics"]["eval_wall_time_sec"],
    }
    if args.profile_phases:
        timing["phases"] = {
            "train_setup_sec": float(train_setup_sec),
            "train_loop_sec": float(train_loop_sec),
            "eval_sec": float(eval_payload["metrics"]["eval_wall_time_sec"]),
            "env_probe": _env_reset_step_probe(train_env_config, n_steps=5, seed=args.seed),
        }
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
