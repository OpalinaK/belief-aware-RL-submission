"""
compare_policies_kl.py — Pairwise KL divergence between trained PPO policies.

Loads checkpoints for v4 baseline, E1, E2', and E3, runs them on the same
fixed episodes, and computes pairwise KL divergence between action distributions
at every step.

Policies compared:
  v4  : PPO baseline (7-dim obs)
  E1  : PPO + oracle belief in obs (9-dim obs)
  E2p : PPO + reward shaping alpha=0.001 (9-dim obs)
  E3  : PPO + inferred belief (9-dim obs)

Obs mismatch fix: generate episodes via RegimeAdapter (9-dim obs).
Feed obs[:7] to v4 and obs[:9] to E1/E2'/E3.

Output: results/policy_kl_comparison/
  kl_matrix.json   — mean pairwise KL over all steps
  kl_by_regime.json — mean pairwise KL split by Val/Mom regime
  kl_over_time.json — mean KL per episode timestep (averaged across episodes)

Usage:
  .venv/bin/python abides-gym/scripts/compare_policies_kl.py
  .venv/bin/python abides-gym/scripts/compare_policies_kl.py --n-episodes 20 --seeds 100 101 102
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import abides_gym  # noqa: F401
from abides_gym.envs.regime_adapter import RegimeAdapter
from regime_belief_wrapper import RegimeRandomizedGymnasiumEnv

import ray

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

ENV_BASE = dict(
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
    debug_mode=False,
    adapter_info_mode="minimal",
    regime_mode="random",
    background_config_extra_kvargs=RMSC04_PARAMS,
)


def _ts_log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _get_action_probs(module, obs: np.ndarray) -> np.ndarray:
    """Return softmax action probabilities (3,) for a single obs."""
    batch = {Columns.OBS: torch.as_tensor(obs[None, :], dtype=torch.float32)}
    out = module.forward_inference(batch)
    for key in (Columns.ACTION_DIST_INPUTS, "action_dist_inputs"):
        if key in out:
            logits = _tensor_to_numpy(out[key])[0]
            logits = logits - logits.max()  # numerical stability
            exp_l = np.exp(logits)
            return exp_l / exp_l.sum()
    # fallback: one-hot on argmax action
    for key in (Columns.ACTIONS, "actions"):
        if key in out:
            a = int(np.asarray(_tensor_to_numpy(out[key])).reshape(-1)[0])
            probs = np.zeros(3, dtype=np.float32)
            probs[a] = 1.0
            return probs
    raise RuntimeError(f"Cannot extract action probs from keys: {list(out.keys())}")


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """KL(p‖q) — clipped to avoid log(0)."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def load_algo(checkpoint_path: str, env_cls, env_config: dict, seed: int = 0):
    """Build a PPO algo with the given env (sets obs space) and restore weights."""
    config = (
        PPOConfig()
        .environment(env=env_cls, env_config=env_config, disable_env_checking=True)
        .framework("torch")
        .resources(num_gpus=0.0)
        .env_runners(num_env_runners=0)
        .rl_module(model_config={"fcnet_hiddens": [64, 64], "fcnet_activation": "tanh"})
        .debugging(seed=seed)
    )
    algo = config.build_algo()
    algo.restore_from_path(checkpoint_path)
    return algo


def collect_episodes(seeds: List[int], env_config: dict) -> List[Dict]:
    """
    Run episodes via RegimeAdapter (9-dim obs) and collect per-step obs + regime.
    Returns list of dicts with keys: regime, obs_9 (T,9), obs_7 (T,7).
    """
    episodes = []
    for seed in seeds:
        env = RegimeAdapter(env_config)
        obs, info = env.reset(seed=seed)
        regime = info.get("regime", "unknown")
        obs_seq = []
        terminated = truncated = False
        while not (terminated or truncated):
            obs_seq.append(np.asarray(obs, dtype=np.float32))
            _, _, terminated, truncated, _ = env.step(1)  # HOLD to collect obs
        env.close()
        obs_arr = np.stack(obs_seq)  # (T, 9)
        episodes.append({
            "seed": seed,
            "regime": regime,
            "obs_9": obs_arr,
            "obs_7": obs_arr[:, :7],
        })
        _ts_log(f"  collected seed={seed} regime={regime} T={len(obs_seq)}")
    return episodes


def compute_kl_matrix(
    policy_names: List[str],
    modules: Dict[str, Any],
    obs_dims: Dict[str, int],
    episodes: List[Dict],
) -> Tuple[Dict, Dict, Dict]:
    """
    Compute pairwise KL divergences across all steps in all episodes.
    Returns (kl_matrix, kl_by_regime, kl_over_time).
    """
    n = len(policy_names)
    # accumulate per-step probs: {name: list of (T,3) arrays}
    all_probs: Dict[str, List[np.ndarray]] = {name: [] for name in policy_names}
    all_regimes: List[str] = []
    all_lengths: List[int] = []

    for ep in episodes:
        T = ep["obs_9"].shape[0]
        all_regimes.append(ep["regime"])
        all_lengths.append(T)
        for name in policy_names:
            dim = obs_dims[name]
            obs = ep["obs_7"] if dim == 7 else ep["obs_9"]
            probs_t = np.stack([_get_action_probs(modules[name], obs[t]) for t in range(T)])
            all_probs[name].append(probs_t)

    # Pairwise mean KL (overall)
    kl_matrix = {a: {b: 0.0 for b in policy_names} for a in policy_names}
    kl_by_regime: Dict[str, Dict] = {}
    for regime in ("val", "mom"):
        kl_by_regime[regime] = {a: {b: 0.0 for b in policy_names} for a in policy_names}

    total_steps = sum(all_lengths)
    regime_steps = {"val": 0, "mom": 0}
    for i, ep in enumerate(episodes):
        regime_steps[ep["regime"]] = regime_steps.get(ep["regime"], 0) + all_lengths[i]

    for a in policy_names:
        for b in policy_names:
            if a == b:
                continue
            kl_vals_all = []
            kl_vals_by_regime: Dict[str, List[float]] = {"val": [], "mom": []}
            for i, ep in enumerate(episodes):
                pa = all_probs[a][i]  # (T, 3)
                pb = all_probs[b][i]  # (T, 3)
                for t in range(pa.shape[0]):
                    kl = kl_divergence(pa[t], pb[t])
                    kl_vals_all.append(kl)
                    kl_vals_by_regime[ep["regime"]].append(kl)
            kl_matrix[a][b] = float(np.mean(kl_vals_all))
            for regime in ("val", "mom"):
                vals = kl_vals_by_regime[regime]
                kl_by_regime[regime][a][b] = float(np.mean(vals)) if vals else float("nan")

    # KL over episode timestep (mean across episodes, for each pair)
    max_T = max(all_lengths)
    kl_over_time: Dict[str, Dict[str, List[float]]] = {}
    for a in policy_names:
        kl_over_time[a] = {}
        for b in policy_names:
            if a == b:
                continue
            t_kls: List[List[float]] = [[] for _ in range(max_T)]
            for i, ep in enumerate(episodes):
                T = all_lengths[i]
                pa = all_probs[a][i]
                pb = all_probs[b][i]
                for t in range(T):
                    t_kls[t].append(kl_divergence(pa[t], pb[t]))
            kl_over_time[a][b] = [float(np.mean(v)) if v else float("nan") for v in t_kls]

    return kl_matrix, kl_by_regime, kl_over_time


def print_matrix(names: List[str], matrix: Dict) -> None:
    col_w = 10
    header = f"{'':12s}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    for a in names:
        row = f"{a:<12s}"
        for b in names:
            if a == b:
                row += f"{'0.000':>{col_w}}"
            else:
                row += f"{matrix[a][b]:>{col_w}.4f}"
        print(row)


def parse_args():
    base = Path(__file__).parent.parent.parent / "results"
    parser = argparse.ArgumentParser(description="Pairwise KL divergence between PPO policies")
    parser.add_argument("--v4-ckpt",   type=str, default=str(base / "ppo_baseline/seed_0/checkpoints"))
    parser.add_argument("--e1-ckpt",   type=str, default=str(base / "e1_belief_obs/seed_0/checkpoints"))
    parser.add_argument("--e2p-ckpt",  type=str, default=str(base / "e2_belief_reward_a001/seed_0/checkpoints"))
    parser.add_argument("--e3-ckpt",   type=str, default=str(base / "e3_inferred_belief/seed_0/checkpoints"))
    parser.add_argument("--e3-model",  type=str, default=str(Path(__file__).parent / "models/mlp_kalman_regime.pt"))
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--seeds",     nargs="+", type=int, default=list(range(200, 220)))
    parser.add_argument("--out-dir",   type=str, default=str(base / "policy_kl_comparison"))
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ray.shutdown()
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    # --- Env configs for each policy ---
    # v4 was trained on plain 7-dim obs — use RegimeRandomizedGymnasiumEnv
    # with belief_mode="none" which produces 7-dim obs matching the checkpoint.
    env_7dim_cfg = {
        "belief_mode": "none",
        "val_prob": 0.5,
        "debug_mode": False,
    }

    env_9dim_oracle = dict(ENV_BASE)  # E1/E2' use RegimeAdapter (9-dim with b*)

    env_9dim_e3 = {  # E3 uses RegimeRandomizedGymnasiumEnv with trained classifier
        "belief_mode": "trained",
        "trained_model_path": str(Path(args.e3_model).resolve()),
        "val_prob": 0.5,
        "debug_mode": False,
    }

    _ts_log("Loading policies...")

    from abides_gym.envs.regime_adapter import RegimeAdapter as _RegimeAdapter

    v4_algo  = load_algo(args.v4_ckpt,  RegimeRandomizedGymnasiumEnv,  env_7dim_cfg,    seed=0)
    e1_algo  = load_algo(args.e1_ckpt,  _RegimeAdapter,               env_9dim_oracle, seed=0)
    e2p_algo = load_algo(args.e2p_ckpt, _RegimeAdapter,               env_9dim_oracle, seed=0)
    e3_algo  = load_algo(args.e3_ckpt,  RegimeRandomizedGymnasiumEnv, env_9dim_e3,     seed=0)

    policy_names = ["v4", "E1", "E2p", "E3"]
    modules = {
        "v4":  v4_algo.get_module(),
        "E1":  e1_algo.get_module(),
        "E2p": e2p_algo.get_module(),
        "E3":  e3_algo.get_module(),
    }
    obs_dims = {"v4": 7, "E1": 9, "E2p": 9, "E3": 9}

    # --- Collect episodes via RegimeAdapter (gives 9-dim obs with oracle b*) ---
    _ts_log(f"Collecting {args.n_episodes} episodes (seeds {args.seeds[:args.n_episodes]})...")
    ep_env_config = dict(ENV_BASE)
    episodes = collect_episodes(args.seeds[:args.n_episodes], ep_env_config)

    # --- Compute KL ---
    _ts_log("Computing pairwise KL divergences...")
    kl_matrix, kl_by_regime, kl_over_time = compute_kl_matrix(
        policy_names, modules, obs_dims, episodes
    )

    # --- Print results ---
    print(f"\n{'='*60}")
    print("MEAN PAIRWISE KL DIVERGENCE  KL(row ‖ col)")
    print(f"{'='*60}")
    print_matrix(policy_names, kl_matrix)

    for regime in ("val", "mom"):
        print(f"\n--- {regime.upper()} regime ---")
        print_matrix(policy_names, kl_by_regime[regime])

    # --- Save ---
    results = {
        "policy_names": policy_names,
        "obs_dims": obs_dims,
        "n_episodes": len(episodes),
        "kl_matrix": kl_matrix,
        "kl_by_regime": kl_by_regime,
        "kl_over_time": kl_over_time,
        "episode_regimes": [ep["regime"] for ep in episodes],
    }
    out_path = out_dir / "kl_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    _ts_log(f"Saved to {out_path}")

    for algo in (v4_algo, e1_algo, e2p_algo, e3_algo):
        algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
