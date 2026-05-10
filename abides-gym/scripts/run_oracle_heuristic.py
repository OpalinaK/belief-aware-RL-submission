"""
run_oracle_heuristic.py — E0 oracle heuristic benchmark
========================================================

Runs MomentumAgentBaseline (Mom regime) and ValueAgentBaseline (Val regime)
under their matched market conditions to establish an oracle ceiling before
any PPO training.

Regime design (ρ = num_momentum_agents / num_value_agents, ρ* = 0.06):
  Val regime: num_momentum_agents=0  → ρ=0.000 < ρ* → b*=(1,0) → VALUE heuristic
  Mom regime: num_momentum_agents=12 → ρ=0.118 ≥ ρ* → b*=(0,1) → MOMENTUM heuristic

Protocol: 20 episodes × 2 regimes × 3 seeds = 120 episodes total.
All episodes use 60s timestep (v4 protocol) to be comparable to ppo_baseline.

Output:
  results/e0_oracle_heuristic/eval_metrics.json   — aggregate + per-regime stats
  results/e0_oracle_heuristic/trajectories/       — obs sequences (--save-trajectories)

Usage:
  .venv/bin/python abides-gym/scripts/run_oracle_heuristic.py
  .venv/bin/python abides-gym/scripts/run_oracle_heuristic.py --episodes 2 --seeds 0
  .venv/bin/python abides-gym/scripts/run_oracle_heuristic.py --save-trajectories
"""

import argparse
import json
import os
import sys
import traceback
import warnings
warnings.filterwarnings("ignore")

import numpy as np

try:
    import gym
    import abides_gym  # noqa: F401
except ImportError:
    sys.exit("ERROR: abides_gym not found. See CLAUDE.md setup section.")

from abides_gym.envs.markets_daily_investor_environment_v0 import (
    SubGymMarketsDailyInvestorEnv_v0,
)

# Import heuristic agents and shared constants from sweep.py
sys.path.insert(0, os.path.dirname(__file__))
from sweep import (
    MomentumAgentBaseline,
    ValueAgentBaseline,
    RMSC04_PARAMS,
    ENV_DEFAULTS,
)


def _compute_max_drawdown(m2m_trace, starting_cash):
    if not m2m_trace:
        return 0.0
    m2m = np.asarray([starting_cash] + list(m2m_trace), dtype=np.float64)
    peaks = np.maximum.accumulate(m2m)
    drawdowns = np.where(peaks > 0.0, (peaks - m2m) / peaks, 0.0)
    return float(np.max(drawdowns))

RHO_STAR = 0.06

REGIMES = {
    "val": {
        "num_momentum_agents": 0,
        "num_value_agents": 102,
        "belief": [1.0, 0.0],
        "agent_cls": ValueAgentBaseline,
        "label": "VALUE heuristic, Val regime (n_mom=0)",
    },
    "mom": {
        "num_momentum_agents": 12,
        "num_value_agents": 102,
        "belief": [0.0, 1.0],
        "agent_cls": MomentumAgentBaseline,
        "label": "MOMENTUM heuristic, Mom regime (n_mom=12)",
    },
}


def _make_env(regime_name: str) -> SubGymMarketsDailyInvestorEnv_v0:
    cfg = REGIMES[regime_name]
    extra = dict(RMSC04_PARAMS)
    extra["num_momentum_agents"] = cfg["num_momentum_agents"]
    extra["num_value_agents"] = cfg["num_value_agents"]
    env_kwargs = dict(ENV_DEFAULTS)
    env_kwargs.pop("debug_mode", None)
    return SubGymMarketsDailyInvestorEnv_v0(
        **env_kwargs,
        debug_mode=True,
        background_config_extra_kvargs=extra,
    )


def run_episode(env, agent, seed: int, save_traj: bool):
    env.seed(seed)
    if hasattr(agent, "reset"):
        agent.reset()

    obs = env.reset()
    env.previous_marked_to_market = env.starting_cash  # BUG 1

    starting_cash = env.starting_cash
    done = False
    step = 0
    total_reward = 0.0
    m2m_trace = []
    obs_sequence = []

    while not done:
        if save_traj:
            obs_sequence.append(obs.flatten().tolist())
        action = agent.act(obs)  # obs is (7,1) — do NOT flatten before act()
        try:
            obs, reward, done, info = env.step(action)
        except AssertionError:
            done = True
            break
        total_reward += float(reward)
        step += 1
        m2m = info.get("marked_to_market", starting_cash)
        m2m_trace.append(float(m2m))

    final_m2m = m2m_trace[-1] if m2m_trace else float(starting_cash)
    final_pnl = final_m2m - starting_cash
    max_dd = _compute_max_drawdown(m2m_trace, starting_cash)

    return {
        "final_pnl": final_pnl,
        "total_reward": total_reward,
        "won": final_pnl >= 0,
        "steps": step,
        "max_drawdown": max_dd,
        "obs_sequence": obs_sequence if save_traj else None,
    }


def _summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def run_regime(regime_name, episodes, seeds, save_traj, traj_dir):
    env = _make_env(regime_name)
    agent = REGIMES[regime_name]["agent_cls"]()
    belief = REGIMES[regime_name]["belief"]

    all_episodes = []
    for seed in seeds:
        for ep in range(episodes):
            ep_seed = seed * 1000 + ep
            try:
                result = run_episode(env, agent, ep_seed, save_traj)
            except Exception as exc:
                print(f"  [WARN] regime={regime_name} seed={seed} ep={ep} error: {exc}")
                traceback.print_exc()
                continue

            result["seed"] = seed
            result["episode"] = ep
            result["regime"] = regime_name
            result["belief"] = belief
            all_episodes.append(result)

            if save_traj and result.get("obs_sequence"):
                fname = os.path.join(
                    traj_dir, f"{regime_name}_seed{seed}_ep{ep:03d}.json"
                )
                with open(fname, "w") as f:
                    json.dump(
                        {
                            "regime": regime_name,
                            "belief": belief,
                            "seed": seed,
                            "obs": result["obs_sequence"],
                        },
                        f,
                    )

            print(
                f"  regime={regime_name} seed={seed} ep={ep:3d}  "
                f"pnl={result['final_pnl']:+.0f}  "
                f"steps={result['steps']}  "
                f"won={'Y' if result['won'] else 'N'}"
            )

    env.close()
    return all_episodes


def compute_metrics(episodes, regime_filter=None):
    eps = [e for e in episodes if regime_filter is None or e["regime"] == regime_filter]
    if not eps:
        return {}
    pnls = [e["final_pnl"] for e in eps]
    wins = [e["won"] for e in eps]
    dds = [e["max_drawdown"] for e in eps]
    pnl_stats = _summarize(pnls)
    std = pnl_stats["std"]
    sharpe = float(pnl_stats["mean"] / std) if std > 1e-12 else 0.0
    return {
        "n_episodes": len(eps),
        "final_pnl_cents": pnl_stats,
        "win_rate": float(np.mean(wins)),
        "sharpe": sharpe,
        "max_drawdown": _summarize(dds),
        "protocol": {"timestep_duration": "60s", "regime": regime_filter or "all"},
    }


def main():
    parser = argparse.ArgumentParser(description="E0 oracle heuristic benchmark")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out-dir", type=str, default="results/e0_oracle_heuristic")
    parser.add_argument("--save-trajectories", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    traj_dir = os.path.join(out_dir, "trajectories")
    os.makedirs(out_dir, exist_ok=True)
    if args.save_trajectories:
        os.makedirs(traj_dir, exist_ok=True)

    all_episodes = []
    for regime_name in ["val", "mom"]:
        print(f"\n{'='*60}")
        print(f"Regime: {REGIMES[regime_name]['label']}")
        print(f"{'='*60}")
        eps = run_regime(regime_name, args.episodes, args.seeds, args.save_trajectories, traj_dir)
        all_episodes.extend(eps)

    metrics = {
        "aggregate": compute_metrics(all_episodes),
        "val": compute_metrics(all_episodes, "val"),
        "mom": compute_metrics(all_episodes, "mom"),
        "seeds": args.seeds,
        "episodes_per_regime_per_seed": args.episodes,
    }

    out_path = os.path.join(out_dir, "eval_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote {out_path}")

    agg = metrics["aggregate"]["final_pnl_cents"]
    print(f"\n{'='*60}")
    print(f"AGGREGATE  n={metrics['aggregate']['n_episodes']}")
    print(f"  mean P&L : {agg['mean']:+.1f} cents")
    print(f"  std      : {agg['std']:.1f}")
    print(f"  Sharpe   : {metrics['aggregate']['sharpe']:.3f}")
    print(f"  win rate : {metrics['aggregate']['win_rate']:.1%}")

    for regime_name in ["val", "mom"]:
        r = metrics[regime_name]
        rp = r["final_pnl_cents"]
        print(f"\n  [{regime_name.upper()}]  n={r['n_episodes']}")
        print(f"    mean P&L : {rp['mean']:+.1f}  Sharpe : {r['sharpe']:.3f}  win : {r['win_rate']:.1%}")


if __name__ == "__main__":
    main()
