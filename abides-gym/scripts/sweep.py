"""
sweep.py  —  Daily investor parameter sweep
============================================

Runs one-at-a-time (OAT) sweeps over background config parameters and env-level
parameters, measuring each of the six baseline strategies' M2M P&L.

Usage
-----
  python sweep.py                          # full sweep, 20 episodes/cell
  python sweep.py --fast                   # 10 eps, 300s timestep, quick pass
  python sweep.py --episodes 5             # custom episode count
  python sweep.py --params mm_wake_up_freq num_value_agents   # subset of params
  python sweep.py --strategies HOLD MR MR_INV                 # subset of strategies
  python sweep.py --out results.json       # custom output file
  python sweep.py --no-plots               # skip plots, save JSON only

Output
------
  sweep_results.json              — one row per (param, value, strategy, seed)
  plots/sweep_<param>.png         — mean P&L bar chart per swept parameter
  plots/sweep_crossover.png       — MR vs MOMENTUM as num_value_agents varies
  stdout                          — progress lines + final summary table
"""

import argparse
import importlib
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── import shared pieces from test.py without duplication ────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
_test = importlib.import_module("test")

# plot helpers from test.py
_style   = _test._style
_save    = _test._save
_c       = _test._c
PALETTE  = _test.PALETTE

HoldBaseline                = _test.HoldBaseline
BuyAndHoldBaseline          = _test.BuyAndHoldBaseline
RandomBaseline              = _test.RandomBaseline
MomentumBaseline            = _test.MomentumBaseline
MeanReversionBaseline       = _test.MeanReversionBaseline
MeanReversionInventoryBaseline = _test.MeanReversionInventoryBaseline
run_episode                 = _test.run_episode
summarise                   = _test.summarise

try:
    import gym
    import abides_gym
except ImportError:
    sys.exit("ERROR: abides_gym not found.\nInstall: see CLAUDE.md setup section.")

# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION & DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

AAPL_PARAMS = dict(
    r_bar               = 19_000,
    kappa               = 1.67e-15,
    kappa_oracle        = 1.67e-16,
    fund_vol            = 1.5e-4,
    sigma_s             = 0,
    lambda_a            = 5.7e-12,
    num_value_agents    = 102,
    num_noise_agents    = 1000,
    num_momentum_agents = 12,
)

ENV_DEFAULTS = dict(
    background_config         = "rmsc04",
    timestep_duration         = "60s",
    starting_cash             = 1_000_000,
    order_fixed_size          = 10,
    state_history_length      = 4,
    market_data_buffer_length = 5,
    first_interval            = "00:05:00",
    reward_mode               = "dense",
    done_ratio                = 0.3,
    debug_mode                = True,
)

ALL_STRATEGIES = {
    "HOLD":     HoldBaseline,
    "BUY_HOLD": BuyAndHoldBaseline,
    "MR":       MeanReversionBaseline,
    "MR_INV":   MeanReversionInventoryBaseline,
    "MOMENTUM": MomentumBaseline,
    "RANDOM":   RandomBaseline,
}

# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

BACKGROUND_SWEEP = {
    # value agent — fundamental anchoring
    "num_value_agents":    [10, 102, 300],
    "lambda_a":            [5.7e-13, 5.7e-12, 5.7e-11],
    "kappa":               [1.67e-16, 1.67e-15, 1.67e-14],
    # market structure
    "mm_wake_up_freq":     ["10S", "60S", "120S"],
    "mm_spread_alpha":     [0.25, 0.75, 0.95],
    "mm_window_size":      [2, 5, "adaptive"],
    "num_noise_agents":    [100, 500, 1000],
    "num_momentum_agents": [0, 12, 50],
    "fund_vol":            [1e-5, 1.5e-4, 5e-4],
    "kappa_oracle":        [1e-17, 1.67e-16, 1e-15],
}

ENV_SWEEP = {
    "timestep_duration": ["30s", "60s", "300s"],
    "order_fixed_size":  [5, 10, 25],
}

# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def make_env(env_kwargs, bg_kwargs):
    env = gym.make(
        "markets-daily_investor-v0",
        **env_kwargs,
        background_config_extra_kvargs=bg_kwargs,
    )
    return env


def run_param(param, val, env_kwargs, bg_kwargs, strategies, n_episodes, results):
    """
    Run all requested strategies for n_episodes each under a single (param, val) config.
    Appends one row per (strategy, episode) to results.
    """
    env = make_env(env_kwargs, bg_kwargs)

    for strat_name, Strat in strategies.items():
        agent = Strat()
        for seed in range(n_episodes):
            t0 = time.time()
            ep = run_episode(env, agent, seed)
            elapsed = time.time() - t0

            tag = "WIN " if ep["won"] else "LOSS"
            print(
                f"  {param}={val!s:<12}  {strat_name:<8}  "
                f"ep {seed+1:>2}/{n_episodes}  {tag}  "
                f"pnl={ep['final_pnl']:>+8.0f}¢  "
                f"len={ep['episode_length']:>4}  "
                f"({elapsed:.1f}s)"
            )

            results.append({
                "param":            param,
                "value":            str(val),
                "strategy":         strat_name,
                "seed":             seed,
                "final_pnl":        ep["final_pnl"],
                "total_reward":     ep["total_reward"],
                "max_drawdown":     ep["max_drawdown"],
                "transaction_cost": ep["transaction_cost"],
                "won":              ep["won"],
                "episode_length":   ep["episode_length"],
            })

    env.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results, strategies):
    try:
        import pandas as pd
    except ImportError:
        print("(pandas not available — skipping summary table)")
        return

    df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print(f"{'param':<22} {'value':<14} {'strategy':<10} "
          f"{'mean_pnl':>10} {'std_pnl':>10} {'sharpe':>8} {'win%':>7} {'n':>4}")
    print("=" * 90)

    for (param, value, strategy), grp in df.groupby(["param", "value", "strategy"]):
        pnls = grp["final_pnl"].values
        mean = np.mean(pnls)
        std  = np.std(pnls)
        sharpe = mean / std if std > 1e-12 else 0.0
        win_rate = np.mean(grp["won"].values)
        print(f"  {param:<20} {value:<14} {strategy:<10} "
              f"{mean:>+10.1f} {std:>10.1f} {sharpe:>8.3f} {win_rate:>6.1%} {len(pnls):>4}")

    print("=" * 90)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_param(df, param, out_dir):
    """
    Bar chart of mean P&L ± 1 std per strategy, one group per sweep value.
    One figure per swept parameter.
    """
    import pandas as pd

    data = df[df["param"] == param]
    values   = list(data["value"].unique())
    strats   = list(data["strategy"].unique())
    n_vals   = len(values)
    n_strats = len(strats)

    x      = np.arange(n_vals)
    width  = 0.8 / n_strats
    offset = np.linspace(-(0.8 - width) / 2, (0.8 - width) / 2, n_strats)

    _style()
    fig, ax = plt.subplots(figsize=(max(8, n_vals * 2.2), 5))

    for i, strat in enumerate(strats):
        grp   = data[data["strategy"] == strat]
        means = [grp[grp["value"] == v]["final_pnl"].mean() for v in values]
        stds  = [grp[grp["value"] == v]["final_pnl"].std()  for v in values]
        ax.bar(x + offset[i], means, width, yerr=stds,
               label=strat, color=_c(i), alpha=0.75,
               error_kw=dict(ecolor="white", elinewidth=1.2, capsize=3))

    ax.axhline(0, color="#ff4444", lw=1.2, ls="--", label="Break-even")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in values], fontsize=9)
    ax.set_xlabel(param)
    ax.set_ylabel("Mean Final P&L  (cents)")
    ax.set_title(f"Strategy P&L vs {param}\n(bars = mean ± 1 std across episodes)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y")
    _save(fig, f"sweep_{param}", out_dir)


def plot_crossover(df, out_dir):
    """
    Line chart: MR vs MOMENTUM mean P&L as num_value_agents varies.
    Highlights the regime crossover point most relevant for RL training.
    """
    data = df[df["param"] == "num_value_agents"]
    if data.empty:
        return

    strats  = ["MR", "MOMENTUM"]
    present = [s for s in strats if s in data["strategy"].unique()]
    if not present:
        return

    values = sorted(data["value"].unique(), key=lambda v: float(v))

    _style()
    fig, ax = plt.subplots(figsize=(7, 4))

    for i, strat in enumerate(present):
        grp   = data[data["strategy"] == strat]
        means = [grp[grp["value"] == v]["final_pnl"].mean() for v in values]
        stds  = [grp[grp["value"] == v]["final_pnl"].std()  for v in values]
        ax.plot(values, means, marker="o", color=_c(i), lw=2, label=strat)
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        ax.fill_between(values, lo, hi, color=_c(i), alpha=0.15)

    ax.axhline(0, color="#ff4444", lw=1.2, ls="--", label="Break-even")
    ax.set_xlabel("num_value_agents  (fundamental anchoring strength)")
    ax.set_ylabel("Mean Final P&L  (cents)")
    ax.set_title("MR vs MOMENTUM crossover\nas value-agent count varies")
    ax.legend(fontsize=9)
    ax.grid()
    _save(fig, "sweep_crossover", out_dir)


def plot_all(results, out_dir):
    try:
        import pandas as pd
    except ImportError:
        print("(pandas not available — skipping plots)")
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)

    params = df["param"].unique()
    for param in params:
        plot_param(df, param, out_dir)

    plot_crossover(df, out_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    all_bg  = list(BACKGROUND_SWEEP.keys())
    all_env = list(ENV_SWEEP.keys())
    p = argparse.ArgumentParser(description="Daily investor parameter sweep")
    p.add_argument("--episodes",   type=int, default=20,
                   help="Episodes per (param, value, strategy) cell (default: 20)")
    p.add_argument("--strategies", nargs="+", default=list(ALL_STRATEGIES.keys()),
                   choices=list(ALL_STRATEGIES.keys()))
    p.add_argument("--params",     nargs="+", default=all_bg + all_env,
                   choices=all_bg + all_env,
                   help="Subset of sweep parameters to run")
    p.add_argument("--fast",       action="store_true",
                   help="Quick mode: 10 episodes, 300s timestep")
    p.add_argument("--out",        type=str, default="sweep_results.json")
    p.add_argument("--plots-dir",  type=str, default="plots")
    p.add_argument("--no-plots",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    n_episodes = 10 if args.fast else args.episodes
    fast_timestep = "300s" if args.fast else None

    strategies = {k: v for k, v in ALL_STRATEGIES.items() if k in args.strategies}
    bg_params  = {k: v for k, v in BACKGROUND_SWEEP.items() if k in args.params}
    env_params = {k: v for k, v in ENV_SWEEP.items()        if k in args.params}

    total_cells = (sum(len(v) for v in bg_params.values()) +
                   sum(len(v) for v in env_params.values()))
    total_episodes = total_cells * len(strategies) * n_episodes

    print("=" * 70)
    print("Daily Investor Parameter Sweep")
    print("=" * 70)
    print(f"  Strategies : {list(strategies.keys())}")
    print(f"  Episodes   : {n_episodes} per cell")
    print(f"  Timestep   : {fast_timestep or ENV_DEFAULTS['timestep_duration']}")
    print(f"  BG params  : {list(bg_params.keys())}")
    print(f"  Env params : {list(env_params.keys())}")
    print(f"  Total eps  : ~{total_episodes}")
    print()

    results = []
    t_start = time.time()

    # ── background config sweeps ──────────────────────────────────────────────
    env_kw = dict(ENV_DEFAULTS)
    if fast_timestep:
        env_kw["timestep_duration"] = fast_timestep

    for param, values in bg_params.items():
        for val in values:
            print(f"{'─'*70}")
            print(f"  param={param}  val={val}")
            bg = {**AAPL_PARAMS, param: val}
            run_param(param, val, env_kw, bg, strategies, n_episodes, results)

    # ── env-level sweeps ──────────────────────────────────────────────────────
    for param, values in env_params.items():
        for val in values:
            print(f"{'─'*70}")
            print(f"  param={param}  val={val}")
            kw = {k: v for k, v in env_kw.items() if k != param}
            kw[param] = val
            run_param(param, val, kw, AAPL_PARAMS, strategies, n_episodes, results)

    # ── save ──────────────────────────────────────────────────────────────────
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults → {args.out}  ({len(results)} rows, {time.time()-t_start:.0f}s total)")

    print_summary(results, strategies)

    # ── plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print(f"\nPlots   → {args.plots_dir}/")
        plot_all(results, args.plots_dir)
        print("Done.")


if __name__ == "__main__":
    main()
