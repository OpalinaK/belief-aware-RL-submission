"""
belief_vs_value.py — Kalman belief signal vs raw direction feature
==================================================================

Compares two value-investing strategies head-to-head across identical seeds:

  ValueAgent (baseline)
    Trades on obs[3] = direction_feature = mid_price - last_transaction.
    This is the raw one-step "price above/below last trade" signal.

  BeliefValueAgent
    Same threshold logic, but trades on belief_dev = (mu_t - mid_t) / r_bar,
    the Kalman-filtered estimate of (fundamental - market price).
    Positive → tracker thinks the stock is undervalued → buy signal.

If BeliefValueAgent consistently outperforms (higher mean P&L, win rate,
or Sharpe), the Kalman filter is adding genuine signal — worth keeping as
a feature for RL training.

Usage
-----
  .venv/bin/python abides-gym/scripts/belief_vs_value.py
  .venv/bin/python abides-gym/scripts/belief_vs_value.py --episodes 30
  .venv/bin/python abides-gym/scripts/belief_vs_value.py --threshold 0.002
  .venv/bin/python abides-gym/scripts/belief_vs_value.py --no-plots
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import gym
    import abides_gym
except ImportError:
    sys.exit("ERROR: abides_gym not found. See CLAUDE.md setup section.")

from belief_tracker import BeliefAugmentedWrapper


# ─────────────────────────────────────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────────────────────────────────────

class ValueAgent:
    """
    Original VALUE baseline: trades on obs[3] = direction_feature.
    Buys when direction_feature > +thr (mid above last trade → momentum up).
    Sells when direction_feature < -thr.
    """
    name = "VALUE (direction_feature)"

    def __init__(self, threshold: float = 0.75):
        self.thr = threshold

    def act(self, obs: np.ndarray) -> int:
        direction = float(obs[3, 0])
        if direction > self.thr:
            return 2   # SELL — mid above last trade, reversion expected
        if direction < -self.thr:
            return 0   # BUY
        return 1


class BeliefValueAgent:
    """
    Kalman belief signal: trades on obs[-2] = belief_dev = (mu - mid) / r_bar.
    Positive belief_dev → Kalman filter thinks stock is underpriced → BUY.
    Threshold is in r_bar-normalised units (e.g. 0.002 = 0.2% of r_bar).
    """
    name = "BELIEF_VALUE (Kalman)"

    def __init__(self, threshold: float = 0.002):
        self.thr = threshold

    def act(self, obs: np.ndarray) -> int:
        belief_dev = float(obs[-2, 0])  # (mu - mid) / r_bar
        if belief_dev > self.thr:
            return 0   # BUY — fundamental above market
        if belief_dev < -self.thr:
            return 2   # SELL
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(env, agent, seed: int) -> dict:
    env.seed(seed)
    obs = env.reset()
    # Bug fix: reset previous_marked_to_market on the unwrapped env
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    starting_cash = unwrapped.starting_cash
    unwrapped.previous_marked_to_market = starting_cash

    done = False
    total_reward = 0.0
    step = 0
    m2m_trace, belief_devs, belief_stds = [], [], []

    while not done:
        action = agent.act(obs)
        try:
            obs, reward, done, info = env.step(action)
        except AssertionError:
            done = True
            break
        step += 1
        total_reward += reward
        m2m_trace.append(float(info.get("marked_to_market", np.nan)))

        # Capture belief features if available (obs has extra rows)
        if obs.shape[0] >= 2:
            belief_devs.append(float(obs[-2, 0]))
            belief_stds.append(float(obs[-1, 0]))

    valid_m2m = [v for v in m2m_trace if not np.isnan(v)]
    final_m2m = valid_m2m[-1] if valid_m2m else float(starting_cash)
    final_pnl = final_m2m - starting_cash

    m2m_arr = np.array([starting_cash] + [v if not np.isnan(v) else starting_cash
                                           for v in m2m_trace], dtype=float)
    peak = np.maximum.accumulate(m2m_arr)
    dd = np.where(peak > 0, (peak - m2m_arr) / peak, 0.0)

    return {
        "seed": seed,
        "final_pnl": final_pnl,
        "total_reward": total_reward,
        "episode_length": step,
        "won": final_pnl >= 0,
        "max_drawdown": float(np.max(dd)),
        "belief_devs": belief_devs,
        "belief_stds": belief_stds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(env, agent, n_episodes: int, label: str) -> list:
    results = []
    for seed in range(n_episodes):
        ep = run_episode(env, agent, seed)
        tag = "WIN " if ep["won"] else "LOSS"
        print(f"  {label:<30}  ep {seed+1:>2}/{n_episodes}  {tag}  "
              f"pnl={ep['final_pnl']:>+8.0f}¢  len={ep['episode_length']:>4}")
        results.append(ep)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Stats + summary
# ─────────────────────────────────────────────────────────────────────────────

def stats(results: list) -> dict:
    pnls = np.array([r["final_pnl"] for r in results])
    wins = np.array([r["won"] for r in results])
    dds  = np.array([r["max_drawdown"] for r in results])
    return {
        "mean_pnl":    float(np.mean(pnls)),
        "std_pnl":     float(np.std(pnls)),
        "sharpe":      float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 0 else 0.0,
        "win_rate":    float(np.mean(wins)),
        "mean_dd":     float(np.mean(dds)),
        "n":           len(results),
    }


def print_summary(label_results: dict):
    print("\n" + "=" * 78)
    print(f"{'Agent':<32} {'mean_pnl':>10} {'std_pnl':>9} {'sharpe':>7} "
          f"{'win%':>6} {'mean_dd':>8}")
    print("=" * 78)
    for label, s in label_results.items():
        print(f"  {label:<30} {s['mean_pnl']:>+10.1f} {s['std_pnl']:>9.1f} "
              f"{s['sharpe']:>7.3f} {s['win_rate']:>5.1%} {s['mean_dd']:>8.4f}")
    print("=" * 78)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {"VALUE (direction_feature)": "#FF9800",
           "BELIEF_VALUE (Kalman)":     "#2196F3"}


def _style():
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor":  "#16213e",
        "axes.edgecolor":   "#444466", "axes.labelcolor": "#ccccee",
        "axes.titlecolor":  "#ffffff", "xtick.color":     "#aaaacc",
        "ytick.color":      "#aaaacc", "grid.color":      "#2a2a4a",
        "grid.linestyle":   "--",      "grid.alpha":       0.5,
        "text.color":       "#ccccee", "legend.facecolor": "#1a1a2e",
        "legend.edgecolor": "#444466", "font.size":        11,
    })


def plot_comparison(all_results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    _style()

    labels = list(all_results.keys())
    colors = [PALETTE.get(l, "#9C27B0") for l in labels]

    # ── 1. P&L distribution ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, results, color in zip(labels, all_results.values(), colors):
        pnls = [r["final_pnl"] for r in results]
        ax.hist(pnls, bins=15, alpha=0.55, label=label, color=color, edgecolor="white", lw=0.4)
    ax.axvline(0, color="#ff4444", lw=1.5, ls="--", label="Break-even")
    ax.set_xlabel("Final P&L (cents)")
    ax.set_ylabel("Episodes")
    ax.set_title("P&L Distribution: Kalman Belief vs Raw Direction Feature")
    ax.legend(fontsize=9)
    ax.grid(axis="x")
    path = os.path.join(out_dir, "belief_pnl_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {path}")

    # ── 2. Win-rate + Sharpe bar chart ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    s_vals = {l: stats(r) for l, r in all_results.items()}

    for ax, metric, ylabel in zip(
        axes,
        ["win_rate", "sharpe"],
        ["Win Rate", "Sharpe Ratio (mean/std P&L)"],
    ):
        vals = [s_vals[l][metric] for l in labels]
        bars = ax.bar(labels, vals, color=colors, alpha=0.8, edgecolor="white", lw=0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="white")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.split(" ")[0] for l in labels], fontsize=9)
        ax.grid(axis="y")

    fig.suptitle("Kalman Belief vs Raw Direction Feature", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "belief_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {path}")

    # ── 3. Belief trajectory from last episode ───────────────────────────────
    belief_eps = all_results.get("BELIEF_VALUE (Kalman)", [])
    if belief_eps:
        last_ep = belief_eps[-1]
        devs = last_ep["belief_devs"]
        stds = last_ep["belief_stds"]
        if devs:
            _style()
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            steps = np.arange(len(devs))
            ax1.plot(steps, devs, color="#2196F3", lw=1.5)
            ax1.axhline(0, color="#ff4444", lw=1, ls="--")
            ax1.set_ylabel("belief_dev  (mu-mid)/r_bar")
            ax1.set_title(f"Kalman Belief Trajectory — episode seed {last_ep['seed']}")
            ax1.grid()
            ax2.plot(steps, stds, color="#4CAF50", lw=1.5)
            ax2.set_ylabel("belief_std  σ/r_bar")
            ax2.set_xlabel("Timestep")
            ax2.grid()
            fig.tight_layout()
            path = os.path.join(out_dir, "belief_trajectory.png")
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Kalman belief vs direction-feature comparison")
    p.add_argument("--episodes",  type=int,   default=20,
                   help="Episodes per agent (default: 20)")
    p.add_argument("--threshold", type=float, default=0.0007,
                   help="BeliefValueAgent threshold in r_bar-normalised units (default: 0.0007)")
    p.add_argument("--dir-threshold", type=float, default=0.75,
                   help="ValueAgent direction_feature threshold (default: 0.75)")
    p.add_argument("--r-bar",        type=int,   default=100_000)
    p.add_argument("--kappa-oracle", type=float, default=1.67e-16)
    p.add_argument("--fund-vol",     type=float, default=5e-5)
    p.add_argument("--timestep",     type=str,   default="60s")
    p.add_argument("--plots-dir",    type=str,   default="plots")
    p.add_argument("--no-plots",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Kalman Belief vs Raw Direction Feature — Head-to-Head")
    print("=" * 70)
    print(f"  episodes       : {args.episodes}")
    print(f"  belief thr     : ±{args.threshold} × r_bar")
    print(f"  direction thr  : ±{args.dir_threshold}")
    print(f"  r_bar          : {args.r_bar}")
    print(f"  kappa_oracle   : {args.kappa_oracle}")
    print(f"  fund_vol       : {args.fund_vol}")
    print()

    # ── VALUE agent on plain env ──────────────────────────────────────────────
    plain_env = gym.make(
        "markets-daily_investor-v0",
        background_config="rmsc04",
        debug_mode=True,
        timestep_duration=args.timestep,
        background_config_extra_kvargs={
            "r_bar":         args.r_bar,
            "kappa_oracle":  args.kappa_oracle,
            "fund_vol":      args.fund_vol,
        },
    )
    value_agent = ValueAgent(threshold=args.dir_threshold)
    print(f"── {value_agent.name} ──")
    value_results = run_agent(plain_env, value_agent, args.episodes, value_agent.name)
    plain_env.close()

    # ── BELIEF VALUE agent on wrapped env ─────────────────────────────────────
    base_env = gym.make(
        "markets-daily_investor-v0",
        background_config="rmsc04",
        debug_mode=True,
        timestep_duration=args.timestep,
        background_config_extra_kvargs={
            "r_bar":         args.r_bar,
            "kappa_oracle":  args.kappa_oracle,
            "fund_vol":      args.fund_vol,
        },
    )
    belief_env = BeliefAugmentedWrapper(
        base_env,
        r_bar=args.r_bar,
        kappa_oracle=args.kappa_oracle,
        fund_vol=args.fund_vol,
        timestep_duration=args.timestep,
    )
    belief_agent = BeliefValueAgent(threshold=args.threshold)
    print(f"\n── {belief_agent.name} ──")
    belief_results = run_agent(belief_env, belief_agent, args.episodes, belief_agent.name)
    belief_env.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    all_results = {
        value_agent.name:  value_results,
        belief_agent.name: belief_results,
    }
    summary = {l: stats(r) for l, r in all_results.items()}
    print_summary(summary)

    # ── Verdict ───────────────────────────────────────────────────────────────
    v_sharpe = summary[value_agent.name]["sharpe"]
    b_sharpe = summary[belief_agent.name]["sharpe"]
    v_win    = summary[value_agent.name]["win_rate"]
    b_win    = summary[belief_agent.name]["win_rate"]

    print("\nVerdict:")
    if b_sharpe > v_sharpe and b_win >= v_win:
        print("  ✓ BeliefValueAgent outperforms on both Sharpe and win rate.")
        print("    Kalman features add genuine signal — worth including in RL obs.")
    elif b_sharpe > v_sharpe:
        print("  ~ BeliefValueAgent has higher Sharpe but similar/lower win rate.")
        print("    Signal is present; threshold tuning may help.")
    elif b_win > v_win:
        print("  ~ BeliefValueAgent has higher win rate but lower Sharpe.")
        print("    Belief captures direction but with noisier P&L — check trajectory plot.")
    else:
        print("  ✗ No clear improvement from Kalman features at this threshold.")
        print("    Try --threshold 0.001 or check belief_trajectory.png for signal shape.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print(f"\nPlots → {args.plots_dir}/")
        plot_comparison(all_results, args.plots_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
