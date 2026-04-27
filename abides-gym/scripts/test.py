"""
run_baseline.py  —  ABIDES daily-investor baseline runner (AAPL calibration)
=============================================================================

Bugs fixed from previous version
---------------------------------
BUG 1 — previous_marked_to_market not reset between episodes (CRITICAL)
  core_environment.reset() rebuilds the kernel but does NOT reset
  `self.previous_marked_to_market` in the daily-investor env.
  After episode 1 ends at some M2M value, episode 2's dense rewards
  compute delta from the PREVIOUS episode's final M2M, not starting_cash.
  Fix: explicitly reset env.previous_marked_to_market = env.starting_cash
  immediately after every env.reset() call.

BUG 2 — HOLD win label wrong
  win condition was `final_pnl > 0`. HOLD always gives pnl = 0 exactly
  (holdings=0 all day, cash unchanged). 0 > 0 is False → every HOLD
  episode printed "LOSS" even though it broke even.
  Fix: win condition is `final_pnl >= 0`.

BUG 3 — env.seed() derives ABIDES seed via np_random, not directly
  env.seed(n) sets self.np_random, then env.reset() calls
  self.np_random.randint() to derive the ABIDES background seed.
  The script was correct to call env.seed() before env.reset(),
  but the seed must be called on the same env instance being reset.
  No code change needed — confirmed correct. Documented here for clarity.

Usage
-----
  python run_baseline.py                         # all 6 strategies, 50 episodes
  python run_baseline.py --episodes 10           # quick smoke test
  python run_baseline.py --strategies HOLD MR    # subset
  python run_baseline.py --timestep 300s         # 5-min steps (faster sims)
  python run_baseline.py --reward_mode sparse    # end-of-episode reward only
  python run_baseline.py --no-plots              # skip graphs, save JSON only

Output
------
  baseline_results.json   — per-episode data + summary stats (traces stripped)
  plots/1_m2m_pnl_distribution.png
  plots/2_reward_distribution.png
  plots/3_episode_length.png
  plots/4_sharpe_ratio.png
  plots/5_win_rate.png
  plots/6_max_drawdown_cdf.png
"""

import argparse
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

try:
    import gym
    import abides_gym
except ImportError:
    sys.exit(
        "ERROR: abides_gym not found.\n"
        "Install with: bash install.sh  (from abides-jpmc-public root)"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# AAPL MARKET CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

AAPL_PARAMS = dict(
    r_bar               = 19_000,     # long-run mean: ~$190 in cents
    kappa               = 1.67e-15,   # agent mean-reversion belief (per ns)
    kappa_oracle        = 1.67e-16,   # actual OU kappa (per ns)
    fund_vol            = 1.5e-4,     # OU vol → ~1.5 % daily move
    sigma_s             = 0,          # no megashocks in baseline
    lambda_a            = 5.7e-12,    # value-agent arrival rate
    num_value_agents    = 102,
    num_noise_agents    = 1000,
    num_momentum_agents = 12,
)

# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE STRATEGIES
# Action space: 0 = MKT BUY | 1 = HOLD | 2 = MKT SELL
# ═══════════════════════════════════════════════════════════════════════════════

class HoldBaseline:
    """Never trades. Reward = 0 every step. Zero-alpha lower bound."""
    name = "HOLD"
    def act(self, obs): return 1

class BuyAndHoldBaseline:
    """Buy on step 1, hold for the rest of the day."""
    name = "BUY_HOLD"
    def __init__(self): self._step = 0
    def reset(self):    self._step = 0
    def act(self, obs):
        self._step += 1
        return 0 if self._step == 1 else 1

class RandomBaseline:
    """Uniform random. High friction, no signal. Absolute floor."""
    name = "RANDOM"
    def __init__(self, seed=0): self._rng = np.random.RandomState(seed)
    def act(self, obs): return int(self._rng.randint(3))

class MomentumBaseline:
    """Follow order-book imbalance (obs[1], depth=3 levels)."""
    name = "MOMENTUM"
    def act(self, obs):
        imb = float(obs[1, 0])
        if   imb > 0.6: return 0   # bid pressure → BUY
        elif imb < 0.4: return 2   # ask pressure → SELL
        return 1

class MeanReversionBaseline:
    """Fade the last mid-price return (obs[4], in cents)."""
    name = "MR"
    def __init__(self, threshold=1.0): self._thr = threshold
    def act(self, obs):
        ret = float(obs[4, 0])
        if   ret < -self._thr: return 0   # dip → BUY
        elif ret >  self._thr: return 2   # rip → SELL
        return 1

class MeanReversionInventoryBaseline:
    """Mean-reversion with a soft inventory cap."""
    name = "MR_INV"
    def __init__(self, threshold=1.0, inventory_cap=5, order_fixed_size=10):
        self._thr       = threshold
        self._max_long  =  inventory_cap * order_fixed_size
        self._max_short = -inventory_cap * order_fixed_size
    def act(self, obs):
        ret  = float(obs[4, 0])
        hold = float(obs[0, 0])
        if ret < -self._thr and hold < self._max_long:  return 0
        if ret >  self._thr and hold > self._max_short: return 2
        return 1

ALL_STRATEGIES = {
    "HOLD":     HoldBaseline,
    "BUY_HOLD": BuyAndHoldBaseline,
    "RANDOM":   RandomBaseline,
    "MOMENTUM": MomentumBaseline,
    "MR":       MeanReversionBaseline,
    "MR_INV":   MeanReversionInventoryBaseline,
}

# ═══════════════════════════════════════════════════════════════════════════════
# EPISODE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_episode(env, agent, seed: int) -> dict:
    """
    Run one episode deterministically.

    Critical fix applied here (BUG 1):
      env.previous_marked_to_market is reset to env.starting_cash after
      every env.reset() call because core_environment.reset() does not
      do this itself — only __init__ does.
    """
    env.seed(seed)

    if hasattr(agent, "reset"):
        agent.reset()

    obs = env.reset()

    # ── BUG 1 FIX ─────────────────────────────────────────────────────────────
    # Without this line, dense rewards in episode N+1 measure delta from
    # episode N's final M2M instead of from starting_cash.
    env.previous_marked_to_market = env.starting_cash
    # ──────────────────────────────────────────────────────────────────────────

    starting_cash = env.starting_cash
    done          = False
    total_reward  = 0.0
    step          = 0

    step_rewards   = []
    step_actions   = []
    m2m_trace      = []
    holdings_trace = []
    spread_trace   = []

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)

        step         += 1
        total_reward += reward
        step_rewards.append(float(reward))
        step_actions.append(int(action))

        m2m_trace.append(     float(info.get("marked_to_market", np.nan)))
        holdings_trace.append(float(info.get("holdings",         np.nan)))
        spread_trace.append(  float(info.get("spread",           np.nan)))

    # ── episode-level metrics ─────────────────────────────────────────────────
    valid_m2m = [v for v in m2m_trace if not np.isnan(v)]
    final_m2m = valid_m2m[-1] if valid_m2m else float(starting_cash)
    final_pnl = final_m2m - starting_cash   # cents

    # Max drawdown over the episode (starting_cash is t=0 baseline)
    m2m_arr      = np.array(
        [starting_cash] + [v if not np.isnan(v) else starting_cash for v in m2m_trace],
        dtype=float
    )
    running_peak = np.maximum.accumulate(m2m_arr)
    dd_arr       = np.where(running_peak > 0,
                            (running_peak - m2m_arr) / running_peak, 0.0)
    max_dd = float(np.max(dd_arr))

    # Transaction cost proxy: ~half-spread per non-HOLD action
    trade_spreads = [spread_trace[i] for i, a in enumerate(step_actions)
                     if a != 1 and not np.isnan(spread_trace[i])]
    txn_cost = float(sum(trade_spreads) / 2) if trade_spreads else 0.0

    return {
        "seed":             seed,
        "total_reward":     float(total_reward),
        "final_m2m":        float(final_m2m),
        "final_pnl":        float(final_pnl),
        "episode_length":   int(step),
        "max_drawdown":     float(max_dd),
        "transaction_cost": float(txn_cost),
        # BUG 2 FIX: >= 0 not > 0  (HOLD gives exactly 0, not a loss)
        "won":              bool(final_pnl >= 0),
        "action_counts":    {"BUY":  step_actions.count(0),
                             "HOLD": step_actions.count(1),
                             "SELL": step_actions.count(2)},
        # kept in memory for plotting; stripped before JSON save
        "_m2m_trace":       m2m_trace,
        "_holdings_trace":  holdings_trace,
        "_spread_trace":    spread_trace,
        "_step_rewards":    step_rewards,
        "_step_actions":    step_actions,
    }


def summarise(episodes: list, starting_cash: int) -> dict:
    def stats(vals):
        a = np.array(vals, dtype=float)
        return {"mean": float(np.mean(a)), "std":  float(np.std(a)),
                "min":  float(np.min(a)),  "max":  float(np.max(a)),
                "p5":   float(np.percentile(a,  5)),
                "p25":  float(np.percentile(a, 25)),
                "p50":  float(np.percentile(a, 50)),
                "p75":  float(np.percentile(a, 75)),
                "p95":  float(np.percentile(a, 95))}

    pnls   = [e["final_pnl"] for e in episodes]
    sharpe = float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 1e-12 else 0.0

    ac  = {k: sum(e["action_counts"][k] for e in episodes) for k in ["BUY","HOLD","SELL"]}
    tot = max(sum(ac.values()), 1)

    return {
        "n_episodes":      len(episodes),
        "total_reward":    stats([e["total_reward"]     for e in episodes]),
        "final_pnl_cents": stats([e["final_pnl"]        for e in episodes]),
        "episode_length":  stats([e["episode_length"]   for e in episodes]),
        "max_drawdown":    stats([e["max_drawdown"]     for e in episodes]),
        "transaction_cost":stats([e["transaction_cost"] for e in episodes]),
        "sharpe":          sharpe,
        "win_rate":        float(np.mean([e["won"]      for e in episodes])),
        "action_pct":      {k: round(v/tot, 4) for k, v in ac.items()},
    }

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

PALETTE = ["#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0","#00BCD4","#FF5722","#607D8B"]

def _style():
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor":  "#16213e",
        "axes.edgecolor":   "#444466", "axes.labelcolor": "#ccccee",
        "axes.titlecolor":  "#ffffff", "xtick.color":     "#aaaacc",
        "ytick.color":      "#aaaacc", "grid.color":      "#2a2a4a",
        "grid.linestyle":   "--",      "grid.alpha":       0.5,
        "text.color":       "#ccccee", "legend.facecolor": "#1a1a2e",
        "legend.edgecolor": "#444466", "font.size":        11,
        "axes.titlesize":   13,        "axes.labelsize":   11,
    })

def _c(i): return PALETTE[i % len(PALETTE)]

def _save(fig, name, out_dir):
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {path}")


def plot_m2m(results, starting_cash, out_dir):
    """Violin + jitter of final P&L per strategy."""
    _style()
    names    = list(results.keys())
    pnl_data = [[e["final_pnl"] for e in results[n]["episodes"]] for n in names]

    fig, ax = plt.subplots(figsize=(max(8, len(names)*1.8), 5))
    rng = np.random.RandomState(42)

    for i, (name, data) in enumerate(zip(names, pnl_data)):
        c = _c(i)
        parts = ax.violinplot([data], positions=[i], widths=0.7,
                               showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(c); pc.set_alpha(0.4)
        parts["cmedians"].set_color(c); parts["cmedians"].set_linewidth(2)
        jitter = rng.uniform(-0.08, 0.08, len(data))
        ax.scatter(i + jitter, data, s=16, color=c, alpha=0.55, zorder=3)
        ax.scatter([i], [np.mean(data)], s=90, color=c, edgecolors="white",
                   linewidths=1.2, zorder=5, marker="D",
                   label=f"{name}  μ={np.mean(data):+.0f}¢")

    ax.axhline(0, color="#ff4444", lw=1.3, ls="--", label="Break-even")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Final P&L  (cents above starting_cash)")
    ax.set_title("Final Mark-to-Market P&L  ·  AAPL calibration\n"
                 "(◆ = mean, line = median, dots = individual episodes)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y")
    _save(fig, "1_m2m_pnl_distribution", out_dir)


def plot_reward(results, out_dir):
    """Box plot of total episode reward per strategy."""
    _style()
    names    = list(results.keys())
    rew_data = [[e["total_reward"] for e in results[n]["episodes"]] for n in names]

    fig, ax = plt.subplots(figsize=(max(8, len(names)*1.8), 5))
    bp = ax.boxplot(rew_data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", lw=2),
                    whiskerprops=dict(color="#888899"),
                    capprops=dict(color="#888899"),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4, color="#888899"))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(_c(i)); patch.set_alpha(0.6)

    means = [np.mean(d) for d in rew_data]
    ax.scatter(range(1, len(names)+1), means, s=80, color="white",
               zorder=5, marker="D", label="Mean")
    ax.axhline(0, color="#ff4444", lw=1.3, ls="--", label="Zero reward")
    ax.set_xticks(range(1, len(names)+1)); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Total Episode Reward  (normalised by order_size × num_steps)")
    ax.set_title("Total Episodic Reward Distribution\n"
                 "(box = IQR, whiskers = 1.5×IQR, ◆ = mean)")
    ax.legend(fontsize=9); ax.grid(axis="y")
    _save(fig, "2_reward_distribution", out_dir)


def plot_episode_length(results, max_steps, out_dir):
    """Overlaid histograms of episode lengths."""
    _style()
    names = list(results.keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, max_steps + 5, 40)

    for i, name in enumerate(names):
        lengths = [e["episode_length"] for e in results[name]["episodes"]]
        ax.hist(lengths, bins=bins, alpha=0.45, color=_c(i), density=True,
                label=f"{name}  (μ={np.mean(lengths):.1f})")

    ax.axvline(max_steps, color="white", lw=1.5, ls="--",
               label=f"Max = {max_steps} steps (full day)")
    ax.set_xlabel("Episode Length  (steps)")
    ax.set_ylabel("Density")
    ax.set_title("Episode Length Distribution\n"
                 "(short episodes = drawdown kill switch triggered)")
    ax.legend(fontsize=9); ax.grid(axis="y")
    _save(fig, "3_episode_length", out_dir)


def plot_sharpe(results, out_dir):
    """Horizontal bar chart of Sharpe ratios, sorted descending."""
    _style()
    names   = list(results.keys())
    sharpes = [results[n]["summary"]["sharpe"] for n in names]
    order   = np.argsort(sharpes)[::-1]
    ns, ss, cs = ([names[i] for i in order],
                  [sharpes[i] for i in order],
                  [_c(i) for i in order])

    fig, ax = plt.subplots(figsize=(8, max(4, len(names)*0.9)))
    bars = ax.barh(ns, ss, color=cs, alpha=0.75, height=0.5)
    for bar, val in zip(bars, ss):
        xoff = 0.02 if val >= 0 else -0.02
        ha   = "left" if val >= 0 else "right"
        ax.text(bar.get_x() + bar.get_width() + xoff,
                bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=10, color="white")
    ax.axvline(0, color="#ff4444", lw=1.3, ls="--")
    ax.set_xlabel("Sharpe Ratio  (mean P&L / std P&L,  across episodes)")
    ax.set_title("Sharpe Ratio per Strategy  ·  higher is better")
    ax.grid(axis="x")
    _save(fig, "4_sharpe_ratio", out_dir)


def plot_win_rate(results, out_dir):
    """Bar chart with 95 % Wilson confidence intervals."""
    from math import sqrt

    def wilson(wins, n, z=1.96):
        if n == 0: return 0.0, 0.0
        p = wins / n
        d = 1 + z**2/n
        c = (p + z**2/(2*n)) / d
        m = z * sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
        return max(0, c-m), min(1, c+m)

    _style()
    names = list(results.keys())
    wrs   = [results[n]["summary"]["win_rate"] for n in names]
    ns_ep = [results[n]["summary"]["n_episodes"] for n in names]

    lo_e, hi_e = [], []
    for wr, n in zip(wrs, ns_ep):
        lo, hi = wilson(round(wr*n), n)
        lo_e.append(wr - lo); hi_e.append(hi - wr)

    fig, ax = plt.subplots(figsize=(max(8, len(names)*1.8), 5))
    x = np.arange(len(names))
    bars = ax.bar(x, wrs, color=[_c(i) for i in range(len(names))],
                  alpha=0.75, width=0.55,
                  yerr=[lo_e, hi_e],
                  error_kw=dict(ecolor="white", elinewidth=1.5, capsize=5))
    for bar, wr in zip(bars, wrs):
        ax.text(bar.get_x() + bar.get_width()/2, min(wr + 0.03, 0.99),
                f"{wr:.1%}", ha="center", va="bottom", fontsize=10, color="white")
    ax.axhline(0.5, color="#ff4444", lw=1.3, ls="--",
               label="50 % reference  (random-walk baseline)")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Win Rate  (fraction of episodes with P&L ≥ 0)")
    ax.set_title("Win Rate per Strategy\n(error bars = 95 % Wilson CI)")
    ax.legend(fontsize=9); ax.grid(axis="y")
    _save(fig, "5_win_rate", out_dir)


def plot_max_drawdown(results, out_dir):
    """CDF of per-episode max drawdown — lower-left is safer."""
    _style()
    names = list(results.keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(names):
        dds = sorted([e["max_drawdown"] for e in results[name]["episodes"]])
        cdf = np.arange(1, len(dds)+1) / len(dds)
        mu  = np.mean(dds)
        ax.plot(dds, cdf, color=_c(i), lw=2, label=f"{name}  (μ={mu:.3f})")
        ax.axvline(mu, color=_c(i), lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("Max Drawdown  (fraction of peak M2M)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Max Drawdown CDF  ·  lower-left = safer\n"
                 "(dotted vertical = mean per strategy)")
    ax.legend(fontsize=9); ax.grid()
    _save(fig, "6_max_drawdown_cdf", out_dir)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",      type=int,   default=50)
    p.add_argument("--strategies",    nargs="+",  default=list(ALL_STRATEGIES.keys()),
                   choices=list(ALL_STRATEGIES.keys()))
    p.add_argument("--timestep",      type=str,   default="60s")
    p.add_argument("--reward_mode",   type=str,   default="dense",
                   choices=["dense","sparse"])
    p.add_argument("--starting_cash", type=int,   default=1_000_000)
    p.add_argument("--order_size",    type=int,   default=10)
    p.add_argument("--done_ratio",    type=float, default=0.3)
    p.add_argument("--out_dir",       type=str,   default="plots")
    p.add_argument("--results_file",  type=str,   default="baseline_results.json")
    p.add_argument("--no-plots",      action="store_true")
    p.add_argument("--seed_offset",   type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # compute theoretical max steps per episode
    ns_day    = int((16 - 9.5) * 60 * 60 * 1e9)
    first_ns  = int(5 * 60 * 1e9)
    step_sec  = int(args.timestep.lower().replace("s",""))
    step_ns   = int(step_sec * 1e9)
    max_steps = (ns_day - first_ns) // step_ns
    seeds     = [args.seed_offset + i for i in range(args.episodes)]

    print("=" * 62)
    print("ABIDES  ·  markets-daily_investor-v0  ·  AAPL calibration")
    print("=" * 62)
    print(f"  Strategies    : {args.strategies}")
    print(f"  Episodes      : {args.episodes}  (seeds {seeds[0]}–{seeds[-1]})")
    print(f"  Timestep      : {args.timestep}  ({max_steps} max steps/ep)")
    print(f"  Reward mode   : {args.reward_mode}")
    print(f"  Order size    : {args.order_size} shares")
    print(f"  Starting cash : {args.starting_cash:,} ¢  "
          f"(= ${args.starting_cash/100:,.2f})")
    print(f"  Done ratio    : {args.done_ratio}  "
          f"(kill at M2M ≤ ${args.done_ratio*args.starting_cash/100:,.0f})")
    print()

    env = gym.make(
        "markets-daily_investor-v0",
        background_config              = "rmsc04",
        mkt_close                      = "16:00:00",
        timestep_duration              = args.timestep,
        starting_cash                  = args.starting_cash,
        order_fixed_size               = args.order_size,
        state_history_length           = 4,
        market_data_buffer_length      = 5,
        first_interval                 = "00:05:00",
        reward_mode                    = args.reward_mode,
        done_ratio                     = args.done_ratio,
        debug_mode                     = True,   # exposes M2M, cash, holdings in info
        background_config_extra_kvargs = AAPL_PARAMS,
    )

    all_results = {}

    for strat_name in args.strategies:
        agent = ALL_STRATEGIES[strat_name]()
        print(f"{'─'*62}")
        print(f"Strategy: {strat_name}")
        print(f"{'─'*62}")

        t0       = time.time()
        episodes = []

        for ep_idx, seed in enumerate(seeds):
            ep = run_episode(env, agent, seed)
            episodes.append(ep)

            tag = "WIN " if ep["won"] else "LOSS"
            print(
                f"  ep {ep_idx+1:>3}/{args.episodes} | seed={seed:>4} | "
                f"{tag} | pnl={ep['final_pnl']:>+8.0f}¢ | "
                f"len={ep['episode_length']:>4} | "
                f"dd={ep['max_drawdown']:.3f} | "
                f"r={ep['total_reward']:>+.2e}"
            )

        summary = summarise(episodes, args.starting_cash)
        all_results[strat_name] = {"summary": summary, "episodes": episodes}

        print(f"\n  ── Summary ({time.time()-t0:.1f}s) ──")
        print(f"  Mean P&L    : {summary['final_pnl_cents']['mean']:+.1f} ¢"
              f"  ±{summary['final_pnl_cents']['std']:.1f}")
        print(f"  Reward μ/σ  : {summary['total_reward']['mean']:+.3e}"
              f"  ±{summary['total_reward']['std']:.3e}")
        print(f"  Sharpe      : {summary['sharpe']:+.3f}")
        print(f"  Win rate    : {summary['win_rate']:.1%}")
        print(f"  Mean length : {summary['episode_length']['mean']:.1f} steps")
        print(f"  Mean max-DD : {summary['max_drawdown']['mean']:.3f}")
        print(f"  Actions     : {summary['action_pct']}\n")

    env.close()

    # ── comparison table ──────────────────────────────────────────────────────
    print("=" * 78)
    print(f"{'Strategy':<12} {'P&L (¢)':>14}  {'Reward':>12}  "
          f"{'Sharpe':>8}  {'Win%':>7}  {'Steps':>7}  {'MaxDD':>7}")
    print("=" * 78)
    for name, data in all_results.items():
        s = data["summary"]
        print(f"{name:<12} "
              f"{s['final_pnl_cents']['mean']:>+9.1f}±{s['final_pnl_cents']['std']:<6.1f} "
              f"{s['total_reward']['mean']:>+11.3e}  "
              f"{s['sharpe']:>+8.3f}  "
              f"{s['win_rate']:>6.1%}  "
              f"{s['episode_length']['mean']:>7.1f}  "
              f"{s['max_drawdown']['mean']:>7.4f}")
    print("=" * 78)

    # ── save JSON (strip per-step traces to keep file small) ─────────────────
    STRIP = {"_m2m_trace","_holdings_trace","_spread_trace",
             "_step_rewards","_step_actions"}
    to_save = {}
    for name, data in all_results.items():
        to_save[name] = {
            "summary":  data["summary"],
            "episodes": [{k: v for k, v in ep.items() if k not in STRIP}
                         for ep in data["episodes"]],
        }
    with open(args.results_file, "w") as f:
        json.dump(to_save, f, indent=2, default=str)
    print(f"\nResults → {args.results_file}")

    # ── plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print(f"Plots   → {args.out_dir}/")
        plot_m2m(           all_results, args.starting_cash, args.out_dir)
        plot_reward(        all_results,                     args.out_dir)
        plot_episode_length(all_results, max_steps,          args.out_dir)
        plot_sharpe(        all_results,                     args.out_dir)
        plot_win_rate(      all_results,                     args.out_dir)
        plot_max_drawdown(  all_results,                     args.out_dir)
        print("Done.")


if __name__ == "__main__":
    main()