"""
sim_agent_baselines.py
======================
Runs N full-day ABIDES-markets simulations directly (no Gym wrapper) and
extracts true behavioural baselines for every agent type as they actually
operate inside the kernel.

What this measures (per agent type, across all agents in all seeds)
--------------------------------------------------------------------
  1. Final Mark-to-Market P&L      — cash + shares × closing_price − starting_cash
  2. Total episode reward proxy     — normalised P&L (comparable to Gym dense reward)
  3. Episode length                 — for direct agents: always full day; used here
                                     to measure active trading window (first to last
                                     ORDER_EXECUTED event, in minutes)
  4. Sharpe ratio                   — mean P&L / std P&L across all agent instances
  5. Win rate                       — fraction of agents with P&L > 0
  6. Max intraday drawdown          — reconstructed from HOLDINGS_UPDATED log events
                                     combined with the market mid-price at each moment

Why this is more faithful than the Gym policy ports
----------------------------------------------------
  • Agents use their REAL wakeup schedules, message passing, and order logic.
  • ValueAgents call the actual oracle with a Kalman filter.
  • MomentumAgents use their actual MA(20)/MA(50) rolling lists.
  • MarketMakers post real multi-level limit order ladders.
  • NoiseAgents draw from the real OrderSizeModel distribution.

Usage
-----
  python sim_agent_baselines.py                  # 10 seeds, full trading day
  python sim_agent_baselines.py --seeds 50       # 50 seeds (~3–4 hrs)
  python sim_agent_baselines.py --seeds 5 --end_time 11:00:00   # shorter day
  python sim_agent_baselines.py --no-plots       # skip graphs, save JSON only

Output
------
  sim_baseline_results.json
  sim_plots/1_m2m_pnl_distribution.png
  sim_plots/2_reward_distribution.png
  sim_plots/3_active_window_distribution.png
  sim_plots/4_sharpe_ratio.png
  sim_plots/5_win_rate.png
  sim_plots/6_max_drawdown_cdf.png

Compatibility fixes applied at import time
------------------------------------------
  • pomegranate replaced with numpy-equivalent OrderSizeModel
  • pandas Timestamp.to_datetime64() → Timestamp.value (newer pandas)
  • coloredlogs and all abides logging suppressed
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Local repo packages (when not installed with pip -e) ───────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _pkg in (_REPO_ROOT / "abides-markets", _REPO_ROOT / "abides-core"):
    _p = str(_pkg)
    if _pkg.is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)

# abides_core.abides imports coloredlogs at module load; optional dependency stub
if "coloredlogs" not in sys.modules:
    import types

    _cl = types.SimpleNamespace()
    _cl.install = lambda *args, **kwargs: None
    sys.modules["coloredlogs"] = _cl

# ── Verify ABIDES is importable ───────────────────────────────────────────────
try:
    from abides_markets.configs import rmsc04
    from abides_core import abides
except ImportError as e:
    sys.exit(
        "ERROR: abides_markets / abides_core could not be imported.\n"
        f"  ({e})\n"
        "From the repo root, run:\n"
        "  pip install -e abides-core -e abides-markets\n"
        "or run this script from the cloned project so abides-markets/ and "
        "abides-core/ sit next to abides-gym/."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# RMSC04 CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

RMSC04_PARAMS = dict(
    starting_cash        = 10_000_000,   # $100,000 per agent
    num_noise_agents     = 1000,
    num_value_agents     = 102,
    num_momentum_agents  = 12,
    r_bar                = 100_000,      # $1,000 fundamental (cents)
    kappa                = 1.67e-15,
    kappa_oracle         = 1.67e-16,
    lambda_a             = 5.7e-12,
    sigma_s              = 0,
    fund_vol             = 5e-5,
    megashock_lambda_a   = 2.77778e-18,
    megashock_mean       = 1000,
    megashock_var        = 50_000,
    mm_window_size       = "adaptive",
    mm_pov               = 0.025,
    mm_num_ticks         = 10,
    mm_wake_up_freq      = "60S",
    mm_min_order_size    = 1,
    mm_skew_beta         = 0,
    mm_price_skew        = 4,
    mm_level_spacing     = 5,
    mm_spread_alpha      = 0.75,
    mm_backstop_quantity = 0,
    mm_cancel_limit_delay= 50,
)

# Agent type names as they appear in ABIDES
AGENT_TYPE_MAP = {
    "NoiseAgent":               "NOISE",
    "ValueAgent":               "VALUE",
    "MomentumAgent":            "MOMENTUM",
    "AdaptiveMarketMakerAgent": "MARKET_MAKER",
}

AGENT_TYPES = list(dict.fromkeys(AGENT_TYPE_MAP.values()))  # preserve insertion order

# ═══════════════════════════════════════════════════════════════════════════════
# DATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_price_lookup(order_book) -> dict:
    """
    Build a {timestamp_ns → mid_price} dict from the exchange order book log.
    Used to mark-to-market holdings at every HOLDINGS_UPDATED event.
    """
    price_lookup = {}
    for entry in order_book.book_log2:
        t    = int(entry["QuoteTime"])
        bids = entry["bids"]
        asks = entry["asks"]
        if len(bids) > 0 and len(asks) > 0:
            mid = (int(bids[0][0]) + int(asks[0][0])) / 2.0
            price_lookup[t] = mid
    return price_lookup


def nearest_price(price_lookup: dict, ts: int, fallback: float) -> float:
    """Return the mid-price at or just before timestamp ts."""
    if not price_lookup:
        return fallback
    keys = sorted(price_lookup.keys())
    # Binary search for the last key ≤ ts
    lo, hi = 0, len(keys) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if keys[mid] <= ts:
            lo = mid
        else:
            hi = mid - 1
    if keys[lo] <= ts:
        return price_lookup[keys[lo]]
    return fallback


def extract_agent_metrics(
    agent,
    closing_price: float,
    price_lookup: dict,
    mkt_open_ns: int,
) -> dict:
    """
    Extract all relevant metrics for a single agent from its post-run state.

    Returns
    -------
    dict with keys:
        pnl, final_shares, n_executed, active_window_min,
        max_drawdown, starting_cash
    """
    starting_cash = agent.starting_cash
    holdings      = agent.holdings
    final_cash    = holdings.get("CASH", 0)
    final_shares  = holdings.get(agent.symbol if hasattr(agent, "symbol") else "ABM", 0)

    final_m2m = final_cash + final_shares * closing_price
    pnl       = final_m2m - starting_cash

    # ── Reconstruct intraday M2M from HOLDINGS_UPDATED events ─────────────────
    h_events = [
        (int(ts), data)
        for ts, ev, data in agent.log
        if ev == "HOLDINGS_UPDATED"
    ]

    # Count executed orders from the log
    n_executed = sum(1 for _, ev, _ in agent.log if ev == "ORDER_EXECUTED")

    # Active trading window: time from first to last executed order (minutes)
    exec_times = [int(ts) for ts, ev, _ in agent.log if ev == "ORDER_EXECUTED"]
    if len(exec_times) >= 2:
        active_window_min = (max(exec_times) - min(exec_times)) / 1e9 / 60.0
    elif len(exec_times) == 1:
        active_window_min = 0.0
    else:
        active_window_min = 0.0

    # ── Max intraday drawdown ─────────────────────────────────────────────────
    # Build a M2M time series: at each HOLDINGS_UPDATED, compute cash + pos × price
    m2m_series = [float(starting_cash)]  # t=0 baseline

    ticker = agent.symbol if hasattr(agent, "symbol") else "ABM"

    for ts, h in h_events:
        cash   = h.get("CASH", 0)
        shares = h.get(ticker, 0)
        price  = nearest_price(price_lookup, ts, closing_price)
        m2m    = cash + shares * price
        m2m_series.append(float(m2m))

    m2m_arr      = np.array(m2m_series, dtype=float)
    running_peak = np.maximum.accumulate(m2m_arr)
    dd_arr       = np.where(
        running_peak > 0,
        (running_peak - m2m_arr) / running_peak,
        0.0,
    )
    max_dd = float(np.max(dd_arr))

    return {
        "pnl":               float(pnl),
        "final_m2m":         float(final_m2m),
        "final_shares":      int(final_shares),
        "n_executed":        int(n_executed),
        "active_window_min": float(active_window_min),
        "max_drawdown":      float(max_dd),
        "starting_cash":     int(starting_cash),
        # normalised reward: comparable to Gym dense reward
        # Gym normalises by order_fixed_size × num_steps
        # Here we normalise by starting_cash to get a fraction
        "reward_proxy":      float(pnl / starting_cash),
    }


def run_simulation(seed: int, end_time: str) -> dict:
    """
    Run one full ABIDES simulation and return per-agent-type metric lists.
    """
    cfg       = rmsc04.build_config(seed=seed, end_time=end_time,
                                    log_orders=True, **RMSC04_PARAMS)
    end_state = abides.run(cfg)
    agents    = end_state["agents"]

    exchange     = agents[0]
    order_book   = exchange.order_books[list(exchange.order_books.keys())[0]]
    closing_price = float(order_book.last_trade)
    mkt_open_ns  = int(exchange.mkt_open)
    price_lookup = build_price_lookup(order_book)

    # Extract mid-price time series for market-level metrics
    mid_series = []
    spread_series = []
    for entry in order_book.book_log2:
        bids = entry["bids"]; asks = entry["asks"]
        if len(bids) > 0 and len(asks) > 0:
            mid    = (int(bids[0][0]) + int(asks[0][0])) / 2.0
            spread = int(asks[0][0]) - int(bids[0][0])
            mid_series.append(mid)
            spread_series.append(spread)

    # Per-agent extraction
    results = {atype: [] for atype in AGENT_TYPES}

    for agent in agents[1:]:  # skip ExchangeAgent
        atype_raw = type(agent).__name__
        atype     = AGENT_TYPE_MAP.get(atype_raw)
        if atype is None:
            continue
        metrics = extract_agent_metrics(
            agent, closing_price, price_lookup, mkt_open_ns
        )
        results[atype].append(metrics)

    return {
        "seed":          seed,
        "closing_price": closing_price,
        "mid_series":    mid_series,
        "spread_series": spread_series,
        "agents":        results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate(all_sim_results: list) -> dict:
    """
    Flatten all per-agent metrics across all seeds into per-type lists,
    then compute summary statistics.
    """
    flat = {atype: [] for atype in AGENT_TYPES}
    for sim in all_sim_results:
        for atype in AGENT_TYPES:
            flat[atype].extend(sim["agents"][atype])

    def stats(vals):
        a = np.array(vals, dtype=float)
        return {
            "mean":   float(np.mean(a)),
            "std":    float(np.std(a)),
            "min":    float(np.min(a)),
            "max":    float(np.max(a)),
            "p5":     float(np.percentile(a,  5)),
            "p25":    float(np.percentile(a, 25)),
            "p50":    float(np.percentile(a, 50)),
            "p75":    float(np.percentile(a, 75)),
            "p95":    float(np.percentile(a, 95)),
        }

    summaries = {}
    for atype, records in flat.items():
        if not records:
            continue
        pnls   = [r["pnl"]               for r in records]
        rews   = [r["reward_proxy"]       for r in records]
        wins   = [r["pnl"] > 0           for r in records]
        dds    = [r["max_drawdown"]       for r in records]
        acts   = [r["active_window_min"]  for r in records]
        sharpe = (float(np.mean(pnls) / np.std(pnls))
                  if np.std(pnls) > 1e-6 else 0.0)

        summaries[atype] = {
            "n_agents":          len(records),
            "pnl":               stats(pnls),
            "reward_proxy":      stats(rews),
            "win_rate":          float(np.mean(wins)),
            "sharpe":            sharpe,
            "max_drawdown":      stats(dds),
            "active_window_min": stats(acts),
            "records":           records,   # full list for plotting
        }

    return summaries


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

# One colour per agent type — consistent across all plots
COLOURS = {
    "NOISE":       "#2196F3",
    "VALUE":       "#4CAF50",
    "MOMENTUM":    "#FF9800",
    "MARKET_MAKER":"#E91E63",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor":   "#16213e",
        "axes.edgecolor":   "#444466",
        "axes.labelcolor":  "#ccccee",
        "axes.titlecolor":  "#ffffff",
        "xtick.color":      "#aaaacc",
        "ytick.color":      "#aaaacc",
        "grid.color":       "#2a2a4a",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
        "text.color":       "#ccccee",
        "legend.facecolor": "#1a1a2e",
        "legend.edgecolor": "#444466",
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.labelsize":   11,
    })


def _save(fig, name: str, out_dir: str):
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {path}")


def plot_pnl_distribution(summaries: dict, out_dir: str):
    """Violin + jitter of final P&L (cents) per agent type."""
    _style()
    types    = [t for t in AGENT_TYPES if t in summaries]
    pnl_data = [[r["pnl"] for r in summaries[t]["records"]] for t in types]

    fig, ax = plt.subplots(figsize=(max(8, len(types) * 2.2), 6))
    rng = np.random.RandomState(42)

    for i, (atype, data) in enumerate(zip(types, pnl_data)):
        c = COLOURS[atype]
        # clip to reasonable range for violin (outliers skew the shape)
        clip  = np.percentile(data, [2, 98])
        clipped = np.clip(data, clip[0], clip[1])

        parts = ax.violinplot([clipped], positions=[i], widths=0.7,
                               showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(c); pc.set_alpha(0.4)
        parts["cmedians"].set_color(c); parts["cmedians"].set_linewidth(2)

        jitter = rng.uniform(-0.12, 0.12, len(clipped))
        ax.scatter(i + jitter, clipped, s=4, color=c, alpha=0.25, zorder=3)
        ax.scatter([i], [np.mean(data)], s=100, color=c, edgecolors="white",
                   linewidths=1.2, zorder=5, marker="D",
                   label=f"{atype}  n={len(data)}  μ={np.mean(data):+.0f}¢")

    ax.axhline(0, color="#ff4444", lw=1.3, ls="--", label="Break-even")
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, fontsize=11)
    ax.set_ylabel("Final Mark-to-Market P&L  (cents above starting_cash)")
    ax.set_title(
        "True Agent P&L  ·  ABIDES-markets native simulation\n"
        "(◆ = mean, centre line = median, dots = individual agents, clipped at p2/p98)"
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y")
    _save(fig, "1_m2m_pnl_distribution", out_dir)


def plot_reward_distribution(summaries: dict, out_dir: str):
    """Box plot of normalised reward proxy per agent type."""
    _style()
    types    = [t for t in AGENT_TYPES if t in summaries]
    rew_data = [[r["reward_proxy"] for r in summaries[t]["records"]] for t in types]

    fig, ax = plt.subplots(figsize=(max(8, len(types) * 2.2), 6))
    bp = ax.boxplot(
        rew_data, patch_artist=True, notch=False,
        medianprops=dict(color="white", lw=2),
        whiskerprops=dict(color="#888899"),
        capprops=dict(color="#888899"),
        flierprops=dict(marker="o", markersize=2, alpha=0.3, color="#888899"),
    )
    for i, (patch, atype) in enumerate(zip(bp["boxes"], types)):
        patch.set_facecolor(COLOURS[atype]); patch.set_alpha(0.65)

    means = [np.mean(d) for d in rew_data]
    ax.scatter(range(1, len(types)+1), means, s=90, color="white",
               zorder=5, marker="D", label="Mean")

    ax.axhline(0, color="#ff4444", lw=1.3, ls="--", label="Zero reward")
    ax.set_xticks(range(1, len(types)+1))
    ax.set_xticklabels(types, fontsize=11)
    ax.set_ylabel("Reward proxy  (P&L / starting_cash)")
    ax.set_title(
        "Normalised Reward Proxy per Agent Type\n"
        "(box = IQR, whiskers = 1.5×IQR, ◆ = mean)"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y")
    _save(fig, "2_reward_distribution", out_dir)


def plot_active_window(summaries: dict, out_dir: str):
    """
    Histogram of active trading window (minutes from first to last fill).
    Replaces 'episode length' — in a native sim every agent runs the full day,
    but their *active* window (first fill → last fill) varies meaningfully.
    """
    _style()
    types = [t for t in AGENT_TYPES if t in summaries]
    fig, ax = plt.subplots(figsize=(9, 5))

    for atype in types:
        windows = [r["active_window_min"] for r in summaries[atype]["records"]]
        # filter agents that actually traded
        active  = [w for w in windows if w > 0]
        if not active:
            continue
        ax.hist(active, bins=30, alpha=0.5, color=COLOURS[atype], density=True,
                label=f"{atype}  (μ={np.mean(active):.1f} min, "
                      f"n_active={len(active)}/{len(windows)})")

    full_day = (16.0 - 9.5) * 60
    ax.axvline(full_day, color="white", lw=1.5, ls="--",
               label=f"Full trading day = {full_day:.0f} min")
    ax.set_xlabel("Active Trading Window  (minutes from first to last fill)")
    ax.set_ylabel("Density")
    ax.set_title(
        "Active Trading Window per Agent Type\n"
        "(agents with zero fills excluded; MARKET_MAKER may show 0 if no fills logged)"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y")
    _save(fig, "3_active_window_distribution", out_dir)


def plot_sharpe(summaries: dict, out_dir: str):
    """Horizontal bar chart of Sharpe ratios, sorted descending."""
    _style()
    types   = [t for t in AGENT_TYPES if t in summaries]
    sharpes = [summaries[t]["sharpe"] for t in types]
    order   = np.argsort(sharpes)[::-1]
    ts, ss  = [types[i] for i in order], [sharpes[i] for i in order]
    cs      = [COLOURS[t] for t in ts]

    fig, ax = plt.subplots(figsize=(8, max(4, len(types) * 1.0)))
    bars = ax.barh(ts, ss, color=cs, alpha=0.78, height=0.5)
    for bar, val in zip(bars, ss):
        xoff = 0.005 if val >= 0 else -0.005
        ha   = "left" if val >= 0 else "right"
        ax.text(bar.get_x() + bar.get_width() + xoff,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=11, color="white")

    ax.axvline(0, color="#ff4444", lw=1.3, ls="--")
    ax.set_xlabel("Sharpe Ratio  (mean P&L / std P&L,  across all agent instances)")
    ax.set_title(
        "Sharpe Ratio per Agent Type  ·  native simulation\n"
        "(higher = better risk-adjusted return)"
    )
    ax.grid(axis="x")
    _save(fig, "4_sharpe_ratio", out_dir)


def plot_win_rate(summaries: dict, out_dir: str):
    """Bar chart of win rates with Wilson 95% CIs."""
    from math import sqrt

    def wilson(wins, n, z=1.96):
        if n == 0: return 0.0, 0.0
        p = wins / n
        d = 1 + z**2 / n
        c = (p + z**2 / (2*n)) / d
        m = z * sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
        return max(0, c-m), min(1, c+m)

    _style()
    types = [t for t in AGENT_TYPES if t in summaries]
    wrs   = [summaries[t]["win_rate"] for t in types]
    ns    = [summaries[t]["n_agents"] for t in types]

    lo_e, hi_e = [], []
    for wr, n in zip(wrs, ns):
        lo, hi = wilson(round(wr * n), n)
        lo_e.append(wr - lo); hi_e.append(hi - wr)

    fig, ax = plt.subplots(figsize=(max(8, len(types) * 2.2), 5))
    x    = np.arange(len(types))
    bars = ax.bar(
        x, wrs, color=[COLOURS[t] for t in types], alpha=0.75, width=0.55,
        yerr=[lo_e, hi_e],
        error_kw=dict(ecolor="white", elinewidth=1.5, capsize=5),
    )
    for bar, wr, n in zip(bars, wrs, ns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                min(wr + 0.03, 0.99),
                f"{wr:.1%}\n(n={n:,})",
                ha="center", va="bottom", fontsize=9, color="white")

    ax.axhline(0.5, color="#ff4444", lw=1.3, ls="--",
               label="50 % reference  (symmetric P&L distribution)")
    ax.set_xticks(x); ax.set_xticklabels(types, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Win Rate  (fraction of agents with P&L > 0)")
    ax.set_title(
        "Win Rate per Agent Type  ·  native simulation\n"
        "(error bars = 95 % Wilson CI)"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y")
    _save(fig, "5_win_rate", out_dir)


def plot_max_drawdown(summaries: dict, out_dir: str):
    """CDF of per-agent max intraday drawdown — lower-left is safer."""
    _style()
    types = [t for t in AGENT_TYPES if t in summaries]
    fig, ax = plt.subplots(figsize=(9, 5))

    for atype in types:
        dds = sorted([r["max_drawdown"] for r in summaries[atype]["records"]])
        cdf = np.arange(1, len(dds)+1) / len(dds)
        mu  = np.mean(dds)
        ax.plot(dds, cdf, color=COLOURS[atype], lw=2,
                label=f"{atype}  (μ={mu:.4f})")
        ax.axvline(mu, color=COLOURS[atype], lw=0.8, ls=":", alpha=0.7)

    ax.set_xlabel("Max Intraday Drawdown  (fraction of peak M2M)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(
        "Max Drawdown CDF per Agent Type  ·  native simulation\n"
        "(lower-left = safer; dotted vertical = mean per type)"
    )
    ax.legend(fontsize=9)
    ax.grid()
    _save(fig, "6_max_drawdown_cdf", out_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="ABIDES-markets native simulation baselines"
    )
    p.add_argument("--seeds",     type=int, default=10,
                   help="Number of simulations to run (default: 10)")
    p.add_argument("--end_time",  type=str, default="16:00:00",
                   help="Market close time (default: 16:00:00 = full day)")
    p.add_argument("--out_dir",   type=str, default="sim_plots")
    p.add_argument("--results_file", type=str,
                   default="sim_baseline_results.json")
    p.add_argument("--no-plots",  action="store_true")
    p.add_argument("--seed_offset", type=int, default=0)
    return p.parse_args()


def main():
    args  = parse_args()
    seeds = [args.seed_offset + i for i in range(args.seeds)]
    os.makedirs(args.out_dir, exist_ok=True)

    n_agents_per_sim = (
        RMSC04_PARAMS["num_noise_agents"]
        + RMSC04_PARAMS["num_value_agents"]
        + RMSC04_PARAMS["num_momentum_agents"]
        + 2   # 2 market makers
    )

    print("=" * 66)
    print("ABIDES-markets  ·  native simulation baselines")
    print("=" * 66)
    print(f"  Simulations   : {args.seeds}  (seeds {seeds[0]}–{seeds[-1]})")
    print(f"  Market hours  : 09:30 → {args.end_time}")
    print(f"  Agents/sim    : {n_agents_per_sim}  "
          f"({RMSC04_PARAMS['num_noise_agents']} noise + "
          f"{RMSC04_PARAMS['num_value_agents']} value + "
          f"{RMSC04_PARAMS['num_momentum_agents']} momentum + 2 MM)")
    print(f"  Total observations: ~{args.seeds * n_agents_per_sim:,} agents")
    print(f"  r_bar         : {RMSC04_PARAMS['r_bar']:,} ¢  "
          f"(= ${RMSC04_PARAMS['r_bar']/100:,.2f})")
    print(f"  starting_cash : {RMSC04_PARAMS['starting_cash']:,} ¢  "
          f"(= ${RMSC04_PARAMS['starting_cash']/100:,.2f})")
    print()

    all_sim_results = []
    t_total = time.time()

    for idx, seed in enumerate(seeds):
        t0 = time.time()
        print(f"  sim {idx+1:>3}/{args.seeds} | seed={seed}", end="", flush=True)
        sim = run_simulation(seed, args.end_time)
        all_sim_results.append(sim)

        # Quick per-type summary for this seed
        parts = []
        for atype in AGENT_TYPES:
            recs = sim["agents"][atype]
            if recs:
                pnls = [r["pnl"] for r in recs]
                parts.append(f"{atype[:4]}={np.mean(pnls):+.0f}¢")
        print(f" | close={sim['closing_price']:.0f}¢ | "
              f"{' '.join(parts)} | {time.time()-t0:.1f}s")

    print(f"\n  Total wall time: {time.time()-t_total:.1f}s")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    summaries = aggregate(all_sim_results)

    # ── Comparison table ──────────────────────────────────────────────────────
    print()
    print("=" * 82)
    print(f"{'Agent':<14} {'n':>7}  {'P&L (¢)':>14}  {'Sharpe':>8}  "
          f"{'Win%':>7}  {'ActiveMin':>10}  {'MaxDD':>8}")
    print("=" * 82)
    for atype in AGENT_TYPES:
        if atype not in summaries:
            continue
        s = summaries[atype]
        p = s["pnl"]
        a = s["active_window_min"]
        print(f"{atype:<14} "
              f"{s['n_agents']:>7,}  "
              f"{p['mean']:>+9.1f}±{p['std']:<6.1f}  "
              f"{s['sharpe']:>+8.3f}  "
              f"{s['win_rate']:>6.1%}  "
              f"{a['mean']:>9.1f}  "
              f"{s['max_drawdown']['mean']:>8.4f}")
    print("=" * 82)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    to_save = {}
    for atype, s in summaries.items():
        to_save[atype] = {
            "n_agents":    s["n_agents"],
            "sharpe":      s["sharpe"],
            "win_rate":    s["win_rate"],
            "pnl":         s["pnl"],
            "reward_proxy":s["reward_proxy"],
            "max_drawdown":s["max_drawdown"],
            "active_window_min": s["active_window_min"],
            # include all individual records (no traces to strip — already compact)
            "records": [
                {k: v for k, v in r.items() if k != "starting_cash"}
                for r in s["records"]
            ],
        }

    with open(args.results_file, "w") as f:
        json.dump(to_save, f, indent=2, default=str)
    print(f"\nResults → {args.results_file}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print(f"Plots   → {args.out_dir}/")
        plot_pnl_distribution(summaries, args.out_dir)
        plot_reward_distribution(summaries, args.out_dir)
        plot_active_window(summaries, args.out_dir)
        plot_sharpe(summaries, args.out_dir)
        plot_win_rate(summaries, args.out_dir)
        plot_max_drawdown(summaries, args.out_dir)
        print("Done.")


if __name__ == "__main__":
    main()