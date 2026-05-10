"""
belief_comparison.py — Five-way belief tracker comparison
==========================================================

Agents
------
  1. VALUE          — raw direction_feature threshold (obs[3])
  2. BELIEF_KALMAN  — Kalman OU filter, threshold on belief_dev
  3. BELIEF_KIM     — Regime-switching Kalman; regime-aware trading logic
  4. BELIEF_PARTICLE — Bootstrap particle filter, threshold on belief_dev
  5. BELIEF_LSTM    — LSTM trained to predict future returns

All agents are evaluated on identical seeds. LSTM uses separate training seeds.

Kim regime-aware trading logic
-------------------------------
  VALUE regime   → contrarian: buy when belief_dev > thr (fundamental above mkt)
  MOMENTUM regime → trend-following: buy when last_return > mom_thr
  NOISE regime   → hold
  Mixed signal   → blend by regime probability weights:
                   signal = p_value * belief_dev - p_momentum * (last_return/r_bar)

Usage
-----
  .venv/bin/python3.9 abides-gym/scripts/belief_comparison.py
  .venv/bin/python3.9 abides-gym/scripts/belief_comparison.py --episodes 20
  .venv/bin/python3.9 abides-gym/scripts/belief_comparison.py --no-lstm
  .venv/bin/python3.9 abides-gym/scripts/belief_comparison.py --train-eps 30 --episodes 20
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

import gym
import abides_gym  # noqa: F401

from belief_tracker      import BeliefAugmentedWrapper
from kim_filter_tracker  import KimBeliefWrapper
from particle_filter_tracker import ParticleBeliefWrapper
from lstm_belief_tracker import LSTMBeliefWrapper, collect_rollouts, train as train_lstm


# ─────────────────────────────────────────────────────────────────────────────
# Shared config
# ─────────────────────────────────────────────────────────────────────────────

R_BAR    = 100_000
ENV_KW   = dict(
    background_config = "rmsc04",
    debug_mode        = True,
    timestep_duration = "60s",
    background_config_extra_kvargs = dict(
        r_bar        = R_BAR,
        kappa_oracle = 1.67e-16,
        fund_vol     = 5e-5,
    ),
)

KALMAN_THR  = 0.0007   # half of steady-state Kalman std / r_bar
KIM_THR     = 0.0007   # same value-signal threshold for Kim
MOM_THR     = 0.0001   # momentum signal threshold (last_return / r_bar)
PARTICLE_THR= 0.0007   # same as Kalman — same steady-state noise floor
LSTM_THR    = 0.0      # sign of predicted future return
DIR_THR     = 0.75     # original VALUE direction threshold

TRAIN_SEED_OFFSET = 200   # LSTM training: seeds 200-229
SEQ_LEN           = 20
LOOKAHEAD         = 3

PALETTE = {
    "VALUE":           "#FF9800",
    "BELIEF_KALMAN":   "#2196F3",
    "BELIEF_KIM":      "#9C27B0",
    "BELIEF_PARTICLE": "#4CAF50",
    "BELIEF_LSTM":     "#F44336",
}


# ─────────────────────────────────────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────────────────────────────────────

class ValueAgent:
    name = "VALUE"
    def act(self, obs, **_):
        d = float(obs[3, 0])
        if d > DIR_THR:  return 2
        if d < -DIR_THR: return 0
        return 1


class BeliefAgent:
    """Generic threshold agent for Kalman and Particle wrappers (obs[-2] = belief_dev)."""
    def __init__(self, name, thr):
        self.name = name
        self.thr  = thr

    def act(self, obs, **_):
        b = float(obs[-2, 0])
        if b > self.thr:  return 0   # fundamental above market → BUY
        if b < -self.thr: return 2
        return 1


class KimRegimeAgent:
    """
    Regime-aware agent for the Kim filter.

    Blends two signals by regime probability:
      value_signal    = belief_dev = (mu - mid) / r_bar
      momentum_signal = -(last_return / r_bar)  [negative: positive return → expect fall... wait]

    Actually:
      value_signal > 0    → BUY (fundamental above market)
      momentum_signal > 0 → BUY (recent return positive, trend continues)

    Combined: signal = p_value * belief_dev + p_momentum * (last_return / r_bar)
    Threshold on this blended signal.
    """
    name = "BELIEF_KIM"

    def __init__(self, thr=KIM_THR, mom_thr=MOM_THR):
        self.thr     = thr
        self.mom_thr = mom_thr

    def act(self, obs, **_):
        belief_dev   = float(obs[-5, 0])    # Kim wrapper: obs[-5] = belief_dev
        p_value      = float(obs[-4, 0])
        p_momentum   = float(obs[-3, 0])
        p_noise      = float(obs[-2, 0])
        last_return  = float(obs[4, 0]) / R_BAR if obs.shape[0] > 4 else 0.0

        # Value signal: buy cheap, sell expensive (contrarian)
        val_signal = belief_dev

        # Momentum signal: follow the trend
        mom_signal = last_return

        # Noise regime: no signal (hold)
        blended = p_value * val_signal + p_momentum * mom_signal

        if blended > self.thr:  return 0   # BUY
        if blended < -self.thr: return 2   # SELL
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(env, agent, seed: int) -> dict:
    env.seed(seed)
    obs       = env.reset()
    unwrapped = env.unwrapped
    starting  = unwrapped.starting_cash
    unwrapped.previous_marked_to_market = starting

    done = False
    total_reward = 0.0
    steps = 0
    m2m_trace, belief_trace, regime_trace = [], [], []

    while not done:
        action = agent.act(obs)
        try:
            obs, reward, done, info = env.step(action)
        except AssertionError:
            done = True
            break
        steps        += 1
        total_reward += reward
        m2m_trace.append(float(info.get("marked_to_market", np.nan)))

        # belief_dev is always at obs[-2] for Kalman/Particle/LSTM,
        # and at obs[-5] for Kim
        belief_trace.append(float(obs[-2, 0]))

        # Regime probs for Kim (obs[-4:-1])
        if obs.shape[0] >= 12:   # Kim wrapper adds 5 features (7+5=12)
            regime_trace.append(obs[-4:-1, 0].tolist())

    valid     = [v for v in m2m_trace if not np.isnan(v)]
    final_m2m = valid[-1] if valid else float(starting)
    final_pnl = final_m2m - starting

    arr  = np.array([starting] + [v if not np.isnan(v) else starting for v in m2m_trace])
    peak = np.maximum.accumulate(arr)
    dd   = np.where(peak > 0, (peak - arr) / peak, 0.0)

    return {
        "seed":          seed,
        "final_pnl":     final_pnl,
        "total_reward":  total_reward,
        "episode_len":   steps,
        "won":           final_pnl >= 0,
        "max_dd":        float(np.max(dd)),
        "belief_trace":  belief_trace,
        "regime_trace":  regime_trace,
    }


def run_agent(env, agent, n_episodes: int) -> list:
    results = []
    for seed in range(n_episodes):
        ep  = run_episode(env, agent, seed)
        tag = "WIN " if ep["won"] else "LOSS"
        print(f"  {agent.name:<18}  ep {seed+1:>2}/{n_episodes}  {tag}  "
              f"pnl={ep['final_pnl']:>+9.0f}¢  len={ep['episode_len']:>4}")
        results.append(ep)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Stats + summary
# ─────────────────────────────────────────────────────────────────────────────

def stats(results):
    pnls = np.array([r["final_pnl"] for r in results])
    return {
        "mean_pnl": float(np.mean(pnls)),
        "std_pnl":  float(np.std(pnls)),
        "sharpe":   float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 0 else 0.0,
        "win_rate": float(np.mean([r["won"] for r in results])),
        "mean_dd":  float(np.mean([r["max_dd"] for r in results])),
    }


def print_summary(all_stats: dict):
    print("\n" + "=" * 84)
    print(f"  {'Agent':<20} {'mean_pnl':>10} {'std_pnl':>10} {'sharpe':>8} "
          f"{'win%':>6} {'mean_dd':>8}")
    print("=" * 84)
    for name, s in all_stats.items():
        print(f"  {name:<20} {s['mean_pnl']:>+10.1f} {s['std_pnl']:>10.1f} "
              f"{s['sharpe']:>8.3f} {s['win_rate']:>5.1%} {s['mean_dd']:>8.4f}")
    print("=" * 84)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor":  "#16213e",
        "axes.edgecolor":   "#444466", "axes.labelcolor": "#ccccee",
        "axes.titlecolor":  "#ffffff", "xtick.color":     "#aaaacc",
        "ytick.color":      "#aaaacc", "grid.color":      "#2a2a4a",
        "grid.linestyle":   "--",      "grid.alpha":       0.5,
        "text.color":       "#ccccee", "legend.facecolor": "#1a1a2e",
        "legend.edgecolor": "#444466", "font.size":        10,
    })


def plot_all(all_results: dict, all_stats: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    _style()
    labels = list(all_results.keys())
    colors = [PALETTE.get(l, "#FFFFFF") for l in labels]

    # ── 1. P&L distributions ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, results, color in zip(labels, all_results.values(), colors):
        pnls = [r["final_pnl"] for r in results]
        ax.hist(pnls, bins=14, alpha=0.45, label=label,
                color=color, edgecolor="white", lw=0.4)
    ax.axvline(0, color="#ff4444", lw=1.5, ls="--")
    ax.set_xlabel("Final P&L (cents)")
    ax.set_ylabel("Episodes")
    ax.set_title("P&L Distribution — All Five Belief Trackers")
    ax.legend(fontsize=8)
    ax.grid(axis="x")
    p = os.path.join(out_dir, "comparison5_pnl.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {p}")

    # ── 2. Metric bars ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, ylabel in zip(
        axes,
        ["mean_pnl", "win_rate", "sharpe"],
        ["Mean P&L (cents)", "Win Rate", "Sharpe Ratio"],
    ):
        vals = [all_stats[l][metric] for l in labels]
        bars = ax.bar(labels, vals, color=colors, alpha=0.8,
                      edgecolor="white", lw=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(max(vals, default=1)) * 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, color="white")
        ax.set_title(ylabel)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace("BELIEF_", "") for l in labels],
                           fontsize=8, rotation=15, ha="right")
        ax.axhline(0, color="#ff4444", lw=0.8, ls="--")
        ax.grid(axis="y")
    fig.suptitle("Five-Way Belief Tracker Comparison", fontsize=12)
    fig.tight_layout()
    p = os.path.join(out_dir, "comparison5_metrics.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {p}")

    # ── 3. Belief signal traces (last episode, non-VALUE agents) ─────────────
    traceable = [(l, all_results[l]) for l in labels if l != "VALUE"]
    fig, axes = plt.subplots(len(traceable), 1,
                             figsize=(11, 2.5 * len(traceable)), sharex=True)
    if len(traceable) == 1:
        axes = [axes]
    for ax, (label, results) in zip(axes, traceable):
        trace = results[-1]["belief_trace"]
        color = PALETTE.get(label, "#FFFFFF")
        if trace:
            ax.plot(trace, color=color, lw=1.2, label=label)
            ax.axhline(0, color="#ff4444", lw=0.8, ls="--")
            ax.set_ylabel("belief_dev")
            ax.set_title(label)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid()
    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Belief Signal Trajectories — Last Test Episode", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, "comparison5_traces.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {p}")

    # ── 4. Kim regime probabilities (last episode) ────────────────────────────
    if "BELIEF_KIM" in all_results:
        regime_trace = all_results["BELIEF_KIM"][-1]["regime_trace"]
        if regime_trace:
            arr = np.array(regime_trace)          # (T, 3): [p_val, p_mom, p_noise]
            _style()
            fig, ax = plt.subplots(figsize=(11, 3.5))
            steps = np.arange(len(arr))
            ax.stackplot(steps, arr[:, 0], arr[:, 1], arr[:, 2],
                         labels=["p_value", "p_momentum", "p_noise"],
                         colors=["#2196F3", "#9C27B0", "#FF9800"], alpha=0.75)
            ax.set_ylabel("Regime probability")
            ax.set_xlabel("Timestep")
            ax.set_title("Kim Filter Regime Beliefs — Last Test Episode")
            ax.legend(loc="upper right", fontsize=9)
            ax.set_ylim(0, 1)
            ax.grid()
            p = os.path.join(out_dir, "comparison5_kim_regimes.png")
            fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  saved → {p}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",    type=int, default=20)
    p.add_argument("--train-eps",   type=int, default=30,
                   help="Hold-only episodes for LSTM training")
    p.add_argument("--no-lstm",     action="store_true",
                   help="Skip LSTM (saves ~10 minutes of training time)")
    p.add_argument("--n-particles", type=int, default=300)
    p.add_argument("--plots-dir",   type=str, default="plots")
    p.add_argument("--no-plots",    action="store_true")
    p.add_argument("--lstm-model",  type=str, default=None,
                   help="Path to a pre-trained LSTM .pt file")
    return p.parse_args()


def make_env():
    return gym.make("markets-daily_investor-v0", **ENV_KW)


def main():
    args = parse_args()

    print("=" * 70)
    print("Five-Way Belief Tracker Comparison")
    print("=" * 70)
    print(f"  episodes       : {args.episodes}")
    print(f"  n_particles    : {args.n_particles}")
    print(f"  LSTM           : {'disabled' if args.no_lstm else 'enabled'}")
    print()

    all_results: dict = {}

    # ── 1. VALUE baseline ─────────────────────────────────────────────────────
    print(f"── VALUE (direction_feature, thr=±{DIR_THR}) ──")
    env = make_env()
    all_results["VALUE"] = run_agent(env, ValueAgent(), args.episodes)
    env.close()

    # ── 2. Kalman ─────────────────────────────────────────────────────────────
    print(f"\n── BELIEF_KALMAN (thr=±{KALMAN_THR}) ──")
    env = BeliefAugmentedWrapper(make_env(), r_bar=R_BAR,
                                 kappa_oracle=1.67e-16, fund_vol=5e-5)
    all_results["BELIEF_KALMAN"] = run_agent(
        env, BeliefAgent("BELIEF_KALMAN", KALMAN_THR), args.episodes)
    env.close()

    # ── 3. Kim filter (regime-aware) ─────────────────────────────────────────
    print(f"\n── BELIEF_KIM (regime-aware, thr=±{KIM_THR}, mom_thr=±{MOM_THR}) ──")
    env = KimBeliefWrapper(make_env(), r_bar=R_BAR,
                           kappa_oracle=1.67e-16, fund_vol=5e-5)
    all_results["BELIEF_KIM"] = run_agent(env, KimRegimeAgent(), args.episodes)
    env.close()

    # ── 4. Particle filter ────────────────────────────────────────────────────
    print(f"\n── BELIEF_PARTICLE (N={args.n_particles}, thr=±{PARTICLE_THR}) ──")
    env = ParticleBeliefWrapper(make_env(), r_bar=R_BAR,
                                kappa_oracle=1.67e-16, fund_vol=5e-5,
                                n_particles=args.n_particles)
    all_results["BELIEF_PARTICLE"] = run_agent(
        env, BeliefAgent("BELIEF_PARTICLE", PARTICLE_THR), args.episodes)
    env.close()

    # ── 5. LSTM (optional) ────────────────────────────────────────────────────
    if not args.no_lstm:
        import torch
        if args.lstm_model:
            from lstm_belief_tracker import LSTMBeliefNet
            print(f"\nLoading LSTM from {args.lstm_model} ...")
            lstm_model = LSTMBeliefNet()
            lstm_model.load_state_dict(torch.load(args.lstm_model, weights_only=True))
            lstm_model.eval()
        else:
            print(f"\nCollecting {args.train_eps} training rollouts (seeds {TRAIN_SEED_OFFSET}+) ...")
            episodes = collect_rollouts(
                n_episodes=args.train_eps, r_bar=R_BAR,
                bg_kwargs=ENV_KW["background_config_extra_kvargs"],
                seed_offset=TRAIN_SEED_OFFSET,
            )
            print("\nTraining LSTM ...")
            lstm_model = train_lstm(episodes, seq_len=SEQ_LEN, lookahead=LOOKAHEAD,
                                    r_bar=R_BAR)

        print(f"\n── BELIEF_LSTM (thr={LSTM_THR}) ──")
        env = LSTMBeliefWrapper(make_env(), lstm_model, seq_len=SEQ_LEN, r_bar=R_BAR)
        all_results["BELIEF_LSTM"] = run_agent(
            env, BeliefAgent("BELIEF_LSTM", LSTM_THR), args.episodes)
        env.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    all_stats = {k: stats(v) for k, v in all_results.items()}
    print_summary(all_stats)

    # ── Ranking ───────────────────────────────────────────────────────────────
    ranked = sorted(all_stats.items(), key=lambda kv: kv[1]["sharpe"], reverse=True)
    print("\nRanked by Sharpe:")
    for rank, (name, s) in enumerate(ranked, 1):
        print(f"  {rank}. {name:<20}  Sharpe={s['sharpe']:>+.3f}  "
              f"win%={s['win_rate']:.0%}  mean_pnl={s['mean_pnl']:>+.0f}¢")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print(f"\nPlots → {args.plots_dir}/")
        plot_all(all_results, all_stats, args.plots_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
