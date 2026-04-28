"""
run_baseline_rmsc03.py  —  ABIDES daily-investor baseline runner (RMSC03 calibration)
=====================================================================================

This is a duplicate of `test.py` with the market-background parameters swapped from
AAPL calibration to the RMSC03 configuration used by the market simulation.

The strategy classes are reused from `test.py`; the change here is the set of
background configuration parameters passed into the environment.
"""

import argparse
import json
import os
import time
import warnings
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

warnings.filterwarnings("ignore")

import gym
import abides_gym


_BASELINE_PATH = Path(__file__).with_name("test.py")
_BASELINE_SPEC = spec_from_file_location("abides_gym_test_baseline", _BASELINE_PATH)
_BASELINE = module_from_spec(_BASELINE_SPEC)
assert _BASELINE_SPEC is not None and _BASELINE_SPEC.loader is not None
_BASELINE_SPEC.loader.exec_module(_BASELINE)


RMSC03_PARAMS = dict(
    starting_cash=10_000_000,
    num_noise_agents=5000,
    num_value_agents=100,
    num_momentum_agents=25,
    execution_agents=True,
    execution_pov=0.1,
    mm_pov=0.025,
    mm_window_size="adaptive",
    mm_min_order_size=1,
    mm_num_ticks=10,
    mm_wake_up_freq="10S",
    mm_skew_beta=0,
    mm_level_spacing=5,
    mm_spread_alpha=0.75,
    mm_backstop_quantity=50_000,
    fund_r_bar=100_000,
    fund_kappa=1.67e-16,
    fund_sigma_s=0,
    fund_vol=1e-3,
    fund_megashock_lambda_a=2.77778e-18,
    fund_megashock_mean=1000,
    fund_megashock_var=50_000,
    val_r_bar=100_000,
    val_kappa=1.67e-15,
    val_vol=1e-8,
    val_lambda_a=7e-11,
)


ALL_STRATEGIES = _BASELINE.ALL_STRATEGIES
run_episode = _BASELINE.run_episode
summarise = _BASELINE.summarise
plot_m2m = _BASELINE.plot_m2m
plot_reward = _BASELINE.plot_reward
plot_episode_length = _BASELINE.plot_episode_length
plot_sharpe = _BASELINE.plot_sharpe
plot_win_rate = _BASELINE.plot_win_rate
plot_max_drawdown = _BASELINE.plot_max_drawdown


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",      type=int,   default=50)
    p.add_argument("--strategies",    nargs="+",  default=list(ALL_STRATEGIES.keys()),
                   choices=list(ALL_STRATEGIES.keys()))
    p.add_argument("--timestep",      type=str,   default="60s")
    p.add_argument("--reward_mode",   type=str,   default="dense",
                   choices=["dense", "sparse"])
    p.add_argument("--starting_cash", type=int,   default=1_000_000)
    p.add_argument("--order_size",    type=int,   default=10)
    p.add_argument("--done_ratio",    type=float, default=0.3)
    p.add_argument("--out_dir",       type=str,   default="plots_rmsc03")
    p.add_argument("--results_file",  type=str,   default="baseline_results_rmsc03.json")
    p.add_argument("--no-plots",      action="store_true")
    p.add_argument("--seed_offset",   type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # compute theoretical max steps per episode
    ns_day    = int((16 - 9.5) * 60 * 60 * 1e9)
    first_ns  = int(5 * 60 * 1e9)
    step_sec  = int(args.timestep.lower().replace("s", ""))
    step_ns   = int(step_sec * 1e9)
    max_steps = (ns_day - first_ns) // step_ns
    seeds     = [args.seed_offset + i for i in range(args.episodes)]

    print("=" * 62)
    print("ABIDES  ·  markets-daily_investor-v0  ·  RMSC03 calibration")
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
        background_config              = "rmsc03",
        mkt_close                      = "16:00:00",
        timestep_duration              = args.timestep,
        starting_cash                  = args.starting_cash,
        order_fixed_size               = args.order_size,
        state_history_length           = 4,
        market_data_buffer_length      = 5,
        first_interval                 = "00:05:00",
        reward_mode                    = args.reward_mode,
        done_ratio                     = args.done_ratio,
        debug_mode                     = True,
        background_config_extra_kvargs = RMSC03_PARAMS,
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
    STRIP = {"_m2m_trace", "_holdings_trace", "_spread_trace",
             "_step_rewards", "_step_actions"}
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
        plot_m2m(all_results, args.starting_cash, args.out_dir)
        plot_reward(all_results, args.out_dir)
        plot_episode_length(all_results, max_steps, args.out_dir)
        plot_sharpe(all_results, args.out_dir)
        plot_win_rate(all_results, args.out_dir)
        plot_max_drawdown(all_results, args.out_dir)
        print("Done.")


if __name__ == "__main__":
    main()