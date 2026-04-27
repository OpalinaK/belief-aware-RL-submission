# Daily Investor Agent Parameter Sweep Plan

## Goal

Understand which market parameters most affect the **daily investor RL agent's**
mark-to-market P&L in the RMSC04 simulation by running controlled parameter sweeps
across the six baseline strategies defined in `test.py`.

The baseline strategies serve as stand-ins for RL policies at different levels of
sophistication: HOLD (zero alpha), BUY_HOLD (directional), RANDOM (noise floor),
MOMENTUM, MR, MR_INV. Sweeping market parameters against these baselines reveals
which conditions make the task easy vs. hard for a learned policy.

---

## Architecture

```
rmsc04 background simulation
├── NoiseAgent × 1000            ─┐
├── ValueAgent × 102             ─┤  fixed-policy, not RL
├── AdaptiveMarketMakerAgent × 2 ─┤
└── MomentumAgent × 12           ─┘
          +
gym daily investor (the RL agent)
  observes : [holdings, imbalance, spread, direction_feature, padded_returns]
  acts      : 0 = MKT BUY  |  1 = HOLD  |  2 = MKT SELL
  reward    : Δ(mark-to-market) per step  (dense)
              final M2M − starting_cash   (sparse)
```

The background agents are not RL agents — they generate realistic market dynamics
for the daily investor to trade against. Market parameters that change their behavior
directly affect the RL agent's observation quality, spread costs, and fill prices.

---

## How P&L Is Computed

The gym env exposes the RL agent's financials via `info` (requires `debug_mode=True`):

```python
obs, reward, done, info = env.step(action)
marked_to_market = info["marked_to_market"]   # cash + holdings × last price
final_pnl        = marked_to_market - env.starting_cash   # cents
```

**Critical bug already fixed in `test.py`:** `env.previous_marked_to_market` must be
reset to `env.starting_cash` after every `env.reset()` call, or dense rewards in
episode N+1 measure delta from episode N's final M2M instead of from starting_cash.

---

## How Background Agents Affect the RL Agent

### Market Maker (`AdaptiveMarketMakerAgent` × 2)

rmsc04 runs 2 MMs with identical parameters. Both share the same `mm_*` config.

| Mechanism | Parameter | Effect on RL Agent |
|---|---|---|
| Spread width | `mm_window_size`, `mm_spread_alpha` | Every BUY or SELL action crosses the spread. Wider spread → higher per-trade cost → all active strategies (MR, MOMENTUM, MR_INV) are penalized; HOLD and BUY_HOLD are not. |
| Book depth | `mm_num_ticks`, `mm_pov` | Thicker book → RL agent's fixed-size orders fill at tighter prices. Thinner book → more slippage, especially for larger `order_fixed_size`. |
| Backstop | `mm_backstop_quantity=0` | No backstop by default. No price floor at extremes; RL agent can fill at very adverse prices during low-liquidity episodes. |
| Wake frequency | `mm_wake_up_freq="60S"` | Slower refresh → stale MM orders sit longer. MR and MOMENTUM strategies can pick off stale quotes after a price move. Faster refresh → MM reprices before RL agent can exploit staleness. |

### Momentum Agents (`MomentumAgent` × 12)

| Mechanism | Effect on RL Agent |
|---|---|
| **Autocorrelation in mid prices** | More momentum agents → stronger short-term trends. MOMENTUM strategy benefits directly; MR strategy is hurt if trends persist longer than the mean-reversion signal. |
| **Adverse fills** | Momentum buying while RL agent sells (or vice versa) means the RL agent is on the wrong side of the trend. Observable in obs[4] (direction_feature). |

### Noise Agents (`NoiseAgent` × 1000)

| Mechanism | Effect on RL Agent |
|---|---|
| **Volume → MM book thickness** | MM sizes orders by `pov × recent_volume`. Fewer noise agents → lower volume → thinner MM book → more RL agent slippage. |
| **Mid-price noise** | Noise trades add jitter to mid price. This degrades the `padded_returns` features in the RL agent's observation, hurting MR and MOMENTUM signals. |

### Value Agents (`ValueAgent` × 102)

Value agents keep the mid price anchored near the fundamental. Their presence
creates mean-reversion tendency in mid prices that MR strategies can exploit.
Fewer value agents → mid price drifts further from fundamental → noisier signal
for the RL agent's return features.

---

## Parameter Sweep Design

All parameters are passed via `background_config_extra_kvargs` to `gym.make()`.

### Value Agent Parameters (fundamental anchoring)

| Parameter | Default | Sweep Values | Hypothesis |
|---|---|---|---|
| `num_value_agents` | `102` | `[10, 102, 300]` | Fewer value agents → mid drifts further from fundamental → MR signal weakens, MOMENTUM improves |
| `lambda_a` | `5.7e-12` | `[5.7e-13, 5.7e-12, 5.7e-11]` | Higher arrival rate → faster price-to-fundamental convergence → tighter MR window, less MOMENTUM persistence |
| `kappa` | `1.67e-15` | `[1.67e-16, 1.67e-15, 1.67e-14]` | Higher kappa → value agents trade more aggressively on deviations → mid price more sticky around fundamental → MR signal stronger but shorter-lived |

### Market Structure Parameters

| Parameter | Default | Sweep Values | Hypothesis |
|---|---|---|---|
| `mm_wake_up_freq` | `"60S"` | `["10S", "60S", "120S"]` | Slower MM → stale quotes → MR/MOMENTUM pick-off improves; HOLD unaffected |
| `mm_spread_alpha` | `0.75` | `[0.25, 0.75, 0.95]` | Higher alpha → spread tracks conditions faster → active strategies pay more variable costs |
| `mm_window_size` | `"adaptive"` | `[2, 5, "adaptive"]` | Fixed narrow window → tighter spread → all active strategies improve |
| `num_noise_agents` | `1000` | `[100, 500, 1000]` | Fewer noise → thinner book + noisier obs → active strategies hurt more than HOLD |
| `num_momentum_agents` | `12` | `[0, 12, 50]` | More momentum → stronger trends → MOMENTUM improves, MR worsens |
| `fund_vol` | `1.5e-4` | `[1e-5, 1.5e-4, 5e-4]` | Higher vol → larger intraday swings → more opportunity and more risk for active strategies |
| `kappa_oracle` | `1.67e-16` | `[1e-17, 1.67e-16, 1e-15]` | Faster mean reversion → stronger MR signal → MR_INV benefits most |

### Env-Level Parameters (not background config)

These are passed directly to `gym.make()`, not via `background_config_extra_kvargs`.

| Parameter | Default | Sweep Values | Hypothesis |
|---|---|---|---|
| `timestep_duration` | `"60s"` | `["30s", "60s", "300s"]` | Longer step → coarser obs → fewer trades → lower total spread cost but worse signal |
| `order_fixed_size` | `10` | `[5, 10, 25]` | Larger order → more market impact per trade → active strategies face more slippage |

---

## Metrics

Computed from `run_episode()` as in `test.py`. All are available with no code changes.

| Metric | How to compute |
|---|---|
| Final M2M P&L | `info["marked_to_market"] - env.starting_cash` at episode end |
| Total episode reward | Sum of step rewards over the episode |
| Sharpe ratio | `mean(pnls) / std(pnls)` across episodes |
| Win rate | `fraction of episodes with final_pnl >= 0` |
| Max drawdown | `max((peak_m2m - m2m_t) / peak_m2m)` over the episode |
| Transaction cost proxy | `sum(spread_t for non-HOLD steps) / 2` |

---

## Smoke Test

```bash
# install (one-time, from belief-aware-RL-project/)
pip3 install -e abides-core -e abides-markets -e abides-gym
```

```python
# copy AAPL_PARAMS and run_episode() from test.py, then:
import gym, abides_gym

env = gym.make(
    "markets-daily_investor-v0",
    background_config              = "rmsc04",
    timestep_duration              = "60s",
    starting_cash                  = 1_000_000,
    order_fixed_size               = 10,
    state_history_length           = 4,
    market_data_buffer_length      = 5,
    first_interval                 = "00:05:00",
    reward_mode                    = "dense",
    done_ratio                     = 0.3,
    debug_mode                     = True,
    background_config_extra_kvargs = AAPL_PARAMS,
)

from test import HoldBaseline, run_episode
ep = run_episode(env, HoldBaseline(), seed=1)
print(ep["final_pnl"], ep["episode_length"])
```

Expected: `final_pnl = 0`, episode runs to full length (HOLD never hits done_ratio).

---

## Sweep Script

```python
import json
import gym
import abides_gym
import numpy as np

# copy HoldBaseline, MomentumBaseline, MeanReversionBaseline,
# MeanReversionInventoryBaseline, BuyAndHoldBaseline, RandomBaseline,
# run_episode(), summarise() from test.py

AAPL_PARAMS = dict(
    r_bar=19_000, kappa=1.67e-15, kappa_oracle=1.67e-16,
    fund_vol=1.5e-4, sigma_s=0, lambda_a=5.7e-12,
    num_value_agents=102, num_noise_agents=1000, num_momentum_agents=12,
)

STRATEGIES = {
    "HOLD":     HoldBaseline,
    "BUY_HOLD": BuyAndHoldBaseline,
    "MR":       MeanReversionBaseline,
    "MR_INV":   MeanReversionInventoryBaseline,
    "MOMENTUM": MomentumBaseline,
    "RANDOM":   RandomBaseline,
}

N_EPISODES = 20  # episodes per (param, value, strategy) cell

# OAT sweep: background config params
background_sweep = {
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

# OAT sweep: env-level params (passed directly to gym.make)
env_sweep = {
    "timestep_duration": ["30s", "60s", "300s"],
    "order_fixed_size":  [5, 10, 25],
}

ENV_DEFAULTS = dict(
    background_config              = "rmsc04",
    timestep_duration              = "60s",
    starting_cash                  = 1_000_000,
    order_fixed_size               = 10,
    state_history_length           = 4,
    market_data_buffer_length      = 5,
    first_interval                 = "00:05:00",
    reward_mode                    = "dense",
    done_ratio                     = 0.3,
    debug_mode                     = True,
    background_config_extra_kvargs = AAPL_PARAMS,
)

results = []

def run_param(param, val, env_kwargs, bg_kwargs):
    env = gym.make("markets-daily_investor-v0", **env_kwargs,
                   background_config_extra_kvargs=bg_kwargs)
    for strat_name, Strat in STRATEGIES.items():
        agent = Strat()
        for seed in range(N_EPISODES):
            ep = run_episode(env, agent, seed)
            results.append({
                "param": param, "value": str(val),
                "strategy": strat_name, "seed": seed,
                **{k: ep[k] for k in
                   ["final_pnl", "total_reward", "max_drawdown",
                    "transaction_cost", "won", "episode_length"]},
            })
    env.close()

# background config sweeps
for param, values in background_sweep.items():
    for val in values:
        bg = {**AAPL_PARAMS, param: val}
        env_kw = {k: v for k, v in ENV_DEFAULTS.items()
                  if k != "background_config_extra_kvargs"}
        run_param(param, val, env_kw, bg)

# env-level sweeps
for param, values in env_sweep.items():
    for val in values:
        env_kw = {k: v for k, v in ENV_DEFAULTS.items()
                  if k not in ("background_config_extra_kvargs", param)}
        env_kw[param] = val
        run_param(param, val, env_kw, AAPL_PARAMS)

# summarise
import pandas as pd
df = pd.DataFrame(results)
summary = (
    df.groupby(["param", "value", "strategy"])["final_pnl"]
    .agg(mean="mean", std="std", sharpe=lambda x: x.mean()/x.std() if x.std()>0 else 0)
    .reset_index()
)
print(summary.to_string())
with open("sweep_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
```

### Fast-run tip

Use `timestep_duration="300s"` (5-min steps) and `N_EPISODES=10` for a first pass.
5-min steps cut simulation time ~5× vs 60s with minimal effect on strategy rankings.

---

## Expected Findings

- **`mm_wake_up_freq` strongest lever for MR** — at 120s the MM goes stale between steps;
  MR and MR_INV can observe a price move and fade it before the MM reprices. At 10s
  the MM reprices faster than the RL agent's timestep, eliminating the edge.
- **`fund_vol` determines P&L magnitude** — higher vol = larger intraday swings = more
  opportunity for MR/MOMENTUM and more risk for BUY_HOLD. HOLD is unaffected.
- **`num_momentum_agents` creates a crossover** — MOMENTUM improves monotonically with
  more momentum agents; MR degrades as trends overpower mean reversion.
- **`order_fixed_size` scales transaction cost** — larger orders move the price more and
  pay more spread. RANDOM and MOMENTUM strategies are hurt most; MR_INV (which caps
  inventory) is most robust to size increases.
- **`num_value_agents` creates a MR vs MOMENTUM crossover** — at low counts the mid
  price drifts freely and MOMENTUM dominates; at high counts fundamental anchoring is
  strong and MR/MR_INV dominate. This crossover point is the most useful finding for
  choosing a market regime to train a belief-aware RL agent in.
- **HOLD is the control** — its P&L should be 0.0 across all parameter values.
  Any deviation signals a bug in `env.previous_marked_to_market` reset (BUG 1).
