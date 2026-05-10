# belief-aware-RL-project

## What This Is

ABIDES-Gym: an agent-based market simulation framework (ABIDES) wrapped as an OpenAI Gym environment.
The goal is to train belief-aware RL agents that trade a single equity over a simulated day.

## Packages

Three editable packages installed into `.venv`:

| Package | Install path | Purpose |
|---|---|---|
| `abides-core` | `./abides-core` | Kernel, agent base class, latency model, utilities |
| `abides-markets` | `./abides-markets` | Market agents, configs, order book, oracle |
| `abides-gym` | `./abides-gym` | Gym env wrappers, baseline strategies, sweep scripts |

## Setup

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python "gym==0.25.2" "numpy<2.0" pandas scipy tqdm matplotlib
uv pip install --python .venv/bin/python -e abides-core -e abides-markets -e abides-gym
```

Do NOT install `ray[rllib]` unless you need RLlib training — the gym env works without it.

For the baseline PPO plan (`plan-v1.md`), RLlib is required but is not currently installed in the
project `.venv`. Do not reinstall from the root `requirements.txt` wholesale: that file is historical
and still pins old Gym/Numpy/Ray/Pomegranate versions that conflict with the current working setup.
Install Ray/RLlib explicitly after choosing a compatible version for Python 3.10 + Gym 0.25.2.

## Running Scripts

Always prefix with `.venv/bin/python`. The venv is at `./belief-aware-RL-project/.venv`.

```bash
# Baseline strategy benchmark (6 strategies × N episodes)
.venv/bin/python abides-gym/scripts/test.py --episodes 20

# Parameter sweep (the sweep plan)
.venv/bin/python abides-gym/scripts/sweep.py

# Baseline PPO training (planned entry point)
.venv/bin/python abides-gym/scripts/train_ppo_daily_investor.py --help

# Quick smoke test
.venv/bin/python -c "import gym, abides_gym; env = gym.make('markets-daily_investor-v0', background_config='rmsc04'); env.seed(1); env.reset(); env.close(); print('ok')"
```

## Known Bugs (fixed in test.py and sweep.py)

**BUG 1 (CRITICAL):** `env.previous_marked_to_market` is NOT reset by `env.reset()`. Must
manually add after every reset:
```python
obs = env.reset()
env.previous_marked_to_market = env.starting_cash  # required
```

**BUG 2:** Win condition must be `final_pnl >= 0`, not `> 0` (HOLD gives exactly 0).

## Background Config Parameters

All passed via `background_config_extra_kvargs` to `gym.make()`.
Key params for the sweep:

| Param | Default | Notes |
|---|---|---|
| `r_bar` | `100_000` | Fundamental long-run mean (cents). AAPL: 19_000 |
| `kappa` | `1.67e-15` | Value agent mean-reversion belief |
| `kappa_oracle` | `1.67e-16` | Actual OU process kappa |
| `lambda_a` | `5.7e-12` | Value agent arrival rate |
| `fund_vol` | `5e-5` | Oracle fundamental volatility |
| `num_value_agents` | `102` | Background Bayesian value agents |
| `num_noise_agents` | `1000` | One-shot random traders |
| `num_momentum_agents` | `12` | MA crossover trend followers |
| `mm_wake_up_freq` | `"60S"` | MM refresh period — must be a string |
| `mm_spread_alpha` | `0.75` | MM spread EWMA decay |
| `mm_window_size` | `"adaptive"` | Int or `"adaptive"` |

## Code Conventions

- All cash values in cents (integer)
- Time values in nanoseconds (integer)
- `mm_wake_up_freq` must be a string like `"60S"` — rmsc04 calls `str_to_ns()` internally
- `window_size` accepts int or `"adaptive"` string
- No ray import at top level (guard with try/except — see `abides_gym/__init__.py`)
- PPO/RLlib training must use a reset-safe env wrapper because RLlib calls `env.reset()` internally

## Speed Optimization Notes (speedip.md)

- Baseline throughput should be measured with the same smoke command and compared using
  `timing.json` (`env_steps_per_sec`, `wall_time_sec`, `eval_wall_time_sec`).
- Keep training speed and evaluation fidelity separate:
  - training path can use minimal info/debug settings
  - eval path keeps required metrics (`marked_to_market`, action distribution, drawdown)
- For Ray 2.55+, avoid deprecated policy inference calls in eval (`compute_single_action`);
  use RLModule inference path.
- Do not run duplicate sweeps while optimizing PPO throughput.

## plan-v2 Experiments (belief-aware RL)

See `plan-v2.md` for full spec. Execution order: E0 → E1+E2a → E2 → (gate) → E3.

**Action encoding** (source of truth from env code — plan-v2 had this wrong):
- `0` = BUY, `1` = HOLD, `2` = SELL

**Regime design** (ρ = num_momentum_agents / num_value_agents, threshold ρ* = 0.06):
- Val regime: ρ < 0.06 → `num_momentum_agents=0`, belief b* = (1, 0)
- Mom regime: ρ ≥ 0.06 → `num_momentum_agents=12`, belief b* = (0, 1)

**RegimeAdapter design**: standalone `gymnasium.Env` (same pattern as `GymnasiumDailyInvestorAdapter`), not a wrapper. Holds two internal `SubGymMarketsDailyInvestorEnv_v0` instances (one per regime). On each `reset()`, samples regime and resets the matching inner env. This avoids env_config patching issues.

**E0 obs shape**: Heuristic agents (`MomentumAgentBaseline`, `ValueAgentBaseline`) from `sweep.py` expect `(7,1)` shaped obs — do NOT flatten before calling `agent.act(obs)`.

**v4 baseline**: mean P&L = 32,854 cents, Sharpe = 2.507, win rate = 100% (60s/386-step episodes, 2 seeds).

**New scripts** (plan-v2):
```bash
# E0 — oracle heuristic benchmark (no training)
.venv/bin/python abides-gym/scripts/run_oracle_heuristic.py --episodes 20 --seeds 0 1 2

# E1 — PPO + belief in obs
.venv/bin/python abides-gym/scripts/train_ppo_oracle_obs.py --seed 0 --timestep 60s

# E2a — alpha sweep for reward shaping (run before E2)
.venv/bin/python abides-gym/scripts/sweep_alpha_e2.py --alphas 1e-4 1e-3 1e-2

# E2 — PPO + reward shaping
.venv/bin/python abides-gym/scripts/train_ppo_oracle_reward.py --seed 0 --alpha 0.001

# E3 — inferred belief (gated on E1 or E2 beating v4)
.venv/bin/python abides-gym/scripts/belief_estimator_train.py
.venv/bin/python abides-gym/scripts/train_ppo_inferred_belief.py --seed 0
```
