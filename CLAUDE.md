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
