## [2026-04-27] Daily investor parameter sweep (value-sweep.md)

### Features Implemented
- `sweep.py`: OAT parameter sweep over 12 market/env params × 6 baseline strategies
- Per-param bar charts (mean P&L ± 1 std) and MR-vs-MOMENTUM crossover plot
- BUG 1 regression test (HOLD pnl == 0 invariant) + row count smoke test
- Full project docs: CLAUDE.md, ARCH.md, TODO.md, code.md

### Files Changed
| File | What changed |
|------|-------------|
| `abides-gym/scripts/sweep.py` | New — full sweep runner, CLI, plots |
| `abides-gym/abides_gym/__init__.py` | Guard `ray` import with try/except |
| `abides-markets/abides_markets/models/order_size_model.py` | Replace pomegranate with numpy mixture sampler |
| `abides-markets/tests/test_sweep_smoke.py` | New — BUG 1 invariant + row count tests |
| `CLAUDE.md` | New — setup, conventions, known bugs |
| `ARCH.md` | New — component map, data flow, obs vector |
| `TODO.md` | New — task tracking, all items complete |
| `code.md` | New — 3-phase code review |
| `value-sweep.md` | New — plan doc (written before implementation) |

### Functions Written
| Function | File | Description |
|----------|------|-------------|
| `make_env` | `sweep.py` | Constructs gym env from env_kwargs + bg_kwargs |
| `run_param` | `sweep.py` | Runs all strategies × N episodes for one (param, val); appends rows to results list |
| `print_summary` | `sweep.py` | Prints grouped mean/std/sharpe/win% table to stdout |
| `plot_param` | `sweep.py` | Grouped bar chart of mean P&L per strategy, one figure per swept param |
| `plot_crossover` | `sweep.py` | MR vs MOMENTUM line chart with ±1 std bands as num_value_agents varies |
| `plot_all` | `sweep.py` | Drives plot_param for every param + plot_crossover |
| `OrderSizeModel.sample` | `order_size_model.py` | Numpy mixture sampler replacing pomegranate |

### Data Structures Created
| Name | File | Description |
|------|------|-------------|
| `BACKGROUND_SWEEP` | `sweep.py` | Dict mapping 10 background config param names → sweep value lists |
| `ENV_SWEEP` | `sweep.py` | Dict mapping 2 env-level param names → sweep value lists |
| `ENV_DEFAULTS` | `sweep.py` | Baseline gym.make kwargs used for all sweep cells |
| `AAPL_PARAMS` | `sweep.py` | AAPL-calibrated background config values (r_bar=19_000, etc.) |
| result row | `sweep.py` | Dict with keys: param, value, strategy, seed, final_pnl, total_reward, max_drawdown, transaction_cost, won, episode_length |

### Notes
- `pomegranate==0.14.5` has no Python 3.10 wheels and cannot be built from source without Python headers. Replaced with a functionally equivalent numpy mixture sampler — same distribution parameters, same `.sample(random_state)` interface.
- `gym==0.18.0` and `==0.21.0` fail to build on Python 3.10 due to a setuptools `extras_require` format change. `gym==0.25.2` is the last version with the old 4-value `step()` API that builds cleanly.
- `numpy<2.0` required — gym 0.25.2 uses `np.bool8` which was removed in numpy 2.0.
- All warnings in the test output are gym 0.25 deprecation notices, not errors.
- To run the full sweep: `.venv/bin/python3 abides-gym/scripts/sweep.py --fast` (~2–3h with 10 eps/cell). For a single param: `--params mm_wake_up_freq --episodes 5`.
