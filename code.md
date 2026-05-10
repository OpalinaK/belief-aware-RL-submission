## Code Review — commit 29a520897e23c9d6011a1b98c58b32ae389775d8 (2026-04-27)

---

## Phase 1: Repo Overview

**Purpose:** ABIDES-Gym — agent-based market simulation framework with OpenAI Gym wrappers.
Three editable packages live as siblings under `belief-aware-RL-project/`:

```
abides-core/      kernel, agent base, latency model, utils
abides-markets/   market agents (value, noise, MM, momentum), configs, order book
abides-gym/       gym env wrappers around abides-markets, experimental scripts
```

**Entry points:**
- `abides-gym/scripts/test.py` — baseline strategy runner (primary entry point for the sweep plan)
- `abides-gym/scripts/gym_runner_daily_investor.py` — original gym runner (simpler, no plots)
- `abides-gym/scripts/rllib_runner.py` — RLlib training entry point (requires ray, not installed)
- `abides-markets/abides_markets/configs/rmsc04.py` — background config builder

**Key data structures:**
- `Kernel` (abides-core) — discrete-event simulation engine, routes messages between agents
- `Agent` (abides-core) — base class; all market participants subclass this
- `AbidesGymCoreEnv` → `AbidesGymMarketsEnv` → `SubGymMarketsDailyInvestorEnv_v0` — gym env hierarchy
- `FinancialGymAgent` — the gym agent injected into the ABIDES simulation at reset
- `OrderBook` — exchange-side matching engine

**Architecture pattern:** Discrete-event message-passing simulation. The gym env rebuilds the
entire kernel on every `reset()` call, injecting a `FinancialGymAgent` that wakes on a fixed
schedule and exposes obs/reward/done/info back to the gym step loop.

**Structural oddities:**
- `abides-gym/__init__.py` unconditionally imported `ray` — fixed to optional try/except
- `pomegranate` (compiled, Python 3.7-3.9 only) used solely for order size sampling — replaced
  with equivalent pure-numpy implementation
- `version_testing/` directory contains ad-hoc scripts, no clear test harness

---

## Phase 2: File-by-File Drill Down

### `abides-gym/abides_gym/envs/core_environment.py`
- Rebuilds full ABIDES kernel on every `reset()` — correct, necessary for seed isolation
- `reset()` does NOT reset `previous_marked_to_market` — this is the BUG 1 documented in test.py
  (must be reset manually after every `env.reset()` call)
- Handles both gym<=0.21 and gym>=0.26 RNG APIs correctly

### `abides-gym/abides_gym/envs/markets_daily_investor_environment_v0.py`
- Action space: Discrete(3) — 0=BUY, 1=HOLD, 2=SELL
- Observation: `[holdings, imbalance, spread, direction_feature] + padded_returns` (shape: (7,1) at default state_history_length=4)
- Dense reward: `(current_M2M - previous_M2M) / order_fixed_size / num_steps_per_episode`
- Sparse reward: `(final_M2M - starting_cash) / order_fixed_size / num_steps_per_episode`
- Done condition: `marked_to_market <= done_ratio * starting_cash`
- `__init__` initialises `self.previous_marked_to_market = self.starting_cash` — but only once;
  subsequent resets do not reset it (BUG 1)

### `abides-markets/abides_markets/configs/rmsc04.py`
- 2 AdaptiveMarketMakerAgents (identical params), 102 ValueAgents, 1000 NoiseAgents, 12 MomentumAgents
- `mm_wake_up_freq` parameter is a string (e.g. "60S"), converted internally via `str_to_ns()`
- Oracle runs from `MKT_OPEN` to `NOISE_MKT_CLOSE` (30min past mkt close) — value agents can
  get fundamental signals after market close, which shouldn't matter for gym RL agent
- All sweep parameters (`kappa`, `lambda_a`, `r_bar`, `fund_vol`, `kappa_oracle`, `num_*_agents`,
  `mm_*`) are valid `build_config` kwargs — passable via `background_config_extra_kvargs`

### `abides-gym/scripts/test.py`
- Canonical baseline runner. BUG 1 fix applied correctly (reset after every env.reset()).
- BUG 2 fix correct (win condition `>= 0`).
- `run_episode()` returns full trace data with `_`-prefixed keys for plotting, stripped for JSON.
- `run_param()` in sweep script creates a new env per (param, val) call — correct for isolation.
- **Issue:** `run_param()` creates a new env per cell but doesn't pass `background_config_extra_kvargs`
  for env-level sweep params — plan's sweep script already handles this correctly by separating
  `background_sweep` from `env_sweep`.

### `abides-markets/abides_markets/models/order_size_model.py`
- Replaced pomegranate with pure-numpy mixture sampler. Functionally equivalent.
- Sampling distribution: LogNormal(2.9, 1.2) with weight 0.2, Normal(100k, 0.15) with 0.7,
  spike normals at 200-1000 for block trades.

### `abides-gym/abides_gym/__init__.py`
- Ray import now guarded with try/except — lean install works without ray.

---

## Phase 3: Cross-Cutting Issues

1. **BUG 1 must be applied in every episodic loop** — `env.previous_marked_to_market` is not
   reset by `env.reset()`. The sweep script must apply the fix (as test.py does) or dense
   rewards will be wrong for all episodes after the first.

2. **`run_param()` in sweep script creates one env per (param, val, strategy-batch)** — this
   is correct but means N_PARAMS × N_VALUES env constructions. Each `gym.make()` is cheap;
   the cost is the per-episode simulation. No issue.

3. **`mm_wake_up_freq` must be a string** — rmsc04 calls `str_to_ns(mm_wake_up_freq)` internally.
   The sweep correctly passes strings like `"10S"`. Passing integers would crash.

4. **`env_sweep` params (`timestep_duration`, `order_fixed_size`) cannot go through
   `background_config_extra_kvargs`** — they are env constructor args, not config args.
   The sweep script handles this correctly by building `env_kw` separately.

5. **No CI** — no `.github/workflows/` directory. All testing is local only.

6. **Existing test (`test_gym_runner.py`)** runs 5 steps with no assertions beyond "doesn't crash".
   It doesn't test BUG 1 fix, reward correctness, or HOLD == 0 invariant.

7. **Most likely failure point:** a sweep cell where `mm_window_size=2` (integer) is passed.
   rmsc04 passes it to `AdaptiveMarketMakerAgent(window_size=2)` — need to verify the MM agent
   accepts integers for window_size, not just "adaptive".

---

## Cross-Reference: Plan vs Codebase

| Plan item | Status |
|---|---|
| `background_config_extra_kvargs` passthrough | Confirmed working in `core_environment.reset()` |
| `mm_wake_up_freq` as string | Confirmed — rmsc04 converts internally |
| BUG 1 fix in sweep script | Not yet written — must be included |
| All sweep params valid kwargs | Confirmed for all 10 background params |
| `env_sweep` params separate from bg params | Confirmed — correct approach in plan |
| Smoke test: HOLD pnl=0 | Confirmed working end-to-end |
| No ray required for gym env | Fixed — try/except guard added |
| pomegranate | Replaced with numpy equivalent |

---

## Code Review — commit 5ffa514af0c2c768452e476b0a0af1b1bd20a225 (2026-05-09)

---

## Phase 1: Repo Overview

**Purpose:** ABIDES-based market simulation project with Gym wrappers and experiment scripts for
daily-investor trading policies. The current plan adds a baseline PPO training/evaluation entry point
for `markets-daily_investor-v0`.

**Directory ownership:**
- `abides-core/` owns the discrete-event kernel, base agent class, message routing, latency, and time utilities.
- `abides-markets/` owns market agents, exchange/order-book logic, oracle models, and RMSC configs.
- `abides-gym/` owns OpenAI Gym/RLlib wrappers plus scripts for baseline rollouts, sweeps, and older RLlib examples.
- `docs/` owns report/writeup artifacts.
- `version_testing/` contains ad-hoc compatibility/regression scripts rather than the primary test suite.
- Root docs (`CLAUDE.md`, `ARCH.md`, `TODO.md`, `LOG.md`, `plan-v1.md`, `value-sweep.md`) hold project context and experiment plans.

**Entry points:**
- `abides-gym/scripts/sweep.py` is the current structured baseline/sweep runner.
- `abides-gym/scripts/test_rmsc04.py` and `test.py` are baseline rollout runners.
- `abides-gym/scripts/rllib_runner.py`, `simple_rllib_runner.py`, and `rllib_exec_custom_metrics.py` are older RLlib examples for execution env/DQN.
- Planned new entry point: `abides-gym/scripts/train_ppo_daily_investor.py`.

**Key objects flowing through the system:**
- `Kernel` runs ABIDES discrete-event simulation.
- `ExchangeAgent`, `ValueAgent`, `NoiseAgent`, `MomentumAgent`, and `AdaptiveMarketMakerAgent` generate the market.
- `FinancialGymAgent` bridges ABIDES state/actions into Gym.
- `SubGymMarketsDailyInvestorEnv_v0` exposes `(obs, reward, done, info)` for PPO.
- Experiment scripts produce JSON metrics, plots, and checkpoints.

**Architectural pattern:** full ABIDES kernel rebuild on every Gym reset, then fixed-interval
`FinancialGymAgent` wakes provide an RL loop over a message-passing market simulator. This makes PPO
training dominated by simulator throughput and reset correctness rather than GPU throughput.

**Structural concerns for this plan:**
- Existing RLlib scripts target `markets-execution-v0` and DQN, not the daily-investor PPO task.
- The daily-investor env has a known reset invariant: `previous_marked_to_market` must be reset after
  each `env.reset()`. RLlib requires this inside an env wrapper/factory, not just hand-written rollout code.
- No `.github/workflows/` files are present, so CI gating is currently unavailable unless added later.

---

## Phase 2: File-by-File Drill Down

### `plan-v1.md`
- Responsibility: implementation plan for baseline PPO on `markets-daily_investor-v0`.
- Inputs/outputs: plan text only; drives creation of a PPO training script, smoke timing outputs,
  checkpoint outputs, and evaluation JSON.
- Key requirement: baseline must use standard observation only; oracle/belief features are explicitly
  out of scope for v1.
- Issue: plan assumes RLlib availability, but `.venv` currently has no `ray` module installed.

### `abides-gym/scripts/sweep.py`
- Responsibility: OAT parameter sweep over baseline heuristic strategies.
- Inputs: CLI params, environment/background kwargs, strategy names, episode counts.
- Outputs: JSON rows, stdout summary, optional plots.
- Relevant to PPO plan because it provides the canonical RMSC04 defaults and metric definitions.
- Reset bug is handled correctly in `run_episode()` via `env.previous_marked_to_market = env.starting_cash`.
- Current user note: a fast/targeted sweep is already running in another session, so PPO implementation
  should not start or duplicate sweep work.

### `abides-gym/abides_gym/envs/markets_daily_investor_environment_v0.py`
- Responsibility: concrete Gym env for daily investor trading.
- Inputs: ABIDES background config plus env args; actions are discrete integers.
- Outputs: observation `(num_state_features, 1)`, scalar reward, done flag, debug info.
- Key functions: `_map_action_space_to_ABIDES_SIMULATOR_SPACE`, `raw_state_to_state`,
  `raw_state_to_reward`, `raw_state_to_update_reward`, `raw_state_to_done`, `raw_state_to_info`.
- Critical issue for RLlib: `previous_marked_to_market` is initialized in `__init__` but not reset
  on `env.reset()`, so the training env factory needs a wrapper or subclass.
- Compatibility concern: observations are shaped `(7, 1)`. RLlib PPO can usually flatten Box
  observations, but the eval path must pass the same shape back into `compute_single_action`.

### `abides-gym/abides_gym/__init__.py`
- Responsibility: registers Gym and optional RLlib env IDs.
- Inputs/outputs: import side effect registers `markets-daily_investor-v0` and `markets-execution-v0`.
- Good: `ray` import is guarded, so non-RL scripts can run without RLlib.
- Issue for this plan: built-in RLlib registration returns the raw env, not the reset-safe wrapper.
  The PPO training script should register its own env factory for training.

### `abides-gym/scripts/rllib_runner.py`
- Responsibility: older RLlib example with WandB/custom callbacks.
- Inputs/outputs: Tune config for `markets-execution-v0`, DQN, checkpointing.
- Not directly reusable: targets execution env, imports WandB, uses DQN, and grid-searches execution
  parameters unrelated to daily-investor PPO.
- Useful only as a reference for old RLlib/Tune style.

### `abides-gym/scripts/simple_rllib_runner.py`
- Responsibility: simpler older RLlib example.
- Inputs/outputs: Tune config for `markets-execution-v0`, DQN.
- Not directly reusable for v1 because it does not register the daily-investor reset wrapper and does
  not implement evaluation/metrics JSON.

### `abides-markets/tests/test_sweep_smoke.py`
- Responsibility: smoke/regression tests for sweep logic and BUG 1 invariant.
- Inputs/outputs: pytest creates `markets-daily_investor-v0` at `300s`, asserts HOLD P&L is zero and
  `run_param()` row counts match.
- Useful model for local smoke tests: fast env construction, `300s` timestep, small seed count.
- Gap: no test yet for a PPO training script, reset-safe wrapper, eval summarization, or timing output.

### `CLAUDE.md`
- Responsibility: project setup and conventions.
- Current state: documents that `ray[rllib]` should not be installed unless needed.
- Needed update: document PPO baseline dependency and the fact that current `.venv` does not have Ray.

### `ARCH.md`
- Responsibility: architecture map and data flow.
- Current state: covers ABIDES/Gym flow but not planned PPO training/eval components.
- Needed update: add planned PPO trainer/evaluator and reset wrapper data flow.

### `TODO.md`
- Responsibility: task tracking.
- Current state: completed sweep tasks only.
- Needed update: add baseline PPO tasks ordered by dependency.

### `requirements.txt`
- Responsibility: historical dependency pin list.
- Issue: pins `gym==0.18.0`, `numpy==1.22.0`, `pomegranate==0.14.5`, and `ray[rllib]==1.7.0`.
  This conflicts with current project memory/setup (`gym==0.25.2`, `numpy<2.0`, pomegranate replaced).
- Risk: blindly installing this file may break the working environment. PPO dependency installation
  should be handled carefully and explicitly, not by reinstalling all requirements.

---

## Phase 3: Cross-Cutting Issues

1. **Main blocker:** current `.venv` does not have `ray` installed, so an RLlib PPO implementation
   cannot run until we choose/install a compatible Ray/RLlib version.

2. **Dependency drift:** `requirements.txt` is stale relative to the working setup. It still includes
   `pomegranate` and old Gym/Numpy/Ray pins. Installing it as-is risks breaking the repo.

3. **Reset invariant is easy to miss:** scripts like `sweep.py` handle BUG 1 in hand-written rollouts,
   but RLlib owns reset calls. The PPO trainer must use a reset-safe env wrapper/factory.

4. **Existing RLlib examples are misleading starting points:** they use `markets-execution-v0`, DQN,
   optional WandB, and old Tune APIs. A new daily-investor PPO script should borrow concepts, not copy
   them.

5. **Most likely failure point:** RLlib/Gym API compatibility. This repo uses Gym's older 4-value
   `step()` API and currently works with `gym==0.25.2`. Newer Ray versions may expect Gymnasium/new API
   wrappers unless compatibility settings are handled.

6. **CI gap:** no `.github/workflows/` exists. All validation for the first PPO chunk will be local
   smoke tests unless CI is added later.

---

## Cross-Reference: Plan v1 vs Codebase

| Plan item | Status |
|---|---|
| Daily-investor env exists | Confirmed: `markets-daily_investor-v0` registered |
| Standard 7-feature observation | Confirmed for `state_history_length=4` |
| Discrete BUY/HOLD/SELL action space | Confirmed: `gym.spaces.Discrete(3)` |
| Dense M2M reward | Confirmed |
| Reset-safe wrapper | Not implemented; required for RLlib |
| PPO training script | Not implemented |
| Evaluation JSON writer | Not implemented |
| Timing JSON writer | Not implemented |
| RLlib dependency | Missing from current `.venv`; must resolve before PPO training can run |
| Fast sweep | Already running separately per user; do not duplicate |
| CI workflows | None present |

---

## Cross-Reference: speedip.md vs Current Codebase

### Already implemented (fully or partially)
- **Gymnasium compatibility adapter:** `abides-gym/scripts/train_ppo_daily_investor.py` now wraps the
  legacy Gym env and exposes Gymnasium reset/step signatures.
- **Observation flattening:** adapter converts `(7, 1)` observation to `(7,)` for Ray 2.55 PPO encoder.
- **Info sanitization:** adapter recursively stringifies info keys, avoiding vector-wrapper crashes due
  to integer nested keys.
- **Timed smoke outputs:** script writes `train_metrics.json`, `timing.json`, checkpoints, and eval files;
  `60s` smoke is currently faster than `300s` in observed runs.

### Dependencies / preconditions for speed plan
- **Stable speed benchmark baseline** exists for `num_workers=0` at `60s`, but worker-parallel runs
  still fail due to worker-side environment resolution/registration issues.
- **No CI workflows** exist, so all speed validation remains local benchmark-driven.
- **Driver/GPU warning** appears from Torch CUDA init (old driver), but training works on CPU mode;
  this is noise for env-throughput work unless GPU training is required.

### Conflicts or mismatches with speedip.md
- **speedip item 1 (train debug off)** is not yet exposed as separate train/eval flags in CLI.
- **speedip item 2 (minimal info mode)** is not yet split by train/eval mode; current adapter always
  sanitizes info recursively.
- **speedip item 3 (num-workers > 0 reliable)** is still unresolved; `num_workers=2` fails because
  worker actors cannot reliably resolve the env construction path.
- **speedip item 4 (phase profiling toggles)** not yet implemented as explicit per-phase timing keys
  beyond top-level train/eval wall-time fields.
- **speedip item 5 (I/O cadence tuning)** not yet implemented; checkpoint/eval frequencies remain basic.
