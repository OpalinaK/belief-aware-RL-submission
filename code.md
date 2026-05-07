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
