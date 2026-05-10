# TODO

## Daily Investor Parameter Sweep (value-sweep.md)

### Infrastructure (prerequisites)
- [x] Create `.venv` with gym 0.25.2, numpy<2.0, pandas, scipy, tqdm, matplotlib
- [x] Install abides-core, abides-markets, abides-gym in editable mode
- [x] Guard `ray` import in `abides_gym/__init__.py` with try/except
- [x] Replace `pomegranate` in `order_size_model.py` with numpy equivalent
- [x] Smoke test: env makes, resets, steps without error

### Chunk 1 — Sweep script scaffold
- [x] Create `abides-gym/scripts/sweep.py`
- [x] Import and re-use all 6 strategy classes and `run_episode()` / `summarise()` from `test.py`
- [x] Define `AAPL_PARAMS`, `ENV_DEFAULTS`, `background_sweep`, `env_sweep` dicts
- [x] Implement `run_param()` helper with BUG 1 fix applied
- [x] Write results to `sweep_results.json`
- [x] Print summary table to stdout

### Chunk 2 — CLI and fast-run mode
- [x] Add `argparse`: `--episodes`, `--strategies`, `--fast` (300s timestep, 10 eps),
      `--params` (subset of sweep keys), `--out` (output file)
- [x] Print progress: param / value / strategy / episode as it runs

### Chunk 3 — Plots
- [x] Bar chart: mean P&L by strategy, faceted by param value (one figure per param)
- [x] Crossover plot: MR vs MOMENTUM P&L as `num_value_agents` varies
- [x] Save to `abides-gym/scripts/plots/sweep_*.png`

### Chunk 4 — Smoke / regression test
- [x] Write `abides-markets/tests/test_sweep_smoke.py`
- [x] Test: HOLD P&L == 0 across all param values (BUG 1 invariant)
- [x] Test: `run_param()` produces correct result shape (N_EPISODES rows per strategy)

## Baseline PPO Agent (plan-v1.md)

### Preconditions
- [x] Decide how to install RLlib/Ray for Python 3.10 + Gym 0.25.2 without reinstalling stale root `requirements.txt`
- [ ] Confirm targeted fast sweep status/results from the existing running tmux job; do not start a duplicate sweep

### Chunk 1 — PPO training scaffold
- [x] Create `abides-gym/scripts/train_ppo_daily_investor.py`
- [x] Add RMSC04 and env defaults matching `sweep.py`
- [x] Register a reset-safe RLlib env factory for `markets-daily_investor-v0`
- [x] Add CLI args for timesteps, timestep duration, seed, out dir, eval seeds, checkpoint frequency

### Chunk 2 — Evaluation and metrics
- [x] Implement deterministic `explore=False` evaluation on held-out seeds
- [x] Write `eval_episodes.json`
- [x] Write `eval_metrics.json` with P&L, Sharpe, win rate, drawdown, episode length, and action distribution
- [x] Reuse metric definitions from existing baseline/sweep scripts where possible

### Chunk 3 — Timed smoke run
- [x] Run `300s` timed smoke test
- [x] Write `timing.json`
- [x] Use smoke throughput to decide whether `60s` full runs are feasible

### Chunk 4 — Full baseline runs
- [ ] Run three independent PPO seeds on available GPUs if timing permits
- [ ] Aggregate seed-level outputs into `results/ppo_baseline/summary.json`
- [ ] Record final runtime and any deviations from `plan-v1.md` in `LOG.md`

## Speed Improvement Plan (speedip.md)

### Preconditions
- [x] Baseline timed smoke exists for `60s` and `300s`
- [x] Current Ray 2.55/Gymnasium-compatible training script runs end-to-end at `num_workers=0`
- [ ] Keep sweep status tracked separately (no duplicate sweep launches during speed work)

### Chunk 1 — Train debug off path
- [x] Add `--train-debug-mode` and `--eval-debug-mode` flags
- [x] Use separate env configs for train vs eval (`debug_mode=False` for train by default)
- [x] Benchmark throughput delta on `60s` smoke

### Chunk 2 — Minimal info path for training
- [x] Add adapter info mode switch (`minimal` vs `full`)
- [x] Keep full info for eval metrics, minimal info for train loop
- [x] Benchmark throughput delta on `60s` smoke

### Chunk 3 — Worker reliability for `num-workers > 0`
- [x] Fix worker-side environment construction path so remote env runners initialize reliably
- [x] Benchmark `num-workers=0,2,4` with same smoke settings
- [x] Pick best worker count for full runs (train throughput: `--num-workers 4` best on smoke host; tune per machine)

### Chunk 4 — Profiling toggles
- [x] Add `--profile-phases` and write phase timings in `timing.json`
- [x] Include setup/train/eval timing breakdown fields

### Chunk 5 — I/O cadence tuning
- [x] Tune checkpoint/eval cadence for speed runs
- [x] Measure wall-time impact at fixed timesteps
