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
- [ ] Write `abides-markets/tests/test_sweep_smoke.py`
- [ ] Test: HOLD P&L == 0 across all param values (BUG 1 invariant)
- [ ] Test: `run_param()` produces correct result shape (N_EPISODES rows per strategy)
