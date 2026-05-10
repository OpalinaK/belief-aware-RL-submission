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

---

## Belief-Aware RL Experiments (plan-v2.md)

### v4 Baseline (prerequisite)
- [x] Seeds 0 and 1 complete (seed_0: 30,528¢ / Sharpe 2.69; seed_1: 35,179¢ / Sharpe 2.32)
- [ ] Re-run seed 2 — failed previously with disk quota error (now resolved: 4.4 TB free)

### E0 — Oracle Heuristic Benchmark (no training)
- [x] Create `abides-gym/scripts/run_oracle_heuristic.py`
  - BUG B fixed: agents receive raw (7,1) obs, not flattened
  - Supports `--save-trajectories` for E3 training data
- [ ] Complete E0 run — currently partial (12/120 episodes, val_seed0 only, run crashed)
  - Re-run: `.venv/bin/python abides-gym/scripts/run_oracle_heuristic.py --episodes 20 --seeds 0 1 2 --save-trajectories`
  - Output: `results/e0_oracle_heuristic/eval_metrics.json`

### E1 — PPO + Belief in Observation
- [x] Create `abides-gym/abides_gym/envs/regime_adapter.py`
  - Standalone `gymnasium.Env` (NOT a wrapper); BUG C fixed
  - Dual inner envs (one per regime); BUG 1 handled via `ResetMarkedToMarketWrapper`
- [x] Create `abides-gym/scripts/train_ppo_oracle_obs.py`
  - Uses `RegimeAdapter`; 9-dim obs; `regime_mode="random"` for train+eval
- [ ] Update `abides-gym/abides_gym/envs/__init__.py` to export `RegimeAdapter`
- [ ] Full runs: `--seed 0 1 2`, `--timesteps 200000`
  - `.venv/bin/python abides-gym/scripts/train_ppo_oracle_obs.py --seed 0`

### E2a — Alpha Sweep for Reward Shaping
- [ ] Create `abides-gym/scripts/sweep_alpha_e2.py`
  - Wrapping order: `RegimeRewardWrapper(RegimeAdapter(...), alpha=α)`
  - Short runs: `--timesteps 50000` per alpha
  - Alphas: `{1e-4, 1e-3, 1e-2}`, seed=0 only
  - Report: episode_reward_mean per alpha, pick best for E2
  - Output: `results/e2_alpha_sweep/alpha_sweep.json`
- [ ] Smoke test: 1 iteration per alpha, verify no crash and JSON written

### E2 — PPO + Reward Shaping
- [ ] Create `abides-gym/abides_gym/envs/regime_reward_wrapper.py`
  - `gymnasium.Wrapper` around `RegimeAdapter`
  - Adds `α * align(action, belief)` to reward
  - `align`: Val (belief[0]>0.5) → +1 if action==0 (BUY); Mom → +1 if action matches return sign (0=BUY if ret>0, 2=SELL if ret<0)
  - Stripped at eval (eval uses `RegimeAdapter` only)
- [ ] Update `abides-gym/abides_gym/envs/__init__.py` to export `RegimeRewardWrapper`
- [ ] Create `abides-gym/scripts/train_ppo_oracle_reward.py`
  - Train: `RegimeRewardWrapper(RegimeAdapter(...))`, eval: `RegimeAdapter` only
  - `--alpha 0.001` (or best α from E2a)
  - Output dir: `results/e2_belief_reward/seed_{seed}/`
- [ ] Smoke test: 3 training iterations, 2 eval seeds
- [ ] Full runs: `--seed 0 1 2`, `--timesteps 200000`

### Gate Check (between E2 and E3)
- [ ] Compare E1 and E2 mean P&L, Sharpe, win rate vs v4 baseline
- [ ] If neither beats v4 on all three metrics: diagnose, then decide (skip E3 or retry)
- [ ] If gate passed: proceed to E3

### E3 — PPO + Inferred Belief (gated)
- [ ] Collect trajectories: run `run_oracle_heuristic.py --save-trajectories`
- [ ] Create `abides-gym/scripts/belief_estimator_train.py`
  - Sliding-window MLP (W=20 steps, input=140, hidden=[128,64], output=2)
  - Cross-entropy loss, Adam lr=1e-3, 50 epochs, 80/20 train/val split
  - Save weights to `results/e3_belief_estimator/estimator.pt`
- [ ] Create `abides-gym/envs/belief_estimator_wrapper.py`
  - Standalone `gymnasium.Env`; no oracle regime
  - Maintains deque of last W obs; calls MLP to get ĥ
  - Appends ĥ to obs → 9-dim (same shape as E1/E2)
- [ ] Update `abides-gym/abides_gym/envs/__init__.py` to export `BeliefEstimatorWrapper`
- [ ] Create `abides-gym/scripts/train_ppo_inferred_belief.py`
  - Output dir: `results/e3_inferred_belief/seed_{seed}/`
- [ ] Smoke test: estimator trains and wrapper steps without crash
- [ ] Full runs: `--seed 0 1 2`
