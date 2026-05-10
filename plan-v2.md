# Plan v2: Oracle and Belief-Aware RL Experiments

## Context

v4 PPO baseline: mean P&L = 32,854 cents, Sharpe = 2.507, win rate = 100% (2 seeds, 60s/386-step episodes).

v5.tex introduces the oracle regime setup. This plan specifies four experiments (E0–E3) and all required code changes.

**Success gate**: E1 or E2 must beat v4 on all three metrics (mean > 32,854, Sharpe > 2.507, win ≥ 100%) to justify E3.

---

## Regime Design

Regime is determined by the **momentum-to-value ratio** `ρ = num_momentum_agents / num_value_agents`. From the sweep data (v5.tex Table 2), VALUE dominates at ρ = 0 and MOMENTUM dominates at ρ ≈ 0.118 (12/102). A threshold of **ρ* = 0.06** bisects these two points.

| Regime | Condition       | Example params                  | Oracle belief `b*` |
|--------|-----------------|---------------------------------|--------------------|
| Val    | ρ < ρ* = 0.06  | `num_momentum_agents=0, num_value_agents=102`  (ρ=0.000) | (1, 0) |
| Mom    | ρ ≥ ρ* = 0.06  | `num_momentum_agents=12, num_value_agents=102` (ρ=0.118) | (0, 1) |

The ratio threshold makes the oracle label a principled function of market composition rather than a hardcoded agent count. Any background config with ρ < 0.06 is labelled Val; any with ρ ≥ 0.06 is labelled Mom. `RegimeAdapter` stores `rho_star = 0.06` and computes the label from the sampled params on each reset.

Episodes sample regime uniformly at random each reset. The oracle belief `b*` is a 2-dim float32 appended to obs (E1) or used to shape reward (E2).

---

## Protocol

All experiments use the **v4 protocol** to be comparable to the baseline:

- `timestep_duration = "60s"` (not "300s")
- `episode_length = 390` steps (one trading day)
- 20 episodes × 3 seeds = 60 total episodes per experiment arm
- Background config: rmsc04 with RMSC04_PARAMS (fund_vol=5e-5, lambda_a=5.7e-12, kappa=1.67e-15, etc.)
- BUG 1 fix applied after every `env.reset()`: `env.previous_marked_to_market = env.starting_cash`

---

## E0: Oracle Heuristic Benchmark (no training)

**Goal**: Establish a ceiling for the oracle signal using hand-coded heuristics. If the oracle heuristic cannot beat random or HOLD, the oracle signal is too weak to be useful.

**Protocol**: 20 episodes × 2 regimes × 3 seeds = 120 episodes total. Use Val regime → VALUE heuristic; Mom regime → MOMENTUM heuristic.

### New file: `abides-gym/scripts/run_oracle_heuristic.py`

Reuse `MomentumAgentBaseline` and `ValueAgentBaseline` from `sweep.py` (import directly).

```
Pseudocode:
  for seed in [0, 1, 2]:
    for regime in ["val", "mom"]:
      n_mom = 0 if regime == "val" else 12
      env = gym.make("markets-daily_investor-v0",
                     background_config="rmsc04",
                     background_config_extra_kvargs={**RMSC04_PARAMS,
                                                      "num_momentum_agents": n_mom})
      env.seed(seed)
      agent = ValueAgentBaseline() if regime == "val" else MomentumAgentBaseline()
      for ep in range(20):
        obs = env.reset()
        env.previous_marked_to_market = env.starting_cash  # BUG 1
        obs = obs.flatten()
        done = False
        while not done:
          action = agent.act(obs)
          obs, reward, done, info = env.step(action)
          obs = obs.flatten()
        record episode final_pnl, win, steps
  write results/e0_oracle_heuristic/eval_metrics.json
```

**Output**: `results/e0_oracle_heuristic/eval_metrics.json` with per-regime and aggregate stats (mean P&L, std, Sharpe, win rate, max drawdown).

**CLI**:
```bash
.venv/bin/python abides-gym/scripts/run_oracle_heuristic.py --episodes 20 --seeds 0 1 2
```

---

## E1: PPO + Belief in Observation

**Goal**: Train PPO with oracle regime belief `b*` appended to the 7-dim observation (→ 9-dim). The agent sees the true regime each episode.

### New file: `abides-gym/envs/regime_adapter.py`

A gymnasium-compatible wrapper that:
1. Accepts `regime_mode ∈ {"val", "mom", "random"}` in env kwargs
2. On each `reset()`: samples regime if `"random"`, sets `num_momentum_agents` accordingly, resets underlying env, appends `b*` to obs
3. On each `step()`: appends the current episode's `b*` to the returned obs
4. Overrides `observation_space` to `Box(-inf, inf, shape=(9,), dtype=float32)`

```python
RHO_STAR = 0.06  # momentum-to-value ratio threshold (Val: ρ < 0.06, Mom: ρ ≥ 0.06)

REGIME_PARAMS = {
    "val": {"num_momentum_agents": 0,  "num_value_agents": 102},
    "mom": {"num_momentum_agents": 12, "num_value_agents": 102},
}

class RegimeAdapter(gym.Wrapper):
    def __init__(self, env, regime_mode="random", rho_star=RHO_STAR):
        super().__init__(env)
        self.regime_mode = regime_mode
        self.rho_star = rho_star
        self._belief = np.zeros(2, dtype=np.float32)
        low  = np.concatenate([env.observation_space.low,  [-np.inf, -np.inf]])
        high = np.concatenate([env.observation_space.high, [ np.inf,  np.inf]])
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def _label(self, n_mom, n_val):
        rho = n_mom / max(n_val, 1)
        return "val" if rho < self.rho_star else "mom"

    def reset(self, **kwargs):
        regime = self.regime_mode if self.regime_mode != "random" else random.choice(["val", "mom"])
        p = REGIME_PARAMS[regime]
        rho = p["num_momentum_agents"] / p["num_value_agents"]
        # label is derived from ratio, not hardcoded — self-consistent by construction
        self._belief = np.array([1.0, 0.0] if rho < self.rho_star else [0.0, 1.0], dtype=np.float32)
        cfg = self.env.env_config["background_config_extra_kvargs"]
        cfg["num_momentum_agents"] = p["num_momentum_agents"]
        cfg["num_value_agents"]    = p["num_value_agents"]
        obs = self.env.reset(**kwargs)
        self.env.previous_marked_to_market = self.env.starting_cash  # BUG 1
        return np.concatenate([obs.flatten(), self._belief])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return np.concatenate([obs.flatten(), self._belief]), reward, done, info
```

**Note**: The underlying env object must expose `env_config` so the wrapper can patch `num_momentum_agents` before each episode. If `GymnasiumDailyInvestorAdapter` does not store this, add `self.env_config = env_config` to its `__init__`.

### New file: `abides-gym/scripts/train_ppo_oracle_obs.py`

Clone of `train_ppo_daily_investor.py` with these changes:
- Import and apply `RegimeAdapter` around the env
- Set `timestep_duration = "60s"` (match v4 protocol)
- Set `obs_space = Box(-inf, inf, (9,), float32)` in RLlib config
- Output dir: `results/e1_belief_obs/seed_{seed}/`
- Eval also uses `RegimeAdapter` with `regime_mode="random"` to match training distribution

**CLI**:
```bash
.venv/bin/python abides-gym/scripts/train_ppo_oracle_obs.py --seed 0
.venv/bin/python abides-gym/scripts/train_ppo_oracle_obs.py --seed 1
.venv/bin/python abides-gym/scripts/train_ppo_oracle_obs.py --seed 2
```

### Changes to existing files

- `abides-gym/envs/__init__.py`: add `from .regime_adapter import RegimeAdapter`
- `abides-gym/scripts/train_ppo_daily_investor.py`: no changes (keep as v4 baseline)

---

## E2: PPO + Belief in Reward (Reward Shaping)

**Goal**: Train PPO with the standard 7-dim obs but add an alignment bonus to the reward during training only. At eval, the wrapper is stripped.

### New file: `abides-gym/envs/regime_reward_wrapper.py`

```python
class RegimeRewardWrapper(gym.Wrapper):
    """
    Adds α * align(action, regime) to reward during training.
    align = +1 if action matches regime heuristic direction, 0 otherwise.

    Regime → preferred action mapping:
      Val  (belief=[1,0]): BUY  = action 2 (or whatever the env encodes buy as)
      Mom  (belief=[0,1]): follows momentum → map to BUY or SELL based on recent return sign
    """
    def __init__(self, env, alpha=0.001):
        super().__init__(env)
        self.alpha = alpha
        self._belief = np.zeros(2, dtype=np.float32)
        self._last_obs = None

    def reset(self, **kwargs):
        # regime sampling delegated to inner RegimeAdapter (wrap order: RegimeRewardWrapper(RegimeAdapter(base_env)))
        obs = self.env.reset(**kwargs)
        self._belief = obs[-2:]  # last 2 dims from RegimeAdapter
        self._last_obs = obs
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._belief = obs[-2:]
        shaped = reward + self.alpha * self._alignment(action, obs)
        self._last_obs = obs
        return obs, shaped, done, info

    def _alignment(self, action, obs):
        # obs[4] is the return feature (from sweep.py: obs[4,0])
        ret = obs[4] if obs.ndim == 1 else obs[4, 0]
        if self._belief[0] > 0.5:          # Val regime → reward BUY
            return 1.0 if action == 2 else 0.0
        else:                              # Mom regime → reward following trend
            preferred = 2 if ret > 0 else 0   # BUY if positive return, SELL if negative
            return 1.0 if action == preferred else 0.0
```

**Alpha search**: run E2 with α ∈ {1e-4, 1e-3, 1e-2}. Use α=1e-3 as default; only tune if E2 with default α fails to beat v4.

### New file: `abides-gym/scripts/train_ppo_oracle_reward.py`

Clone of `train_ppo_oracle_obs.py` with:
- Wrapping order: `RegimeRewardWrapper(RegimeAdapter(base_env, regime_mode="random"), alpha=0.001)`
- **Eval wrapper**: `RegimeAdapter` only (no reward shaping) — strip `RegimeRewardWrapper` at eval time
- Obs space: 9-dim (from `RegimeAdapter`, same as E1)
- Output dir: `results/e2_belief_reward/seed_{seed}/`

**CLI**:
```bash
.venv/bin/python abides-gym/scripts/train_ppo_oracle_reward.py --seed 0 --alpha 0.001
```

---

## E3: PPO + Inferred Belief Estimator (conditional on E1 or E2 passing gate)

**Goal**: Replace oracle b* with an estimated belief ĥ from a sliding-window estimator trained on historical obs. The agent sees (7-dim obs, ĥ) = 9-dim at inference.

### Phase 3a: Collect regime-labelled data

Run `run_oracle_heuristic.py` with `--save-trajectories` to dump obs sequences + regime labels to `results/e0_oracle_heuristic/trajectories/`.

Each trajectory: `{"obs": [T × 7 float32], "regime": "val" | "mom", "seed": int}`

### New file: `abides-gym/scripts/belief_estimator_train.py`

Trains a sliding-window MLP (or LSTM) to predict regime from the last `W=20` steps of obs.

Architecture:
- Input: `(W × 7)` flattened = 140-dim
- Hidden: [128, 64]
- Output: softmax over 2 classes (val, mom)
- Loss: cross-entropy
- Optimizer: Adam lr=1e-3, 50 epochs, train/val 80/20 split on trajectories

Saves model weights to `results/e3_belief_estimator/estimator.pt`.

### New file: `abides-gym/envs/belief_estimator_wrapper.py`

```python
class BeliefEstimatorWrapper(gym.Wrapper):
    """Replaces oracle b* with MLP estimate ĥ. Obs: 7-dim raw → 9-dim with ĥ appended."""
    def __init__(self, env, model_path, window=20):
        super().__init__(env)
        self.model = load_estimator(model_path)
        self.window = window
        self._obs_buffer = deque(maxlen=window)
        low  = np.concatenate([env.observation_space.low,  [-np.inf, -np.inf]])
        high = np.concatenate([env.observation_space.high, [ np.inf,  np.inf]])
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.env.previous_marked_to_market = self.env.starting_cash  # BUG 1
        self._obs_buffer.clear()
        self._obs_buffer.append(obs.flatten()[:7])
        h_hat = self._estimate()
        return np.concatenate([obs.flatten()[:7], h_hat])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._obs_buffer.append(obs.flatten()[:7])
        h_hat = self._estimate()
        return np.concatenate([obs.flatten()[:7], h_hat]), reward, done, info

    def _estimate(self):
        buf = np.array(self._obs_buffer)   # (≤W, 7)
        if len(buf) < self.window:
            buf = np.pad(buf, ((self.window - len(buf), 0), (0, 0)))
        return self.model.predict_proba(buf.flatten())  # (2,) float32
```

### New file: `abides-gym/scripts/train_ppo_inferred_belief.py`

Clone of `train_ppo_oracle_obs.py` with `BeliefEstimatorWrapper` instead of `RegimeAdapter`. No regime oracle at all — env uses default `num_momentum_agents=12` (normal rmsc04).

Output dir: `results/e3_inferred_belief/seed_{seed}/`

---

## Results Directory Structure

```
results/
├── ppo_baseline/           # v4 — already done
│   ├── seed_0/eval_metrics.json
│   └── seed_1/eval_metrics.json
├── e0_oracle_heuristic/
│   ├── eval_metrics.json   # aggregate across seeds and regimes
│   └── trajectories/       # obs sequences for E3 training data (if --save-trajectories)
├── e1_belief_obs/
│   ├── seed_0/eval_metrics.json
│   ├── seed_1/eval_metrics.json
│   └── seed_2/eval_metrics.json
├── e2_belief_reward/
│   ├── seed_0/eval_metrics.json
│   ├── seed_1/eval_metrics.json
│   └── seed_2/eval_metrics.json
└── e3_inferred_belief/
    ├── estimator.pt
    ├── seed_0/eval_metrics.json
    ├── seed_1/eval_metrics.json
    └── seed_2/eval_metrics.json
```

---

## New Files Summary

| File | Purpose |
|------|---------|
| `abides-gym/scripts/run_oracle_heuristic.py` | E0: heuristic benchmark under oracle regime |
| `abides-gym/envs/regime_adapter.py` | E1/E2: appends b* to obs, samples regime per episode |
| `abides-gym/scripts/train_ppo_oracle_obs.py` | E1: PPO training with b* in obs |
| `abides-gym/envs/regime_reward_wrapper.py` | E2: reward shaping with alignment bonus |
| `abides-gym/scripts/train_ppo_oracle_reward.py` | E2: PPO training with shaped reward |
| `abides-gym/scripts/belief_estimator_train.py` | E3: train sliding-window MLP on regime-labelled data |
| `abides-gym/envs/belief_estimator_wrapper.py` | E3: replace oracle with estimated ĥ |
| `abides-gym/scripts/train_ppo_inferred_belief.py` | E3: PPO training with inferred belief |

## Modified Files

| File | Change |
|------|--------|
| `abides-gym/envs/__init__.py` | Export `RegimeAdapter`, `RegimeRewardWrapper`, `BeliefEstimatorWrapper` |
| `abides-gym/envs/gymnasium_daily_investor_env.py` | Add `self.env_config = env_config` to `__init__` if not present, so `RegimeAdapter` can patch it per episode |

---

## Execution Order

```
E0  →  (always run regardless of outcome, establishes oracle ceiling)
E1  →  (run E1 and E2 in parallel if compute allows)
E2  →  (run in parallel with E1)
       ↓
       Gate check: does E1 or E2 beat v4? (mean > 32,854, Sharpe > 2.507, win ≥ 100%)
       YES → E3
       NO  → diagnose (check learning curves, α sweep for E2, regime balance)
```

---

## Metric Reporting Format

Each `eval_metrics.json`:
```json
{
  "mean_pnl": float,
  "std_pnl": float,
  "sharpe": float,
  "win_rate": float,
  "max_drawdown": float,
  "n_episodes": int,
  "seeds": [int],
  "protocol": {"timestep_duration": "60s", "num_regimes": 2}
}
```

Sharpe = mean_pnl / std_pnl (same convention as v4 baseline).
