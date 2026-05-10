# abides-gym/scripts — Belief-Aware RL Scripts

All scripts must be run from the repository root with the project virtual environment:

```bash
# Always prefix with:
.venv/bin/python abides-gym/scripts/<script>.py
```

---

## File Overview

| File | Purpose |
|---|---|
| `belief_tracker.py` | Kalman filter estimator for the OU fundamental (original) |
| `particle_filter_tracker.py` | Bootstrap SIR particle filter; drop-in replacement for Kalman |
| `kim_filter_tracker.py` | Kim (1994) regime-switching Kalman filter for K=3 market regimes |
| `lstm_belief_tracker.py` | LSTM-based fundamental tracker (experimental) |
| `belief_comparison.py` | Benchmarks all fundamental trackers vs oracle |
| `regime_classifier.py` | Trains MLP/LSTM classifiers to output (p_val, p_mom) per step |
| `regime_belief_wrapper.py` | Gym wrappers that augment observations with regime belief |
| `test.py` | Baseline heuristic strategies benchmark (6 strategies × N episodes) |
| `sweep.py` | Parameter sweep over background config |
| `train_ppo_daily_investor.py` | PPO training entry point (RLlib) |
| `baselines.py` | Heuristic strategy implementations (Hold, MR, Momentum, etc.) |

---

## Quick Smoke Tests

```bash
# Verify base gym environment
.venv/bin/python -c "
import gym, abides_gym
env = gym.make('markets-daily_investor-v0', background_config='rmsc04')
env.seed(1); env.reset(); env.close(); print('ok')
"

# Verify regime belief wrapper (oracle + trained modes)
.venv/bin/python abides-gym/scripts/regime_belief_wrapper.py \
    --trained-model abides-gym/scripts/models/mlp_kalman_regime.pt
```

---

## Regime Classifier

Trains classifiers to produce per-step regime probabilities `(p_val, p_mom)` from observable
market features, replacing the oracle regime label in PPO training.

### Train and Save

```bash
# Full training run (recommended: 60 train + 20 eval episodes per regime)
.venv/bin/python abides-gym/scripts/regime_classifier.py \
    --train-eps 60 --eval-eps 20 --n-epochs 40 \
    --save-dir abides-gym/scripts/models

# Quick test run (no save)
.venv/bin/python abides-gym/scripts/regime_classifier.py \
    --train-eps 20 --eval-eps 10 --n-epochs 40 --no-save
```

### CLI Arguments

| Flag | Default | Description |
|---|---|---|
| `--train-eps N` | 60 | Episodes per regime for training |
| `--eval-eps N` | 20 | Episodes per regime for evaluation |
| `--seq-len N` | 20 | Sliding window length fed to the model |
| `--n-epochs N` | 40 | Training epochs |
| `--hidden N` | 32 | LSTM hidden size (MLP uses 4× this) |
| `--save-dir PATH` | `models` | Directory to save `.pt` checkpoints |
| `--no-save` | — | Skip saving (dry run) |
| `--no-kalman` | — | Skip Kalman-augmented feature variants |

### What Gets Trained

Four variants are trained in sequence:

| Model | Features | Overall acc | p_val\|val | p_mom\|mom |
|---|---|---|---|---|
| LSTM (raw) | 6 market features | 70% | 0.50 | 0.50 |
| MLP (raw) | 6 market features | **100%** | 0.56 | 0.58 |
| LSTM (+Kalman) | 8 features (+ belief_dev, belief_std) | 70% | 0.50 | 0.50 |
| MLP (+Kalman) | 8 features (+ belief_dev, belief_std) | **100%** | 0.59 | 0.61 |

**Recommended model: `mlp_kalman_regime.pt`** — highest confidence margins.

### Input Features (per timestep)

| Index | Name | Source | Notes |
|---|---|---|---|
| 0 | `imbalance` | `obs[1]` | Bid vol / total vol; elevated in Mom |
| 1 | `spread` | `obs[2] / r_bar` | Bid-ask spread, normalised |
| 2 | `direction` | `obs[3] / r_bar` | Mid − last transaction |
| 3 | `ret_lag1` | `obs[4] / r_bar` | Most-recent 60s return |
| 4 | `ret_lag2` | `obs[5] / r_bar` | 2-step lag |
| 5 | `ret_lag3` | `obs[6] / r_bar` | 3-step lag |
| 6 | `belief_dev` | Kalman tracker | `(mu_t − mid_t) / r_bar` (+Kalman only) |
| 7 | `belief_std` | Kalman tracker | `sigma_t / r_bar` (+Kalman only) |

### Saved Checkpoint Format

Each `.pt` file is a dict:
```python
{
    "state_dict":  OrderedDict,   # model weights
    "model_type":  "mlp" | "lstm",
    "use_kalman":  bool,
    "seq_len":     int,           # window length (must match at inference)
    "n_features":  int,           # 6 (raw) or 8 (+Kalman)
    "hidden_size": int,
}
```

Load with:
```python
from regime_classifier import load_classifier
model, model_type, use_kalman, seq_len = load_classifier("models/mlp_kalman_regime.pt")
```

---

## Regime Belief Wrapper

Gym/Gymnasium wrappers that augment the 7-dim daily-investor observation with
2 belief features `(p_val, p_mom)`, producing a 9-dim observation for PPO.

### Belief Modes

| Mode | Description | Oracle? | Per-step? |
|---|---|---|---|
| `"none"` | No augmentation; obs shape (7,) | — | — |
| `"oracle"` | Hard one-hot b* = (1,0) or (0,1); constant per episode | Yes | No |
| `"trained"` | Soft MLP classifier output; updates every step | No | Yes |
| `"kim"` | Kim (1994) filter output; **does not distinguish Val/Mom** | No | Yes |

### Usage with RegimeRandomizedGymnasiumEnv

```python
from regime_belief_wrapper import RegimeRandomizedGymnasiumEnv

# E1 — oracle belief (upper bound)
env = RegimeRandomizedGymnasiumEnv({
    "belief_mode": "oracle",
    "val_prob":    0.5,          # 50% Val / 50% Mom per episode
})

# E3 — learned belief (trained MLP+Kalman)
env = RegimeRandomizedGymnasiumEnv({
    "belief_mode":        "trained",
    "trained_model_path": "abides-gym/scripts/models/mlp_kalman_regime.pt",
    "val_prob":           0.5,
})

obs, info = env.reset()       # obs shape (9,)
obs, r, term, trunc, info = env.step(action)

# info keys: "regime" ("val"/"mom"), "belief" (np.ndarray shape (2,))
```

### Usage with RLlib PPO

Pass `RegimeRandomizedGymnasiumEnv` as the environment class and `env_config` as below:

```python
from ray.rllib.algorithms.ppo import PPOConfig
from regime_belief_wrapper import RegimeRandomizedGymnasiumEnv

config = (
    PPOConfig()
    .environment(
        env=RegimeRandomizedGymnasiumEnv,
        env_config={
            "belief_mode":        "trained",
            "trained_model_path": "abides-gym/scripts/models/mlp_kalman_regime.pt",
            "val_prob":           0.5,
        },
    )
    ...
)
```

### Smoke Test

```bash
# Tests none + oracle + trained modes (3 short rollouts)
.venv/bin/python abides-gym/scripts/regime_belief_wrapper.py \
    --trained-model abides-gym/scripts/models/mlp_kalman_regime.pt
```

---

## Baseline Benchmarks

```bash
# 6 strategies × 20 episodes on rmsc04
.venv/bin/python abides-gym/scripts/test.py --episodes 20

# Custom background config
.venv/bin/python abides-gym/scripts/test.py --episodes 20 \
    --config rmsc04 --num-momentum 0
```

---

## Known Issues

**BUG 1 (CRITICAL):** `env.previous_marked_to_market` is not reset by `env.reset()`.
Always add after every reset:
```python
obs = env.reset()
env.previous_marked_to_market = env.starting_cash
```
All wrappers in `regime_belief_wrapper.py` handle this automatically.

**Kim filter non-functional for regime detection:** `KimRegimeBeliefWrapper` and
`belief_mode="kim"` produce near-identical output in Val and Mom regimes. The Kim filter
detects OU dynamics (present in both regimes), not agent-count structure. Use
`belief_mode="trained"` instead.

---

## Saved Models

```
abides-gym/scripts/models/
├── mlp_kalman_regime.pt    # MLP + Kalman features — RECOMMENDED (100% acc, p≈0.60)
├── mlp_raw_regime.pt       # MLP, market features only (100% acc, p≈0.57)
├── lstm_kalman_regime.pt   # LSTM + Kalman — fails on small data (70% acc)
└── lstm_raw_regime.pt      # LSTM, raw only — fails on small data (45% acc)
```

Retrain with more episodes (`--train-eps 60`) for tighter confidence margins before PPO.
