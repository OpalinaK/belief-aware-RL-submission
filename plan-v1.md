# Plan v1: Baseline PPO Agent on ABIDES Daily Investor

## Goal

Train a baseline PPO policy on the ABIDES `markets-daily_investor-v0` environment using only the
standard public market observation `o_t`. This is the control policy for later oracle and
belief-aware variants.

The baseline policy should answer:

> How well can PPO trade in the ABIDES daily-investor setting without oracle regime information or
> learned belief features?

## Scope

This plan implements only the baseline RL agent:

- PPO policy trained on the standard daily-investor observation.
- No oracle belief vector.
- No learned belief module.
- No reward shaping.
- Evaluation on held-out seeds with the same metrics used by the existing baseline and sweep scripts.

The next experiment after this should be PPO with oracle regime information as an observation.

## Existing Environment

Use the existing Gym environment:

```text
markets-daily_investor-v0
```

Default observation with `state_history_length = 4`:

```text
obs[0] = holdings
obs[1] = order book imbalance
obs[2] = spread
obs[3] = direction feature
obs[4:7] = lagged mid-price returns
```

Action space:

```text
0 = market buy
1 = hold
2 = market sell
```

Reward:

```text
dense reward = normalized change in mark-to-market wealth
```

Important implementation invariant:

```python
obs = env.reset()
env.previous_marked_to_market = env.starting_cash
```

The reset fix is required because the environment does not reset `previous_marked_to_market`
internally.

## Implementation Steps

### 1. Add PPO Training Script

Create:

```text
abides-gym/scripts/train_ppo_daily_investor.py
```

The script should:

- Register/import `abides_gym`.
- Build `markets-daily_investor-v0` with the RMSC04 background config.
- Use the same environment defaults as `abides-gym/scripts/sweep.py`.
- Train PPO with the standard observation only.
- Save config, checkpoints, training metrics, and evaluation metrics.

Initial environment defaults:

```python
ENV_DEFAULTS = dict(
    background_config="rmsc04",
    mkt_close="16:00:00",
    timestep_duration="300s",  # start fast; switch to 60s after smoke tests
    starting_cash=1_000_000,
    order_fixed_size=10,
    state_history_length=4,
    market_data_buffer_length=5,
    first_interval="00:05:00",
    reward_mode="dense",
    done_ratio=0.3,
    debug_mode=True,
)
```

Use the existing RMSC04 background parameters from `sweep.py`.

### 2. Use RLlib PPO First

Start with RLlib PPO because the repository already contains RLlib-style runners.

Ray/RLlib has significant API differences in current versions (for example, Ray 2.55). The baseline
implementation should use the current `AlgorithmConfig` method names and the new RLModule inference
path in evaluation.

Initial PPO config:

```python
config = {
    "env": "markets-daily_investor-v0",
    "env_config": env_config,
    "framework": "torch",
    "num_env_runners": 0,
    "num_gpus": 1,
    "gamma": 1.0,
    "lr": 1e-4,
    "train_batch_size": 2000,
    "minibatch_size": 128,
    "num_epochs": 5,
    "entropy_coeff": 0.01,
    "env_runners": {
        "observation_filter": "MeanStdFilter",
    },
    "model_config": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "tanh",
    },
}
```

Keep the first run small and boring. The goal is to make the pipeline reliable before tuning.

### 3. Add A Reset-Safe Env Wrapper

RLlib calls `env.reset()` internally, so the manual reset fix must live inside a wrapper rather than
only inside hand-written rollout code.

Add a wrapper in the training script:

```python
class ResetMarkedToMarketWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.env.previous_marked_to_market = self.env.starting_cash
        return obs
```

Then register RLlib's env factory to return the wrapped environment.

For Ray 2.55 compatibility with this codebase, the env adapter should also:

- Expose Gymnasium-compatible `reset`/`step` signatures.
- Flatten observations from `(7, 1)` to `(7,)` so the default PPO encoder can be built.
- Sanitize `info` keys to strings (the native env uses nested integer keys under order book levels,
  which can break Gymnasium vector wrappers).

### 4. Timed Smoke Test Before Full Training

Run one short timed training job:

```bash
time \
.venv/bin/python abides-gym/scripts/train_ppo_daily_investor.py \
  --timesteps 10000 \
  --timestep 300s \
  --seed 0 \
  --out-dir results/ppo_baseline_smoke
```

The smoke test is not just a correctness check. It should measure wall-clock throughput so we can
decide what is feasible in the remaining project time.

Success criteria:

- The env resets and steps through RLlib without crashing.
- PPO produces at least one checkpoint.
- Evaluation produces non-empty metrics.
- Action distribution is not degenerate unless the policy has learned to mostly hold.
- The script reports total wall time, environment steps per second, and training iterations per hour.

The training script should write a timing file:

```text
results/ppo_baseline_smoke/timing.json
```

with:

```json
{
  "timesteps": 10000,
  "timestep_duration": "300s",
  "wall_time_sec": 0,
  "env_steps_per_sec": 0,
  "train_iterations": 0,
  "train_iterations_per_hour": 0,
  "eval_wall_time_sec": 0
}
```

Use this timing result to choose the full run:

- If the `300s` smoke run is fast and stable, try a second timed smoke with `60s`.
- If the `60s` smoke run projects to less than a day for three seeds, use `60s` for the final result.
- If the `60s` projection is too slow, keep `300s` for the main baseline and clearly label it as the
  fast training setting.
- If PPO setup or RLlib overhead dominates, reduce the target timesteps before changing the experiment
  definition.

### 5. Full Baseline Runs

Use the three Titan GPUs for independent seeds, not one large distributed run:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python abides-gym/scripts/train_ppo_daily_investor.py \
  --timesteps 200000 --timestep 60s --seed 0 --out-dir results/ppo_baseline/seed_0

CUDA_VISIBLE_DEVICES=1 .venv/bin/python abides-gym/scripts/train_ppo_daily_investor.py \
  --timesteps 200000 --timestep 60s --seed 1 --out-dir results/ppo_baseline/seed_1

CUDA_VISIBLE_DEVICES=2 .venv/bin/python abides-gym/scripts/train_ppo_daily_investor.py \
  --timesteps 200000 --timestep 60s --seed 2 --out-dir results/ppo_baseline/seed_2
```

If runtime is too high, keep `300s` timestep for the first defensible result and clearly label it as
the fast training setting.

### 6. Evaluation

After each checkpoint, evaluate on held-out seeds that were not used for training.

Evaluation should be a deterministic rollout loop. With current Ray versions, avoid
`compute_single_action` and use RLModule inference:

```python
def evaluate(algo, env, seeds):
    module = algo.get_module()
    episodes = []

    for seed in seeds:
        env.seed(seed)
        obs = env.reset()
        env.previous_marked_to_market = env.starting_cash

        done = False
        total_reward = 0.0
        actions = []
        m2m_trace = []
        spread_trace = []

        while not done:
            batch = {Columns.OBS: torch.as_tensor(obs[None, :], dtype=torch.float32)}
            out = module.forward_inference(batch)
            logits = out.get(Columns.ACTION_DIST_INPUTS) or out["action_dist_inputs"]
            action = int(np.argmax(np.asarray(logits)[0]))  # deterministic policy action
            obs, reward, done, info = env.step(action)

            total_reward += reward
            actions.append(int(action))
            m2m_trace.append(float(info["marked_to_market"]))
            spread_trace.append(float(info.get("spread", 0.0)))

        final_m2m = m2m_trace[-1] if m2m_trace else float(env.starting_cash)
        final_pnl = final_m2m - env.starting_cash

        episodes.append({
            "seed": seed,
            "total_reward": float(total_reward),
            "final_m2m": float(final_m2m),
            "final_pnl": float(final_pnl),
            "episode_length": len(actions),
            "won": bool(final_pnl >= 0),
            "action_counts": {
                "BUY": actions.count(0),
                "HOLD": actions.count(1),
                "SELL": actions.count(2),
            },
        })

    return summarize_eval(episodes)
```

Evaluation contract:

- Use held-out seeds, for example train with seeds `0`, `1`, `2` and evaluate on seeds `100` through
  `119`.
- Use `explore=False` so PPO acts deterministically.
- Reset `previous_marked_to_market` after every environment reset.
- Save both per-episode rows and summary statistics.
- Compare PPO against the heuristic baselines using the same metric definitions.

Metrics:

- Mean final P&L.
- Sharpe ratio across episodes.
- Win rate.
- Mean max drawdown.
- Mean episode length.
- Action distribution: buy / hold / sell.
- Total reward.

Evaluation output:

```text
results/ppo_baseline/
  seed_0/
    config.json
    train_metrics.json
    eval_metrics.json
    eval_episodes.json
    timing.json
    checkpoints/
  seed_1/
  seed_2/
  summary.json
```

Example `eval_metrics.json` shape:

```json
{
  "n_episodes": 20,
  "final_pnl_cents": {
    "mean": 0,
    "std": 0,
    "min": 0,
    "max": 0
  },
  "sharpe": 0,
  "win_rate": 0,
  "episode_length": {
    "mean": 0
  },
  "action_pct": {
    "BUY": 0,
    "HOLD": 0,
    "SELL": 0
  },
  "eval_wall_time_sec": 0
}
```

### 7. Reporting

The report should describe this as the belief-blind control:

> Baseline PPO is trained on public ABIDES market observations only. It receives holdings,
> order-book imbalance, spread, direction feature, and lagged returns, but no oracle regime label and
> no learned belief vector. Oracle and belief-aware policies are compared against this same training
> and evaluation protocol.

## Next Experiment: Oracle As Observation

After the baseline PPO result exists, implement the first belief-aware stepping stone:

```text
obs'_t = concat(o_t, b_t^{oracle})
```

where `b_t^{oracle}` is a one-hot regime label derived from the background configuration. This is the
cleanest test of whether regime information helps the learned policy before spending time on a
learned belief classifier.

## Stretch Experiment: Oracle Reward Shaping

After oracle-as-observation, test a reward-shaped variant:

```text
r'_t = r_t + lambda * regime_action_bonus(o_t, a_t, b_t^{oracle})
```

This changes the objective, so it should be reported separately from the clean belief-aware
comparison.

## Non-Goal For v1

Do not prioritize SAC in this version. The daily-investor environment has a discrete action space
(`BUY`, `HOLD`, `SELL`), so PPO is a more natural first baseline. If an algorithm comparison is
needed, DQN is lower-risk because the repository already contains RLlib DQN runner examples.
