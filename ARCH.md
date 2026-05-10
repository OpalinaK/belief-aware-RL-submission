# Architecture

## Component Responsibilities

```
abides-core
  Kernel          — discrete-event simulation engine; routes timestamped messages
  Agent           — base class; agents wake on messages, send messages back
  LatencyModel    — per-agent-pair message latency (used for realism, not critical for RL)

abides-markets
  ExchangeAgent   — order book matching, publishes L1/L2 market data
  ValueAgent      — Bayesian OU-belief agent; trades when mid deviates from posterior
  NoiseAgent      — one-shot random direction trader; fires once per simulation
  MomentumAgent   — MA20/MA50 crossover; wakes on Poisson schedule
  AdaptiveMarketMakerAgent — POV ladder MM; reprices on fixed schedule
  SparseMeanRevertingOracle — OU fundamental process; queried by value agents
  OrderSizeModel  — mixture sampler for background agent order sizes (numpy, no pomegranate)
  rmsc04          — background config builder; wires all agents together

abides-gym
  AbidesGymCoreEnv         — gym.Env base; rebuilds kernel on reset()
  AbidesGymMarketsEnv      — adds financial agent (FinancialGymAgent) and market data buffer
  SubGymMarketsDailyInvestorEnv_v0 — concrete env: action=BUY/HOLD/SELL, obs=7-vector
  FinancialGymAgent        — injected ABIDES agent that bridges simulation → gym step loop

planned PPO baseline
  train_ppo_daily_investor.py — RLlib PPO trainer/evaluator for daily_investor
  ResetMarkedToMarketWrapper — resets previous_marked_to_market after every env.reset()
  GymnasiumDailyInvestorAdapter — gym->gymnasium API bridge + obs flattening
  eval loop                  — RLModule-based deterministic held-out rollouts, writes JSON metrics
  speed profile path         — timed smoke runs and throughput fields in timing.json
```

## Data Flow

```
gym.make() / env.reset()
  └─ builds rmsc04 background config (all background agents)
  └─ injects FinancialGymAgent into simulation
  └─ runs Kernel until FinancialGymAgent's first wakeup (first_interval)
  └─ returns initial obs

env.step(action)
  └─ FinancialGymAgent translates action → ABIDES order
  └─ Kernel advances simulation until next FinancialGymAgent wakeup
  └─ raw_state collected (order book snapshot, holdings, cash)
  └─ raw_state_to_state()   → obs (7×1 array)
  └─ raw_state_to_reward()  → scalar reward
  └─ raw_state_to_done()    → bool
  └─ raw_state_to_info()    → dict (debug_mode=True for M2M, spread, etc.)
```

## Planned PPO Training Flow

```
train_ppo_daily_investor.py
  └─ creates GymnasiumDailyInvestorAdapter
      └─ gym.make("markets-daily_investor-v0", rmsc04 defaults)
      └─ wraps env with ResetMarkedToMarketWrapper
      └─ flattens obs (7,1)->(7,) and sanitizes info keys
  └─ trains PPO on standard observation only
  └─ saves checkpoints + train_metrics.json + timing.json
  └─ evaluates checkpoint with RLModule inference on held-out seeds
  └─ writes eval_episodes.json + eval_metrics.json
```

## Observation Vector (daily_investor-v0, state_history_length=4)

```
obs[0] — holdings (shares held, signed)
obs[1] — order book imbalance (bid_vol / (bid_vol + ask_vol), depth=3)
obs[2] — spread (best_ask - best_bid, cents)
obs[3] — direction feature (mid_price - last_transaction, cents)
obs[4] — return t-2 → t-1
obs[5] — return t-1 → t
obs[6] — return t   (most recent)
shape: (7, 1) float32
```

## Reward

Dense: `(M2M_t - M2M_{t-1}) / order_fixed_size / num_steps_per_episode`
Sparse: `(M2M_final - starting_cash) / order_fixed_size / num_steps_per_episode`

## Key Design Decisions

- **Full kernel rebuild on reset:** ensures clean seed isolation between episodes
- **Two MM agents:** rmsc04 uses 2 AdaptiveMarketMakerAgents with identical params for thicker book
- **No POVExecutionAgent:** rmsc04 (unlike rmsc03) has no large directional buyer; market is symmetric
- **BUG 1:** `previous_marked_to_market` survives across resets — callers must reset it manually
- **Baseline PPO first:** v1 trains a belief-blind PPO policy before oracle or learned-belief variants
- **Sweep separation:** targeted fast sweep results are used for regime selection and should not be duplicated by PPO training
- **Speed plan:** optimize env-side throughput first (debug/info path, worker reliability, profiling), then scale timesteps

---

## plan-v2 Extensions (Belief-Aware RL)

### New Components

```
abides-gym/envs/regime_adapter.py
  RegimeAdapter (gymnasium.Env)
    — standalone env (not a wrapper); holds TWO SubGymMarketsDailyInvestorEnv_v0 instances
    — one per regime (val: n_mom=0, mom: n_mom=12)
    — on reset(): samples regime, seeds+resets the matching inner env
    — appends b* = (1,0) or (0,1) to flattened obs → 9-dim obs
    — RLlib sees: RegimeAdapter as env class, env_config includes regime_mode

abides-gym/envs/regime_reward_wrapper.py
  RegimeRewardWrapper (gymnasium.Wrapper)
    — wraps RegimeAdapter; adds α*align(action, regime) to reward during training
    — stripped at eval time (eval uses RegimeAdapter only)
    — align: Val → BUY (action=0); Mom → follow return sign (BUY=0 if ret>0, SELL=2 if ret<0)
    — α = 0.001 default; swept over {1e-4, 1e-3, 1e-2} in E2a before full E2

abides-gym/envs/belief_estimator_wrapper.py (E3, gated)
  BeliefEstimatorWrapper (gymnasium.Env)
    — no oracle; uses trained MLP to estimate regime from sliding window of obs
    — appends ĥ (2-dim softmax) instead of b* → same 9-dim obs shape as E1/E2

Scripts:
  run_oracle_heuristic.py   — E0: no training, oracle regime → best heuristic
  sweep_alpha_e2.py         — E2a: short PPO runs per α value; outputs alpha_sweep.json
  train_ppo_oracle_obs.py   — E1: PPO on 9-dim obs with oracle b*
  train_ppo_oracle_reward.py— E2: PPO on 9-dim obs with α-shaped reward during training
  belief_estimator_train.py — E3: trains MLP regime classifier on labelled trajectories
  train_ppo_inferred_belief.py — E3: PPO with BeliefEstimatorWrapper
```

### plan-v2 Training Data Flow (E1 example)

```
RegimeAdapter.reset()
  └─ sample regime ∈ {val, mom}
  └─ seed + reset matching SubGymMarketsDailyInvestorEnv_v0
  └─ flatten obs (7,1)→(7,) + append b* (2,) → obs (9,)

RegimeAdapter.step(action)
  └─ delegate to current inner env
  └─ flatten obs + append b* → obs (9,)

PPO trains on 9-dim obs; at eval, same RegimeAdapter with regime_mode="random"
```

### Regime Definition

- ρ = num_momentum_agents / num_value_agents
- ρ* = 0.06 (threshold derived from sweep: VALUE wins at ρ=0, MOMENTUM wins at ρ=0.118)
- Val: ρ < ρ*, b* = (1, 0)
- Mom: ρ ≥ ρ*, b* = (0, 1)

### Success Gate

E3 is only run if E1 or E2 beats v4 on ALL three: mean P&L > 32,854, Sharpe > 2.507, win ≥ 100%.
