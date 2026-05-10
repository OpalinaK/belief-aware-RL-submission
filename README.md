# Belief-aware RL with ABIDES-Gym

PPO experiments on an ABIDES daily-investor market env: oracle regime belief (observations or reward shaping) vs baselines. **Write-up:** see `docs/results/` (figures and LaTeX).

---

## Requirements

- **Python 3.10**  
- **`uv`** (recommended) or `pip`  
- Disk: RLlib checkpoints/logs can be large (`./results/`).

---

## One-time setup

From the repo root:

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python \
  "gym==0.25.2" "numpy<2.0" pandas scipy tqdm matplotlib
uv pip install --python .venv/bin/python -e abides-core -e abides-markets -e abides-gym
```

**Training scripts** (`train_ppo_*.py`) need RLlib + PyTorch. Install builds that match your machine (CPU is fine for small runs):

```bash
uv pip install --python .venv/bin/python "ray[rllib]" torch
```

Do **not** use the repo’s historical `requirements.txt` wholesale; pins there conflict with the setup above (`CLAUDE.md`).

---

## Check that the env loads

```bash
.venv/bin/python -c "import gym, abides_gym; env = gym.make('markets-daily_investor-v0', background_config='rmsc04'); env.seed(1); env.reset(); env.close(); print('ok')"
```

---

## What are the experiments?

In plain words: **an experiment here is “run one script once, save metrics and checkpoints under `results/`.”** Each run answers one concrete question inside the same simulated market (ABIDES “daily investor” Gym env).

Rough map (names match the paper/pipeline shorthand):

| Name | Idea | Typical script |
|------|------|----------------|
| **E0** | Run a fixed **oracle/heuristic trader** across seeds and optionally save episodes (reference behavior + data). | `run_oracle_heuristic.py` |
| **Baseline PPO** | Train **generic PPO** on the usual market observations (no special belief setup). | `train_ppo_daily_investor.py` |
| **E1** | Train PPO while **giving the agent the true regime** as extra observation dimensions (“oracle belief in obs”—an upper-bound style setting). | `train_ppo_oracle_obs.py` |
| **E2 / E2a** | Train PPO with **reward shaping** that uses the oracle regime; **E2a** sweeps how strong that shaping is (**α**) before E2. | `sweep_alpha_e2.py`, then `train_ppo_oracle_reward.py` |

So: **E0 ≈ scripted baseline rollout**, **E1/E2 ≈ RL comparisons** where the simulator’s regime is labeled and we either expose it as input (E1) or through the reward (E2).

---

## How we run experiments

Use the **repo root** as the working directory. Prefer **`./run_exp.sh <tag> <script> …`** so big outputs go via `/tmp` and then sync into **`results/<tag>/`**.

### Smallest commands to try first

Sanity check (no learning; environment only):

```bash
.venv/bin/python abides-gym/scripts/test.py --episodes 5
```

E0-style rollout (few episodes; increase for real runs):

```bash
./run_exp.sh e0_oracle_heuristic abides-gym/scripts/run_oracle_heuristic.py \
  --episodes 5 --seeds 0 1 --save-trajectories
```

Baseline PPO (short run; bump `--timesteps` for publication-scale):

```bash
./run_exp.sh ppo_baseline abides-gym/scripts/train_ppo_daily_investor.py \
  --seed 0 --timesteps 50000
```

**E1** — oracle regime in observations:

```bash
./run_exp.sh e1_try abides-gym/scripts/train_ppo_oracle_obs.py \
  --seed 0 --timesteps 50000 --timestep 60s --num-workers 0
```

**E2** — shaped reward (needs a strength **α**; example **0.001**):

```bash
./run_exp.sh e2_try abides-gym/scripts/train_ppo_oracle_reward.py \
  --seed 0 --timesteps 50000 --timestep 60s --num-workers 0 --alpha 0.001
```

**E2a** is just “try several **α** values” via `abides-gym/scripts/sweep_alpha_e2.py` (or `./run_pipeline.sh`, which runs the sweep automatically before E2).

To run **only E1+E2** at publication timesteps **without** resweeping:

```bash
SKIP_E2A=1 FORCE_ALPHA=0.001 ./run_pipeline.sh
```

CLI details and options: **`abides-gym/scripts/README.md`** and **`--help`** on each training script.

### Full pipeline shortcut

**`./run_pipeline.sh`** runs **E2a → E1 (seeds 0,1) → E2 (seeds 0,1)** after optional wait for **E0** (see comments in `run_pipeline.sh`). Environment variables: **`TIMESTEPS`**, **`E2A_TIMESTEPS`**, **`SKIP_E2A`**, **`FORCE_ALPHA`**.

```bash
./run_pipeline.sh
```

---

## Repro / reporting

- Plots and tables for the submission live under **`docs/results/`** (e.g. `v6/`).  
- Raw training outputs land under **`results/`** (tags match `run_exp.sh` / pipeline names).

For **shorter** smoke runs, lower `--timesteps`, `--episodes`, or `TIMESTEPS` before full-length jobs.
