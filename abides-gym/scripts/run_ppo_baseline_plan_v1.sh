#!/usr/bin/env bash
# Plan v1 baseline PPO: three independent training seeds + held-out eval (seeds 100–119).
# Usage:
#   chmod +x abides-gym/scripts/run_ppo_baseline_plan_v1.sh
#   ./abides-gym/scripts/run_ppo_baseline_plan_v1.sh
#
# Default is sequential seeds (one job at a time): ABIDES envs are CPU-heavy; three parallel
# runs each with many env runners oversubscribes cores and slows everyone.
# Optional: PARALLEL=1 runs three jobs on GPU 0/1/2 at once only if you have spare CPU/RAM.
# Override defaults with env vars, e.g.:
#   TIMESTEPS=50000 NUM_WORKERS=16 ./abides-gym/scripts/run_ppo_baseline_plan_v1.sh
#   PARALLEL=1 NUM_WORKERS=8 ./abides-gym/scripts/run_ppo_baseline_plan_v1.sh
#
# Writes under /tmp by default (NFS $HOME quota); on success, rsyncs into the repo.
#   OUT=/path/to/staging FINAL_OUT=/path/in/repo ./abides-gym/scripts/run_ppo_baseline_plan_v1.sh
#   OUT="${ROOT}/results/ppo_baseline"  # disable staging: same path as FINAL_OUT, no sync
#
set -euo pipefail

# Line-buffered Python stdout/stderr (helps tee and log files).
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

# Keep Ray/Python temps off $HOME until quota is raised.
export TMPDIR="${TMPDIR:-/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${USER}}"
mkdir -p "${RAY_TMPDIR}"

PYTHON="${ROOT}/.venv/bin/python"
TRAIN="${ROOT}/abides-gym/scripts/train_ppo_daily_investor.py"

if [[ ! -f "${PYTHON}" ]]; then
  echo "error: expected venv interpreter at ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN}" ]]; then
  echo "error: missing ${TRAIN}" >&2
  exit 1
fi

# --- plan v1 knobs (override with env) ---
TIMESTEPS="${TIMESTEPS:-200000}"
TIMESTEP="${TIMESTEP:-60s}"
TRAIN_BATCH="${TRAIN_BATCH_SIZE:-2000}"
MINIBATCH="${MINIBATCH_SIZE:-128}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
LR="${LR:-1e-4}"
ENTROPY="${ENTROPY_COEFF:-0.01}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-50}"
NUM_WORKERS="${NUM_WORKERS:-16}"
# 0 = one seed after another (recommended for CPU-bound ABIDES sims).
# 1 = three jobs at once on CUDA_VISIBLE_DEVICES 0,1,2 (can thrash CPU; lower NUM_WORKERS if used).
PARALLEL="${PARALLEL:-0}"
FINAL_OUT="${FINAL_OUT:-${ROOT}/results/ppo_baseline}"
OUT="${OUT:-/tmp/belief-aware-ppo-baseline-$(id -un)}"
# Per-GPU learner; env sim stays CPU-heavy. Use 0 if CUDA/driver issues.
NUM_GPUS="${NUM_GPUS:-1}"

echo "Staging run under: ${OUT}"
echo "Will sync to repo at: ${FINAL_OUT} (after all seeds + summary succeed)"

mkdir -p "${OUT}/seed_0" "${OUT}/seed_1" "${OUT}/seed_2"

# Held-out eval seeds 100–119 (plan v1 §6)
mapfile -t EVAL_SEEDS < <(seq 100 119)

common_args=(
  "${TRAIN}"
  --timesteps "${TIMESTEPS}"
  --timestep "${TIMESTEP}"
  --train-batch-size "${TRAIN_BATCH}"
  --minibatch-size "${MINIBATCH}"
  --num-epochs "${NUM_EPOCHS}"
  --lr "${LR}"
  --entropy-coeff "${ENTROPY}"
  --checkpoint-freq "${CHECKPOINT_FREQ}"
  --num-workers "${NUM_WORKERS}"
  --num-gpus "${NUM_GPUS}"
  --train-debug-mode false
  --eval-debug-mode true
  --train-info-mode minimal
  --eval-info-mode full
  --eval-seeds "${EVAL_SEEDS[@]}"
)

run_one() {
  local seed="$1"
  local gpu="$2"
  local log="${OUT}/seed_${seed}/train.log"
  echo "=== seed=${seed} CUDA_VISIBLE_DEVICES=${gpu} log=${log} ==="
  mkdir -p "${OUT}/seed_${seed}"
  if command -v stdbuf >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="${gpu}" stdbuf -oL -eL "${PYTHON}" "${common_args[@]}" \
      --seed "${seed}" \
      --out-dir "${OUT}/seed_${seed}" \
      2>&1 | stdbuf -oL tee "${log}"
  else
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${common_args[@]}" \
      --seed "${seed}" \
      --out-dir "${OUT}/seed_${seed}" \
      2>&1 | tee "${log}"
  fi
}

if [[ "${PARALLEL}" == "1" ]]; then
  echo "Starting three runs in parallel (GPUs 0,1,2). Expect CPU contention; consider NUM_WORKERS=8 or less per job."
  run_one 0 0 &
  pid0=$!
  run_one 1 1 &
  pid1=$!
  run_one 2 2 &
  pid2=$!
  ec=0
  wait "${pid0}" || ec=1
  wait "${pid1}" || ec=1
  wait "${pid2}" || ec=1
  if [[ "${ec}" -ne 0 ]]; then
    echo "error: one or more training jobs failed (see logs under ${OUT}/seed_*/train.log)" >&2
    exit 1
  fi
else
  echo "Running seeds 0,1,2 sequentially on GPU 0."
  run_one 0 0
  run_one 1 0
  run_one 2 0
fi

echo "Aggregating summary -> ${OUT}/summary.json"
"${PYTHON}" - "${OUT}" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for seed in (0, 1, 2):
    d = out / f"seed_{seed}"
    em = d / "eval_metrics.json"
    tm = d / "timing.json"
    row = {"train_seed": seed, "dir": str(d)}
    if em.is_file():
        row["eval_metrics"] = json.loads(em.read_text())
    if tm.is_file():
        row["timing"] = json.loads(tm.read_text())
    rows.append(row)

summary = {"runs": rows}
if all("eval_metrics" in r for r in rows):
    pnls = [r["eval_metrics"]["final_pnl_cents"]["mean"] for r in rows]
    sharpes = [r["eval_metrics"].get("sharpe", 0.0) for r in rows]
    wins = [r["eval_metrics"].get("win_rate", 0.0) for r in rows]
    summary["aggregate"] = {
        "mean_of_seed_means_final_pnl_cents": sum(pnls) / len(pnls),
        "mean_sharpe_across_seeds": sum(sharpes) / len(sharpes),
        "mean_win_rate_across_seeds": sum(wins) / len(wins),
    }

out.joinpath("summary.json").write_text(json.dumps(summary, indent=2))
print("Wrote", out / "summary.json")
PY

# Copy staging tree into the repo (same layout as before); skip if already writing in-repo.
_out_rp="$(realpath -m "${OUT}")"
_final_rp="$(realpath -m "${FINAL_OUT}")"
if [[ "${_out_rp}" != "${_final_rp}" ]]; then
  mkdir -p "$(dirname "${FINAL_OUT}")"
  rsync -a --mkpath "${OUT}/" "${FINAL_OUT}/"
  echo "Synced staging -> ${FINAL_OUT}"
  if [[ "${OUT}" == /tmp/belief-aware-ppo-baseline-* ]]; then
    rm -rf "${OUT}"
    echo "Removed staging dir ${OUT}"
  fi
else
  echo "OUT and FINAL_OUT resolve to the same path; no rsync."
fi

echo "Done. Outputs:"
echo "  ${FINAL_OUT}/seed_{0,1,2}/"
echo "  ${FINAL_OUT}/summary.json"
