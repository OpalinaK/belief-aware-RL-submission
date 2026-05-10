#!/usr/bin/env bash
# Plan v1 baseline PPO: three independent training seeds + held-out eval (seeds 100–119).
# Usage:
#   chmod +x abides-gym/scripts/run_ppo_baseline_plan_v1.sh
#   ./abides-gym/scripts/run_ppo_baseline_plan_v1.sh
#
# Override defaults with env vars, e.g.:
#   TIMESTEPS=50000 NUM_WORKERS=16 PARALLEL=0 ./abides-gym/scripts/run_ppo_baseline_plan_v1.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

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
NUM_WORKERS="${NUM_WORKERS:-12}"
# 1 = three jobs at once (CUDA_VISIBLE_DEVICES 0,1,2); 0 = one after another
PARALLEL="${PARALLEL:-1}"
OUT="${OUT:-${ROOT}/results/ppo_baseline}"
# Per-GPU learner; env sim stays CPU-heavy. Use 0 if CUDA/driver issues.
NUM_GPUS="${NUM_GPUS:-1}"

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
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${common_args[@]}" \
    --seed "${seed}" \
    --out-dir "${OUT}/seed_${seed}" \
    2>&1 | tee "${log}"
}

if [[ "${PARALLEL}" == "1" ]]; then
  echo "Starting three runs in parallel (GPUs 0,1,2). Set PARALLEL=0 for sequential."
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

echo "Done. Outputs:"
echo "  ${OUT}/seed_{0,1,2}/"
echo "  ${OUT}/summary.json"
