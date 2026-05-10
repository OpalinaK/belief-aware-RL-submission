#!/usr/bin/env bash
# run_exp.sh — run an experiment script via /tmp to avoid home quota, then rsync back.
#
# Usage:
#   ./run_exp.sh <experiment-tag> <python-script> [script-args...]
#
# Example:
#   ./run_exp.sh e0 abides-gym/scripts/run_oracle_heuristic.py --episodes 20 --seeds 0 1 2 --save-trajectories
#   ./run_exp.sh e1_seed0 abides-gym/scripts/train_ppo_oracle_obs.py --seed 0 --timesteps 200000
#   ./run_exp.sh baseline_seed2 abides-gym/scripts/train_ppo_daily_investor.py --seed 2
#
# The script:
#   1. Creates /tmp/belief-rl/<tag>/ as the output directory
#   2. Runs the python script with --out-dir pointing at /tmp
#   3. On exit (success or failure), rsyncs /tmp/belief-rl/<tag>/ to results/<tag>/
#   4. Prints a summary of what was synced

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
TMP_BASE="/tmp/belief-rl"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <experiment-tag> <python-script> [script-args...]"
    exit 1
fi

TAG="$1"; shift
PY_SCRIPT="$1"; shift

TMP_DIR="$TMP_BASE/$TAG"
RESULTS_DIR="$SCRIPT_DIR/results/$TAG"

mkdir -p "$TMP_DIR"
mkdir -p "$RESULTS_DIR"

LOG_FILE="$TMP_DIR/run.log"

echo "[run_exp] tag=$TAG"
echo "[run_exp] tmp=$TMP_DIR"
echo "[run_exp] results=$RESULTS_DIR"
echo "[run_exp] cmd: $PYTHON $PY_SCRIPT --out-dir $TMP_DIR $*"
echo "---"

# Rsync on exit regardless of success/failure.
cleanup() {
    local exit_code=$?
    echo ""
    echo "[run_exp] exit_code=$exit_code — syncing $TMP_DIR -> $RESULTS_DIR"
    rsync -av --progress "$TMP_DIR/" "$RESULTS_DIR/"
    echo "[run_exp] sync complete"
    exit $exit_code
}
trap cleanup EXIT

# Run the experiment. Tee stdout+stderr to both terminal and log file.
cd "$SCRIPT_DIR"
"$PYTHON" "$PY_SCRIPT" --out-dir "$TMP_DIR" "$@" 2>&1 | tee "$LOG_FILE"
