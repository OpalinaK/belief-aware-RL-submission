#!/usr/bin/env bash
# run_pipeline.sh — sequential experiment pipeline: E0 → E2a → E1 → E2
#
# Waits for the already-running E0 process, then runs:
#   E2a  — alpha sweep (50K steps, picks best alpha)
#   E1   — PPO + oracle belief in obs, seeds 0 and 1
#   E2   — PPO + reward shaping, seeds 0 and 1 (alpha from E2a)
#
# All output goes to /tmp/belief-rl/<tag>/ and is rsynced to results/<tag>/ on exit.
# Set E0_PID below if you want to wait for an already-running E0 process.
#
# Usage:
#   ./run_pipeline.sh                        # E0_PID auto-detected from /proc
#   E0_PID=3169530 ./run_pipeline.sh         # explicit PID
#   SKIP_E2A=1 FORCE_ALPHA=0.001 ./run_pipeline.sh  # skip alpha sweep

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
TMP_BASE="/tmp/belief-rl"
RESULTS_BASE="$SCRIPT_DIR/results"

TIMESTEPS="${TIMESTEPS:-200000}"
E2A_TIMESTEPS="${E2A_TIMESTEPS:-50000}"
SKIP_E2A="${SKIP_E2A:-0}"
FORCE_ALPHA="${FORCE_ALPHA:-}"

log() { echo "[pipeline $(date '+%H:%M:%S')] $*" | tee -a "$SCRIPT_DIR/pipeline.log"; }

# ---------------------------------------------------------------------------
# Helper: run one experiment via /tmp, rsync on exit
# ---------------------------------------------------------------------------
run_exp() {
    local tag="$1"; shift
    local script="$1"; shift
    local tmp_dir="$TMP_BASE/$tag"
    local results_dir="$RESULTS_BASE/$tag"
    mkdir -p "$tmp_dir" "$results_dir"

    log "START $tag"
    local t0=$SECONDS

    "$PYTHON" "$script" --out-dir "$tmp_dir" "$@" 2>&1 | tee "$tmp_dir/run.log"
    local rc=${PIPESTATUS[0]}

    log "SYNC $tag -> $results_dir"
    rsync -a "$tmp_dir/" "$results_dir/"

    local elapsed=$(( SECONDS - t0 ))
    if [[ $rc -eq 0 ]]; then
        log "DONE $tag in ${elapsed}s"
    else
        log "FAIL $tag (exit $rc) after ${elapsed}s"
        exit $rc
    fi
}

# ---------------------------------------------------------------------------
# Step 0: wait for E0
# ---------------------------------------------------------------------------
E0_PID="${E0_PID:-}"
if [[ -z "$E0_PID" ]]; then
    # Auto-detect by process name
    E0_PID=$(pgrep -f "run_oracle_heuristic.py" 2>/dev/null | head -1 || true)
fi

if [[ -n "$E0_PID" ]] && kill -0 "$E0_PID" 2>/dev/null; then
    log "Waiting for E0 (PID $E0_PID) to finish..."
    while kill -0 "$E0_PID" 2>/dev/null; do
        traj_count=$(ls "$TMP_BASE/e0_oracle_heuristic/trajectories/" 2>/dev/null | wc -l || echo 0)
        log "  E0 still running — $traj_count trajectories so far"
        sleep 60
    done
    log "E0 finished."
    # Final rsync of E0
    if [[ -d "$TMP_BASE/e0_oracle_heuristic" ]]; then
        log "Final sync of E0 results"
        rsync -a "$TMP_BASE/e0_oracle_heuristic/" "$RESULTS_BASE/e0_oracle_heuristic/"
    fi
else
    log "E0 already finished (PID not running). Continuing."
fi

# ---------------------------------------------------------------------------
# Step 1: E2a alpha sweep
# ---------------------------------------------------------------------------
if [[ "$SKIP_E2A" -eq 1 ]]; then
    log "SKIP_E2A=1: skipping alpha sweep"
else
    run_exp e2_alpha_sweep \
        abides-gym/scripts/sweep_alpha_e2.py \
        --alphas 1e-4 1e-3 1e-2 \
        --timesteps "$E2A_TIMESTEPS" \
        --seed 0
fi

# Parse best alpha from sweep results (or use FORCE_ALPHA override)
if [[ -n "$FORCE_ALPHA" ]]; then
    BEST_ALPHA="$FORCE_ALPHA"
    log "FORCE_ALPHA=$BEST_ALPHA"
else
    SWEEP_JSON="$RESULTS_BASE/e2_alpha_sweep/alpha_sweep.json"
    if [[ ! -f "$SWEEP_JSON" ]]; then
        # Try /tmp fallback
        SWEEP_JSON="$TMP_BASE/e2_alpha_sweep/alpha_sweep.json"
    fi
    BEST_ALPHA=$("$PYTHON" -c "
import json, sys
d = json.load(open('$SWEEP_JSON'))
print(d['best_alpha'])
")
    log "Best alpha from E2a: $BEST_ALPHA"
fi

# ---------------------------------------------------------------------------
# Step 2: E1 — PPO + oracle belief in obs (seeds 0 and 1)
# ---------------------------------------------------------------------------
for SEED in 0 1; do
    run_exp "e1_belief_obs/seed_$SEED" \
        abides-gym/scripts/train_ppo_oracle_obs.py \
        --seed "$SEED" \
        --timesteps "$TIMESTEPS" \
        --timestep 60s \
        --num-workers 0
done

# ---------------------------------------------------------------------------
# Step 3: E2 — PPO + reward shaping (seeds 0 and 1)
# ---------------------------------------------------------------------------
for SEED in 0 1; do
    run_exp "e2_belief_reward/seed_$SEED" \
        abides-gym/scripts/train_ppo_oracle_reward.py \
        --seed "$SEED" \
        --timesteps "$TIMESTEPS" \
        --timestep 60s \
        --num-workers 0 \
        --alpha "$BEST_ALPHA"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "Pipeline complete."
log ""
log "Results:"
for tag in e2_alpha_sweep e1_belief_obs/seed_0 e1_belief_obs/seed_1 e2_belief_reward/seed_0 e2_belief_reward/seed_1; do
    json="$RESULTS_BASE/$tag/eval_metrics.json"
    if [[ -f "$json" ]]; then
        summary=$("$PYTHON" -c "
import json
d = json.load(open('$json'))
pnl = d.get('final_pnl_cents', {}).get('mean', 'N/A')
sharpe = d.get('sharpe', 'N/A')
win = d.get('win_rate', 'N/A')
print(f'mean_pnl={pnl:.0f}  sharpe={sharpe:.3f}  win={win:.1%}' if isinstance(pnl, float) else 'no metrics')
" 2>/dev/null || echo "parse error")
        log "  $tag: $summary"
    else
        log "  $tag: no eval_metrics.json"
    fi
done
