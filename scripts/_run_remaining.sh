#!/bin/bash
# Chained runner: waits for in-flight seq_len sweep to finish (PID $SWEEP_PID),
# then in order:
#   1. retrain final GAME-Mal on best (max_seq_len, truncation) — full corpus,
#      50 epochs max, 3-fold, seeded — and overwrite results/models/
#   2. BiLSTM 3-fold using the same best config, full corpus
#   3. deletion test on the new saved model
#   4. regenerate per-class CSV with corrected balanced-accuracy column
#   5. aggregate FINAL_REPORT.md
#   6. logical-consistency analysis -> PROJECT_ANALYSIS.md
#   7. write results/PIPELINE_DONE marker
set -u
cd /Users/aravjain/PERSONAL/btech-project

LOG=logs/remaining.log
mkdir -p logs results

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" >> "$LOG"; }

log "=== chained runner started ==="
SWEEP_PID=${SWEEP_PID:-93407}
log "waiting for seq_len sweep PID=$SWEEP_PID"
while kill -0 "$SWEEP_PID" 2>/dev/null; do
    sleep 30
done
log "seq_len sweep finished"

run_step () {
    local name=$1
    local cmd=$2
    local logfile=$3
    log "--- $name ---"
    eval "$cmd" >> "$logfile" 2>&1
    local rc=$?
    log "$name exit=$rc"
    if [ "$rc" -ne 0 ]; then
        log "STEP FAILED: $name. See $logfile. Aborting pipeline."
        touch results/PIPELINE_FAILED
        exit "$rc"
    fi
}

run_step "final GAME-Mal 3-fold (best seq_len/truncation)" \
    "python3 -u scripts/run_game_mal_final.py" \
    logs/game_mal_final.log

run_step "BiLSTM 3-fold (same config)" \
    "python3 -u scripts/run_bilstm.py" \
    logs/bilstm.log

run_step "deletion test on new saved model" \
    "python3 -u scripts/run_deletion_test.py" \
    logs/deletion.log

run_step "regenerate per-class CSV (corrected column name)" \
    "python3 -u scripts/regen_per_class.py" \
    logs/regen_per_class.log

run_step "aggregate FINAL_REPORT.md" \
    "python3 -u scripts/aggregate_final_results.py" \
    logs/aggregate.log

run_step "project coherence analysis" \
    "python3 -u scripts/analyze_project.py" \
    logs/analyze.log

touch results/PIPELINE_DONE
log "=== chained runner DONE ==="
