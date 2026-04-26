#!/usr/bin/env bash
# ============================================================
# Chained runner: plain transformer retrain → SHAP → attention rollout
# Usage:  bash scripts/run_plain_and_visualize.sh
# Logs:   results/logs/
# ============================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO/results/logs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo " Step 1: Retrain plain transformer (3-fold, 50 epochs)"
echo "========================================"
python3 "$REPO/scripts/run_plain_transformer_final.py" \
    2>&1 | tee "$LOG_DIR/plain_transformer_retrain.log"

if [ $? -ne 0 ]; then
    echo "ERROR: plain transformer retrain failed. Aborting."
    exit 1
fi
echo "✓ Plain transformer retrain complete."

echo ""
echo "========================================"
echo " Step 2: SHAP analysis (RF + Transformer)"
echo "========================================"
python3 "$REPO/scripts/shap_analysis.py" \
    2>&1 | tee "$LOG_DIR/shap_analysis.log"

if [ $? -ne 0 ]; then
    echo "ERROR: SHAP analysis failed. Aborting."
    exit 1
fi
echo "✓ SHAP analysis complete."

echo ""
echo "========================================"
echo " Step 3: Attention rollout visualisation"
echo "========================================"
python3 "$REPO/scripts/attention_rollout.py" \
    2>&1 | tee "$LOG_DIR/attention_rollout.log"

if [ $? -ne 0 ]; then
    echo "ERROR: Attention rollout failed. Aborting."
    exit 1
fi
echo "✓ Attention rollout complete."

echo ""
echo "========================================"
echo " ALL DONE — figures in results/figures/"
echo "========================================"
ls "$REPO/results/figures/"*.png 2>/dev/null | while read f; do
    echo "  $(basename "$f")"
done

touch "$REPO/results/VISUALIZE_DONE"
echo ""
echo "Pipeline marker: results/VISUALIZE_DONE"
