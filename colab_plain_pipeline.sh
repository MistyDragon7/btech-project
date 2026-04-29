#!/bin/bash
set -e

echo "=========================================="
echo "1. Installing Dependencies"
echo "=========================================="
pip install -r requirements.txt

echo "=========================================="
echo "2. Training Plain Transformer"
echo "=========================================="
python scripts/run_plain_transformer_final.py

echo "=========================================="
echo "3. Running Explainability Deletion Test"
echo "=========================================="
python scripts/run_deletion_test.py --model_prefix plain_transformer

echo "=========================================="
echo "4. Generating Attention Visualizations"
echo "=========================================="
python scripts/visualize_attention.py --model_prefix plain_transformer

echo "=========================================="
echo "5. Zipping Results for Download"
echo "=========================================="
zip -r results_plain_run.zip results/

echo "=========================================="
echo "Pipeline Complete! Please download results_plain_run.zip"
echo "=========================================="
