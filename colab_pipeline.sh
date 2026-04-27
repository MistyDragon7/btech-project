#!/bin/bash
set -e

echo "=========================================="
echo "1. Installing Dependencies"
echo "=========================================="
pip install -r requirements.txt

echo "=========================================="
echo "2. Training Plain Transformer"
echo "=========================================="
# This will train the model, find the best fold, and save the checkpoint to results/models/
python scripts/run_plain_transformer_final.py

echo "=========================================="
echo "3. Running Explainability Deletion Test"
echo "=========================================="
# This uses the new [CLS] attention maps to validate faithfulness
python scripts/run_deletion_test.py

echo "=========================================="
echo "4. Generating Attention Visualizations"
echo "=========================================="
# This generates the heatmaps and sink plots
python scripts/visualize_attention.py

echo "=========================================="
echo "5. Zipping Results for Download"
echo "=========================================="
# Zips the entire results directory so you can easily download it from Colab
zip -r results_colab_run.zip results/

echo "=========================================="
echo "Pipeline Complete! Please download results_colab_run.zip"
echo "=========================================="
