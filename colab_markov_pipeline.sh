#!/bin/bash
set -e

echo "=========================================="
echo "1. Installing Dependencies"
echo "=========================================="
pip install -r requirements.txt

echo "=========================================="
echo "2. Training Markov-Initialized Transformer"
echo "=========================================="
python scripts/run_markov_embedding_experiment.py

echo "=========================================="
echo "3. Running Explainability Deletion Test"
echo "=========================================="
python scripts/run_deletion_test.py --model_prefix markov_transformer

echo "=========================================="
echo "4. Generating Attention Visualizations"
echo "=========================================="
python scripts/visualize_attention.py --model_prefix markov_transformer

echo "=========================================="
echo "5. Generating Markov Embeddings Visualization"
echo "=========================================="
python scripts/visualize_markov_embeddings.py

echo "=========================================="
echo "6. Zipping Results for Download"
echo "=========================================="
zip -r results_markov_run.zip results/

echo "=========================================="
echo "Pipeline Complete! Please download results_markov_run.zip"
echo "=========================================="
