# Explainable Android Malware Classification via Transformer with Markov-Initialized Embeddings

A B.Tech research project combining:
1. Reflection-aware parsing of Droidmon dynamic-analysis traces,
2. A Transformer classifier using a learnable `[CLS]` token,
3. Markov-initialized embeddings via SVD factorization of k-spaced API transition matrices (inspired by D'Angelo et al., 2023),
4. Intrinsic per-sample explanations derived from `[CLS]` attention maps, validated via faithfulness deletion tests.

## Results (3-fold stratified CV, 8-family UMD subset, 9,337 samples)

| Model              | Accuracy | Macro F1 | Macro AUROC |
|--------------------|--------:|---------:|------------:|
| GaussianNB         |  0.860  |  0.782   |  0.925      |
| DecisionTree       |  0.923  |  0.836   |  0.910      |
| LinearSVM          |  0.923  |  0.844   |  0.980      |
| MarkovTransformer  |  0.927  |  0.858   |  0.976      |
| PlainTransformer   |  0.940  |  0.884   |  0.981      |
| BiLSTM             |  0.948  |  0.894   |  0.992      |
| RandomForest       |  0.950  |  0.893   |  0.993      |

See `PROJECT_SUMMARY.md` for the full analysis including the Markov embedding experiment and faithfulness deletion test results.

## Repo Layout

```
src/
  preprocessing.py          # Reflection-aware parsing + vocabulary
  markov.py                 # k-spaced rule mining + SVD embedding generation
  model.py                  # Transformer with learnable [CLS] token
  train.py                  # PyTorch train/eval loop
  baselines.py              # sklearn classifiers (RF, SVM, DT, NB)
  bilstm.py                 # BiLSTM baseline model

scripts/
  data_extractor.py                  # Raw corpus extraction
  run_plain_transformer_final.py     # Plain Transformer 3-fold CV
  run_markov_embedding_experiment.py # Markov Transformer ablation
  run_deletion_test.py               # CLS attention faithfulness test
  visualize_attention.py             # CLS attention heatmaps
  visualize_markov_embeddings.py     # Cosine similarity visualization
  run_bilstm.py                      # BiLSTM 3-fold baseline
  run_seq_len_sweep.py               # Sequence length / truncation sweep
  run_markov_sweep.py                # Markov rule hyperparameter sweep
  run_plain_analysis.py              # Per-class metrics + confusion matrix
  build_sweep_summary.py             # Aggregate sweep results

results/
  results_summary.csv                # All models, 3-fold aggregates
  plain_transformer_final.json       # Plain Transformer fold details
  markov_transformer_final.json      # Markov Transformer fold details
  deletion_test.json                 # Plain Transformer faithfulness
  markov_transformer_deletion_test.json  # Markov faithfulness
  figures/                           # All generated plots
  models/                            # Saved weights + configs

colab_markov_pipeline.sh             # One-command Colab automation
```

## Reproducing

Prerequisites: Python 3.9+, PyTorch 2.2+ (with MPS or CUDA), scikit-learn 1.4+.

```bash
pip install -r requirements.txt

# Plain Transformer (primary model)
python scripts/run_plain_transformer_final.py

# Markov Transformer (ablation)
python scripts/run_markov_embedding_experiment.py

# Deletion test (explainability validation)
python scripts/run_deletion_test.py --model_prefix plain_transformer
python scripts/run_deletion_test.py --model_prefix markov_transformer

# Attention visualizations
python scripts/visualize_attention.py --model_prefix plain_transformer
python scripts/visualize_attention.py --model_prefix markov_transformer
```

Or on Google Colab:
```bash
!bash colab_markov_pipeline.sh
```

The raw corpus (`extracted_data/`) is not checked in — see `scripts/data_extractor.py` for the extraction routine.

## Author

Arav Jain
