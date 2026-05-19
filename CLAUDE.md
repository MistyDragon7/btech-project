# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

GAME-Mal: Explainable Android malware family classification (8 families, UMD corpus, 9,337 samples). Input is Droidmon dynamic-analysis JSONL traces; output is a family label. The primary model is a plain Transformer encoder with a learnable `[CLS]` token and attention-rollout explainability (Abnar & Zuidema 2020). GAME-Mal (gated attention variant) is kept as an ablation comparison.

## Environment

- **Always use `python3`** — `python` resolves to Python 2.7 on this machine.
- PyTorch backend: Apple Silicon MPS (auto-detected). MPS deadlocks at seq_len=768; SHAP gradient passes require CPU.
- Install: `pip install -r requirements.txt` (torch, numpy, pandas, scikit-learn, matplotlib, seaborn).

## Common commands

```bash
# Primary model: plain transformer 3-fold CV (epochs=100, patience=20)
python3 scripts/run_plain_transformer_final.py

# Per-class metrics + confusion matrix for best fold
python3 scripts/run_plain_analysis.py

# SHAP analysis (RF TreeExplainer + Transformer GradientExplainer)
python3 scripts/shap_analysis.py

# Attention rollout figures
python3 scripts/attention_rollout.py

# Full chain (retrain → analysis → SHAP → rollout)
bash scripts/run_plain_and_visualize.sh

# Ablation / comparison runs
python3 scripts/run_markov_embedding_experiment.py   # Markov-init embeddings
python3 scripts/run_bilstm.py                        # BiLSTM baseline
python3 scripts/run_deletion_test.py --model_prefix plain_transformer
python3 scripts/run_seq_len_sweep.py                 # 256/512/768 × head/tail
python3 scripts/run_markov_sweep.py                  # 135-config Markov sweep
```

All scripts resolve `REPO_ROOT` as `Path(__file__).resolve().parents[1]`, so they must be run from the repo root or invoked with their full path.

## Architecture

### Data pipeline (`src/preprocessing.py`)
- Loads Droidmon JSONL from `extracted_data/<FamilyName>/*.apk`.
- `resolve_api()`: reflection-aware — uses `hooked_class.hooked_method` for reflected calls, prefixed `REFL:`.
- `APIVocabulary`: PAD=0, UNK=1, then APIs by frequency (min_freq=2). Vocab size=1,118.
- `pad_with_truncation(side='head'|'tail')`: head truncation is the canonical setting (sweeps confirmed head ≫ tail).

### Model (`src/model.py`)
`MalwareTransformer` — plain Transformer encoder:
- Learnable `cls_token` parameter prepended to every sequence; classification from CLS final hidden state.
- Pre-norm blocks (`TransformerBlock`): LayerNorm → MHA → residual → LayerNorm → FFN → residual.
- `forward(x, return_attention=True)` returns logits + dict with `attn_weights` (list of `(B, H, S, S)` tensors per layer), enabling attention rollout without post-hoc wrappers.
- Padding mask: PAD keys are masked out; CLS is never masked.

### Training (`src/train.py`)
`train_transformer()` (aliased as `train_game_mal` for back-compat):
- AdamW + cosine LR (`T_max=epochs`), grad-clip 1.0, early stopping on macro-F1.
- Class-weighted CE loss (inverse frequency) to handle 35× Airpush imbalance.
- Canonical config: `d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.15, lr=5e-4, epochs=100, patience=20, max_seq_len=512`.

### Baselines (`src/baselines.py`, `src/bilstm.py`, `src/markov.py`)
- `baselines.py`: RF, SVM, DT, GNB sklearn wrappers + `compute_metrics()` used project-wide.
- `bilstm.py`: 2-layer BiLSTM, d_model=128, mean-pool, class-weighted CE.
- `markov.py`: k-spaced rule mining, `build_class_graphs` → `(class_graphs, global_rules)`, SVD embedding generation.

## Results files

| File | Contents |
|---|---|
| `results/results_summary.csv` | All models, 3-fold aggregates |
| `results/plain_transformer_final.json` | Fold-level metrics |
| `results/models/plain_transformer_best.pt` | Best-fold weights (fold3, F1=0.900) |
| `results/models/plain_transformer_config.json` | Config + `best_fold` index (0-based) |
| `results/figures/` | All plots (confusion matrix, SHAP, rollout) |

## Key design decisions

- **Gate ablation result**: gated attention hurts macro-F1 by 1.4pp at this scale. Plain transformer is the primary contribution; GAME-Mal (gated) is the ablation.
- **Explainability**: attention rollout (not raw attention, not SHAP) is the primary intrinsic method. SHAP figures exist for supplementary analysis.
- **SHAP on CPU**: `shap_analysis.py` forces CPU device — MPS gradient passes are unstable for GradientExplainer.
- **`TransformerFromEmbedding` wrapper in SHAP**: passes token-only embeddings (no positional) to the wrapper; positional embeddings are added inside `forward()`. This is the fix for a padding bug where non-zero positional values were breaking attention masks.
- **Fold selection**: scripts that load a checkpoint read `best_fold` from `plain_transformer_config.json` rather than hardcoding.
