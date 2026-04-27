# Transformer with Attention Map Explainability

**Plain Transformer over Markov Embeddings for Explainable Android Malware Classification**

A B.Tech research project combining:
1. Reflection-aware parsing of Droidmon dynamic-analysis traces,
2. A plain Transformer classifier using a `[CLS]` token,
3. A faithful reproduction of D'Angelo et al.'s (2023) Markov associative-rule classifier as a baseline,
4. Intrinsic per-sample explanations derived directly from the `[CLS]` token attention maps.

## Results (3-fold stratified CV, 8-family UMD subset, 9,337 samples)

| Model          | Accuracy | Macro F1 | Macro AUROC |
|----------------|---------:|---------:|------------:|
| GaussianNB     |  0.860   |  0.782   |  0.925      |
| DecisionTree   |  0.923   |  0.836   |  0.910      |
| LinearSVM      |  0.923   |  0.844   |  0.980      |
| MarkovPruning  |  0.829   |  0.709   |  0.942      |
| **PlainTransformer**| **0.939**| **0.884**| **0.980**   |
| RandomForest   |  0.950   |  0.893   |  0.993      |

The Plain Transformer matches the Random Forest baseline on all macro metrics within one
fold-std, while improving macro F1 by **+17.5 points** over the D'Angelo
associative-rule baseline and providing intrinsic per-sample explanations via `[CLS]` attention maps.

The reported numbers are the final retrained model (`scripts/run_plain_transformer_final.py`)
with the optimal config from the sequence-length sweep:
$L_{\max}=512$, **head**-truncation, dropout=0.15, lr=5e-4, 50 epochs,
patience=12, seed=42 (numpy + torch).

## Repo layout

```
.
├── src/
│   ├── preprocessing.py      # reflection-aware parsing + vocab
│   ├── markov.py             # k-spaced rule mining + pruning
│   ├── baselines.py          # sklearn + MarkovPruning classifiers
│   ├── model.py              # Plain Transformer with [CLS] token
│   ├── train.py              # PyTorch train/eval loop
│   └── explain.py            # [CLS] attention extraction
├── run_experiments.py        # full 3-fold pipeline
├── scripts/
│   ├── data_extractor.py             # raw-corpus extraction
│   ├── train_final.py                # single final model + weight saving
│   ├── run_plain_transformer_final.py# 3-fold final retrain at best config
│   ├── run_seq_len_sweep.py          # max_seq_len × {head,tail} grid
│   ├── run_bilstm.py                 # BiLSTM 3-fold sequence baseline
│   ├── run_deletion_test.py          # CLS attention faithfulness test
│   ├── run_markov_sweep.py           # MarkovPruning hyperparameter sweep
│   ├── compare_attributions.py       # attention vs GradientInput overlap
│   ├── regen_per_class.py            # rebuild per-class CSV with corrected fields
│   ├── build_api_semantic_groups.py  # heuristic bucket histograms
│   ├── aggregate_final_results.py    # FINAL_REPORT.md
│   └── analyze_project.py            # PROJECT_ANALYSIS.md (coherence audit)
├── results/
│   ├── results_summary.csv   # 3-fold aggregate metrics
│   ├── plain_transformer_per_class.csv
│   ├── top_apis_per_family.json
│   ├── sparsity_stats.json
│   ├── figures/              # confusion matrices, attention rollout, etc.
│   └── models/               # saved weights + vocab + config
├── paper/                    # IEEE conference draft + figures
│   ├── main.tex
│   ├── sections/01_introduction.tex … 08_conclusion.tex
│   ├── references.bib
│   └── figures/
├── STUDY_GUIDE.md            # thesis defense prep
└── DL_CONCEPTS.md            # deep-learning background primer
```

## Reproducing

Prerequisites: Python 3.9+, PyTorch 2.2+ (with MPS or CUDA), scikit-learn 1.4+.

```bash
pip install -r requirements.txt

# Full 3-fold pipeline (~2.5 h on Apple M-series via MPS)
python3 -u run_experiments.py

# Single-fold final model with weight saving (~1 h)
python3 -u scripts/train_final.py

# 3-fold retrain at the swept-best (max_seq_len, truncation), 50 epochs
python3 -u scripts/run_plain_transformer_final.py

# Sequence-length / truncation sweep (slow on long lengths via MPS)
python3 -u scripts/run_seq_len_sweep.py
```

The raw corpus (`extracted_data/`) is not checked in — see `scripts/data_extractor.py`
for the extraction routine.

## Citation

Work in progress. See `paper/main.tex` for the latest draft.

## Author

Arav Jain — `aravjain.int@urbancompany.com`
