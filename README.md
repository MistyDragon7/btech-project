# GAME-Mal

**Gated Attention over Markov Embeddings for Explainable Android Malware Classification**

A B.Tech research project combining:
1. Reflection-aware parsing of Droidmon dynamic-analysis traces,
2. Sigmoid-gated multi-head attention (G1 placement of Qiu et al., 2025) for family classification,
3. A faithful reproduction of D'Angelo et al.'s (2023) Markov associative-rule classifier as a baseline,
4. Intrinsic per-sample explanations read directly off the gate activations.

## Results (3-fold stratified CV, 8-family UMD subset, 9,337 samples)

| Model          | Accuracy | Macro F1 | Macro AUROC |
|----------------|---------:|---------:|------------:|
| GaussianNB     |  0.860   |  0.782   |  0.925      |
| DecisionTree   |  0.923   |  0.836   |  0.910      |
| LinearSVM      |  0.923   |  0.844   |  0.980      |
| MarkovPruning  |  0.829   |  0.709   |  0.942      |
| **GAME-Mal**   | **0.940**| **0.886**| **0.984**   |
| RandomForest   |  0.950   |  0.893   |  0.993      |

GAME-Mal matches the Random Forest baseline on all macro metrics within one
fold-std, while improving macro F1 by **+17.7 points** over the D'Angelo
associative-rule baseline and providing intrinsic per-sample explanations.

## Repo layout

```
.
├── src/
│   ├── preprocessing.py      # reflection-aware parsing + vocab
│   ├── markov.py             # k-spaced rule mining + pruning
│   ├── baselines.py          # sklearn + MarkovPruning classifiers
│   ├── model.py              # GAME-Mal gated-attention transformer
│   ├── train.py              # PyTorch train/eval loop
│   └── explain.py            # gate-activation aggregation
├── run_experiments.py        # full 3-fold pipeline
├── scripts/
│   ├── data_extractor.py     # raw-corpus extraction
│   └── train_final.py        # single final model + weight saving
├── results/
│   ├── results_summary.csv   # 3-fold aggregate metrics
│   ├── game_mal_per_class.csv
│   ├── top_apis_per_family.json
│   ├── sparsity_stats.json
│   ├── figures/              # confusion matrices, gate histograms, etc.
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
```

The raw corpus (`extracted_data/`) is not checked in — see `scripts/data_extractor.py`
for the extraction routine.

## Citation

Work in progress. See `paper/main.tex` for the latest draft.

## Author

Arav Jain — `aravjain.int@urbancompany.com`
