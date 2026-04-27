"""
Regenerate results/game_mal_per_class.csv with the corrected
balanced-accuracy column name (was previously mislabelled as "auc").

Replays the saved best fold of GAME-Mal: loads model + vocab, rebuilds
the matched-prep test split (3-fold stratified, seed=42, len>=30 filter),
identifies the best fold by predicted F1, and computes per-class metrics.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.baselines import compute_per_class_metrics
from src.model import GAMEMal
from src.preprocessing import load_dataset, pad_sequences

MODEL_DIR = REPO_ROOT / "results" / "models"
SEED = 42
N_FOLDS = 3


def main() -> None:
    with open(MODEL_DIR / "config.json") as f:
        cfg = json.load(f)
    with open(MODEL_DIR / "vocab.pkl", "rb") as f:
        v = pickle.load(f)
    with open(MODEL_DIR / "family_names.json") as f:
        family_names = json.load(f)
    api2idx = v["api2idx"]
    unk = api2idx.get("<UNK>", 1)

    sequences, labels, _ = load_dataset(REPO_ROOT / "extracted_data")
    keep = [i for i, s in enumerate(sequences) if len(s) >= 30]
    sequences = [sequences[i] for i in keep]
    labels = [labels[i] for i in keep]
    encoded = [[api2idx.get(t, unk) for t in s] for s in sequences]
    padded = pad_sequences(encoded, max_len=cfg["max_seq_len"])
    y = np.array(labels)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model = GAMEMal(
        vocab_size=cfg["vocab_size"],
        num_classes=cfg["num_classes"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(
        torch.load(MODEL_DIR / "game_mal_best.pt", map_location=device)
    )
    model.eval()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(padded, y))
    best_fold = cfg.get("best_fold", 0)
    if not isinstance(best_fold, int) or best_fold >= N_FOLDS:
        best_fold = 0
    _, te = splits[best_fold]
    X_te, y_te = padded[te], y[te]

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_te), 64):
            x = torch.from_numpy(X_te[i : i + 64]).long().to(device)
            logits, _ = model(x, return_attention=False)
            preds.append(logits.argmax(dim=-1).cpu().numpy())
    y_pred = np.concatenate(preds)

    per_class = compute_per_class_metrics(y_te, y_pred, family_names)

    out_path = REPO_ROOT / "results" / "game_mal_per_class.csv"
    fieldnames = [
        "family",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "balanced_accuracy",
        "f_score",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for fam, m in per_class.items():
            w.writerow(
                {
                    "family": fam,
                    "accuracy": m["accuracy"],
                    "sensitivity": m["sensitivity"],
                    "specificity": m["specificity"],
                    "precision": m["precision"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "f_score": m["f_score"],
                }
            )
    print(f"Wrote {out_path} (column 'auc' renamed to 'balanced_accuracy')")
    for fam, m in per_class.items():
        print(
            f"  {fam:14s}  P={m['precision']:.3f}  R={m['sensitivity']:.3f}  F1={m['f_score']:.3f}  bal_acc={m['balanced_accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
