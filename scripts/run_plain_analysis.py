"""
Post-training analysis for the plain transformer:
  - Per-family precision / recall / F1 / support (best fold)
  - Confusion matrix figure
  - Training-curve figure (loss + F1 vs epoch, from log)
  - Saves plain_transformer_per_class.csv and figures

Run after run_plain_transformer_final.py completes.
"""
from __future__ import annotations
import csv, json, pickle, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.preprocessing import load_dataset, APIVocabulary, pad_with_truncation
from src.model import GAMEMal
from src.train import create_dataloader, evaluate

MODELS  = REPO / "results" / "models"
RESULTS = REPO / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 42


def load_plain_model():
    pt_cfg  = MODELS / "plain_transformer_config.json"
    cfg = json.loads(pt_cfg.read_text())

    with open(MODELS / "vocab.pkl", "rb") as f:
        vobj = pickle.load(f)

    model = GAMEMal(
        vocab_size=cfg["vocab_size"], num_classes=cfg["num_classes"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"], max_seq_len=cfg["max_seq_len"], dropout=0.0,
        use_gate=False,
    )
    state = torch.load(MODELS / "plain_transformer_best.pt",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model, vobj, cfg


def run_analysis():
    print("Loading plain transformer …")
    model, vobj, cfg = load_plain_model()
    device = torch.device("cpu")   # inference only — CPU is fine
    model.to(device)

    print("Loading data …")
    sequences, labels, family_names = load_dataset(REPO / "extracted_data")
    y = np.array(labels)
    num_cls = len(family_names)

    vocab_obj = APIVocabulary(min_freq=2)
    vocab_obj.api2idx = vobj["api2idx"]
    vocab_obj.idx2api = vobj["idx2api"]

    encoded = [vocab_obj.encode_sequence(s) for s in sequences]
    padded  = pad_with_truncation(encoded, max_len=cfg["max_seq_len"],
                                  truncation=cfg.get("truncation", "head"))

    # Reconstruct the SAME fold splits as training
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    splits = list(skf.split(padded, y))

    best_fold = cfg.get("best_fold", 0)   # 0-indexed
    tr_idx, te_idx = splits[best_fold]
    X_te = padded[te_idx]
    y_te = y[te_idx]

    print(f"Running inference on best fold ({best_fold+1}/3), n={len(te_idx)} …")
    counts = np.bincount(y[tr_idx], minlength=num_cls).astype(np.float32)
    class_w = len(tr_idx) / (num_cls * np.maximum(counts, 1.0))
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_w).to(device))

    te_loader = create_dataloader(X_te, y_te, batch_size=64, shuffle=False)
    _, metrics, y_true, y_pred = evaluate(model, te_loader, criterion, device, num_cls)

    # ── Per-class report ────────────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred, target_names=family_names,
        output_dict=True, zero_division=0)

    rows = []
    for fn in family_names:
        r = report[fn]
        rows.append({
            "family":    fn,
            "precision": round(r["precision"], 4),
            "recall":    round(r["recall"], 4),
            "f1":        round(r["f1-score"], 4),
            "support":   int(r["support"]),
            "share_pct": round(100 * r["support"] / len(y_te), 1),
        })

    out_csv = RESULTS / "plain_transformer_per_class.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved per-class CSV: {out_csv}")
    for r in rows:
        print(f"  {r['family']:15s}  P={r['precision']:.3f}  R={r['recall']:.3f}"
              f"  F1={r['f1']:.3f}  n={r['support']}")

    # ── Confusion matrix ────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = range(num_cls)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(family_names, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(family_names, fontsize=8)
    for i in range(num_cls):
        for j in range(num_cls):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if cm[i,j] > 0.5 else "black")
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(
        f"Plain Transformer — Confusion Matrix (best fold {best_fold+1}, normalised by true class)\n"
        f"acc={metrics['accuracy']:.4f}  F1={metrics['f_score']:.4f}  AUC={metrics['auc']:.4f}",
        fontsize=10,
    )
    plt.tight_layout()
    out_cm = FIGURES / "confusion_plain_transformer.png"
    plt.savefig(out_cm, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved confusion matrix: {out_cm}")

    # ── Summary ─────────────────────────────────────────────────────────────
    final = json.loads((RESULTS / "plain_transformer_final.json").read_text())
    print(f"\nPlain Transformer 3-fold summary:")
    print(f"  acc  = {final['accuracy_mean']:.4f} ± {final['accuracy_std']:.4f}")
    print(f"  F1   = {final['f_score_mean']:.4f} ± {final['f_score_std']:.4f}")
    print(f"  AUC  = {final['auc_mean']:.4f} ± {final['auc_std']:.4f}")


if __name__ == "__main__":
    run_analysis()
