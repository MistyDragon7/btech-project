"""
Sequence-length + truncation sweep for GAME-Mal.

Grid: max_seq_len in {256, 512, 768} x truncation in {head, tail}  (6 configs)
Same architecture + training recipe as run_gated_matched.py, on the
len>=30 subset (8,085 samples), 3-fold stratified CV (seed=42).

Outputs:
    results/seq_len_sweep.csv          (per-fold per-config metrics)
    results/seq_len_sweep_summary.csv  (aggregate mean/std per config)

Usage:
    python3 -u scripts/run_seq_len_sweep.py
    python3 -u scripts/run_seq_len_sweep.py --quick-check   # 1 fold, 5 epochs
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import load_dataset, APIVocabulary, pad_with_truncation
from src.model import GAMEMal
from src.train import create_dataloader, train_epoch, evaluate


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("seq_len_sweep")


class Cfg:
    min_freq = 2
    d_model = 128
    n_heads = 4
    n_layers = 2
    d_ff = 256
    dropout = 0.1
    batch_size = 32
    epochs = 25
    lr = 3e-4
    weight_decay = 1e-4
    patience = 7
    n_folds = 3
    seed = 42

    seq_lens = (256, 512, 768)
    truncations = ("head", "tail")


def train_one(
    X_train, y_train, X_test, y_test,
    vocab_size, num_classes, max_seq_len, device,
):
    model = GAMEMal(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=Cfg.d_model,
        n_heads=Cfg.n_heads,
        n_layers=Cfg.n_layers,
        d_ff=Cfg.d_ff,
        max_seq_len=max_seq_len,
        dropout=Cfg.dropout,
        use_gate=True,
    ).to(device)

    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_w = len(y_train) / (num_classes * counts)
    class_w_t = torch.from_numpy(class_w).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w_t)

    optim = torch.optim.AdamW(model.parameters(), lr=Cfg.lr, weight_decay=Cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=Cfg.epochs)

    train_loader = create_dataloader(X_train, y_train, batch_size=Cfg.batch_size, shuffle=True)
    test_loader = create_dataloader(X_test, y_test, batch_size=Cfg.batch_size, shuffle=False)

    best_metrics = None
    best_f1 = -1.0
    stale = 0
    for ep in range(1, Cfg.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optim, criterion, device)
        val_loss, metrics, _, _ = evaluate(model, test_loader, criterion, device, num_classes)
        sched.step()
        if metrics["f_score"] > best_f1:
            best_f1 = metrics["f_score"]
            best_metrics = metrics
            stale = 0
        else:
            stale += 1
            if stale >= Cfg.patience:
                log.info("    early stop ep=%d best_f1=%.4f", ep, best_f1)
                break
    return best_metrics


def main(quick: bool = False) -> None:
    if quick:
        Cfg.n_folds = 1
        Cfg.epochs = 5
        Cfg.patience = 3
        log.info("QUICK CHECK: 1 fold, %d epochs", Cfg.epochs)

    log.info("Loading corpus...")
    sequences, labels, family_names = load_dataset(REPO_ROOT / "extracted_data")
    keep = [i for i, s in enumerate(sequences) if len(s) >= 30]
    sequences = [sequences[i] for i in keep]
    labels = [labels[i] for i in keep]
    y = np.array(labels)
    log.info("Loaded %d samples across %d families (len>=30 filter)",
             len(sequences), len(family_names))

    vocab = APIVocabulary(min_freq=Cfg.min_freq)
    vocab.build(sequences)
    encoded = [vocab.encode_sequence(s) for s in sequences]

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    log.info("Device: %s", device)

    num_classes = len(family_names)
    if Cfg.n_folds > 1:
        skf = StratifiedKFold(n_splits=Cfg.n_folds, shuffle=True, random_state=Cfg.seed)
        splits = list(skf.split(encoded, y))
    else:
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(np.arange(len(y)), test_size=0.3,
                                  stratify=y, random_state=Cfg.seed)
        splits = [(tr, te)]

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / ("seq_len_sweep_quick.csv" if quick else "seq_len_sweep.csv")
    summary_path = out_dir / ("seq_len_sweep_summary_quick.csv" if quick
                              else "seq_len_sweep_summary.csv")

    rows = []
    t_start = time.time()
    for max_len in Cfg.seq_lens:
        for trunc in Cfg.truncations:
            log.info("=" * 60)
            log.info("CONFIG: max_seq_len=%d  truncation=%s", max_len, trunc)
            log.info("=" * 60)
            padded = pad_with_truncation(encoded, max_len=max_len, truncation=trunc)

            for fold_idx, (tr_idx, te_idx) in enumerate(splits, start=1):
                log.info("  fold %d/%d (train=%d test=%d)",
                         fold_idx, len(splits), len(tr_idx), len(te_idx))
                X_train, X_test = padded[tr_idx], padded[te_idx]
                y_train, y_test = y[tr_idx], y[te_idx]
                metrics = train_one(
                    X_train, y_train, X_test, y_test,
                    vocab_size=len(vocab), num_classes=num_classes,
                    max_seq_len=max_len, device=device,
                )
                row = {
                    "max_seq_len": max_len,
                    "truncation": trunc,
                    "fold": fold_idx,
                    "accuracy": metrics["accuracy"],
                    "sensitivity": metrics["sensitivity"],
                    "precision": metrics["precision"],
                    "f_score": metrics["f_score"],
                    "auc": metrics["auc"],
                }
                rows.append(row)
                log.info("  -> acc=%.4f f1=%.4f auc=%.4f",
                         metrics["accuracy"], metrics["f_score"], metrics["auc"])

                # Persist after every fold so partial progress survives a crash.
                with open(rows_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

    # Aggregate
    summary = {}
    for r in rows:
        key = (r["max_seq_len"], r["truncation"])
        summary.setdefault(key, []).append(r)

    summary_rows = []
    for (max_len, trunc), group in summary.items():
        rec = {"max_seq_len": max_len, "truncation": trunc, "n_folds": len(group)}
        for k in ["accuracy", "sensitivity", "precision", "f_score", "auc"]:
            vals = [g[k] for g in group]
            rec[f"{k}_mean"] = float(np.mean(vals))
            rec[f"{k}_std"] = float(np.std(vals))
        summary_rows.append(rec)

    summary_rows.sort(key=lambda r: r["f_score_mean"], reverse=True)
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    log.info("=" * 60)
    log.info("SWEEP COMPLETE in %.1f min", (time.time() - t_start) / 60.0)
    log.info("Results:    %s", rows_path)
    log.info("Summary:    %s", summary_path)
    log.info("Best config (by mean macro-F1):")
    best = summary_rows[0]
    log.info("  max_seq_len=%d  truncation=%s  f1=%.4f (+/- %.4f)  acc=%.4f",
             best["max_seq_len"], best["truncation"],
             best["f_score_mean"], best["f_score_std"], best["accuracy_mean"])

    # Also dump JSON for memory-update convenience.
    with open(out_dir / ("seq_len_sweep_summary_quick.json" if quick
                         else "seq_len_sweep_summary.json"), "w") as f:
        json.dump({
            "configs": summary_rows,
            "best": best,
            "n_folds": Cfg.n_folds,
            "epochs_max": Cfg.epochs,
        }, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick-check", action="store_true",
                   help="1 fold, 5 epochs — wiring smoke test only.")
    args = p.parse_args()
    main(quick=args.quick_check)
