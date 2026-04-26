"""
Apples-to-apples companion to scripts/run_ablation.py.

Runs the GATED GAME-Mal (use_gate=True) under the *exact same* recipe,
sequence-length filter (>=30), and stratified folds as the no-gate
ablation, so the two can be compared without a dataset-size confound.
Writes results/gated_matched_summary.{csv,json}.

Usage:
    python3 -u scripts/run_gated_matched.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import load_dataset, APIVocabulary, pad_sequences
from src.model import GAMEMal
from src.train import create_dataloader, train_epoch, evaluate


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gated_matched")


class Cfg:
    # Must match run_ablation.py exactly
    min_freq = 2
    max_seq_len = 512
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


def main() -> None:
    # Reproducibility: seed torch as well as numpy/sklearn.
    torch.manual_seed(Cfg.seed)
    np.random.seed(Cfg.seed)
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
    padded = pad_sequences(encoded, max_len=Cfg.max_seq_len)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    log.info("Device: %s", device)

    num_classes = len(family_names)
    skf = StratifiedKFold(n_splits=Cfg.n_folds, shuffle=True, random_state=Cfg.seed)

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(padded, y), start=1):
        log.info("=" * 60)
        log.info("FOLD %d/%d (train=%d, test=%d)", fold, Cfg.n_folds, len(train_idx), len(test_idx))
        log.info("=" * 60)

        X_train, X_test = padded[train_idx], padded[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = GAMEMal(
            vocab_size=len(vocab),
            num_classes=num_classes,
            d_model=Cfg.d_model,
            n_heads=Cfg.n_heads,
            n_layers=Cfg.n_layers,
            d_ff=Cfg.d_ff,
            max_seq_len=Cfg.max_seq_len,
            dropout=Cfg.dropout,
            use_gate=True,  # <-- gated (the one difference from run_ablation.py)
        ).to(device)
        log.info("Gated-transformer parameters: %d", sum(p.numel() for p in model.parameters()))

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
            log.info(
                "Epoch %3d | train=%.4f | test=%.4f | acc=%.4f | f1=%.4f",
                ep, tr_loss, val_loss, metrics["accuracy"], metrics["f_score"],
            )
            if metrics["f_score"] > best_f1:
                best_f1 = metrics["f_score"]
                best_metrics = metrics
                stale = 0
            else:
                stale += 1
                if stale >= Cfg.patience:
                    log.info("Early stop at epoch %d (best F1=%.4f)", ep, best_f1)
                    break

        log.info("FOLD %d best: acc=%.4f f1=%.4f auc=%.4f",
                 fold, best_metrics["accuracy"], best_metrics["f_score"], best_metrics["auc"])
        fold_metrics.append(best_metrics)

    keys = ["accuracy", "sensitivity", "precision", "f_score", "auc"]
    agg = {k: (
        float(np.mean([m[k] for m in fold_metrics])),
        float(np.std([m[k] for m in fold_metrics])),
    ) for k in keys}

    out = REPO_ROOT / "results" / "gated_matched_summary.csv"
    with open(out, "w") as f:
        f.write("model,accuracy_avg,accuracy_std,f_score_avg,f_score_std,auc_avg,auc_std\n")
        f.write(
            f"GatedGAMEMal,"
            f"{agg['accuracy'][0]:.4f},{agg['accuracy'][1]:.4f},"
            f"{agg['f_score'][0]:.4f},{agg['f_score'][1]:.4f},"
            f"{agg['auc'][0]:.4f},{agg['auc'][1]:.4f}\n"
        )

    payload = {
        "model": "GAME-Mal gated (matched to ablation)",
        "n_folds": Cfg.n_folds,
        "per_fold": fold_metrics,
        "aggregate": {k: {"mean": v[0], "std": v[1]} for k, v in agg.items()},
    }
    with open(REPO_ROOT / "results" / "gated_matched_summary.json", "w") as f:
        json.dump(payload, f, indent=2, default=float)

    log.info("=" * 60)
    log.info("Gated matched-prep aggregate:")
    for k, (m, s) in agg.items():
        log.info("  %s: %.4f (+/- %.4f)", k, m, s)
    log.info("Saved to %s", out)


if __name__ == "__main__":
    main()
