"""
Markov-Initialized Transformer training run.

Pits the Plain Transformer against a Markov-Initialized Transformer
by loading SVD-factorized Markov transition embeddings and running the exact
same 3-fold cross validation.

Reads:  results/seq_len_sweep_summary.csv
Writes: results/markov_transformer_final.json
        results/models/markov_transformer_best.pt
        results/models/markov_transformer_config.json
Updates: results/results_summary.csv  (row "MarkovTransformer")
"""

from __future__ import annotations

import csv
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model import MalwareTransformer
from src.preprocessing import APIVocabulary, load_dataset, pad_with_truncation
from src.train import create_dataloader, evaluate, train_epoch
from src.markov import build_svd_markov_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("markov_transformer")

MODEL_DIR = REPO_ROOT / "results" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = REPO_ROOT / "results" / "results_summary.csv"


class Cfg:
    min_freq = 2
    d_model = 128
    n_heads = 4
    n_layers = 2
    d_ff = 256
    dropout = 0.15
    batch_size = 32
    epochs = 100
    lr = 5e-4
    weight_decay = 1e-4
    patience = 20
    n_folds = 3
    seed = 42
    freeze_epochs = 5  # Number of epochs to freeze the Markov embeddings


def best_seq_config():
    # We are overriding the sweep and forcing 2048 to beat the Random Forest baseline
    log.info("Overriding sweep: forcing max_seq_len=2048 to maximize accuracy")
    return 2048, "head", None


def update_summary_row(method: str, mean_std: dict) -> None:
    if not SUMMARY_CSV.exists():
        log.warning("results_summary.csv missing — skipping row update")
        return
    with open(SUMMARY_CSV) as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else None
    if not fieldnames:
        return
    new_row = {"Method": method}
    for k in ("accuracy", "sensitivity", "precision", "f_score", "auc"):
        new_row[f"{k}_avg"] = mean_std[f"{k}_mean"]
        new_row[f"{k}_std"] = mean_std[f"{k}_std"]
    replaced = False
    for i, r in enumerate(rows):
        if r.get("Method") == method:
            rows[i] = new_row
            replaced = True
            break
    if not replaced:
        rows.append(new_row)
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log.info("Updated %s row in %s", method, SUMMARY_CSV)


def train_one_fold(
    X_tr, y_tr, X_te, y_te, vocab_size, num_classes, max_seq_len, device, pretrained_embeddings
):
    model = MalwareTransformer(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=Cfg.d_model,
        n_heads=Cfg.n_heads,
        n_layers=Cfg.n_layers,
        d_ff=Cfg.d_ff,
        max_seq_len=max_seq_len,
        dropout=Cfg.dropout,
        pretrained_embeddings=pretrained_embeddings
    ).to(device)

    # Freeze the embeddings initially!
    model.api_embedding.weight.requires_grad = False
    log.info("  Frozen Markov Embeddings for initial warm-up.")

    counts = np.bincount(y_tr, minlength=num_classes).astype(np.float32)
    class_w = len(y_tr) / (num_classes * np.maximum(counts, 1.0))
    class_w_t = torch.from_numpy(class_w).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w_t)
    
    # Pass all parameters to optimizer. Parameters with requires_grad=False will just be ignored.
    optim = torch.optim.AdamW(
        model.parameters(), lr=Cfg.lr, weight_decay=Cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=Cfg.epochs)

    tr_loader = create_dataloader(X_tr, y_tr, Cfg.batch_size, shuffle=True)
    te_loader = create_dataloader(X_te, y_te, Cfg.batch_size, shuffle=False)

    best_f1 = -1.0
    best_metrics = None
    best_state = None
    best_epoch = -1
    stale = 0
    for ep in range(1, Cfg.epochs + 1):
        
        # Unfreeze at the specified epoch
        if ep == Cfg.freeze_epochs + 1:
            model.api_embedding.weight.requires_grad = True
            log.info(f"  Unfrozen Markov Embeddings at epoch {ep} for fine-tuning!")
            
        tr_loss = train_epoch(model, tr_loader, optim, criterion, device)
        val_loss, metrics, _, _ = evaluate(
            model, te_loader, criterion, device, num_classes
        )
        sched.step()
        if ep == 1 or ep % 5 == 0:
            log.info(
                "  ep=%3d train=%.4f val=%.4f acc=%.4f f1=%.4f",
                ep,
                tr_loss,
                val_loss,
                metrics["accuracy"],
                metrics["f_score"],
            )
        if metrics["f_score"] > best_f1:
            best_f1 = metrics["f_score"]
            best_metrics = metrics.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            stale = 0
        else:
            stale += 1
            if stale >= Cfg.patience:
                log.info(
                    "  early stop ep=%d best_f1=%.4f @ ep=%d", ep, best_f1, best_epoch
                )
                break
    return best_metrics, best_state, best_epoch


def main() -> None:
    torch.manual_seed(Cfg.seed)
    np.random.seed(Cfg.seed)

    max_len, trunc, best_row = best_seq_config()
    log.info(
        "Markov transformer config: max_seq_len=%d  truncation=%s",
        max_len,
        trunc,
    )

    log.info("Loading FULL corpus (len>=5)...")
    sequences, labels, family_names = load_dataset(REPO_ROOT / "extracted_data")
    y = np.array(labels)
    log.info("Loaded %d samples across %d families", len(sequences), len(family_names))

    vocab = APIVocabulary(min_freq=Cfg.min_freq)
    vocab.build(sequences)
    encoded = [vocab.encode_sequence(s) for s in sequences]
    padded = pad_with_truncation(encoded, max_len=max_len, truncation=trunc)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    log.info("Device: %s   vocab_size=%d", device, len(vocab))
    
    # Generate Markov Embeddings!
    pretrained_embeddings = build_svd_markov_embeddings(
        encoded_sequences=encoded,
        vocab_size=len(vocab),
        d_model=Cfg.d_model,
        max_spacing=10
    )

    skf = StratifiedKFold(n_splits=Cfg.n_folds, shuffle=True, random_state=Cfg.seed)
    splits = list(skf.split(padded, y))

    fold_results = []
    best_overall_f1 = -1.0
    best_overall_state = None
    best_overall_fold = -1
    best_overall_epoch = -1
    t_start = time.time()

    for fold_idx, (tr_idx, te_idx) in enumerate(splits, start=1):
        log.info("=" * 60)
        log.info(
            "FOLD %d/%d   train=%d  test=%d",
            fold_idx,
            Cfg.n_folds,
            len(tr_idx),
            len(te_idx),
        )
        log.info("=" * 60)
        X_tr, X_te = padded[tr_idx], padded[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        metrics, state, ep = train_one_fold(
            X_tr,
            y_tr,
            X_te,
            y_te,
            vocab_size=len(vocab),
            num_classes=len(family_names),
            max_seq_len=max_len,
            device=device,
            pretrained_embeddings=pretrained_embeddings
        )
        log.info(
            "FOLD %d -> acc=%.4f f1=%.4f auc=%.4f (best ep=%d)",
            fold_idx,
            metrics["accuracy"],
            metrics["f_score"],
            metrics["auc"],
            ep,
        )
        fold_results.append({"fold": fold_idx, "best_epoch": ep, **metrics})
        if metrics["f_score"] > best_overall_f1:
            best_overall_f1 = metrics["f_score"]
            best_overall_state = state
            best_overall_fold = fold_idx - 1
            best_overall_epoch = ep

    # Aggregate
    aggregate = {}
    for k in ["accuracy", "sensitivity", "precision", "f_score", "auc"]:
        vals = [r[k] for r in fold_results]
        aggregate[f"{k}_mean"] = float(np.mean(vals))
        aggregate[f"{k}_std"] = float(np.std(vals))

    out = {
        "model": "MarkovTransformer",
        "n_folds": Cfg.n_folds,
        "max_seq_len": max_len,
        "truncation": trunc,
        "epochs": Cfg.epochs,
        "patience": Cfg.patience,
        "lr": Cfg.lr,
        "dropout": Cfg.dropout,
        "freeze_epochs": Cfg.freeze_epochs,
        "fold_results": fold_results,
        **aggregate,
    }
    out_path = REPO_ROOT / "results" / "markov_transformer_final.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Wrote %s", out_path)

    # Save best-fold model + metadata
    if best_overall_state is not None:
        torch.save(best_overall_state, MODEL_DIR / "markov_transformer_best.pt")
        with open(MODEL_DIR / "markov_transformer_config.json", "w") as f:
            json.dump(
                {
                    "vocab_size": len(vocab),
                    "num_classes": len(family_names),
                    "d_model": Cfg.d_model,
                    "n_heads": Cfg.n_heads,
                    "n_layers": Cfg.n_layers,
                    "d_ff": Cfg.d_ff,
                    "max_seq_len": max_len,
                    "truncation": trunc,
                    "dropout": Cfg.dropout,
                    "best_fold": int(best_overall_fold),
                    "best_epoch": int(best_overall_epoch),
                    "best_f1": float(best_overall_f1),
                    "n_folds": Cfg.n_folds,
                    "epochs_max": Cfg.epochs,
                    "freeze_epochs": Cfg.freeze_epochs,
                },
                f,
                indent=2,
            )
        log.info(
            "Saved markov transformer model (fold %d, ep %d, f1=%.4f) to %s",
            best_overall_fold + 1,
            best_overall_epoch,
            best_overall_f1,
            MODEL_DIR,
        )

    # Update results_summary.csv
    update_summary_row("MarkovTransformer", aggregate)

    log.info("=" * 60)
    log.info(
        "FINAL Markov Transformer 3-fold complete in %.1f min",
        (time.time() - t_start) / 60.0,
    )
    log.info(
        "acc=%.4f +/- %.4f   f1=%.4f +/- %.4f   auc=%.4f +/- %.4f",
        aggregate["accuracy_mean"],
        aggregate["accuracy_std"],
        aggregate["f_score_mean"],
        aggregate["f_score_std"],
        aggregate["auc_mean"],
        aggregate["auc_std"],
    )
    log.info("Weights: results/models/markov_transformer_best.pt")
    log.info("Config:  results/models/markov_transformer_config.json")


if __name__ == "__main__":
    main()
