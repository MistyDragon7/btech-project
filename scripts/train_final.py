"""
Train a single final GAME-Mal model on a 90/10 split of the full dataset
and save weights + metadata for downstream inference / explainability.

Run from repo root:
    python3 -u scripts/train_final.py
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Make sure `src` package is importable when run from project root.
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
log = logging.getLogger("train_final")


MODEL_DIR = REPO_ROOT / "results" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class Cfg:
    min_freq = 2
    max_seq_len = 512
    d_model = 128
    n_heads = 4
    n_layers = 2
    d_ff = 256
    dropout = 0.1
    batch_size = 32
    epochs = 30
    lr = 3e-4
    weight_decay = 1e-4
    patience = 10
    test_frac = 0.1
    seed = 42


def main() -> None:
    log.info("Loading corpus...")
    sequences, labels, family_names = load_dataset(REPO_ROOT / "extracted_data")
    # Apply min-length filter to match the main pipeline
    min_len = 30
    keep = [i for i, s in enumerate(sequences) if len(s) >= min_len]
    sequences = [sequences[i] for i in keep]
    labels = [labels[i] for i in keep]
    log.info(
        "Loaded %d samples across %d families: %s",
        len(sequences),
        len(family_names),
        family_names,
    )

    log.info("Building vocabulary (min_freq=%d)...", Cfg.min_freq)
    vocab = APIVocabulary(min_freq=Cfg.min_freq)
    vocab.build(sequences)
    log.info("Vocab size: %d", len(vocab))

    encoded = [vocab.encode_sequence(s) for s in sequences]
    padded = pad_sequences(encoded, max_len=Cfg.max_seq_len)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        padded, y, test_size=Cfg.test_frac, stratify=y, random_state=Cfg.seed
    )
    log.info("Train %d | Test %d", len(X_train), len(X_test))

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    log.info("Device: %s", device)

    num_classes = len(family_names)
    model = GAMEMal(
        vocab_size=len(vocab),
        num_classes=num_classes,
        d_model=Cfg.d_model,
        n_heads=Cfg.n_heads,
        n_layers=Cfg.n_layers,
        d_ff=Cfg.d_ff,
        max_seq_len=Cfg.max_seq_len,
        dropout=Cfg.dropout,
    ).to(device)

    # Class-weighted loss
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_w = len(y_train) / (num_classes * counts)
    class_w_t = torch.from_numpy(class_w).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w_t)

    optim = torch.optim.AdamW(model.parameters(), lr=Cfg.lr, weight_decay=Cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=Cfg.epochs)

    train_loader = create_dataloader(X_train, y_train, batch_size=Cfg.batch_size, shuffle=True)
    test_loader = create_dataloader(X_test, y_test, batch_size=Cfg.batch_size, shuffle=False)

    best_f1 = -1.0
    best_epoch = -1
    stale = 0
    for ep in range(1, Cfg.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optim, criterion, device)
        val_loss, metrics, _, _ = evaluate(
            model, test_loader, criterion, device, num_classes
        )
        sched.step()
        log.info(
            "Epoch %3d | train=%.4f | test=%.4f | acc=%.4f | f1=%.4f",
            ep,
            tr_loss,
            val_loss,
            metrics["accuracy"],
            metrics["f_score"],
        )
        if metrics["f_score"] > best_f1:
            best_f1 = metrics["f_score"]
            best_epoch = ep
            stale = 0
            torch.save(model.state_dict(), MODEL_DIR / "game_mal_best.pt")
        else:
            stale += 1
            if stale >= Cfg.patience:
                log.info("Early stopping at epoch %d (best=%d, f1=%.4f)", ep, best_epoch, best_f1)
                break

    # persist vocab + metadata
    with open(MODEL_DIR / "vocab.pkl", "wb") as f:
        pickle.dump({"api2idx": vocab.api2idx, "idx2api": vocab.idx2api, "min_freq": vocab.min_freq}, f)
    with open(MODEL_DIR / "family_names.json", "w") as f:
        json.dump(family_names, f, indent=2)
    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump(
            {
                "vocab_size": len(vocab),
                "num_classes": num_classes,
                "d_model": Cfg.d_model,
                "n_heads": Cfg.n_heads,
                "n_layers": Cfg.n_layers,
                "d_ff": Cfg.d_ff,
                "max_seq_len": Cfg.max_seq_len,
                "dropout": Cfg.dropout,
                "best_epoch": int(best_epoch),
                "best_f1": float(best_f1),
                "train_size": int(len(y_train)),
                "test_size": int(len(y_test)),
            },
            f,
            indent=2,
        )
    log.info("Saved model + metadata to %s (best F1=%.4f @ epoch %d)", MODEL_DIR, best_f1, best_epoch)


if __name__ == "__main__":
    main()
