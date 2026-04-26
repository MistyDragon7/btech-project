"""
BiLSTM baseline — 3-fold stratified CV on the same len>=30 subset
used for the matched-prep ablation (8,085 samples).

Outputs:
    results/bilstm_summary.json   (per-fold + aggregate)

Usage:
    python3 -u scripts/run_bilstm.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import load_dataset, APIVocabulary, pad_with_truncation
from src.bilstm import train_bilstm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bilstm")


SEED = 42
N_FOLDS = 3
DEFAULT_MAX_LEN = 512
DEFAULT_TRUNCATION = "tail"
MIN_FREQ = 2


def best_seq_config():
    """Read best (max_seq_len, truncation) from the sweep summary, falling
    back to the defaults above if the sweep file is absent."""
    import csv
    p = REPO_ROOT / "results" / "seq_len_sweep_summary.csv"
    if not p.exists():
        return DEFAULT_MAX_LEN, DEFAULT_TRUNCATION
    with open(p) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return DEFAULT_MAX_LEN, DEFAULT_TRUNCATION
    rows.sort(key=lambda r: float(r["f_score_mean"]), reverse=True)
    best = rows[0]
    return int(best["max_seq_len"]), best["truncation"]


def main() -> None:
    # Reproducibility: torch RNG was previously unseeded, so model init and
    # dropout varied run-to-run. Seed it here.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED) if hasattr(torch.mps, "manual_seed") else None
    max_len, trunc = best_seq_config()
    log.info("BiLSTM config: max_seq_len=%d  truncation=%s", max_len, trunc)

    log.info("Loading FULL corpus (matches GAME-Mal benchmark)...")
    sequences, labels, family_names = load_dataset(REPO_ROOT / "extracted_data")
    y = np.array(labels)
    log.info("Loaded %d samples across %d families",
             len(sequences), len(family_names))

    vocab = APIVocabulary(min_freq=MIN_FREQ)
    vocab.build(sequences)
    encoded = [vocab.encode_sequence(s) for s in sequences]
    padded = pad_with_truncation(encoded, max_len=max_len, truncation=trunc)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    log.info("Device: %s", device)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    t_start = time.time()
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(padded, y), start=1):
        log.info("=" * 60)
        log.info("FOLD %d/%d   train=%d  test=%d", fold_idx, N_FOLDS, len(tr_idx), len(te_idx))
        log.info("=" * 60)
        X_tr, X_te = padded[tr_idx], padded[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        _, metrics = train_bilstm(
            vocab_size=len(vocab),
            num_classes=len(family_names),
            X_train=X_tr, y_train=y_tr,
            X_test=X_te, y_test=y_te,
            device=device,
        )
        log.info("FOLD %d -> acc=%.4f f1=%.4f auc=%.4f",
                 fold_idx, metrics["accuracy"], metrics["f_score"], metrics["auc"])
        fold_results.append({"fold": fold_idx, **metrics})

    out = {
        "model": "BiLSTM",
        "n_folds": N_FOLDS,
        "max_seq_len": max_len,
        "truncation": trunc,
        "fold_results": fold_results,
    }
    for k in ["accuracy", "sensitivity", "precision", "f_score", "auc"]:
        vals = [r[k] for r in fold_results]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))

    out_path = REPO_ROOT / "results" / "bilstm_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("=" * 60)
    log.info("BiLSTM 3-fold complete in %.1f min", (time.time() - t_start) / 60.0)
    log.info("acc=%.4f +/- %.4f   f1=%.4f +/- %.4f   auc=%.4f +/- %.4f",
             out["accuracy_mean"], out["accuracy_std"],
             out["f_score_mean"], out["f_score_std"],
             out["auc_mean"], out["auc_std"])
    log.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
