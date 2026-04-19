"""
Minimal inference example: load the trained GAME-Mal checkpoint and
classify one sample. Also prints the top-k APIs by gate activation
for that sample (per-sample explanation).

    python3 scripts/predict_demo.py [optional path to a .apk JSONL trace]

If no path is given, a random sample from extracted_data/ is used.
"""
from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import load_family_samples, APIVocabulary, pad_sequences
from src.model import GAMEMal

MODEL_DIR = REPO_ROOT / "results" / "models"


def load_checkpoint(device: torch.device):
    with open(MODEL_DIR / "config.json") as f:
        cfg = json.load(f)
    with open(MODEL_DIR / "vocab.pkl", "rb") as f:
        v = pickle.load(f)
    with open(MODEL_DIR / "family_names.json") as f:
        family_names = json.load(f)

    api2idx = v["api2idx"]
    idx2api = {i: a for a, i in api2idx.items()}

    model = GAMEMal(
        vocab_size=cfg["vocab_size"],
        num_classes=cfg["num_classes"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
        use_gate=True,
    ).to(device)
    state = torch.load(MODEL_DIR / "game_mal_best.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, api2idx, idx2api, family_names, cfg


def main() -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model, api2idx, idx2api, family_names, cfg = load_checkpoint(device)

    # Pick or read a sample
    if len(sys.argv) > 1:
        sample_path = Path(sys.argv[1])
        family_dir = sample_path.parent
        true_family = family_dir.name
        seqs = load_family_samples(family_dir)
        # Just take the first sequence from this directory; good enough for a demo
        seq = seqs[0]
    else:
        families = [d for d in (REPO_ROOT / "extracted_data").iterdir() if d.is_dir()]
        fam = random.choice(families)
        true_family = fam.name
        seq = load_family_samples(fam)[0]

    pad = api2idx.get("<PAD>", 0)
    unk = api2idx.get("<UNK>", 1)
    encoded = [api2idx.get(tok, unk) for tok in seq]
    padded = pad_sequences([encoded], max_len=cfg["max_seq_len"])
    x = torch.from_numpy(padded).long().to(device)

    with torch.no_grad():
        logits, info = model(x, return_attention=True)
        probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze(0)

    pred_idx = int(probs.argmax())
    print(f"True family:      {true_family}")
    print(f"Predicted family: {family_names[pred_idx]}  (p={probs[pred_idx]:.3f})")
    print("\nTop-3 predictions:")
    for i in probs.argsort()[::-1][:3]:
        print(f"  {family_names[int(i)]:<14} {probs[int(i)]:.3f}")

    # Per-sample explanation: top tokens by mean gate activation across heads+layers
    gate_tensors = [g for g in info["gate_scores"] if g is not None]
    if gate_tensors:
        # shape per layer: (1, n_heads, seq_len, d_k); mean over heads, d_k, layers
        stacked = torch.stack(gate_tensors, dim=0)  # (L, 1, H, N, d_k)
        mean_gate = stacked.mean(dim=(0, 1, 2, -1)).cpu().numpy()  # (N,)
        tokens = padded.squeeze(0)
        # rank non-pad tokens
        valid = [(int(tok), float(mean_gate[i])) for i, tok in enumerate(tokens) if tok != pad]
        # unique APIs with their mean gate (over positions)
        by_api = {}
        counts = {}
        for tok, g in valid:
            by_api.setdefault(tok, 0.0)
            by_api[tok] += g
            counts[tok] = counts.get(tok, 0) + 1
        ranked = sorted(
            ((idx2api.get(t, f"<{t}>"), by_api[t] / counts[t]) for t in by_api),
            key=lambda p: -p[1],
        )
        print("\nTop-5 gate-activated APIs in this sample:")
        for api, g in ranked[:5]:
            print(f"  {g:.3f}  {api}")


if __name__ == "__main__":
    main()
