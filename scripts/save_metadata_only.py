"""
Build and save the vocab + config metadata accompanying
`results/models/game_mal_best.pt` (checkpoint saved early by a paused
training run). Lets the checkpoint be reloaded for inference.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import load_dataset, APIVocabulary

MODEL_DIR = REPO_ROOT / "results" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

sequences, labels, family_names = load_dataset(REPO_ROOT / "extracted_data")
# same min-length filter as training
sequences = [s for s in sequences if len(s) >= 30]

vocab = APIVocabulary(min_freq=2)
vocab.build(sequences)

with open(MODEL_DIR / "vocab.pkl", "wb") as f:
    pickle.dump({"api2idx": vocab.api2idx, "idx2api": vocab.idx2api, "min_freq": vocab.min_freq}, f)
with open(MODEL_DIR / "family_names.json", "w") as f:
    json.dump(family_names, f, indent=2)
with open(MODEL_DIR / "config.json", "w") as f:
    json.dump(
        {
            "vocab_size": len(vocab),
            "num_classes": len(family_names),
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 256,
            "max_seq_len": 512,
            "dropout": 0.1,
            "notes": (
                "Checkpoint saved at epoch 2 of a paused 30-epoch run (F1=0.77). "
                "For the published 3-fold results (F1=0.886), rerun "
                "scripts/train_final.py to completion."
            ),
        },
        f,
        indent=2,
    )
print(f"Wrote metadata to {MODEL_DIR}")
