"""
Deletion test for explainability faithfulness.

For each malware family we take up to N test samples, compute the gate
activation magnitude for every non-pad token, mask the top-k tokens
(set them to <PAD>=0 so attention/pooling skips them), and re-run the
model. We measure the drop in predicted probability for the true class
between the original and masked inputs.

A faithful explanation should show a meaningful drop. We additionally
compare against a *random* baseline that masks k tokens chosen uniformly
at random per sample.

Outputs:
    results/deletion_test.json
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import load_dataset, pad_sequences
from src.model import GAMEMal


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("deletion")

MODEL_DIR = REPO_ROOT / "results" / "models"
SAMPLES_PER_FAMILY = 30
K_LIST = [5, 10, 20]
PAD_IDX = 0


def load_model(device: torch.device):
    with open(MODEL_DIR / "config.json") as f:
        cfg = json.load(f)
    with open(MODEL_DIR / "vocab.pkl", "rb") as f:
        v = pickle.load(f)
    with open(MODEL_DIR / "family_names.json") as f:
        family_names = json.load(f)
    api2idx = v["api2idx"]

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
    return model, api2idx, family_names, cfg["max_seq_len"]


@torch.no_grad()
def predict_proba(model: GAMEMal, x: torch.Tensor) -> np.ndarray:
    logits, _ = model(x, return_attention=False)
    return torch.softmax(logits, dim=-1).cpu().numpy()


def gate_magnitude_per_token(model: GAMEMal, x: torch.Tensor) -> np.ndarray:
    """Mean (over layers, heads, gate-dims) of sigmoid gate scores per token.

    Returns array of shape (seq_len,). Padded positions get 0.
    """
    with torch.no_grad():
        _, info = model(x, return_attention=True)
    # info["gate_scores"] is list per layer of (B, n_heads, N, d_k) tensors
    per_layer = []
    for g in info["gate_scores"]:
        if g is None:
            continue
        # mean over heads + gate dim → (B, N)
        per_layer.append(g.mean(dim=(1, 3)))
    if not per_layer:
        N = x.shape[1]
        return np.zeros(N, dtype=np.float32)
    stacked = torch.stack(per_layer, dim=0).mean(dim=0)  # (B, N)
    mag = stacked.squeeze(0).cpu().numpy()
    pad_positions = (x.squeeze(0) == PAD_IDX).cpu().numpy()
    mag[pad_positions] = 0.0
    return mag


def mask_topk(seq: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    """Set the top-k highest-scoring (non-pad) positions to PAD."""
    out = seq.copy()
    nonpad = np.where(seq != PAD_IDX)[0]
    if len(nonpad) == 0:
        return out
    k_eff = min(k, len(nonpad))
    # rank non-pad positions by score desc
    np_scores = scores[nonpad]
    top_local = np.argsort(-np_scores)[:k_eff]
    top_global = nonpad[top_local]
    out[top_global] = PAD_IDX
    return out


def mask_random(seq: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    out = seq.copy()
    nonpad = np.where(seq != PAD_IDX)[0]
    if len(nonpad) == 0:
        return out
    k_eff = min(k, len(nonpad))
    chosen = rng.choice(nonpad, size=k_eff, replace=False)
    out[chosen] = PAD_IDX
    return out


def main() -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    log.info("Device: %s", device)

    model, api2idx, family_names, max_len = load_model(device)
    unk = api2idx.get("<UNK>", 1)

    log.info("Loading corpus...")
    sequences, labels, _ = load_dataset(REPO_ROOT / "extracted_data")
    keep = [i for i, s in enumerate(sequences) if len(s) >= 30]
    sequences = [sequences[i] for i in keep]
    labels = [labels[i] for i in keep]
    encoded = [[api2idx.get(tok, unk) for tok in seq] for seq in sequences]
    padded = pad_sequences(encoded, max_len=max_len)
    y = np.array(labels)

    rng = np.random.default_rng(42)

    per_family: Dict[str, Dict] = {}
    overall = defaultdict(lambda: defaultdict(list))  # k -> mode -> list of deltas

    for fam_idx, fam_name in enumerate(family_names):
        fam_mask = y == fam_idx
        fam_pos = np.where(fam_mask)[0]
        if len(fam_pos) == 0:
            continue
        take = min(SAMPLES_PER_FAMILY, len(fam_pos))
        sampled = rng.choice(fam_pos, size=take, replace=False)

        per_k: Dict[str, Dict] = {}
        for k in K_LIST:
            base_probs: List[float] = []
            gate_probs: List[float] = []
            rand_probs: List[float] = []

            for idx in sampled:
                seq = padded[idx]
                x = torch.from_numpy(seq[None, :]).long().to(device)
                gates = gate_magnitude_per_token(model, x)

                p_base = predict_proba(model, x)[0, fam_idx]

                masked_gate = mask_topk(seq, gates, k)
                xg = torch.from_numpy(masked_gate[None, :]).long().to(device)
                p_gate = predict_proba(model, xg)[0, fam_idx]

                masked_rand = mask_random(seq, k, rng)
                xr = torch.from_numpy(masked_rand[None, :]).long().to(device)
                p_rand = predict_proba(model, xr)[0, fam_idx]

                base_probs.append(float(p_base))
                gate_probs.append(float(p_gate))
                rand_probs.append(float(p_rand))

            base_arr = np.array(base_probs)
            gate_arr = np.array(gate_probs)
            rand_arr = np.array(rand_probs)
            d_gate = base_arr - gate_arr
            d_rand = base_arr - rand_arr

            per_k[f"k_{k}"] = {
                "n_samples": int(take),
                "mean_p_base": float(base_arr.mean()),
                "mean_p_after_gate_mask": float(gate_arr.mean()),
                "mean_p_after_random_mask": float(rand_arr.mean()),
                "mean_delta_gate": float(d_gate.mean()),
                "mean_delta_random": float(d_rand.mean()),
                "delta_gate_minus_random": float(d_gate.mean() - d_rand.mean()),
            }
            overall[k]["gate"].extend(d_gate.tolist())
            overall[k]["random"].extend(d_rand.tolist())

            log.info(
                "[%s] k=%2d  base=%.3f  gate->%.3f (Δ=%.3f)  rand->%.3f (Δ=%.3f)",
                fam_name, k,
                base_arr.mean(), gate_arr.mean(), d_gate.mean(),
                rand_arr.mean(), d_rand.mean(),
            )
        per_family[fam_name] = per_k

    overall_summary = {}
    for k in K_LIST:
        g = np.array(overall[k]["gate"])
        r = np.array(overall[k]["random"])
        overall_summary[f"k_{k}"] = {
            "n_samples": int(len(g)),
            "mean_delta_gate": float(g.mean()),
            "mean_delta_random": float(r.mean()),
            "delta_gate_minus_random": float(g.mean() - r.mean()),
        }

    out = {
        "samples_per_family": SAMPLES_PER_FAMILY,
        "k_values": K_LIST,
        "per_family": per_family,
        "overall": overall_summary,
        "notes": (
            "Mean drop in predicted probability for the true class after "
            "masking top-k highest-gate tokens vs k random tokens. Larger "
            "delta_gate_minus_random indicates the gate identifies tokens "
            "that the model genuinely relies on (faithful explanation)."
        ),
    }
    out_path = REPO_ROOT / "results" / "deletion_test.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved %s", out_path)
    for k in K_LIST:
        s = overall_summary[f"k_{k}"]
        log.info(
            "OVERALL k=%2d  Δgate=%.4f  Δrand=%.4f  diff=%.4f",
            k, s["mean_delta_gate"], s["mean_delta_random"], s["delta_gate_minus_random"],
        )


if __name__ == "__main__":
    main()
