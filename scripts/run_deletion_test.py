"""
Deletion test for explainability faithfulness (CLS-attention).

For each malware family we take up to N test samples, compute a per-token
importance score derived from attention weights *from the class token* ([CLS])
to all non-pad tokens, mask the top-k tokens (set them to <PAD>=0 so attention
skips them), and re-run the model.

We measure the drop in predicted probability for the true class between the
original and masked inputs.

A faithful explanation should show a meaningful drop. We additionally compare
against a *random* baseline that masks k tokens chosen uniformly at random per
sample.

IMPORTANT:
- This test assumes the model has an explicit [CLS] token at position 0, and
  that `return_attention=True` returns per-layer attention matrices of shape
  (B, n_heads, seq, seq).

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

from src.model import MalwareTransformer
from src.preprocessing import load_dataset, pad_sequences

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
    with open(MODEL_DIR / "plain_transformer_config.json") as f:
        cfg = json.load(f)
    with open(MODEL_DIR / "vocab.pkl", "rb") as f:
        v = pickle.load(f)
    with open(MODEL_DIR / "family_names.json") as f:
        family_names = json.load(f)
    api2idx = v["api2idx"]

    # NOTE: gating has been removed from the project; deletion test now uses CLS-attention.
    model = MalwareTransformer(
        vocab_size=cfg["vocab_size"],
        num_classes=cfg["num_classes"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
    ).to(device)
    state = torch.load(MODEL_DIR / "plain_transformer_best.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, api2idx, family_names, cfg["max_seq_len"]


@torch.no_grad()
def predict_proba(model: MalwareTransformer, x: torch.Tensor) -> np.ndarray:
    logits, _ = model(x, return_attention=False)
    return torch.softmax(logits, dim=-1).cpu().numpy()


def cls_attention_importance_per_token(model: MalwareTransformer, x: torch.Tensor) -> np.ndarray:
    """Derive per-token importance from attention from [CLS] to tokens.

    We take attention weights from the [CLS] query position (index 0) to all
    key positions, average across heads, and then average across layers.

    Returns array of shape (seq_len,). Padded positions get 0.
    """
    with torch.no_grad():
        _, info = model(x, return_attention=True)

    # info["attn_weights"] is list per layer of (B, n_heads, N, N)
    per_layer = []
    for a in info.get("attn_weights", []):
        if a is None:
            continue
        # a[:, :, cls, :] -> (B, n_heads, N)
        cls_row = a[:, :, 0, :]
        # mean over heads -> (B, N)
        per_layer.append(cls_row.mean(dim=1))

    if not per_layer:
        N = x.shape[1]
        return np.zeros(N, dtype=np.float32)

    stacked = torch.stack(per_layer, dim=0).mean(dim=0)  # (B, N)
    imp = stacked.squeeze(0).cpu().numpy()

    pad_positions = (x.squeeze(0) == PAD_IDX).cpu().numpy()
    imp[pad_positions] = 0.0
    # Never allow [CLS] itself to be masked
    if len(imp) > 0:
        imp[0] = 0.0
    return imp


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
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
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
                attn_imp = cls_attention_importance_per_token(model, x)

                p_base = predict_proba(model, x)[0, fam_idx]

                masked_top = mask_topk(seq, attn_imp, k)
                xg = torch.from_numpy(masked_top[None, :]).long().to(device)
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
                "[%s] k=%2d  base=%.3f  top-attn->%.3f (Δ=%.3f)  rand->%.3f (Δ=%.3f)",
                fam_name,
                k,
                base_arr.mean(),
                gate_arr.mean(),
                d_gate.mean(),
                rand_arr.mean(),
                d_rand.mean(),
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
            "masking top-k CLS-attention-ranked tokens vs k random tokens. Larger "
            "delta_gate_minus_random indicates the attention map identifies tokens "
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
            k,
            s["mean_delta_gate"],
            s["mean_delta_random"],
            s["delta_gate_minus_random"],
        )


if __name__ == "__main__":
    main()
