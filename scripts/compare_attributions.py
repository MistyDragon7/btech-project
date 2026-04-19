"""
Compare gate-activation top-APIs per family against a post-hoc
input-gradient saliency attribution over the same model.

We use embedding-gradient-times-embedding (a standard baseline
attribution for transformers; also known as GradientInput). For each
test sample correctly classified into family c, we compute:

    saliency_t = || d(logit_c) / d(E[t]) * E[t] ||_1
    (summed over the embedding dimension, taken at the embedding lookup)

We then rank APIs by mean saliency over family-c samples and compare
against `results/top_apis_per_family.json` (the gate ranking). The
metric is top-5 and top-10 overlap percentage.

Requires a trained model at results/models/game_mal_best.pt.
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
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import load_dataset, APIVocabulary, pad_sequences
from src.model import GAMEMal


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("attrib")

MODEL_DIR = REPO_ROOT / "results" / "models"
SAMPLES_PER_FAMILY = 30  # keep runtime bounded
TOP_K_LIST = [5, 10]


def load_model(device: torch.device) -> Tuple[GAMEMal, Dict, Dict, List[str]]:
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
    return model, api2idx, idx2api, family_names


def input_gradient_saliency(
    model: GAMEMal, x: torch.Tensor, target_class: int
) -> np.ndarray:
    """Compute ||grad_E * E||_1 per token for a single sample."""
    # Embed manually so we can attach grads to the embedding output
    B, N = x.shape
    pad_mask = (x == 0)
    positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
    emb = model.api_embedding(x)
    emb.retain_grad()
    pos_emb = model.pos_embedding(positions)
    h = model.embed_dropout(emb + pos_emb)

    for block in model.blocks:
        h, _, _ = block(h, pad_mask, return_attention=False)
    h = model.final_norm(h)
    mask_expanded = (~pad_mask).unsqueeze(-1).float()
    h_sum = (h * mask_expanded).sum(dim=1)
    lengths = mask_expanded.sum(dim=1).clamp(min=1)
    h_pooled = h_sum / lengths
    logits = model.classifier(h_pooled)

    model.zero_grad()
    logits[0, target_class].backward()
    grad = emb.grad  # (1, N, D)
    saliency = (grad * emb.detach()).abs().sum(dim=-1).squeeze(0)  # (N,)
    return saliency.cpu().numpy()


def rank_apis_by_saliency(
    model: GAMEMal,
    X_fam: np.ndarray,
    target_class: int,
    idx2api: Dict[int, str],
    device: torch.device,
) -> List[Tuple[str, float]]:
    """Average token saliency across a family's samples, rank APIs."""
    token_total = defaultdict(float)
    token_count = defaultdict(int)
    for i in range(len(X_fam)):
        x = torch.from_numpy(X_fam[i : i + 1]).long().to(device)
        sal = input_gradient_saliency(model, x, target_class)
        seq = X_fam[i]
        for pos, tok in enumerate(seq):
            if tok == 0:  # pad
                continue
            token_total[int(tok)] += float(sal[pos])
            token_count[int(tok)] += 1
    means = []
    for tok, total in token_total.items():
        if token_count[tok] >= 3:  # require at least 3 occurrences
            means.append((idx2api.get(tok, f"<{tok}>"), total / token_count[tok]))
    means.sort(key=lambda p: -p[1])
    return means


def overlap_at_k(list_a: List[str], list_b: List[str], k: int) -> float:
    top_a = set(list_a[:k])
    top_b = set(list_b[:k])
    if not top_a:
        return 0.0
    return len(top_a & top_b) / k


def main() -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    log.info("Device: %s", device)

    log.info("Loading model from %s", MODEL_DIR)
    model, api2idx, idx2api, family_names = load_model(device)

    # Rebuild eval data (same recipe as training: min-length filter)
    log.info("Loading corpus...")
    sequences, labels, _ = load_dataset(REPO_ROOT / "extracted_data")
    keep = [i for i, s in enumerate(sequences) if len(s) >= 30]
    sequences = [sequences[i] for i in keep]
    labels = [labels[i] for i in keep]

    # encode using saved vocab
    pad = api2idx.get("<PAD>", 0)
    unk = api2idx.get("<UNK>", 1)
    encoded = [
        [api2idx.get(tok, unk) for tok in seq] for seq in sequences
    ]
    padded = pad_sequences(encoded, max_len=512)
    y = np.array(labels)

    # Gate-based top-APIs
    with open(REPO_ROOT / "results" / "top_apis_per_family.json") as f:
        gate_top = json.load(f)  # {family: [[api, score], ...]}

    overlaps_by_family = {}
    saliency_top = {}
    for fam_idx, fam_name in enumerate(family_names):
        fam_mask = y == fam_idx
        fam_idx_list = np.where(fam_mask)[0]
        if len(fam_idx_list) == 0:
            continue
        take = min(SAMPLES_PER_FAMILY, len(fam_idx_list))
        rng = np.random.default_rng(42 + fam_idx)
        sampled = rng.choice(fam_idx_list, size=take, replace=False)
        X_fam = padded[sampled]

        log.info("[%s] Computing saliency over %d samples...", fam_name, take)
        sal_ranked = rank_apis_by_saliency(model, X_fam, fam_idx, idx2api, device)
        saliency_top[fam_name] = [{"api": a, "saliency": s} for a, s in sal_ranked[:15]]

        sal_apis = [a for a, _ in sal_ranked]
        gate_apis = [entry[0] for entry in gate_top.get(fam_name, [])]

        ov = {f"top_{k}": overlap_at_k(gate_apis, sal_apis, k) for k in TOP_K_LIST}
        overlaps_by_family[fam_name] = ov
        log.info("  overlap @5=%.2f  @10=%.2f", ov["top_5"], ov["top_10"])

    mean_overlap_5 = float(np.mean([v["top_5"] for v in overlaps_by_family.values()]))
    mean_overlap_10 = float(np.mean([v["top_10"] for v in overlaps_by_family.values()]))

    out = {
        "samples_per_family": SAMPLES_PER_FAMILY,
        "per_family_overlap": overlaps_by_family,
        "mean_top5_overlap": mean_overlap_5,
        "mean_top10_overlap": mean_overlap_10,
        "saliency_top_apis_per_family": saliency_top,
    }
    out_path = REPO_ROOT / "results" / "attribution_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("=" * 60)
    log.info("Mean top-5 overlap (gate vs input-gradient):  %.2f", mean_overlap_5)
    log.info("Mean top-10 overlap (gate vs input-gradient): %.2f", mean_overlap_10)
    log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
