"""
visualize_attention.py
======================

CLS-attention visualizations (gating-free).

Assumptions:
- The model prepends a learnable [CLS] token at index 0.
- `return_attention=True` returns per-layer attention weights shaped (B, n_heads, S, S).
- We interpret token importance via attention from CLS -> tokens.

Outputs go to results/figures/.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.model import MalwareTransformer
from src.preprocessing import load_dataset, pad_with_truncation

RESULTS = REPO / "results"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"
FIGURES.mkdir(parents=True, exist_ok=True)

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {DEVICE}")


def load_model_and_vocab():
    cfg = json.loads((MODELS / "plain_transformer_config.json").read_text())
    with open(MODELS / "vocab.pkl", "rb") as f:
        vocab_obj = pickle.load(f)

    # vocab.pkl may store APIVocabulary or a dict wrapper
    if isinstance(vocab_obj, dict) and "api2idx" in vocab_obj:
        vocab = vocab_obj["api2idx"]
    elif hasattr(vocab_obj, "api2idx"):
        vocab = vocab_obj.api2idx
    else:
        vocab = vocab_obj

    family_names = json.loads((MODELS / "family_names.json").read_text())

    model = MalwareTransformer(
        vocab_size=cfg["vocab_size"],
        num_classes=cfg["num_classes"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        dropout=0.0,
    )
    ckpt = torch.load(MODELS / "plain_transformer_best.pt", map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, vocab, family_names, cfg


def encode_sequences(seqs, vocab, max_seq_len, truncation="head"):
    encoded = []
    unk_id = vocab.get("<UNK>", 1)
    for s in seqs:
        ids = [vocab.get(t, unk_id) for t in s]
        encoded.append(ids)
    return pad_with_truncation(encoded, max_seq_len, truncation=truncation)


def get_attention(model, token_ids: torch.Tensor):
    with torch.no_grad():
        logits, info = model(token_ids.to(DEVICE), return_attention=True)
    attn_list = info.get("attn_weights", [])
    attn = [a.detach().cpu() for a in attn_list if a is not None]
    token_ids_with_cls = info.get("token_ids_with_cls", None)
    if token_ids_with_cls is None:
        # Fallback: prepend dummy CLS (PAD=0)
        bsz = token_ids.shape[0]
        cls_col = torch.zeros((bsz, 1), dtype=token_ids.dtype)
        token_ids_with_cls = torch.cat([cls_col, token_ids], dim=1)
    return logits.detach().cpu(), attn, token_ids_with_cls.cpu()


def cls_attention_score(
    attn_list: List[torch.Tensor],
    token_ids_with_cls: torch.Tensor,
    cls_index: int = 0,
    pad_idx: int = 0,
) -> torch.Tensor:
    if not attn_list:
        raise ValueError("No attention weights returned by model.")

    stacked = torch.stack(attn_list, dim=0)  # (L, B, H, S, S)
    cls_rows = stacked[:, :, :, cls_index, :]  # (L, B, H, S)
    scores = cls_rows.mean(dim=(0, 2))  # (B, S)

    scores = scores.clone()
    scores[:, cls_index] = 0.0
    pad_mask = token_ids_with_cls == pad_idx
    scores[pad_mask] = 0.0
    return scores


def fig1_per_family_heatmap(model, all_seqs, family_names, vocab, vocab_inv, cfg):
    """
    Heatmap: top-20 tokens per family vs transformer layers (CLS attention).
    """
    print("Building Figure 1: CLS-attention heatmap per family …")
    n_families = len(family_names)
    n_layers = cfg["n_layers"]
    fig, axes = plt.subplots(n_families, 1, figsize=(12, n_families * 3.2))
    if n_families == 1:
        axes = [axes]

    for fi, fname in enumerate(family_names):
        seqs = all_seqs[fname][:10]
        if not seqs:
            axes[fi].axis("off")
            continue

        ids = encode_sequences(
            seqs, vocab, cfg["max_seq_len"], cfg.get("truncation", "head")
        )
        ids_t = torch.tensor(ids, dtype=torch.long)
        _, attn_list, token_ids_with_cls = get_attention(model, ids_t)

        # Build per-token, per-layer CLS attention
        token_layer_scores: Dict[str, Dict[int, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for layer_idx, attn in enumerate(attn_list):
            cls_row = attn[:, :, 0, :].mean(dim=1)  # (B, S)
            for b in range(cls_row.shape[0]):
                seq_ids = token_ids_with_cls[b].numpy().tolist()
                for pos, tid in enumerate(seq_ids):
                    if tid == 0 or pos == 0:
                        continue
                    tok = vocab_inv.get(tid, "<UNK>")
                    if tok in ("<PAD>", "<CLS>", "<UNK>"):
                        continue
                    token_layer_scores[tok][layer_idx].append(float(cls_row[b, pos]))

        # Rank tokens by mean CLS attention across layers
        token_means = {}
        for tok, layer_map in token_layer_scores.items():
            vals = []
            for layer_idx in range(n_layers):
                vals.extend(layer_map.get(layer_idx, []))
            token_means[tok] = float(np.mean(vals)) if vals else 0.0

        top_tokens = sorted(token_means, key=token_means.get, reverse=True)[:20]
        if not top_tokens:
            axes[fi].axis("off")
            continue

        mat = np.zeros((len(top_tokens), n_layers))
        for i, tok in enumerate(top_tokens):
            for layer_idx in range(n_layers):
                vals = token_layer_scores[tok].get(layer_idx, [])
                mat[i, layer_idx] = float(np.mean(vals)) if vals else 0.0

        ax = axes[fi]
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_yticks(range(len(top_tokens)))
        ax.set_yticklabels(
            [t.replace("REFL:", "R:").split(".")[-1][:28] for t in top_tokens],
            fontsize=7,
        )
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f"L{l + 1}" for l in range(n_layers)], fontsize=8)
        ax.set_title(
            f"{fname} — CLS attention to top tokens", fontsize=9, fontweight="bold"
        )
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    plt.tight_layout()
    out = FIGURES / "cls_attn_heatmap_per_family.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def fig2_head_specialisation(model, all_seqs, family_names, vocab, vocab_inv, cfg):
    """
    Head specialisation based on CLS attention per head.
    """
    print("Building Figure 2: head specialisation (CLS attention) …")
    n_heads = cfg["n_heads"]
    n_layers = cfg["n_layers"]
    fig, axes = plt.subplots(
        n_layers, n_heads, figsize=(18, 3 * n_layers), squeeze=False
    )

    family_colors = {
        fname: plt.get_cmap("tab10")(i / max(len(family_names), 1))
        for i, fname in enumerate(family_names)
    }

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            ax = axes[layer_idx][head_idx]
            tok_family_attn = defaultdict(lambda: defaultdict(list))

            for fname in family_names:
                seqs = all_seqs[fname][:8]
                if not seqs:
                    continue
                ids = encode_sequences(
                    seqs, vocab, cfg["max_seq_len"], cfg.get("truncation", "head")
                )
                ids_t = torch.tensor(ids, dtype=torch.long)
                _, attn_list, token_ids_with_cls = get_attention(model, ids_t)
                attn = attn_list[layer_idx]  # (B, H, S, S)
                head_cls = attn[:, head_idx, 0, :]  # (B, S)

                for b in range(head_cls.shape[0]):
                    seq_ids = token_ids_with_cls[b].numpy().tolist()
                    for pos, tid in enumerate(seq_ids):
                        if tid == 0 or pos == 0:
                            continue
                        tok = vocab_inv.get(tid, "<UNK>")
                        if tok in ("<PAD>", "<CLS>", "<UNK>"):
                            continue
                        tok_family_attn[tok][fname].append(float(head_cls[b, pos]))

            tok_overall = {
                tok: np.mean([v for vals in fam_dict.values() for v in vals])
                for tok, fam_dict in tok_family_attn.items()
                if sum(len(v) for v in fam_dict.values()) >= 3
            }
            top10 = sorted(tok_overall, key=lambda t: tok_overall[t], reverse=True)[:10]

            bottom = np.zeros(len(top10))
            for fname in family_names:
                heights = []
                for tok in top10:
                    vals = tok_family_attn[tok].get(fname, [])
                    heights.append(np.mean(vals) if vals else 0.0)
                ax.bar(
                    range(len(top10)),
                    heights,
                    bottom=bottom,
                    color=family_colors[fname],
                    label=fname,
                    alpha=0.85,
                )
                bottom += np.array(heights)

            labels = [t.replace("REFL:", "R:").split(".")[-1][:18] for t in top10]
            ax.set_xticks(range(len(top10)))
            ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)
            ax.set_title(
                f"L{layer_idx + 1} H{head_idx + 1}", fontsize=8, fontweight="bold"
            )
            ax.set_ylabel("Mean CLS attn", fontsize=7)
            ax.tick_params(axis="y", labelsize=6)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=family_colors[f]) for f in family_names
    ]
    fig.legend(
        handles,
        family_names,
        loc="lower center",
        ncol=min(len(family_names), n_heads * n_layers),
        fontsize=7,
        title="Family",
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    out = FIGURES / "cls_attention_head_specialisation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def fig3_cls_attention_sink(model, all_seqs, family_names, vocab, cfg):
    """
    Mean CLS attention per position (averaged over layers/heads/samples).
    """
    print("Building Figure 3: CLS attention sink …")
    position_attn = np.zeros(cfg["max_seq_len"] + 1, dtype=np.float64)
    n_samples_total = 0

    for fname in family_names:
        seqs = all_seqs[fname][:12]
        if not seqs:
            continue
        ids = encode_sequences(
            seqs, vocab, cfg["max_seq_len"], cfg.get("truncation", "head")
        )
        ids_t = torch.tensor(ids, dtype=torch.long)
        _, attn_list, token_ids_with_cls = get_attention(model, ids_t)

        scores = cls_attention_score(attn_list, token_ids_with_cls)
        position_attn[: scores.shape[1]] += scores.mean(dim=0).numpy()
        n_samples_total += 1

    if n_samples_total > 0:
        position_attn /= n_samples_total

    show_len = min(60, len(position_attn))
    x = np.arange(show_len)
    colors = np.where(x == 0, "#d62728", "#1f77b4")  # red for CLS position
    plt.figure(figsize=(10, 4))
    plt.bar(x, position_attn[:show_len], color=colors, alpha=0.8)
    plt.title("Mean CLS Attention per Position (first 60 positions)")
    plt.xlabel("Sequence position (0 = CLS)")
    plt.ylabel("Mean CLS attention")
    plt.tight_layout()

    out = FIGURES / "cls_attention_sink.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading model and vocab …")
    model, vocab, family_names, cfg = load_model_and_vocab()
    vocab_inv = {v: k for k, v in vocab.items()}

    print("Loading data …")
    data_dir = REPO / "extracted_data"
    all_seqs = defaultdict(list)
    all_seqs_flat, all_labels_int, data_family_names = load_dataset(data_dir)
    for seq, lbl_idx in zip(all_seqs_flat, all_labels_int):
        fname = data_family_names[lbl_idx]
        if len(seq) >= 5:
            all_seqs[fname].append(seq)

    print(
        f"Loaded {sum(len(v) for v in all_seqs.values())} samples across "
        f"{len(all_seqs)} families."
    )

    fig1_per_family_heatmap(model, all_seqs, family_names, vocab, vocab_inv, cfg)
    fig2_head_specialisation(model, all_seqs, family_names, vocab, vocab_inv, cfg)
    fig3_cls_attention_sink(model, all_seqs, family_names, vocab, cfg)

    print("\nAll figures saved to results/figures/:")
    for p in sorted(FIGURES.glob("cls_*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
