"""
attention_rollout.py
====================
Attention Rollout visualisation for the plain (non-gated) transformer.

Method: Abnar & Zuidema (2020) — "Quantifying Attention Flow in Transformers"
        as described in https://medium.com/@nivonl/exploring-visual-attention-in-transformer-models-ab538c06083a

rollout(attentions):
    rollout = I (identity)
    for each layer:
        A_avg = mean over heads
        A_aug  = A_avg + I       (add residual)
        A_norm = A_aug / rowsum  (normalise rows)
        rollout = rollout @ A_norm
    return rollout

Outputs (results/figures/):
  rollout_per_family.png       — Mean rollout score per token position, 8 families
  rollout_top_tokens.png       — Top-15 API tokens by mean rollout score, per family
  rollout_sample_heatmap.png   — Single-sample rollout heatmap (first 60 positions)
  rollout_vs_gate.png          — Rollout (plain) vs gate-score (gated) top-token comparison
"""

from __future__ import annotations
import json, pickle, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.preprocessing import load_dataset, prepare_splits, pad_with_truncation
from src.model import GAMEMal

RESULTS     = REPO / "results"
FIGURES     = RESULTS / "figures"
MODELS      = RESULTS / "models"
FIGURES.mkdir(parents=True, exist_ok=True)

DEVICE      = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
MAX_SEQ_LEN = 512
TRUNCATION  = "head"
SEED        = 42
N_SAMPLES   = 30   # samples per family for mean rollout

def _load_family_names():
    p = MODELS / "family_names.json"
    if p.exists():
        return json.loads(p.read_text())
    return ["Airpush","DroidKungFu","Fusob","Genpua","GinMaster","Jisut","Opfake","SmsPay"]

FAMILY_NAMES = _load_family_names()


# ── core rollout function (exactly as provided) ────────────────────────────

def attention_rollout(attentions: list[torch.Tensor]) -> torch.Tensor:
    """
    attentions: list of (B, n_heads, seq_len, seq_len) tensors, one per layer.
    Returns rollout: (B, seq_len, seq_len)
    """
    # Initialize rollout with identity matrix
    B = attentions[0].size(0)
    S = attentions[0].size(-1)
    rollout = torch.eye(S).unsqueeze(0).expand(B, -1, -1).to(attentions[0].device)

    for attention in attentions:
        attention_heads_fused = attention.mean(dim=1)                                             # Average attention across heads  (B, S, S)
        attention_heads_fused += torch.eye(S).unsqueeze(0).to(attention_heads_fused.device)      # A + I
        attention_heads_fused /= attention_heads_fused.sum(dim=-1, keepdim=True)                  # Normalizing A
        rollout = torch.matmul(rollout, attention_heads_fused)                                    # Multiplication

    return rollout   # (B, S, S)


# ── helpers ────────────────────────────────────────────────────────────────

def load_model(use_gate: bool):
    # Choose the right checkpoint + config depending on gate flag
    if use_gate:
        cfg_path = MODELS / "config.json"
        ckpt_path = MODELS / "game_mal_best.pt"
    else:
        # Prefer the dedicated plain-transformer checkpoint if it exists,
        # otherwise fall back to the gated checkpoint (dropping gate keys).
        pt_cfg = MODELS / "plain_transformer_config.json"
        pt_ckpt = MODELS / "plain_transformer_best.pt"
        cfg_path  = pt_cfg  if pt_cfg.exists()  else MODELS / "config.json"
        ckpt_path = pt_ckpt if pt_ckpt.exists() else MODELS / "game_mal_best.pt"

    cfg = json.loads(cfg_path.read_text())
    with open(MODELS / "vocab.pkl", "rb") as f:
        vobj = pickle.load(f)
    vocab = vobj["api2idx"] if isinstance(vobj, dict) and "api2idx" in vobj else (
        vobj.api2idx if hasattr(vobj, "api2idx") else vobj)

    model = GAMEMal(
        vocab_size=cfg["vocab_size"], num_classes=cfg["num_classes"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"], max_seq_len=cfg["max_seq_len"], dropout=0.0,
        use_gate=use_gate,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    # Strip any keys that don't exist in this model (e.g. gate params when loading
    # a gated checkpoint into a plain model)
    own = model.state_dict()
    state = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    own.update(state)
    model.load_state_dict(own)
    model.to(DEVICE).eval()
    print(f"  Loaded {'gated' if use_gate else 'plain'} model from {ckpt_path.name}")
    return model, vocab, cfg


def encode(seqs, vocab, max_seq, trunc):
    enc = [[vocab.get(t, vocab.get("<UNK>", 1)) for t in s] for s in seqs]
    return pad_with_truncation(enc, max_seq, truncation=trunc)


def get_rollout_and_gate(model_plain, model_gated,
                          token_ids: torch.Tensor):
    """Run both models, return rollout (plain) and mean gate score (gated)."""
    ids = token_ids.to(DEVICE)

    # Plain transformer — rollout
    with torch.no_grad():
        _, info_plain = model_plain(ids, return_attention=True)
    attn_list = [a for a in info_plain["attn_weights"]]   # list[(B,h,S,S)]
    rollout = attention_rollout(attn_list)                 # (B, S, S)
    # CLS token (pos 0) rollout to all other positions
    cls_rollout = rollout[:, 0, :]                         # (B, S) — influence on [CLS]

    # Gated transformer — gate scores
    with torch.no_grad():
        _, info_gated = model_gated(ids, return_attention=True)
    # mean gate over layers, heads, d_k → (B, S)
    gate_list = [g for g in info_gated["gate_scores"] if g is not None]
    if gate_list:
        stacked = torch.stack([g.mean(dim=(1, 3)) for g in gate_list], dim=0)
        gate_score = stacked.mean(dim=0).cpu()
    else:
        gate_score = None

    return cls_rollout.cpu(), gate_score


# ── Figure 1: mean rollout per position, per family ────────────────────────

def fig1_rollout_per_family(model_plain, model_gated, all_seqs, vocab, cfg):
    print("Building Figure 1: mean rollout per family …")
    show = 80
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
    axes = axes.flatten()
    fig.suptitle(
        "Attention Rollout — Mean CLS-to-Token Score per Sequence Position\n"
        "(Plain Transformer, averaged over 30 samples per family; "
        "higher = more influence on classification)",
        fontsize=11,
    )

    for ci, fname in enumerate(FAMILY_NAMES):
        ax = axes[ci]
        seqs = all_seqs[fname][:N_SAMPLES]
        if not seqs:
            ax.axis("off"); continue

        ids = torch.tensor(encode(seqs, vocab, cfg["max_seq_len"], TRUNCATION), dtype=torch.long)
        rollout_cls, _ = get_rollout_and_gate(model_plain, model_gated, ids)
        mean_r = rollout_cls.numpy()[:, :show].mean(axis=0)

        norm = plt.Normalize(mean_r.min(), mean_r.max())
        colors = plt.get_cmap("Blues")(norm(mean_r))
        ax.bar(np.arange(show), mean_r, color=colors, width=1.0, edgecolor="none")
        ax.set_title(fname, fontsize=9, fontweight="bold")
        ax.set_xlabel("Token position (head-truncated)", fontsize=7)
        ax.set_ylabel("Mean rollout (CLS→pos)", fontsize=7)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    out = FIGURES / "rollout_per_family.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


# ── Figure 2: top-15 API tokens by mean rollout, per family ────────────────

def fig2_rollout_top_tokens(model_plain, model_gated, all_seqs, vocab, cfg):
    print("Building Figure 2: top tokens by rollout …")
    idx2api = {v: k for k, v in vocab.items()}

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    fig.suptitle(
        "Attention Rollout — Top-15 APIs by Mean CLS-Rollout Score, per Family\n"
        "(REFL: = call resolved through Method.invoke; "
        "bars show mean rollout score across 30 samples)",
        fontsize=11,
    )

    for ci, fname in enumerate(FAMILY_NAMES):
        ax = axes[ci]
        seqs = all_seqs[fname][:N_SAMPLES]
        if not seqs:
            ax.axis("off"); continue

        ids_np = encode(seqs, vocab, cfg["max_seq_len"], TRUNCATION)
        ids_t  = torch.tensor(ids_np, dtype=torch.long)
        rollout_cls, _ = get_rollout_and_gate(model_plain, model_gated, ids_t)

        # Aggregate by token name
        tok_scores = defaultdict(list)
        for bi in range(len(seqs)):
            for pos, tid in enumerate(ids_np[bi]):
                if tid == 0:
                    break
                tok = idx2api.get(int(tid), "<UNK>")
                if tok in ("<PAD>", "<CLS>", "<UNK>"):
                    continue
                tok_scores[tok].append(rollout_cls[bi, pos].item())

        ranked = sorted(
            {t: np.mean(v) for t, v in tok_scores.items()}.items(),
            key=lambda x: x[1], reverse=True
        )[:15]

        lbls = [t.replace("REFL:", "R:").split(".")[-1][:28] for t, _ in ranked]
        vals = [v for _, v in ranked]

        cmap = plt.get_cmap("Blues")
        norm = plt.Normalize(min(vals), max(vals))
        colors = [cmap(norm(v)) for v in vals]

        ax.barh(range(len(vals)), vals[::-1], color=colors[::-1], alpha=0.9)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(lbls[::-1], fontsize=6)
        ax.set_title(fname, fontsize=9, fontweight="bold")
        ax.set_xlabel("Mean rollout score", fontsize=7)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    out = FIGURES / "rollout_top_tokens.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


# ── Figure 3: single-sample heatmap (rollout matrix) ──────────────────────

def fig3_sample_heatmap(model_plain, model_gated, all_seqs, vocab, cfg):
    print("Building Figure 3: sample rollout heatmap …")
    idx2api = {v: k for k, v in vocab.items()}
    show = 40   # first 40 tokens — readable heatmap

    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    axes = axes.flatten()
    fig.suptitle(
        "Attention Rollout Heatmap (first 40 positions) — One Sample per Family\n"
        "Row = query token, Column = key token; colour = rollout score",
        fontsize=11,
    )

    for ci, fname in enumerate(FAMILY_NAMES):
        ax = axes[ci]
        seqs = all_seqs[fname]
        if not seqs:
            ax.axis("off"); continue

        # Pick sample closest to median length
        lengths = [len(s) for s in seqs[:50]]
        med = int(np.median(lengths))
        sample = min(seqs[:50], key=lambda s: abs(len(s) - med))

        ids_np = encode([sample], vocab, cfg["max_seq_len"], TRUNCATION)
        ids_t  = torch.tensor(ids_np, dtype=torch.long)
        actual_len = min(len(sample), cfg["max_seq_len"], show)

        with torch.no_grad():
            _, info = model_plain(ids_t.to(DEVICE), return_attention=True)
        attn_list = [a for a in info["attn_weights"]]
        rollout_mat = attention_rollout(attn_list)[0].cpu().numpy()  # (S, S)

        sub = rollout_mat[:actual_len, :actual_len]

        # Token labels for axes
        def short_tok(tid):
            t = idx2api.get(int(tid), "?")
            return t.replace("REFL:", "R:").split(".")[-1][:12]

        tick_labels = [short_tok(ids_np[0][i]) for i in range(actual_len)]

        im = ax.imshow(sub, aspect="auto", cmap="Blues", interpolation="nearest")
        step = max(1, actual_len // 10)
        ax.set_xticks(range(0, actual_len, step))
        ax.set_xticklabels(tick_labels[::step], rotation=55, ha="right", fontsize=5)
        ax.set_yticks(range(0, actual_len, step))
        ax.set_yticklabels(tick_labels[::step], fontsize=5)
        ax.set_title(fname, fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    plt.tight_layout()
    out = FIGURES / "rollout_sample_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


# ── Figure 4: rollout vs gate score comparison ────────────────────────────

def fig4_rollout_vs_gate(model_plain, model_gated, all_seqs, vocab, cfg):
    print("Building Figure 4: rollout vs gate score …")
    idx2api = {v: k for k, v in vocab.items()}

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    fig.suptitle(
        "Attention Rollout (Plain Transformer) vs Gate Score (Gated) — Top-10 APIs\n"
        "Two independent explanation methods; overlap reveals robust signal",
        fontsize=11,
    )

    for ci, fname in enumerate(FAMILY_NAMES):
        ax = axes[ci]
        seqs = all_seqs[fname][:N_SAMPLES]
        if not seqs:
            ax.axis("off"); continue

        ids_np = encode(seqs, vocab, cfg["max_seq_len"], TRUNCATION)
        ids_t  = torch.tensor(ids_np, dtype=torch.long)
        rollout_cls, gate_score = get_rollout_and_gate(model_plain, model_gated, ids_t)

        # Aggregate rollout by API name
        roll_tok = defaultdict(list)
        gate_tok = defaultdict(list)
        for bi in range(len(seqs)):
            for pos, tid in enumerate(ids_np[bi]):
                if tid == 0:
                    break
                tok = idx2api.get(int(tid), "<UNK>")
                if tok in ("<PAD>", "<CLS>", "<UNK>"):
                    continue
                roll_tok[tok].append(rollout_cls[bi, pos].item())
                if gate_score is not None:
                    gate_tok[tok].append(gate_score[bi, pos].item())

        roll_mean = {t: np.mean(v) for t, v in roll_tok.items()}
        gate_mean = {t: np.mean(v) for t, v in gate_tok.items()} if gate_tok else {}

        top10_roll = sorted(roll_mean, key=lambda t: roll_mean[t], reverse=True)[:10]
        top10_gate = sorted(gate_mean, key=lambda t: gate_mean[t], reverse=True)[:10] if gate_mean else []

        overlap = set(top10_roll) & set(top10_gate)
        all_top = list(dict.fromkeys(top10_roll + [t for t in top10_gate if t not in top10_roll]))[:12]

        x = np.arange(len(all_top))
        w = 0.35
        roll_vals = [roll_mean.get(t, 0) for t in all_top]
        gate_vals = [gate_mean.get(t, 0) for t in all_top]
        lbls = [t.replace("REFL:", "R:").split(".")[-1][:18] for t in all_top]

        bars1 = ax.bar(x - w/2, roll_vals, w, label="Rollout (plain)", color="#1f77b4", alpha=0.8)
        bars2 = ax.bar(x + w/2, gate_vals, w, label="Gate score (gated)", color="#ff7f0e", alpha=0.8)

        # Mark overlap tokens with a star
        for i, tok in enumerate(all_top):
            if tok in overlap:
                ax.text(i, max(roll_vals[i], gate_vals[i]) * 1.02, "★",
                        ha="center", fontsize=8, color="green")

        ax.set_xticks(x)
        ax.set_xticklabels(lbls, rotation=50, ha="right", fontsize=5)
        ax.set_title(f"{fname} (★={len(overlap)} overlap)", fontsize=8, fontweight="bold")
        ax.set_ylabel("Score", fontsize=7)
        if ci == 0:
            ax.legend(fontsize=6)

    plt.tight_layout()
    out = FIGURES / "rollout_vs_gate.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading models …")
    model_plain, vocab, cfg = load_model(use_gate=False)
    model_gated, _, _       = load_model(use_gate=True)
    idx2api = {v: k for k, v in vocab.items()}

    print("Loading data …")
    data_dir = REPO / "extracted_data"
    all_seqs_flat, all_labels_int, data_family_names = load_dataset(data_dir)
    all_seqs = defaultdict(list)
    for seq, lbl_idx in zip(all_seqs_flat, all_labels_int):
        fname = data_family_names[lbl_idx]
        if len(seq) >= 5:
            all_seqs[fname].append(seq)
    print(f"  {sum(len(v) for v in all_seqs.values())} samples across {len(all_seqs)} families")

    fig1_rollout_per_family(model_plain, model_gated, all_seqs, vocab, cfg)
    fig2_rollout_top_tokens(model_plain, model_gated, all_seqs, vocab, cfg)
    fig3_sample_heatmap    (model_plain, model_gated, all_seqs, vocab, cfg)
    fig4_rollout_vs_gate   (model_plain, model_gated, all_seqs, vocab, cfg)

    print("\nAll rollout figures saved to results/figures/:")
    for p in sorted(FIGURES.glob("rollout_*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
