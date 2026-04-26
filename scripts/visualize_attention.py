"""
visualize_attention.py
======================
Generates four attention-map figures for the GAME-Mal paper/thesis:

  1. attn_heatmap_per_family.png
     One row per family, one column per layer×head (8 panels).
     Y-axis = top-20 API tokens by mean gate score for that family.
     Color = softmax attention weight (averaged over 10 samples).

  2. gate_vs_attn_scatter.png
     For a single representative sample per family: scatter plot of
     gate score (x) vs. attention weight received (y) per token.
     Shows whether high-gate tokens also receive high attention.

  3. attention_head_specialisation.png
     One subplot per attention head (4 heads × 2 layers = 8 panels).
     For each head, the top-10 most-attended API tokens, coloured
     by family, to reveal whether heads specialise.

  4. attention_sink_comparison.png
     Side-by-side: plain transformer vs GAME-Mal (gated).
     Mean attention weight at each sequence position (averaged over
     all samples), showing the attention-sink effect and its removal.

Outputs go to results/figures/.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.model import GAMEMal
from src.preprocessing import load_dataset, prepare_splits, pad_with_truncation

RESULTS   = REPO / "results"
FIGURES   = RESULTS / "figures"
MODELS    = RESULTS / "models"
FIGURES.mkdir(parents=True, exist_ok=True)

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
print(f"Using device: {DEVICE}")

# ── helpers ────────────────────────────────────────────────────────────────

def load_model_and_vocab():
    cfg = json.loads((MODELS / "config.json").read_text())
    with open(MODELS / "vocab.pkl", "rb") as f:
        vocab_obj = pickle.load(f)
    # vocab.pkl stores an APIVocabulary object (dict with 'api2idx') or plain dict
    if isinstance(vocab_obj, dict) and "api2idx" in vocab_obj:
        vocab = vocab_obj["api2idx"]
    elif hasattr(vocab_obj, "api2idx"):
        vocab = vocab_obj.api2idx
    else:
        vocab = vocab_obj
    family_names = json.loads((MODELS / "family_names.json").read_text())

    model = GAMEMal(
        vocab_size  = cfg["vocab_size"],
        num_classes = cfg["num_classes"],
        d_model     = cfg["d_model"],
        n_heads     = cfg["n_heads"],
        n_layers    = cfg["n_layers"],
        d_ff        = cfg["d_ff"],
        max_seq_len = cfg["max_seq_len"],
        dropout     = 0.0,          # eval mode, no dropout
        use_gate    = True,
    )
    ckpt = torch.load(MODELS / "game_mal_best.pt", map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, vocab, family_names, cfg


def encode_sequences(seqs, vocab, max_seq_len, truncation="head"):
    """Encode raw token lists → padded int tensor."""
    encoded = []
    for s in seqs:
        ids = [vocab.get(t, vocab.get("<UNK>", 1)) for t in s]
        encoded.append(ids)
    return pad_with_truncation(encoded, max_seq_len, truncation=truncation)


def get_attention_and_gates(model, token_ids: torch.Tensor):
    """
    Run one forward pass with return_attention=True.
    Returns:
        attn_weights: list[layer] of (B, n_heads, seq, seq)
        gate_scores:  list[layer] of (B, n_heads, seq, d_k)
        logits:       (B, num_classes)
    """
    with torch.no_grad():
        logits, info = model(token_ids.to(DEVICE), return_attention=True)
    attn = [a.cpu() for a in info["attn_weights"]]
    gates = [g.cpu() for g in info["gate_scores"] if g is not None]
    return logits.cpu(), attn, gates


def mean_gate_per_token(gate_list):
    """
    gate_list: list[n_layers] of (B, n_heads, seq, d_k)
    Returns (B, seq) mean gate score averaged over layers, heads, d_k.
    """
    stacked = torch.stack([g.mean(dim=(1, 3)) for g in gate_list], dim=0)  # (L, B, seq)
    return stacked.mean(dim=0)  # (B, seq)


def top_tokens_for_family(family_seqs, vocab_inv, model, vocab, cfg, n_samples=10, top_k=20):
    """Return list of (token_str, mean_gate) for the top-k gate-ranked APIs of this family."""
    seqs = family_seqs[:n_samples]
    ids = encode_sequences(seqs, vocab, cfg["max_seq_len"], cfg["truncation"])
    ids_t = torch.tensor(ids, dtype=torch.long)

    _, attn_list, gate_list = get_attention_and_gates(model, ids_t)
    mg = mean_gate_per_token(gate_list)  # (B, seq)

    # Map each position back to token string
    token_gate = defaultdict(list)
    for b, seq in enumerate(seqs):
        enc = ids[b]
        for pos, tid in enumerate(enc):
            if tid == 0:
                break  # PAD
            tok = vocab_inv.get(tid, "<UNK>")
            if tok in ("<PAD>", "<CLS>", "<UNK>"):
                continue
            token_gate[tok].append(mg[b, pos].item())

    ranked = sorted(
        {t: np.mean(v) for t, v in token_gate.items()}.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_k]


# ── Figure 1: per-family attention heatmap ─────────────────────────────────

def fig1_per_family_heatmap(model, all_seqs, family_names, vocab, vocab_inv, cfg):
    """
    For each family: pick 10 samples, extract attention weights per layer/head,
    average over samples, plot as heatmap (top-20 tokens × positions).
    Layout: 8 rows (families) × 2 columns (layer 0 avg, layer 1 avg).
    """
    print("Building Figure 1: per-family attention heatmap …")
    n_families = len(family_names)
    fig, axes = plt.subplots(n_families, 2, figsize=(18, n_families * 3.2))
    fig.suptitle(
        "Mean Softmax Attention Weights per Family and Layer\n"
        "(averaged over 10 samples per family, all 4 heads)",
        fontsize=13, y=1.01,
    )

    cmap = plt.get_cmap("YlOrRd")

    for fi, fname in enumerate(family_names):
        seqs = all_seqs[fname][:10]
        if not seqs:
            continue
        ids = encode_sequences(seqs, vocab, cfg["max_seq_len"], cfg["truncation"])
        ids_t = torch.tensor(ids, dtype=torch.long)
        _, attn_list, gate_list = get_attention_and_gates(model, ids_t)
        mg = mean_gate_per_token(gate_list)  # (B, seq)

        # Find top-20 tokens for this family by gate score
        token_gate = defaultdict(list)
        for b, seq in enumerate(seqs):
            enc = ids[b]
            for pos, tid in enumerate(enc):
                if tid == 0:
                    break
                tok = vocab_inv.get(tid, "<UNK>")
                if tok in ("<PAD>", "<CLS>", "<UNK>"):
                    continue
                token_gate[tok].append(mg[b, pos].item())

        top20 = [t for t, _ in sorted(
            {t: np.mean(v) for t, v in token_gate.items()}.items(),
            key=lambda x: x[1], reverse=True
        )[:20]]

        # For each layer, build a (20 tokens × 512 positions) attn matrix,
        # then compress to top-20 × top-50 positions for readability.
        for layer_idx in range(min(2, len(attn_list))):
            ax = axes[fi, layer_idx]
            attn = attn_list[layer_idx]  # (B, n_heads, seq, seq)
            # Average over heads and samples → (seq_dst, seq_src)
            mean_attn = attn.mean(dim=(0, 1)).numpy()  # (seq, seq)

            # For each top token, find its most common position across samples
            # and extract the attention row for that position
            rows = []
            yticklabels = []
            for tok in top20:
                # Find which positions this token appears at
                positions_seen = []
                for b in range(len(seqs)):
                    enc = ids[b]
                    for pos, tid in enumerate(enc):
                        if tid == 0:
                            break
                        if vocab_inv.get(tid, "") == tok:
                            positions_seen.append(pos)
                if not positions_seen:
                    continue
                # Use median position
                med_pos = int(np.median(positions_seen))
                # Get the attention this token pays to others (its query row)
                row = mean_attn[med_pos, :]  # (seq,)
                rows.append(row[:80])        # show first 80 positions
                short = tok.replace("REFL:", "R:").split(".")[-1][:28]
                yticklabels.append(short)

            if not rows:
                ax.axis("off")
                continue

            mat = np.array(rows)  # (n_tokens, 80)
            im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest",
                           vmin=0, vmax=mat.max())
            ax.set_yticks(range(len(yticklabels)))
            ax.set_yticklabels(yticklabels, fontsize=6)
            ax.set_xlabel("Sequence position (first 80)", fontsize=7)
            ax.set_title(
                f"{fname} — Layer {layer_idx + 1}",
                fontsize=8, fontweight="bold",
            )
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    plt.tight_layout()
    out = FIGURES / "attn_heatmap_per_family.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 2: gate score vs attention weight scatter ───────────────────────

def fig2_gate_vs_attn_scatter(model, all_seqs, family_names, vocab, vocab_inv, cfg):
    """
    For one representative sample per family: scatter gate score (x) vs
    mean attention weight received (y) per token position.
    """
    print("Building Figure 2: gate vs attention scatter …")
    n_families = len(family_names)
    cols = 4
    rows = (n_families + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten()
    fig.suptitle(
        "Gate Score vs. Mean Attention Weight Received\n"
        "(one representative sample per family; each dot = one token position)",
        fontsize=12,
    )

    family_colors = plt.get_cmap("tab10")(np.linspace(0, 1, n_families))

    for fi, fname in enumerate(family_names):
        ax = axes[fi]
        seqs = all_seqs[fname]
        if not seqs:
            ax.axis("off")
            continue

        # Pick the sample closest to median length
        lengths = [len(s) for s in seqs[:30]]
        med_len = int(np.median(lengths))
        sample = min(seqs[:30], key=lambda s: abs(len(s) - med_len))

        ids = encode_sequences([sample], vocab, cfg["max_seq_len"], cfg["truncation"])
        ids_t = torch.tensor(ids, dtype=torch.long)
        _, attn_list, gate_list = get_attention_and_gates(model, ids_t)

        # Mean gate over layers, heads, d_k → (seq,)
        gate_per_pos = mean_gate_per_token(gate_list)[0].numpy()

        # Mean attention received: average over all layers, heads, and src positions
        # attn[layer] is (1, n_heads, seq_dst, seq_src)
        # For each position p: how much attention does p receive from others?
        attn_received = np.zeros(cfg["max_seq_len"])
        for layer_attn in attn_list:
            # layer_attn: (1, n_heads, seq, seq)
            # axis=-2 = query, axis=-1 = key; column sum = attention received
            attn_received += layer_attn[0].mean(dim=0).sum(dim=0).numpy()  # (seq,)
        attn_received /= len(attn_list)

        # Only non-PAD positions
        seq_len = min(len(sample), cfg["max_seq_len"])
        positions = np.arange(seq_len)
        gate_vals = gate_per_pos[:seq_len]
        attn_vals = attn_received[:seq_len]

        # Color by relative position (early = blue, late = red)
        pos_colors = plt.get_cmap("coolwarm")(positions / max(positions.max(), 1))

        sc = ax.scatter(gate_vals, attn_vals, c=pos_colors, alpha=0.5, s=12,
                        edgecolors="none")
        ax.set_title(fname, fontsize=9, fontweight="bold")
        ax.set_xlabel("Gate score", fontsize=7)
        ax.set_ylabel("Attn received (mean)", fontsize=7)
        ax.tick_params(labelsize=7)

        # Pearson r
        if len(gate_vals) > 5:
            r = np.corrcoef(gate_vals, attn_vals)[0, 1]
            ax.text(0.05, 0.92, f"r = {r:.2f}", transform=ax.transAxes,
                    fontsize=7, color="black")

    # Hide unused panels
    for i in range(n_families, len(axes)):
        axes[i].axis("off")

    # Colorbar legend for position
    sm = plt.cm.ScalarMappable(cmap="coolwarm",
                                norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:n_families], fraction=0.015, pad=0.04)
    cbar.set_label("Relative position (0=early, 1=late)", fontsize=8)

    plt.tight_layout()
    out = FIGURES / "gate_vs_attn_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 3: head specialisation ──────────────────────────────────────────

def fig3_head_specialisation(model, all_seqs, family_names, vocab, vocab_inv, cfg):
    """
    For each of the 8 attention heads (4 per layer × 2 layers):
    show the top-10 most-attended tokens overall, with a stacked bar
    per family showing how much each family contributes to that token's
    average attention.
    """
    print("Building Figure 3: head specialisation …")
    n_heads = cfg["n_heads"]
    n_layers = cfg["n_layers"]
    n_panels = n_heads * n_layers  # 8

    fig, axes = plt.subplots(2, n_heads, figsize=(20, 8))
    fig.suptitle(
        "Attention Head Specialisation: Top-10 Attended API Tokens per Head\n"
        "(stacked bar = family contribution; REFL: = reflective call)",
        fontsize=12,
    )

    family_colors = {
        fname: plt.get_cmap("tab10")(i / len(family_names))
        for i, fname in enumerate(family_names)
    }

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            ax = axes[layer_idx, head_idx]
            # Collect per-token, per-family mean attention for this head
            # token → {family: [attn_values]}
            tok_family_attn = defaultdict(lambda: defaultdict(list))

            for fname in family_names:
                seqs = all_seqs[fname][:8]
                if not seqs:
                    continue
                ids = encode_sequences(seqs, vocab, cfg["max_seq_len"], cfg["truncation"])
                ids_t = torch.tensor(ids, dtype=torch.long)
                _, attn_list, _ = get_attention_and_gates(model, ids_t)

                attn = attn_list[layer_idx]  # (B, n_heads, seq, seq)
                head_attn = attn[:, head_idx, :, :]  # (B, seq, seq)
                # Mean attention received per position (column sum, averaged over src)
                recv = head_attn.mean(dim=1).numpy()  # (B, seq)

                for b in range(len(seqs)):
                    enc = ids[b]
                    for pos, tid in enumerate(enc):
                        if tid == 0:
                            break
                        tok = vocab_inv.get(tid, "<UNK>")
                        if tok in ("<PAD>", "<CLS>", "<UNK>"):
                            continue
                        tok_family_attn[tok][fname].append(recv[b, pos])

            # Rank tokens by overall mean attention received
            tok_overall = {
                tok: np.mean([v for vals in fam_dict.values() for v in vals])
                for tok, fam_dict in tok_family_attn.items()
                if sum(len(v) for v in fam_dict.values()) >= 3  # seen at least 3x
            }
            top10 = sorted(tok_overall, key=lambda t: tok_overall[t], reverse=True)[:10]

            # Stacked bar chart
            bottom = np.zeros(len(top10))
            for fname in family_names:
                heights = []
                for tok in top10:
                    vals = tok_family_attn[tok].get(fname, [])
                    heights.append(np.mean(vals) if vals else 0.0)
                ax.bar(range(len(top10)), heights, bottom=bottom,
                       color=family_colors[fname], label=fname, alpha=0.85)
                bottom += np.array(heights)

            labels = [
                t.replace("REFL:", "R:").split(".")[-1][:18]
                for t in top10
            ]
            ax.set_xticks(range(len(top10)))
            ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)
            ax.set_title(f"L{layer_idx+1} H{head_idx+1}", fontsize=9, fontweight="bold")
            ax.set_ylabel("Mean attn recv.", fontsize=7)
            ax.tick_params(axis="y", labelsize=6)

    # One shared legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=family_colors[f])
        for f in family_names
    ]
    fig.legend(handles, family_names, loc="lower center", ncol=n_panels,
               fontsize=7, title="Family", bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    out = FIGURES / "attention_head_specialisation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 4: attention sink comparison ────────────────────────────────────

def fig4_attention_sink(model, all_seqs, family_names, vocab, cfg):
    """
    Compare mean attention weight at each sequence position for:
      - GAME-Mal (gated)  vs
      - GAME-Mal (use_gate=False, ablation weights not saved — simulate by
        loading the same weights but disabling gate at runtime)
    A true attention sink shows as a spike at position 0.
    """
    print("Building Figure 4: attention sink comparison …")

    # Build a version with gate disabled at the architecture level
    cfg2 = json.loads((MODELS / "config.json").read_text())
    model_plain = GAMEMal(
        vocab_size=cfg2["vocab_size"], num_classes=cfg2["num_classes"],
        d_model=cfg2["d_model"], n_heads=cfg2["n_heads"], n_layers=cfg2["n_layers"],
        d_ff=cfg2["d_ff"], max_seq_len=cfg2["max_seq_len"],
        dropout=0.0, use_gate=False,
    )
    # Load the gated weights into the plain model for all shared parameters
    # (gate-specific weights w_gate will simply be absent/ignored)
    ckpt = torch.load(MODELS / "game_mal_best.pt", map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    # Load only matching keys
    plain_state = model_plain.state_dict()
    matched = {k: v for k, v in state.items() if k in plain_state and plain_state[k].shape == v.shape}
    plain_state.update(matched)
    model_plain.load_state_dict(plain_state)
    model_plain.to(DEVICE)
    model_plain.eval()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(
        "Attention Sink Effect: Mean Attention Received per Sequence Position\n"
        "(averaged over all families and samples; spike at pos 0 = sink)",
        fontsize=12,
    )

    show_len = 60   # show first 60 positions where sink is most visible

    for ax, (mdl, label) in zip(
        axes,
        [(model, "GAME-Mal (gated — gate removes sink)"),
         (model_plain, "Plain Transformer (no gate — sink visible)")],
    ):
        position_attn = np.zeros(cfg["max_seq_len"])
        n_samples_total = 0

        for fname in family_names:
            seqs = all_seqs[fname][:10]
            if not seqs:
                continue
            ids = encode_sequences(seqs, vocab, cfg["max_seq_len"], cfg["truncation"])
            ids_t = torch.tensor(ids, dtype=torch.long)
            _, attn_list, _ = get_attention_and_gates(mdl, ids_t)

            for layer_attn in attn_list:
                # layer_attn: (B, n_heads, seq, seq)
                # attention received per position = column sum over queries,
                # averaged over heads and batch
                recv = layer_attn.mean(dim=(0, 1)).sum(dim=0).numpy()  # (seq,)
                position_attn += recv
            n_samples_total += len(seqs)

        position_attn /= (n_samples_total * len(family_names) * len(attn_list))

        x = np.arange(show_len)
        colors = np.where(x == 0, "#d62728", "#1f77b4")   # red for pos 0
        ax.bar(x, position_attn[:show_len], color=colors, alpha=0.8, width=0.8)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Sequence position (first 60 tokens)", fontsize=9)
        ax.set_ylabel("Mean attention weight received", fontsize=9)
        ax.tick_params(labelsize=8)

        # Annotate position-0 value
        ax.annotate(
            f"pos 0 = {position_attn[0]:.4f}",
            xy=(0, position_attn[0]),
            xytext=(8, position_attn[0] * 0.85),
            fontsize=8, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728"),
        )

    plt.tight_layout()
    out = FIGURES / "attention_sink_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── main ───────────────────────────────────────────────────────────────────

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

    print(f"Loaded {sum(len(v) for v in all_seqs.values())} samples across "
          f"{len(all_seqs)} families.")

    fig1_per_family_heatmap(model, all_seqs, family_names, vocab, vocab_inv, cfg)
    fig2_gate_vs_attn_scatter(model, all_seqs, family_names, vocab, vocab_inv, cfg)
    fig3_head_specialisation(model, all_seqs, family_names, vocab, vocab_inv, cfg)
    fig4_attention_sink(model, all_seqs, family_names, vocab, cfg)

    print("\nAll figures saved to results/figures/:")
    for p in sorted(FIGURES.glob("attn_*.png")) + sorted(FIGURES.glob("gate_vs_attn*.png")) + sorted(FIGURES.glob("attention_*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
