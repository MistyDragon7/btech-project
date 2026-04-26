"""
shap_analysis.py
================
SHAP explainability analysis for two models:
  1. Random Forest (over Markov rule features) — shap.TreeExplainer (fast, exact)
  2. Plain Transformer (no gate)               — shap.GradientExplainer on embeddings

Outputs (all to results/figures/):
  shap_rf_beeswarm.png         — SHAP beeswarm: top-20 rules, all test samples
  shap_rf_bar_per_class.png    — Mean |SHAP| per API (aggregated from rules), per class
  shap_transformer_summary.png — Mean |SHAP| per token position, averaged by family
  shap_rf_vs_transformer.png   — Side-by-side top-20 API importance: RF vs Transformer
"""

from __future__ import annotations
import json, pickle, sys, warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import shap
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.preprocessing import load_dataset, prepare_splits, APIVocabulary, pad_with_truncation
from src.markov import build_class_graphs, compute_support_confidence, prune_rules, build_rule_feature_matrix
from src.baselines import BASELINE_MODELS
from src.model import GAMEMal

RESULTS = REPO / "results"
FIGURES = RESULTS / "figures"
MODELS  = RESULTS / "models"
FIGURES.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")   # SHAP gradient passes are more stable on CPU
MAX_SEQ_LEN  = 512
TRUNCATION   = "head"
MAX_SPACING  = 10
MIN_SUPPORT  = 1e-4
MIN_CONF     = 0.8
SEED         = 42
N_SHAP_BACK  = 50   # background samples for GradientExplainer
N_SHAP_TEST  = 80   # test samples to explain
TOP_K        = 20   # top features to show

FAMILY_NAMES = ["Airpush","DroidKungFu","Fusob","Genpua","GinMaster","Jisut","Opfake","SmsPay"]
COLORS = plt.get_cmap("tab10")(np.linspace(0, 1, len(FAMILY_NAMES)))

# ── helpers ────────────────────────────────────────────────────────────────

def shorten(rule_str: str, maxlen: int = 32) -> str:
    """Shorten a rule string for axis labels."""
    return rule_str.replace("REFL:", "R:").split(".")[-1][:maxlen]

def load_data():
    data_dir = REPO / "extracted_data"
    seqs, labels, _ = load_dataset(data_dir)
    seqs   = [s for s, l in zip(seqs, labels) if len(s) >= 5]
    labels = [l for s, l in zip(seqs, labels) if len(s) >= 5]
    # redo with correct filter
    all_seqs, all_labels, _ = load_dataset(data_dir)
    seqs_f, lbls_f = [], []
    for s, l in zip(all_seqs, all_labels):
        if len(s) >= 5:
            seqs_f.append(s)
            lbls_f.append(l)
    return seqs_f, np.array(lbls_f)

def build_vocab_and_encode(train_seqs, all_seqs):
    vocab = APIVocabulary(min_freq=2)
    vocab.build(train_seqs)
    encoded = [vocab.encode(s) for s in all_seqs]
    return vocab, encoded

def load_transformer(use_gate: bool = False):
    # Use dedicated plain-transformer checkpoint when available
    if use_gate:
        cfg_path  = MODELS / "config.json"
        ckpt_path = MODELS / "game_mal_best.pt"
    else:
        pt_cfg  = MODELS / "plain_transformer_config.json"
        pt_ckpt = MODELS / "plain_transformer_best.pt"
        cfg_path  = pt_cfg  if pt_cfg.exists()  else MODELS / "config.json"
        ckpt_path = pt_ckpt if pt_ckpt.exists() else MODELS / "game_mal_best.pt"

    cfg = json.loads(cfg_path.read_text())
    with open(MODELS / "vocab.pkl", "rb") as f:
        vocab_obj = pickle.load(f)
    vocab = vocab_obj["api2idx"] if isinstance(vocab_obj, dict) and "api2idx" in vocab_obj else (
        vocab_obj.api2idx if hasattr(vocab_obj, "api2idx") else vocab_obj)

    model = GAMEMal(
        vocab_size=cfg["vocab_size"], num_classes=cfg["num_classes"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"], max_seq_len=cfg["max_seq_len"], dropout=0.0, use_gate=use_gate,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    own = model.state_dict()
    state = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    own.update(state)
    model.load_state_dict(own)
    model.to(DEVICE).eval()
    print(f"  Loaded {'gated' if use_gate else 'plain'} transformer from {ckpt_path.name}")
    return model, vocab, cfg

# ── Transformer wrapper for GradientExplainer ──────────────────────────────

class TransformerFromEmbedding(torch.nn.Module):
    """Wraps the transformer so its input is the embedding matrix (float),
    not integer token IDs. This lets shap.GradientExplainer work on it."""

    def __init__(self, model: GAMEMal):
        super().__init__()
        self.model = model

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        """
        embed: (B, seq_len, d_model) — token embeddings only (no positional).
        Padding positions are exact zero (padding_idx=0 in nn.Embedding).
        Positional embeddings are added here so the pad mask works correctly.
        Returns logits: (B, num_classes)
        """
        B, N, D = embed.shape
        # Detect padding before adding positional embeddings (zero = padding)
        pad_mask = (embed.abs().sum(dim=-1) == 0)  # (B, N)

        # Add positional embeddings (mirrors GAMEMal.forward exactly)
        positions = torch.arange(N, device=embed.device).unsqueeze(0).expand(B, -1)
        h = embed + self.model.pos_embedding(positions)

        h = self.model.embed_dropout(h)
        for block in self.model.blocks:
            h, _, _ = block(h, pad_mask, return_attention=False)
        h = self.model.final_norm(h)
        mask_exp = (~pad_mask).unsqueeze(-1).float()
        h_pooled = (h * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        return self.model.classifier(h_pooled)


def get_embeddings(model: GAMEMal, token_ids: torch.Tensor) -> torch.Tensor:
    """Return token embeddings only (no positional).
    Padding positions (token_id=0) map to exact-zero vectors via padding_idx=0,
    letting TransformerFromEmbedding correctly detect them as padding.
    Positional embeddings are added inside the wrapper's forward()."""
    with torch.no_grad():
        return model.api_embedding(token_ids)  # (B, N, d_model)

# ── Part 1: RF SHAP ────────────────────────────────────────────────────────

def run_rf_shap(seqs, labels):
    print("=" * 60)
    print("Part 1: RF SHAP (TreeExplainer)")
    print("=" * 60)

    np.random.seed(SEED)
    splits = prepare_splits(seqs, labels.tolist(), n_folds=3, seed=SEED)
    # Use the same best fold as the saved model
    _pt_cfg = MODELS / "plain_transformer_config.json"
    _best_fold = json.load(open(_pt_cfg))["best_fold"] if _pt_cfg.exists() else 2
    train_idx, test_idx = splits[_best_fold]

    train_seqs = [seqs[i] for i in train_idx]
    test_seqs  = [seqs[i] for i in test_idx]
    y_train    = labels[train_idx]
    y_test     = labels[test_idx]
    num_cls    = len(FAMILY_NAMES)

    print(f"  Train: {len(train_seqs)}   Test: {len(test_seqs)}")

    # Build vocab for integer encoding (needed for rule extraction)
    vocab = APIVocabulary(min_freq=2)
    vocab.build(train_seqs)

    train_enc = [vocab.encode_sequence(s) for s in train_seqs]
    test_enc  = [vocab.encode_sequence(s) for s in test_seqs]

    print("  Extracting Markov rules …")
    class_graphs, _ = build_class_graphs(train_enc, y_train.tolist(), num_cls, MAX_SPACING)
    support, confidence = compute_support_confidence(class_graphs, num_cls)
    selected_rules = prune_rules(support, confidence, MIN_SUPPORT, MIN_CONF)
    print(f"  {len(selected_rules)} rules survive pruning")

    X_train = build_rule_feature_matrix(train_enc, selected_rules, MAX_SPACING)
    X_test  = build_rule_feature_matrix(test_enc,  selected_rules, MAX_SPACING)

    print("  Training Random Forest …")
    rf = BASELINE_MODELS["RandomForest"]()
    rf.fit(X_train, y_train)
    acc = rf.score(X_test, y_test)
    print(f"  RF test accuracy: {acc:.4f}")

    # SHAP TreeExplainer
    print("  Running TreeExplainer …")
    explainer = shap.TreeExplainer(rf, X_train[:200])
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_test), size=min(N_SHAP_TEST, len(X_test)), replace=False)
    X_explain = X_test[idx]
    y_explain  = y_test[idx]
    # shap 0.49+: returns (n_samples, n_features, n_classes)
    shap_values = explainer.shap_values(X_explain)          # (n, n_feat, n_cls)
    shap_values = np.array(shap_values)
    if shap_values.ndim == 2:                               # binary edge case
        shap_values = shap_values[:, :, np.newaxis]
    # (n_samples, n_features, n_classes)

    # Build human-readable rule labels using vocab inverse
    idx2api = {v: k for k, v in vocab.api2idx.items()}
    rule_labels = []
    for (a, b) in selected_rules:
        na = idx2api.get(a, f"tok{a}")
        nb = idx2api.get(b, f"tok{b}")
        na = na.replace("REFL:", "R:").split(".")[-1][:20]
        nb = nb.replace("REFL:", "R:").split(".")[-1][:20]
        rule_labels.append(f"{na}→{nb}")

    # ── Figure A: Global beeswarm ─────────────────────────────────────────
    print("  Plotting beeswarm …")
    # Mean |SHAP| across classes: (n_samples, n_features)
    shap_abs = np.abs(shap_values).mean(axis=-1)
    # Global feature importance: mean over samples
    feat_importance = shap_abs.mean(axis=0)                 # (n_features,)
    top_idx = np.argsort(feat_importance)[-TOP_K:][::-1]    # top-20 features

    shap_top = shap_abs[:, top_idx]    # (n_samples, TOP_K)
    X_top    = X_explain[:, top_idx]   # (n_samples, TOP_K)

    fig, ax = plt.subplots(figsize=(11, 7))
    for i in range(TOP_K):
        vals = shap_top[:, i]           # (n_samples,)
        feat_raw = X_top[:, i]
        feat_norm = (feat_raw - feat_raw.min()) / (feat_raw.max() - feat_raw.min() + 1e-9)
        jitter = np.random.default_rng(i).uniform(-0.3, 0.3, size=len(vals))
        sc = ax.scatter(vals, np.full(len(vals), i) + jitter,
                        c=feat_norm, cmap="RdBu_r", alpha=0.55, s=10, vmin=0, vmax=1)

    ax.set_yticks(range(TOP_K))
    ax.set_yticklabels([rule_labels[j] for j in top_idx], fontsize=7)
    ax.set_xlabel("Mean |SHAP| (averaged over classes)", fontsize=9)
    ax.set_title("RF SHAP — Top-20 Rules by Global Importance\n"
                 "(colour = feature value: blue=low, red=high)", fontsize=10)
    plt.colorbar(sc, ax=ax, label="Feature value (normalised)", fraction=0.02)
    plt.tight_layout()
    out = FIGURES / "shap_rf_beeswarm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")

    # ── Figure B: Per-class SHAP bar chart ───────────────────────────────
    print("  Plotting per-class bar chart …")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    fig.suptitle("RF SHAP — Mean |SHAP| per Rule, per Family (top-15)", fontsize=12)

    for ci, fname in enumerate(FAMILY_NAMES):
        ax = axes[ci]
        sv_cls = np.abs(shap_values[:, :, ci])    # (n_samples, n_features)
        mean_sv = sv_cls.mean(axis=0)              # (n_features,)
        top15   = np.argsort(mean_sv)[-15:][::-1]
        vals    = mean_sv[top15]
        lbls    = [rule_labels[j] for j in top15]
        ax.barh(range(len(vals)), vals[::-1], color=COLORS[ci], alpha=0.85)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(lbls[::-1], fontsize=6)
        ax.set_title(fname, fontsize=9, fontweight="bold", color=COLORS[ci])
        ax.set_xlabel("|SHAP|", fontsize=7)
        ax.tick_params(labelsize=6)

    for i in range(len(FAMILY_NAMES), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    out = FIGURES / "shap_rf_bar_per_class.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")

    return shap_abs, top_idx, rule_labels, selected_rules, vocab

# ── Part 2: Transformer SHAP ───────────────────────────────────────────────

def run_transformer_shap(seqs, labels):
    print("\n" + "=" * 60)
    print("Part 2: Transformer SHAP (GradientExplainer on embeddings)")
    print("=" * 60)

    model, vocab, cfg = load_transformer(use_gate=False)
    max_seq = cfg["max_seq_len"]

    splits = prepare_splits(seqs, labels.tolist(), n_folds=3, seed=SEED)
    _pt_cfg = MODELS / "plain_transformer_config.json"
    _best_fold = json.load(open(_pt_cfg))["best_fold"] if _pt_cfg.exists() else 2
    train_idx, test_idx = splits[_best_fold]

    train_seqs = [seqs[i] for i in train_idx]
    test_seqs  = [seqs[i] for i in test_idx]
    y_test     = labels[test_idx]

    def encode_batch(seq_list):
        enc = [[vocab.get(t, vocab.get("<UNK>", 1)) for t in s] for s in seq_list]
        arr = pad_with_truncation(enc, max_seq, truncation=TRUNCATION)
        return torch.tensor(arr, dtype=torch.long)

    print("  Computing embeddings for background and test sets …")
    rng = np.random.default_rng(SEED)
    back_idx = rng.choice(len(train_seqs), size=N_SHAP_BACK, replace=False)
    test_ex_idx = rng.choice(len(test_seqs), size=min(N_SHAP_TEST, len(test_seqs)), replace=False)

    back_ids  = encode_batch([train_seqs[i] for i in back_idx])
    test_ids  = encode_batch([test_seqs[i]  for i in test_ex_idx])
    y_explain = y_test[test_ex_idx]

    back_emb = get_embeddings(model, back_ids)   # (50, 512, 128)
    test_emb = get_embeddings(model, test_ids)   # (80, 512, 128)

    wrapper = TransformerFromEmbedding(model)
    wrapper.eval()

    print("  Running GradientExplainer (this may take ~2 min) …")
    explainer = shap.GradientExplainer(wrapper, back_emb)
    # shap_values: list[n_classes] of (n_test, seq_len, d_model)
    shap_vals = explainer.shap_values(test_emb)

    # Reduce d_model dimension: L2 norm over embedding dims → (n_classes, n_test, seq_len)
    shap_per_pos = np.array([
        np.linalg.norm(sv, axis=-1)   # (n_test, seq_len)
        for sv in shap_vals
    ])  # (n_classes, n_test, seq_len)

    # ── Figure C: per-family mean SHAP over sequence positions ─────────────
    print("  Plotting transformer SHAP summary …")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
    axes = axes.flatten()
    fig.suptitle(
        "Transformer SHAP — Mean |SHAP| per Sequence Position, per Family\n"
        "(GradientExplainer on embedding layer; each bar = one token position)",
        fontsize=11,
    )

    show_positions = 80   # first 80 positions are most informative after head-trunc

    for ci, fname in enumerate(FAMILY_NAMES):
        ax = axes[ci]
        # Samples of this class
        cls_mask = (y_explain == ci)
        if cls_mask.sum() == 0:
            ax.axis("off"); continue

        mean_shap = shap_per_pos[ci][cls_mask].mean(axis=0)[:show_positions]
        x = np.arange(show_positions)

        # Colour by magnitude
        norm = plt.Normalize(mean_shap.min(), mean_shap.max())
        colors = plt.get_cmap("YlOrRd")(norm(mean_shap))
        ax.bar(x, mean_shap, color=colors, width=1.0, edgecolor="none")
        ax.set_title(f"{fname} (n={cls_mask.sum()})", fontsize=8, fontweight="bold")
        ax.set_xlabel("Token position (head-truncated)", fontsize=7)
        ax.set_ylabel("Mean |SHAP|", fontsize=7)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    out = FIGURES / "shap_transformer_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")

    # ── Figure D: top-20 tokens by transformer SHAP (averaged over families) ─
    # Map positions → API names using the saved vocab inverse
    idx2api = {v: k for k, v in vocab.items()}

    # For each test sample, get the actual token at each position
    token_shap = defaultdict(list)
    for bi in range(len(test_ex_idx)):
        ids_np = test_ids[bi].numpy()
        # sum |SHAP| over classes for this sample → (seq_len,)
        sample_shap = shap_per_pos[:, bi, :].sum(axis=0)
        for pos, tid in enumerate(ids_np):
            if tid == 0:
                break
            tok = idx2api.get(int(tid), "<UNK>")
            if tok in ("<PAD>", "<CLS>", "<UNK>"):
                continue
            token_shap[tok].append(sample_shap[pos])

    token_mean = {t: np.mean(v) for t, v in token_shap.items() if len(v) >= 3}
    top_transformer = sorted(token_mean, key=lambda t: token_mean[t], reverse=True)[:TOP_K]

    return shap_per_pos, token_mean, top_transformer, test_ids, y_explain, idx2api

# ── Figure E: RF vs Transformer top-API comparison ─────────────────────────

def plot_rf_vs_transformer(rf_shap_abs, rf_top_idx, rf_rule_labels,
                            transformer_token_mean, transformer_top):
    print("\nPlotting RF vs Transformer comparison …")

    # RF: aggregate rules → API by extracting each API from "A→B" and summing
    rf_api_importance = defaultdict(float)
    for j, lbl in enumerate(rf_rule_labels):
        mean_imp = rf_shap_abs[:, j].mean()
        for part in lbl.split("→"):
            rf_api_importance[part.strip()] += mean_imp

    rf_top20 = sorted(rf_api_importance, key=lambda t: rf_api_importance[t], reverse=True)[:TOP_K]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        "Top-20 API Importance: Random Forest (SHAP) vs Plain Transformer (Gradient SHAP)\n"
        "RF importance is aggregated from rule-pair SHAP values; "
        "Transformer importance is gradient × embedding norm per token",
        fontsize=10,
    )

    # RF bar
    rf_vals  = [rf_api_importance[t] for t in rf_top20]
    rf_lbls  = [t[:30] for t in rf_top20]
    ax1.barh(range(len(rf_vals)), rf_vals[::-1], color="#1f77b4", alpha=0.85)
    ax1.set_yticks(range(len(rf_vals)))
    ax1.set_yticklabels(rf_lbls[::-1], fontsize=7)
    ax1.set_title("Random Forest (TreeExplainer SHAP)", fontsize=9, fontweight="bold")
    ax1.set_xlabel("Aggregated |SHAP| (sum over rules containing this API)", fontsize=8)

    # Transformer bar
    tr_vals = [transformer_token_mean[t] for t in transformer_top]
    tr_lbls = [t.replace("REFL:", "R:").split(".")[-1][:30] for t in transformer_top]
    ax2.barh(range(len(tr_vals)), tr_vals[::-1], color="#d62728", alpha=0.85)
    ax2.set_yticks(range(len(tr_vals)))
    ax2.set_yticklabels(tr_lbls[::-1], fontsize=7)
    ax2.set_title("Plain Transformer (GradientExplainer SHAP)", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Mean |SHAP| summed over classes", fontsize=8)

    # Mark tokens that appear in both top-20 lists
    rf_set = set(rf_top20)
    tr_set = set(transformer_top)
    overlap = rf_set & tr_set
    print(f"  API overlap between RF top-{TOP_K} and Transformer top-{TOP_K}: "
          f"{len(overlap)} / {TOP_K}")
    if overlap:
        print(f"  Shared: {', '.join(sorted(overlap)[:8])}")

    for ax in (ax1, ax2):
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    out = FIGURES / "shap_rf_vs_transformer.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading data …")
    seqs, labels = load_data()
    print(f"  {len(seqs)} samples, {len(np.unique(labels))} families")

    rf_shap_abs, rf_top_idx, rf_rule_labels, selected_rules, vocab_obj = run_rf_shap(seqs, labels)
    shap_per_pos, token_mean, top_transformer, test_ids, y_explain, idx2api = run_transformer_shap(seqs, labels)
    plot_rf_vs_transformer(rf_shap_abs, rf_top_idx, rf_rule_labels, token_mean, top_transformer)

    print("\nAll SHAP figures saved to results/figures/:")
    for p in sorted(FIGURES.glob("shap_*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
