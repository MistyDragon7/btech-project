"""
Explainability utilities.

This module computes **token importance from CLS attention maps**.

Assumptions:
- The model prepends a learnable [CLS] token at position 0.
- The classifier reads the final [CLS] representation.
- Attention weights returned by the model are standard softmax self-attention
  weights of shape (B, n_heads, seq, seq).

What we extract:
- `per_token_importance`: (n_samples, seq_len) importance score per token,
  derived from how much the CLS token attends to each input position.

Notes on validity:
- Attention is not guaranteed to be a faithful explanation in general.
  You should validate with deletion tests (mask top-k by this score and
  measure confidence drop vs random).
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import MalwareTransformer
from src.preprocessing import APIVocabulary

logger = logging.getLogger(__name__)


def _cls_attention_to_token_importance(
    attn_weights: torch.Tensor,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Convert attention weights into per-token importance using CLS attention.

    Args:
        attn_weights: (B, n_heads, S, S) attention for a single layer.
        pad_mask: (B, S) True where PAD. Must include CLS position at index 0.

    Returns:
        importance: (B, S) where importance[b, j] = mean_h attn[b, h, CLS, j],
                    with PAD positions forced to 0 and CLS position set to 0.
    """
    # CLS query row: (B, n_heads, S)
    cls_row = attn_weights[:, :, 0, :]
    imp = cls_row.mean(dim=1)  # (B, S)

    # Zero out CLS itself and PAD
    imp = imp.clone()
    imp[:, 0] = 0.0
    imp[pad_mask] = 0.0

    return imp


@torch.no_grad()
def extract_cls_attention_scores(
    model: MalwareTransformer,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
    layer_aggregation: str = "mean",
) -> Dict[str, np.ndarray]:
    """
    Extract token-importance scores from CLS attention maps.

    Args:
        model: classifier model that supports return_attention=True
        X: (n_samples, seq_len) integer-encoded sequences INCLUDING CLS at index 0
        device: torch device
        batch_size: batch size
        layer_aggregation:
            - "last": use last layer only
            - "mean": mean over layers

    Returns:
        dict with:
            per_token_importance: (n_samples, seq_len) float importance per token
    """
    model.eval()
    all_importance: List[np.ndarray] = []

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        x_batch = torch.from_numpy(X[start:end]).long().to(device)

        # pad_mask includes CLS position; CLS is never PAD, so pad_mask[:,0] should be False
        pad_mask = x_batch == 0

        _, info = model(x_batch, return_attention=True)
        attn_list = info.get("attn_weights", [])

        if not attn_list:
            raise ValueError(
                "Model did not return attention weights (info['attn_weights'] is empty)."
            )

        if layer_aggregation not in ("last", "mean"):
            raise ValueError("layer_aggregation must be 'last' or 'mean'")

        if layer_aggregation == "last":
            imp = _cls_attention_to_token_importance(attn_list[-1], pad_mask)
        else:
            per_layer = []
            for a in attn_list:
                per_layer.append(_cls_attention_to_token_importance(a, pad_mask))
            imp = torch.stack(per_layer, dim=0).mean(dim=0)

        all_importance.append(imp.cpu().numpy())

    return {"per_token_importance": np.concatenate(all_importance, axis=0)}


def get_top_apis_per_family(
    X: np.ndarray,
    y: np.ndarray,
    token_importance: np.ndarray,
    vocab: APIVocabulary,
    family_names: List[str],
    top_k: int = 15,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    For each malware family, find the top-K most important API calls
    based on average token-importance scores.

    In the new setup, token importance is derived from CLS attention maps
    (how strongly CLS attends to each token), not from gated attention.
    """
    results = {}

    for class_idx, family in enumerate(family_names):
        mask = y == class_idx
        if mask.sum() == 0:
            results[family] = []
            continue

        family_X = X[mask]
        family_importance = token_importance[mask]

        # Accumulate importance per API
        api_importance: Dict[int, List[float]] = {}
        for i in range(len(family_X)):
            for j in range(family_X.shape[1]):
                token_id = family_X[i, j]
                if token_id == 0:  # padding
                    continue
                imp = family_importance[i, j]
                if token_id not in api_importance:
                    api_importance[token_id] = []
                api_importance[token_id].append(imp)

        # Average importance per API
        api_avg = {
            token_id: np.mean(scores) for token_id, scores in api_importance.items()
        }

        # Sort and get top K
        sorted_apis = sorted(api_avg.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results[family] = [
            (vocab.idx2api.get(token_id, f"UNK_{token_id}"), score)
            for token_id, score in sorted_apis
        ]

    return results


def compute_sparsity_stats(
    token_importance: np.ndarray, X: np.ndarray
) -> Dict[str, float]:
    """
    Compute distribution statistics of token-importance scores.

    (This is no longer "gate sparsity" in the Qiu et al. sense; it's a
    summary of the attention-derived importance distribution.)
    """
    # Only consider non-padding positions
    non_pad = X != 0
    scores = token_importance[non_pad]

    return {
        "mean_gate_score": float(np.mean(scores)),
        "median_gate_score": float(np.median(scores)),
        "std_gate_score": float(np.std(scores)),
        "pct_below_0.1": float((scores < 0.1).mean()),
        "pct_below_0.3": float((scores < 0.3).mean()),
        "pct_below_0.5": float((scores < 0.5).mean()),
    }


# ── Visualization ───────────────────────────────────────────────────────────


def plot_top_apis(
    top_apis: Dict[str, List[Tuple[str, float]]],
    output_dir: Path,
    top_k: int = 10,
):
    """Bar chart of top-K important APIs per family."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for family, apis in top_apis.items():
        if not apis:
            continue
        apis_to_plot = apis[:top_k]
        names = [a[0].split(".")[-1] if "." in a[0] else a[0] for a in apis_to_plot]
        full_names = [a[0] for a in apis_to_plot]
        scores = [a[1] for a in apis_to_plot]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(
            range(len(names)), scores, color=sns.color_palette("viridis", len(names))
        )
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(
            [f"{n}\n({fn})" for n, fn in zip(names, full_names)], fontsize=8
        )
        ax.invert_yaxis()
        ax.set_xlabel("Average Token Importance (CLS-attention-derived)")
        ax.set_title(f"Top-{top_k} Important API Calls — {family}")
        plt.tight_layout()
        plt.savefig(output_dir / f"top_apis_{family}.png", dpi=150)
        plt.close()


def plot_gate_score_distribution(
    token_importance: np.ndarray,
    X: np.ndarray,
    output_dir: Path,
):
    """
    Distribution of token-importance scores.

    Kept function name for minimal churn, but this is now attention-derived
    importance (CLS attention), not gating scores.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    non_pad = X != 0
    scores = token_importance[non_pad]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        scores, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white"
    )
    ax.axvline(
        np.mean(scores),
        color="red",
        linestyle="--",
        label=f"Mean = {np.mean(scores):.3f}",
    )
    ax.set_xlabel("Token Importance (CLS-attention-derived)")
    ax.set_ylabel("Normalized Density")
    ax.set_title("Transformer: CLS-Attention Token Importance Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "gate_score_distribution.png", dpi=150)
    plt.close()


def plot_training_history(history: List[Dict], output_dir: Path):
    """Plot training/test loss and F1 over epochs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    test_loss = [h["test_loss"] for h in history]
    test_f1 = [h["test_f_score"] for h in history]
    test_acc = [h["test_accuracy"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss, label="Train Loss", color="steelblue")
    axes[0].plot(epochs, test_loss, label="Test Loss", color="coral")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Test Loss")
    axes[0].legend()

    axes[1].plot(epochs, test_acc, label="Accuracy", color="seagreen")
    axes[1].plot(epochs, test_f1, label="F1-Score", color="darkorange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Test Accuracy and F1-Score")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=150)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    family_names: List[str],
    title: str,
    output_dir: Path,
):
    """Plot confusion matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    from sklearn.metrics import confusion_matrix as cm_fn

    cm = cm_fn(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=family_names,
        yticklabels=family_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    safe_title = title.replace(" ", "_").replace("/", "_")
    plt.savefig(output_dir / f"confusion_{safe_title}.png", dpi=150)
    plt.close()
