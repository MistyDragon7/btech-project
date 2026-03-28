"""
Explainability module for GAME-Mal.

Extracts gating scores from the trained model and produces:
1. Per-family top-K most important API calls (by gating score)
2. Gating score heatmaps
3. Attention weight visualizations
4. Sparsity analysis (matching Qiu et al. findings)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import GAMEMal
from src.preprocessing import APIVocabulary

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_gate_scores(
    model: GAMEMal,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """
    Extract gating scores for all samples.

    Returns dict with:
        gate_scores: list of (n_heads, seq_len, d_k) per layer
        per_token_importance: (n_samples, seq_len) — mean gate score per token
    """
    model.eval()
    all_token_importance = []

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        x_batch = torch.from_numpy(X[start:end]).long().to(device)
        logits, info = model(x_batch, return_attention=True)

        # Average gate scores across heads and d_k dimensions, use last layer
        # gate_scores shape: (B, n_heads, N, d_k)
        last_gate = info["gate_scores"][-1]  # last layer
        # Mean over heads and d_k → (B, N)
        token_importance = last_gate.mean(dim=(1, 3)).cpu().numpy()
        all_token_importance.append(token_importance)

    return {
        "per_token_importance": np.concatenate(all_token_importance, axis=0),
    }


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
    based on average gating scores.

    This IS the explainability: the gating mechanism learned which APIs
    matter most for classifying each family.
    """
    results = {}

    for class_idx, family in enumerate(family_names):
        mask = (y == class_idx)
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
            token_id: np.mean(scores)
            for token_id, scores in api_importance.items()
        }

        # Sort and get top K
        sorted_apis = sorted(api_avg.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results[family] = [
            (vocab.idx2api.get(token_id, f"UNK_{token_id}"), score)
            for token_id, score in sorted_apis
        ]

    return results


def compute_sparsity_stats(token_importance: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    """
    Compute sparsity statistics of gating scores (to compare with Qiu et al.).
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
        bars = ax.barh(range(len(names)), scores, color=sns.color_palette("viridis", len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([f"{n}\n({fn})" for n, fn in zip(names, full_names)], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Average Gating Score (Importance)")
        ax.set_title(f"Top-{top_k} Important API Calls — {family}")
        plt.tight_layout()
        plt.savefig(output_dir / f"top_apis_{family}.png", dpi=150)
        plt.close()


def plot_gate_score_distribution(
    token_importance: np.ndarray,
    X: np.ndarray,
    output_dir: Path,
):
    """Distribution of gating scores (compare with Fig. 3 in Qiu et al.)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    non_pad = X != 0
    scores = token_importance[non_pad]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(np.mean(scores), color="red", linestyle="--",
               label=f"Mean = {np.mean(scores):.3f}")
    ax.set_xlabel("Gating Score")
    ax.set_ylabel("Normalized Density")
    ax.set_title("GAME-Mal: SDPA Output Gating Score Distribution")
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=family_names, yticklabels=family_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    safe_title = title.replace(" ", "_").replace("/", "_")
    plt.savefig(output_dir / f"confusion_{safe_title}.png", dpi=150)
    plt.close()
