"""
Training and evaluation loop for the Transformer-based malware classifier.

This module trains/evaluates the plain Transformer model defined in `src/model.py`
(which prepends a learnable [CLS] token and can optionally return attention maps
for explainability).
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines import compute_metrics
from src.model import MalwareTransformer

logger = logging.getLogger(__name__)


def create_dataloader(
    sequences: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """Create a PyTorch DataLoader from numpy arrays."""
    X = torch.from_numpy(sequences).long()
    y = torch.from_numpy(labels).long()
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(
    model: MalwareTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits, _ = model(X_batch, return_attention=False)
        loss = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: MalwareTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model. Returns loss, metrics, predictions, and probabilities."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits, _ = model(X_batch, return_attention=False)
        loss = criterion(logits, y_batch)

        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_probs)

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(y_true, y_pred, y_score, num_classes)

    return avg_loss, metrics, y_pred, y_score


def train_transformer(
    vocab_size: int,
    num_classes: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 256,
    max_seq_len: int = 512,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10,
    device: torch.device = None,
) -> Tuple[MalwareTransformer, Dict[str, float], List[Dict], np.ndarray]:
    """
    Full training pipeline for the plain Transformer classifier.

    Returns:
        model: trained MalwareTransformer model
        best_metrics: metrics on test set at best epoch
        history: list of per-epoch metrics
        y_pred: predictions on test set
    """
    if device is None:
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    logger.info("Using device: %s", device)

    # Create model
    model = MalwareTransformer(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d (%.2f K)", param_count, param_count / 1000)

    # Handle class imbalance with weighted loss
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float32)
    class_weights = class_weights / class_weights.sum() * num_classes
    weight_tensor = torch.from_numpy(class_weights).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = create_dataloader(X_train, y_train, batch_size, shuffle=True)
    test_loader = create_dataloader(X_test, y_test, batch_size, shuffle=False)

    # Training loop with early stopping
    best_f1 = 0.0
    best_metrics = {}
    best_state = None
    best_pred = None
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, metrics, y_pred, y_score = evaluate(
            model, test_loader, criterion, device, num_classes
        )
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                **{f"test_{k}": v for k, v in metrics.items()},
            }
        )

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d | train_loss=%.4f | test_loss=%.4f | acc=%.4f | f1=%.4f",
                epoch,
                train_loss,
                test_loss,
                metrics["accuracy"],
                metrics["f_score"],
            )

        # Early stopping on F1
        if metrics["f_score"] > best_f1:
            best_f1 = metrics["f_score"]
            best_metrics = metrics.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_pred = y_pred.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d (best F1=%.4f)", epoch, best_f1)
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, best_metrics, history, best_pred


# Backward-compatible alias for legacy callers
train_game_mal = train_transformer
