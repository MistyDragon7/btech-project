"""
BiLSTM baseline for GAME-Mal benchmark.

Standard 2-layer bidirectional LSTM with mean-pooling over non-pad positions
and a linear classifier head. Same training recipe as GAME-Mal:
class-weighted CE, AdamW + cosine LR + grad clip + early stop on macro-F1.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) long
        mask = (x != self.pad_idx).float().unsqueeze(-1)  # (B, T, 1)
        emb = self.embedding(x)
        out, _ = self.lstm(emb)  # (B, T, 2*d_model)
        # Mean-pool over non-pad positions
        summed = (out * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def train_bilstm(
    vocab_size: int,
    num_classes: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    d_model: int = 128,
    n_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 12,
    device: torch.device = None,
) -> Tuple[nn.Module, Dict[str, float]]:
    from src.train import create_dataloader, evaluate
    from src.baselines import compute_metrics

    if device is None:
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=d_model,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_w = len(y_train) / (num_classes * np.maximum(counts, 1.0))
    class_w_t = torch.from_numpy(class_w).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w_t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = create_dataloader(X_train, y_train, batch_size, shuffle=True)
    test_loader = create_dataloader(X_test, y_test, batch_size, shuffle=False)

    best_f1 = -1.0
    best_metrics: Dict[str, float] = {}
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        total = 0.0
        n = 0
        for X_b, y_b in train_loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
            n += 1
        scheduler.step()
        train_loss = total / max(n, 1)

        # eval
        model.eval()
        all_pred, all_true, all_score = [], [], []
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b = X_b.to(device)
                logits = model(X_b)
                probs = torch.softmax(logits, dim=-1)
                all_pred.append(logits.argmax(dim=-1).cpu().numpy())
                all_true.append(y_b.numpy())
                all_score.append(probs.cpu().numpy())
        y_pred = np.concatenate(all_pred)
        y_true = np.concatenate(all_true)
        y_score = np.concatenate(all_score)
        metrics = compute_metrics(y_true, y_pred, y_score, num_classes)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "BiLSTM ep=%3d train_loss=%.4f acc=%.4f f1=%.4f",
                epoch, train_loss, metrics["accuracy"], metrics["f_score"],
            )

        if metrics["f_score"] > best_f1:
            best_f1 = metrics["f_score"]
            best_metrics = metrics.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("BiLSTM early stop ep=%d best_f1=%.4f", epoch, best_f1)
                break

    return model, best_metrics
