"""
Plain Transformer encoder for Android malware family classification with
CLS-token-based attention explainability.

Key design choices:
- Learnable [CLS] token prepended to every sequence.
- Classification uses the final hidden state of [CLS].
- Explainability uses attention maps specifically "with respect to the class token":
  i.e., for each layer/head we expose the attention row from CLS -> all tokens.

This file intentionally removes all gated-attention logic to reduce redundancy and
avoid "explainability by gate" claims. Attention maps are returned for downstream
analysis and faithfulness tests (e.g., deletion tests masking top-k CLS-attended tokens).

Tensor shapes:
- Input x: (B, N) where N is max_seq_len, padded with 0 (PAD).
- Internally we create: (B, N+1) with CLS at position 0.
- Attention weights per layer: (B, H, N+1, N+1)

Note:
- Padding mask is applied so PAD tokens are not attended-to.
- CLS is never masked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformerInfo:
    """
    Lightweight container for explainability.

    attn_weights: list of length n_layers.
                  Each entry is a tensor (B, n_heads, S, S) where S = N+1 (CLS+tokens).
    cls_index: index of CLS token in the sequence (always 0).
    token_ids_with_cls: the input ids with CLS prepended: (B, S)
    pad_mask_with_cls: True where padded: (B, S)
    """

    attn_weights: List[torch.Tensor]
    cls_index: int
    token_ids_with_cls: torch.Tensor
    pad_mask_with_cls: torch.Tensor


class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention that can optionally return attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # for numerical stability when all tokens are masked (should not happen due to CLS)
        self._neg_inf = -1e9

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, S, D)
            pad_mask: (B, S) True for PAD positions (these keys are masked out)
            return_attention: whether to return attention weights

        Returns:
            out: (B, S, D)
            attn: (B, H, S, S) if return_attention else None
        """
        B, S, D = x.shape

        # Project to Q, K, V and reshape to (B, H, S, d_k)
        Q = self.w_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores: (B, H, S, S)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if pad_mask is not None:
            # Mask out PAD *keys* so nobody attends to padding.
            # pad_mask: (B, S) -> (B, 1, 1, S)
            key_mask = pad_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_mask, self._neg_inf)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, S, d_k)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.w_o(out)
        out = self.out_dropout(out)

        if return_attention:
            return out, attn.detach()
        return out, None


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer encoder block (MHA + FFN).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.norm1(x)
        attn_out, attn_w = self.attn(
            h, pad_mask=pad_mask, return_attention=return_attention
        )
        x = x + attn_out

        h = self.norm2(x)
        x = x + self.ffn(h)

        return x, attn_w


class MalwareTransformer(nn.Module):
    """
    Plain Transformer encoder classifier with a learnable CLS token.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx

        # Token embeddings (+1 because we prepend CLS, but CLS itself is a vector param)
        self.api_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # Positional embeddings for sequence WITH CLS => length max_seq_len+1
        self.pos_embedding = nn.Embedding(max_seq_len + 1, d_model)

        # Learnable CLS token embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.embed_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def _prepend_cls(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepend CLS token to token id tensor.

        Args:
            x: (B, N)

        Returns:
            x_with_cls: (B, N+1)
            pad_mask_with_cls: (B, N+1) True where PAD
        """
        B, N = x.shape
        if N != self.max_seq_len:
            # We expect already padded/truncated inputs at max_seq_len.
            # If a script passes a different length, positional embeddings will still work
            # up to max_seq_len, but we keep the check to surface pipeline mistakes early.
            if N > self.max_seq_len:
                raise ValueError(
                    f"Input seq_len={N} exceeds max_seq_len={self.max_seq_len}"
                )

        pad_mask = x == self.pad_idx  # (B, N)

        # Create a dummy CLS id column (not used for embedding lookup)
        cls_col = torch.full(
            (B, 1), fill_value=self.pad_idx, device=x.device, dtype=x.dtype
        )
        x_with_cls = torch.cat([cls_col, x], dim=1)  # (B, N+1)

        # CLS is never padding
        cls_pad = torch.zeros((B, 1), device=x.device, dtype=torch.bool)
        pad_mask_with_cls = torch.cat([cls_pad, pad_mask], dim=1)  # (B, N+1)

        return x_with_cls, pad_mask_with_cls

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: (B, N) int token ids (PAD=0)
            return_attention: if True returns attention weights for each block

        Returns:
            logits: (B, C)
            info: dict containing attention maps and metadata for explainability
        """
        B, N = x.shape
        x_with_cls, pad_mask_with_cls = self._prepend_cls(x)  # (B, S)
        S = x_with_cls.shape[1]
        cls_index = 0

        # Build embeddings (CLS uses parameter vector, other tokens use embedding table)
        token_emb = self.api_embedding(x)  # (B, N, D)
        cls_emb = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        h = torch.cat([cls_emb, token_emb], dim=1)  # (B, S, D)

        # Add positional embeddings
        pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)  # (B, S)
        h = h + self.pos_embedding(pos)

        h = self.embed_dropout(h)

        attn_weights: List[torch.Tensor] = []
        for blk in self.blocks:
            h, attn_w = blk(
                h, pad_mask=pad_mask_with_cls, return_attention=return_attention
            )
            if return_attention and attn_w is not None:
                attn_weights.append(attn_w)

        h = self.final_norm(h)

        # Classification from [CLS] final hidden state
        cls_h = h[:, cls_index, :]  # (B, D)
        logits = self.classifier(cls_h)

        if not return_attention:
            return logits, {}

        info = TransformerInfo(
            attn_weights=attn_weights,
            cls_index=cls_index,
            token_ids_with_cls=x_with_cls.detach(),
            pad_mask_with_cls=pad_mask_with_cls.detach(),
        )

        # Return as a plain dict to keep backward compatibility with existing callers.
        return logits, {
            "attn_weights": info.attn_weights,
            "cls_index": info.cls_index,
            "token_ids_with_cls": info.token_ids_with_cls,
            "pad_mask_with_cls": info.pad_mask_with_cls,
        }


# Backward-compat alias removed.
