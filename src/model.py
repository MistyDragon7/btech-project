"""
GAME-Mal: Gated Attention over Markov Embeddings for Malware classification.

Architecture:
1. API Embedding layer (learns representations of API calls)
2. Positional encoding (sequence position matters for behavioral patterns)
3. Multi-head self-attention with SDPA output sigmoid gating (from Qiu et al. 2025)
4. Classification head with pooling

The sigmoid gating after SDPA:
- Introduces non-linearity (breaks low-rank bottleneck between V and W_O)
- Creates input-dependent sparsity (learns which patterns matter)
- Gating scores ARE the explanation (no post-hoc method needed)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with head-specific sigmoid gating after SDPA.

    This is the G1 position from Qiu et al. (2025), which they found to be
    the most effective: it introduces non-linearity upon the low-rank mapping
    and applies query-dependent sparse gating scores to SDPA output.

    gate_scores = sigmoid(X @ W_gate)  # head-specific, element-wise
    output = gate_scores * softmax(QK^T/sqrt(d_k)) @ V
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # QKV projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model)

        # HEAD-SPECIFIC SIGMOID GATE (the key innovation from Qiu et al.)
        # Produces gating scores with shape (batch, n_heads, seq_len, d_k)
        # Each head gets its own gate projection
        self.w_gate = nn.Linear(d_model, d_model)

        # Initialize gate bias negative to encourage sparsity (sigmoid(-2) ≈ 0.12)
        # This matches the mean gating score of ~0.116 reported in Qiu et al.
        nn.init.constant_(self.w_gate.bias, -2.0)
        nn.init.normal_(self.w_gate.weight, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Storage for explainability
        self._last_gate_scores = None
        self._last_attn_weights = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) — True for padding positions
            return_attention: if True, return attention weights and gate scores

        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, n_heads, seq_len, seq_len) or None
            gate_scores: (batch, n_heads, seq_len, d_k) or None
        """
        B, N, D = x.shape

        # QKV projections → (B, n_heads, N, d_k)
        Q = self.w_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # mask shape: (B, N) → (B, 1, 1, N)
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # SDPA output
        sdpa_out = torch.matmul(attn_weights, V)  # (B, n_heads, N, d_k)

        # ═══════════════════════════════════════════════════════
        # GATING: Head-specific sigmoid gate after SDPA (G1)
        # This is the core contribution adapted from Qiu et al.
        # ═══════════════════════════════════════════════════════
        gate_input = self.w_gate(x)  # (B, N, D)
        gate_input = gate_input.view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        gate_scores = torch.sigmoid(gate_input)  # (B, n_heads, N, d_k)

        # Apply gate: element-wise multiplication
        gated_out = sdpa_out * gate_scores  # sparse, query-dependent filtering

        # Concatenate heads and project
        gated_out = gated_out.transpose(1, 2).contiguous().view(B, N, D)
        output = self.w_o(gated_out)
        output = self.dropout(output)

        # Store for explainability
        if return_attention:
            self._last_gate_scores = gate_scores.detach()
            self._last_attn_weights = attn_weights.detach()
            return output, attn_weights.detach(), gate_scores.detach()

        return output, None, None


class GAMEMalBlock(nn.Module):
    """Single transformer block with gated attention + FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = GatedMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None, return_attention=False):
        # Pre-norm architecture
        normed = self.norm1(x)
        attn_out, attn_w, gate_s = self.attention(normed, mask, return_attention)
        x = x + attn_out

        normed = self.norm2(x)
        x = x + self.ffn(normed)

        return x, attn_w, gate_s


class GAMEMal(nn.Module):
    """
    GAME-Mal: Gated Attention Markov Embeddings for Malware classification.

    Takes encoded API call sequences, applies gated attention, and classifies
    into malware families. The gating scores provide built-in explainability.
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
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # Embeddings
        self.api_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer blocks with gated attention
        self.blocks = nn.ModuleList([
            GAMEMalBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: (batch, seq_len) — integer-encoded API call sequences
            return_attention: if True, collect attention/gate info for explainability

        Returns:
            logits: (batch, num_classes)
            info: dict with 'gate_scores', 'attn_weights' per layer (if requested)
        """
        B, N = x.shape

        # Padding mask: True where padded
        pad_mask = (x == 0)

        # Embeddings
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.api_embedding(x) + self.pos_embedding(positions)
        h = self.embed_dropout(h)

        # Transformer blocks
        info = {"gate_scores": [], "attn_weights": []}
        for block in self.blocks:
            h, attn_w, gate_s = block(h, pad_mask, return_attention)
            if return_attention:
                info["gate_scores"].append(gate_s)
                info["attn_weights"].append(attn_w)

        h = self.final_norm(h)

        # Global average pooling (ignoring padding)
        mask_expanded = (~pad_mask).unsqueeze(-1).float()  # (B, N, 1)
        h_sum = (h * mask_expanded).sum(dim=1)  # (B, d_model)
        lengths = mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
        h_pooled = h_sum / lengths

        logits = self.classifier(h_pooled)

        return logits, info
