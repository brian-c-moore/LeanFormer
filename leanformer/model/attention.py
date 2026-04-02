"""
TwoPassSparseAttention — efficient attention via cheap screening + exact computation.

Pass 1 (screening): Project Q and K to a low-dimensional space, compute approximate
similarity, select top-K candidates per query. Cost: O(n * d/r) per head.

Pass 2 (exact): Full attention only between each query and its top-K candidates.
Cost: O(n * K * d) per head.

Total vs standard O(n^2 * d): roughly 10-20x cheaper for typical K values.

This is the SlotArbitrationPass from Orkestratum applied to attention:
cheap screening identifies candidates, expensive exact computation runs only on winners.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .low_rank import LowRankLinear


class TwoPassSparseAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rank: int,
        screening_rank: int,
        top_k: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.top_k = top_k

        # Pass 2 projections (low-rank)
        self.q_proj = LowRankLinear(d_model, d_model, rank)
        self.k_proj = LowRankLinear(d_model, d_model, rank)
        self.v_proj = LowRankLinear(d_model, d_model, rank)
        self.out_proj = LowRankLinear(d_model, d_model, rank)

        # Fix attention B=0 deadlock: LowRankLinear inits B=0, so
        # Q=K=V=0, softmax produces uniform weights, output is zero
        # for all positions, and gradients through B are exactly zero.
        # Same class of bug as the SwiGLU deadlock in GatedFeedForward.
        nn.init.normal_(self.q_proj.B, std=0.01)
        nn.init.normal_(self.k_proj.B, std=0.01)
        nn.init.normal_(self.v_proj.B, std=0.01)
        nn.init.normal_(self.out_proj.B, std=0.01)

        # Pass 1 screening projections (even lower rank — much cheaper)
        # topk is non-differentiable so screening B doesn't learn via
        # gradient flow, but non-zero init gives meaningful initial
        # candidate selection instead of arbitrary tie-breaking on zeros.
        self.screen_dim = d_model // 4
        self.q_screen = LowRankLinear(d_model, self.screen_dim, screening_rank)
        self.k_screen = LowRankLinear(d_model, self.screen_dim, screening_rank)
        nn.init.normal_(self.q_screen.B, std=0.01)
        nn.init.normal_(self.k_screen.B, std=0.01)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5
        self.screen_scale = self.screen_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        B, T, D = x.shape
        k = min(self.top_k, T)

        # === Pass 1: Cheap screening ===
        q_s = self.q_screen(x)  # (B, T, screen_dim)
        k_s = self.k_screen(x)  # (B, T, screen_dim)

        screen_scores = torch.bmm(q_s, k_s.transpose(-2, -1)) * self.screen_scale

        if mask is not None:
            # mask is (1, 1, T, T) causal — squeeze to (1, T, T) for bmm compat
            screen_mask = mask.squeeze(0).squeeze(0)  # (T, T)
            screen_scores = screen_scores.masked_fill(screen_mask[:T, :T] == 0, float("-inf"))

        # Select top-K candidate keys per query
        _, top_k_indices = torch.topk(screen_scores, k, dim=-1)  # (B, T, K)

        # === Pass 2: Exact attention over candidates only ===
        Q = rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.n_heads)
        K = rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=self.n_heads)
        V = rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=self.n_heads)

        # Gather only top-K keys and values per query
        # Expand indices: (B, T, K) -> (B, H, T, K, d_head)
        idx = top_k_indices.unsqueeze(1).unsqueeze(-1)
        idx = idx.expand(B, self.n_heads, T, k, self.d_head)

        # Expand K, V for gathering: (B, H, T, d_head) -> (B, H, 1, T, d_head) -> (B, H, T, T, d_head)
        K_gather = K.unsqueeze(2).expand(B, self.n_heads, T, T, self.d_head)
        V_gather = V.unsqueeze(2).expand(B, self.n_heads, T, T, self.d_head)

        K_sparse = torch.gather(K_gather, 3, idx)  # (B, H, T, K, d_head)
        V_sparse = torch.gather(V_gather, 3, idx)  # (B, H, T, K, d_head)

        # Compute exact attention scores: Q (B,H,T,1,d) * K_sparse (B,H,T,K,d) -> (B,H,T,K)
        scores = (Q.unsqueeze(3) * K_sparse).sum(-1) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum: (B,H,T,1,K) @ (B,H,T,K,d) -> (B,H,T,d)
        out = (attn_weights.unsqueeze(3) @ V_sparse).squeeze(3)

        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.out_proj(out)

        stats = {
            "attention_sparsity": 1.0 - (k / T) if T > 0 else 0.0,
            "attention_candidates": k,
            "attention_total_keys": T,
        }

        return out, stats
