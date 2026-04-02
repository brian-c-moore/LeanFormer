"""
GatedFeedForward — sparse activation via a cheap gate predictor.

At training time: all neurons compute (for gradient flow), but a gate network
is trained as a side objective to predict which neurons will activate significantly.

At inference time: the gate screens neurons, only predicted-active ones compute.
Target: 70-80% of neurons skipped.

This is the visibility buffer principle: identify winners cheaply before doing
expensive computation. The gate is the SlotArbitrationPass over the neuron space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .low_rank import LowRankLinear


class GatedFeedForward(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        rank: int,
        gate_rank: int,
        sparsity_target: float = 0.8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.sparsity_target = sparsity_target

        # Main computation (low-rank SwiGLU)
        self.up_proj = LowRankLinear(d_model, d_ff, rank)
        self.gate_proj = LowRankLinear(d_model, d_ff, rank)
        self.down_proj = LowRankLinear(d_ff, d_model, rank)

        # Fix SwiGLU multiplicative dead zone: LowRankLinear inits B=0
        # (LoRA pattern). For SwiGLU gating hidden = up(x) * SiLU(gate(x)),
        # both outputs are zero when B=0, so neither receives gradients —
        # a permanent deadlock. Break it by giving all three FF projection
        # B matrices small random values. down_proj also needs this:
        # even with non-zero hidden, down_proj(hidden) = hidden @ A @ 0 = 0,
        # making the entire FF contribution zero on the first forward pass.
        nn.init.normal_(self.up_proj.B, std=0.01)
        nn.init.normal_(self.gate_proj.B, std=0.01)
        nn.init.normal_(self.down_proj.B, std=0.01)

        # Cheap activation gate — predicts which neurons matter
        self.activation_gate = LowRankLinear(d_model, d_ff, gate_rank, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        training: bool = False,
    ) -> tuple[torch.Tensor, dict]:

        if training:
            return self._forward_train(x)
        else:
            return self._forward_inference(x)

    def _forward_train(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Full computation + gate training objective."""
        gate_logits = self.activation_gate(x)

        up = self.up_proj(x)
        gate = self.act(self.gate_proj(x))
        hidden = up * gate  # SwiGLU

        # Gate loss: train to predict which neurons activate significantly
        # Use _with_logits for mixed precision (autocast) safety
        with torch.no_grad():
            actual_active = (hidden.abs() > 0.01).float()
        gate_loss = F.binary_cross_entropy_with_logits(gate_logits, actual_active)

        out = self.dropout(self.down_proj(hidden))

        stats = {
            "gate_loss": gate_loss,
            "ff_sparsity": 0.0,  # no sparsity during training
            "ff_active_neurons": self.d_ff,
            "ff_total_neurons": self.d_ff,
        }
        return out, stats

    def _forward_inference(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Sparse computation — only predicted-active neurons fire."""
        gate_scores = torch.sigmoid(self.activation_gate(x))

        # Keep only the top (1 - sparsity_target) fraction of neurons
        k = max(1, int(self.d_ff * (1.0 - self.sparsity_target)))
        original_shape = gate_scores.shape
        flat_scores = gate_scores.reshape(-1, self.d_ff)  # (B*T, d_ff)

        # Build a mask by selecting exactly the top-k neurons per row.
        # This guarantees the sparsity target even when scores are tied.
        _, topk_indices = torch.topk(flat_scores, k, dim=-1)
        active_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
        active_mask.scatter_(1, topk_indices, True)
        active_mask = active_mask.reshape(original_shape)

        sparsity = 1.0 - active_mask.float().mean().item()

        up = self.up_proj(x)
        gate = self.act(self.gate_proj(x))
        hidden = up * gate

        # Zero out inactive neurons
        # (In a production CUDA kernel, these would be skipped entirely)
        hidden = hidden * active_mask.float()

        out = self.down_proj(hidden)

        active_count = int(active_mask.float().sum(-1).mean().item())
        stats = {
            "ff_sparsity": sparsity,
            "ff_active_neurons": active_count,
            "ff_total_neurons": self.d_ff,
        }
        return out, stats
