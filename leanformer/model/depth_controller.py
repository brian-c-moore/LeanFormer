"""
DepthController — convergence-based adaptive computation depth.

Decides whether the model has converged sufficiently at each layer and can
exit early. The criterion: if hidden state changes by less than a threshold
between consecutive layers, the model has converged for this input.

This is the ConvergenceGovernor from Orkestratum applied to forward pass depth.
Simple inputs use fewer layers. Complex inputs use all of them.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class DepthController(nn.Module):

    def __init__(
        self,
        d_model: int,
        min_depth: int = 2,
        threshold: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.min_depth = min_depth
        self.threshold = threshold

        # Learned exit classifier: predicts whether to exit at this layer
        # Exit head outputs raw logits (no sigmoid) — sigmoid applied
        # manually where needed, and loss uses _with_logits for autocast safety
        self.exit_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
        )

    def should_exit(
        self,
        hidden: torch.Tensor,
        prev_hidden: torch.Tensor,
        layer_idx: int,
        training: bool = False,
    ) -> tuple[bool, float]:
        """
        Returns (should_exit, exit_confidence).
        Never exits during training — exit heads are trained via auxiliary loss only.
        """
        if training or layer_idx < self.min_depth:
            return False, 0.0

        # Measure how much the hidden state changed (convergence residual)
        with torch.no_grad():
            residual = (hidden - prev_hidden).norm(dim=-1).mean()
            prev_norm = prev_hidden.norm(dim=-1).mean().clamp(min=1e-8)
            residual_normalized = (residual / prev_norm).item()

            # Learned exit classifier (apply sigmoid to raw logits)
            # Pool over sequence dimension, average over batch
            exit_logits = self.exit_head(hidden.mean(dim=1))
            exit_prob = torch.sigmoid(exit_logits).mean().item()

        # Exit requires BOTH criteria: small residual AND classifier confidence
        should = residual_normalized < self.threshold and exit_prob > 0.7
        return should, exit_prob

    def compute_exit_loss(
        self,
        hidden: torch.Tensor,
        target_depth: int,
        current_depth: int,
    ) -> torch.Tensor:
        """
        Auxiliary loss that trains the exit classifier.
        Uses _with_logits for mixed precision (autocast) safety.
        """
        exit_logits = self.exit_head(hidden.mean(dim=1))

        if current_depth >= target_depth:
            target = torch.ones_like(exit_logits)
        else:
            target = torch.zeros_like(exit_logits)

        return F.binary_cross_entropy_with_logits(exit_logits, target)

    def get_exit_confidence(self, hidden: torch.Tensor) -> float:
        """Get exit probability for the current hidden state (for demo/logging)."""
        with torch.no_grad():
            return torch.sigmoid(self.exit_head(hidden.mean(dim=1))).mean().item()
