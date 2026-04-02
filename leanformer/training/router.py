"""
Gradient Router

Routes individual samples to relevant parameter groups via a small MLP.
Sigmoid activation (independent group scores), top-k selection with
straight-through estimator. Entropy regularization prevents routing
collapse. Gradient masking zeros non-selected group gradients post-backward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any


@dataclass
class RouterConfig:
    """Configuration for the gradient router."""
    # Router architecture
    hidden_dim_divisor: int = 4  # hidden = model_dim // divisor

    # Top-k selection
    min_active_groups: int = 2
    max_active_groups: int | None = None  # None = num_groups // 2

    # Entropy regularization
    entropy_coeff: float = 0.01
    # Increase entropy coeff if any group falls below this utilization
    utilization_floor: float = 0.1
    entropy_boost_factor: float = 2.0

    # Warmup: observation mode for first N steps (no gradient masking)
    warmup_steps: int = 50

    # Routing tracking window for load balance
    balance_window: int = 100


class GradientRouter(nn.Module):
    """Routes samples to parameter groups for selective gradient computation.

    Architecture: Linear(model_dim, hidden) → GELU → Linear(hidden, num_groups) → Sigmoid
    Selection: top-k with straight-through estimator for gradient flow.
    """

    def __init__(
        self,
        model_dim: int,
        num_groups: int,
        config: RouterConfig | None = None,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_groups = num_groups
        self.config = config or RouterConfig()

        hidden_dim = max(model_dim // self.config.hidden_dim_divisor, 16)
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_groups),
        )

        self.max_active = self.config.max_active_groups or max(num_groups // 2, 1)
        self.min_active = min(self.config.min_active_groups, num_groups)

        # Tracking for load balance
        self._routing_counts: torch.Tensor | None = None  # [num_groups]
        self._total_samples: int = 0
        self._current_step: int = 0

        # Adaptive entropy coefficient
        self._entropy_coeff = self.config.entropy_coeff

    def forward(
        self,
        hidden_states: torch.Tensor,
        group_ids: list[str],
    ) -> dict[str, Any]:
        """Compute routing scores and selection mask.

        Args:
            hidden_states: [batch, seq_len, model_dim] from layer 0.
            group_ids: list of parameter group IDs (ordering matches output dim).

        Returns dict with:
            - scores: [batch, num_groups] routing probabilities
            - mask: [batch, num_groups] binary selection mask (1 = selected)
            - entropy_loss: scalar entropy regularization loss
            - selected_groups: list of sets, one per batch element
        """
        # Pool over sequence dimension
        pooled = hidden_states.mean(dim=1)  # [batch, model_dim]

        # Router scores
        logits = self.net(pooled)  # [batch, num_groups]
        scores = torch.sigmoid(logits)  # Independent group probabilities

        # Top-k selection with straight-through estimator
        k = min(self.max_active, self.num_groups)
        _, top_indices = torch.topk(scores, k, dim=-1)

        # Binary mask
        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(1, top_indices, 1.0)

        # Ensure minimum active groups
        if self.min_active > k:
            hard_mask.fill_(1.0)  # All active

        # Straight-through estimator: forward uses hard mask, backward uses soft scores
        mask = hard_mask - scores.detach() + scores

        # Entropy regularization
        # Entropy of per-sample routing distribution
        # Higher entropy = more spread out routing (less collapse)
        eps = 1e-8
        entropy_per_sample = -(
            scores * torch.log(scores + eps) +
            (1 - scores) * torch.log(1 - scores + eps)
        ).mean(dim=-1)  # [batch]
        entropy_loss = -self._entropy_coeff * entropy_per_sample.mean()

        # Update routing counts for load balance
        with torch.no_grad():
            batch_counts = hard_mask.sum(dim=0)  # [num_groups]
            if self._routing_counts is None:
                self._routing_counts = batch_counts.clone()
            else:
                self._routing_counts = (
                    self._routing_counts.to(batch_counts.device) + batch_counts
                )
            self._total_samples += hidden_states.shape[0]

        # Build per-sample selected groups
        selected = []
        for b in range(hidden_states.shape[0]):
            selected.append({
                group_ids[i] for i in range(self.num_groups)
                if hard_mask[b, i] > 0.5
            })

        return {
            "scores": scores,
            "mask": mask,
            "hard_mask": hard_mask,
            "entropy_loss": entropy_loss,
            "selected_groups": selected,
        }

    def step(self):
        """Called after each training step. Updates load balance and entropy coeff."""
        self._current_step += 1

        # Check load balance and adjust entropy coefficient
        if (self._routing_counts is not None
                and self._total_samples > 0
                and self._current_step % self.config.balance_window == 0):

            utilization = self._routing_counts / max(self._total_samples, 1)
            min_util = utilization.min().item()

            if min_util < self.config.utilization_floor:
                self._entropy_coeff = min(
                    self.config.entropy_coeff * self.config.entropy_boost_factor,
                    1.0,
                )
            else:
                # Decay back toward base
                self._entropy_coeff = max(
                    self.config.entropy_coeff,
                    self._entropy_coeff * 0.9,
                )

            # Reset counters
            self._routing_counts = None
            self._total_samples = 0

    @property
    def in_warmup(self) -> bool:
        """Whether the router is in observation mode (no gradient masking)."""
        return self._current_step < self.config.warmup_steps

    def get_load_balance_stats(self) -> dict[str, float] | None:
        """Return current load balance statistics."""
        if self._routing_counts is None or self._total_samples == 0:
            return None
        utilization = self._routing_counts / self._total_samples
        return {
            "min_utilization": utilization.min().item(),
            "max_utilization": utilization.max().item(),
            "mean_utilization": utilization.mean().item(),
            "std_utilization": utilization.std().item(),
            "entropy_coeff": self._entropy_coeff,
        }


def apply_gradient_mask(
    model: nn.Module,
    groups: dict[str, Any],
    routing_result: dict[str, Any],
    group_ids: list[str],
):
    """Zero gradients for parameter groups not selected by the router.

    This is called AFTER backward(). For each sample in the batch,
    groups not in the routing mask have their accumulated gradients zeroed.

    Since we accumulate gradients across the batch in standard PyTorch,
    we use the batch-aggregate mask: if a group is selected by ANY sample
    in the batch, its gradients are kept. Groups selected by NO sample
    are zeroed.
    """
    # Aggregate mask across batch: group is active if selected by any sample
    hard_mask = routing_result["hard_mask"]  # [batch, num_groups]
    group_active = hard_mask.sum(dim=0) > 0  # [num_groups] bool

    for i, gid in enumerate(group_ids):
        if not group_active[i]:
            # Zero all gradients for this group
            if gid in groups:
                for p in groups[gid].params:
                    if p.grad is not None:
                        p.grad.zero_()
