"""
Federated Budget Allocator

Distributes gradient compute across active parameter groups proportional
to learning need. The invariant sum(allocations) <= master_budget holds
at every step. Implementation: effective learning rate multiplier per group.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .convergence import ConvergenceGovernor, ConvergenceState


@dataclass
class BudgetConfig:
    """Configuration for the federated budget allocator."""
    # Master budget (1.0 = total available compute)
    master_budget: float = 1.0

    # Floor/ceiling per group as fraction of master budget
    floor_fraction: float = 0.02
    ceiling_fraction: float = 0.40

    # Learning-need scoring weights
    alpha: float = 0.33  # gradient magnitude EMA weight
    beta: float = 0.33   # loss contribution EMA weight
    gamma: float = 0.34  # (1 - convergence_progress) weight

    # How often to recompute allocations (steps)
    eval_window: int = 50

    # EMA decay for loss contribution tracking
    loss_ema_decay: float = 0.95


class FederatedBudget:
    """Distributes compute budget across parameter groups by learning need.

    The budget allocation determines the effective learning rate multiplier
    per group. High-allocation groups get higher effective LR; low-allocation
    groups get lower. The invariant sum(allocations) <= master_budget holds
    at all times.
    """

    def __init__(
        self,
        group_ids: list[str],
        governors: dict[str, ConvergenceGovernor],
        config: BudgetConfig | None = None,
        log_path: Path | str | None = None,
    ):
        self.group_ids = group_ids
        self.governors = governors
        self.config = config or BudgetConfig()

        # Current allocations (uniform initial)
        n = len(group_ids)
        uniform = self.config.master_budget / n if n > 0 else 0
        self.allocations: dict[str, float] = {gid: uniform for gid in group_ids}

        # Learning-need tracking
        self._initial_gradient_emas: dict[str, float | None] = {gid: None for gid in group_ids}
        self._loss_contribution_emas: dict[str, float] = {gid: 0.0 for gid in group_ids}
        self._step_count: int = 0

        # Logging
        self._log_path = Path(log_path) if log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def step(self, step_num: int | None = None) -> dict[str, float]:
        """Update allocations based on current learning-need scores.

        Should be called every step, but only recomputes allocations
        every eval_window steps.

        Returns dict of group_id -> allocation (effective LR multiplier).
        """
        self._step_count += 1

        # Track initial gradient EMAs (for convergence_progress computation)
        for gid in self.group_ids:
            if gid in self.governors:
                gov = self.governors[gid]
                if self._initial_gradient_emas[gid] is None and gov._ema_initialized:
                    self._initial_gradient_emas[gid] = max(gov.gradient_ema, 1e-8)

        # Recompute allocations every eval_window steps
        if self._step_count % self.config.eval_window == 0:
            self._recompute_allocations()

            if self._log_path:
                self._log_allocation(step_num or self._step_count)

        return dict(self.allocations)

    def update_loss_contribution(self, group_id: str, loss_contribution: float):
        """Update loss contribution EMA for a group."""
        if group_id in self._loss_contribution_emas:
            alpha = self.config.loss_ema_decay
            self._loss_contribution_emas[group_id] = (
                alpha * self._loss_contribution_emas[group_id]
                + (1 - alpha) * loss_contribution
            )

    def _recompute_allocations(self):
        """Recompute budget allocations based on learning-need scores."""
        # Compute learning-need scores for active groups
        needs: dict[str, float] = {}
        active_ids = []

        for gid in self.group_ids:
            if gid not in self.governors:
                continue
            gov = self.governors[gid]

            # Converged groups get maintenance allocation only
            if gov.state == ConvergenceState.CONVERGED:
                self.allocations[gid] = self.config.floor_fraction * self.config.master_budget
                continue

            active_ids.append(gid)

            # Learning-need scoring
            grad_ema = gov.gradient_ema
            loss_contrib = self._loss_contribution_emas.get(gid, 0.0)

            # Convergence progress: ratio of current to initial gradient EMA
            initial = self._initial_gradient_emas.get(gid)
            if initial and initial > 0:
                convergence_progress = min(1.0, grad_ema / initial)
            else:
                convergence_progress = 1.0  # No progress info yet

            need = (
                self.config.alpha * grad_ema
                + self.config.beta * loss_contrib
                + self.config.gamma * (1.0 - convergence_progress)
            )
            needs[gid] = max(need, 1e-10)  # Avoid zero

        if not active_ids:
            return

        # Budget available for active groups (total minus converged maintenance)
        converged_cost = sum(
            self.allocations[gid] for gid in self.group_ids
            if gid in self.governors
            and self.governors[gid].state == ConvergenceState.CONVERGED
        )
        available = self.config.master_budget - converged_cost

        # Normalize needs to probability distribution
        total_need = sum(needs.values())
        if total_need <= 0:
            # Uniform allocation among active groups
            for gid in active_ids:
                self.allocations[gid] = available / len(active_ids)
            return

        # Allocate proportionally
        for gid in active_ids:
            raw = (needs[gid] / total_need) * available
            # Enforce floor and ceiling
            floor = self.config.floor_fraction * self.config.master_budget
            ceiling = self.config.ceiling_fraction * self.config.master_budget
            self.allocations[gid] = max(floor, min(ceiling, raw))

        # Re-normalize to maintain budget invariant
        total_allocated = sum(self.allocations.values())
        if total_allocated > self.config.master_budget:
            scale = self.config.master_budget / total_allocated
            for gid in self.allocations:
                self.allocations[gid] *= scale

    def get_lr_multipliers(self) -> dict[str, float]:
        """Convert allocations to per-group LR multipliers.

        Multiplier = allocation / uniform_allocation, so a group with
        twice the average allocation gets 2x the base LR.
        """
        n = len(self.group_ids)
        if n == 0:
            return {}
        uniform = self.config.master_budget / n
        if uniform <= 0:
            return {gid: 1.0 for gid in self.group_ids}
        return {
            gid: alloc / uniform
            for gid, alloc in self.allocations.items()
        }

    def verify_invariant(self) -> bool:
        """Check that sum(allocations) <= master_budget."""
        return sum(self.allocations.values()) <= self.config.master_budget + 1e-8

    def _log_allocation(self, step: int):
        """Log allocation decision to JSON-lines."""
        with open(self._log_path, "a") as f:
            record = {
                "type": "budget_allocation",
                "step": step,
                "allocations": dict(self.allocations),
                "learning_needs": {
                    gid: {
                        "gradient_ema": self.governors[gid].gradient_ema,
                        "loss_contribution": self._loss_contribution_emas.get(gid, 0.0),
                        "state": self.governors[gid].state.value,
                    }
                    for gid in self.group_ids
                    if gid in self.governors
                },
                "total_allocated": sum(self.allocations.values()),
                "master_budget": self.config.master_budget,
                "invariant_holds": self.verify_invariant(),
            }
            f.write(json.dumps(record) + "\n")

    def state_dict(self) -> dict:
        return {
            "allocations": dict(self.allocations),
            "initial_gradient_emas": dict(self._initial_gradient_emas),
            "loss_contribution_emas": dict(self._loss_contribution_emas),
            "step_count": self._step_count,
        }

    def load_state_dict(self, state: dict):
        self.allocations = state["allocations"]
        self._initial_gradient_emas = state["initial_gradient_emas"]
        self._loss_contribution_emas = state["loss_contribution_emas"]
        self._step_count = state["step_count"]
