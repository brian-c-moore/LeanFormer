"""
Forge Readiness Gating

Forge activates only when target parameter groups are CONVERGED for
a stability_window. Automated delta validation pipeline. Forge-training
feedback signal.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable

from .convergence import ConvergenceGovernor, ConvergenceState, ConvergenceSignal


@dataclass
class ForgeGateConfig:
    """Configuration for forge readiness gating."""
    # Steps a group must be CONVERGED before forge activates
    stability_window: int = 100

    # Quality threshold for accepting forged deltas
    acceptance_threshold: float = 0.5

    # Feedback: increase budget for groups with low forge quality
    feedback_budget_boost: float = 1.5


class ForgeReadinessGate:
    """Gates knowledge forge activation on base model stability.

    The forge pipeline does not begin producing deltas until the
    parameter groups it targets have been CONVERGED for stability_window
    consecutive steps.
    """

    def __init__(
        self,
        governors: dict[str, ConvergenceGovernor],
        target_groups: dict[str, list[str]],  # domain -> list of group_ids
        config: ForgeGateConfig | None = None,
    ):
        self.governors = governors
        self.target_groups = target_groups
        self.config = config or ForgeGateConfig()

        # Track how long each group has been CONVERGED
        self._converged_steps: dict[str, int] = {gid: 0 for gid in governors}

        # Track forge activation per domain
        self._activated_domains: set[str] = set()
        self._suspended_domains: set[str] = set()

        # Delta validation results
        self._delta_results: list[dict] = []

        # Signal subscribers
        self._subscribers: list[Callable] = []

    def subscribe(self, callback: Callable):
        """Subscribe to forge activation/suspension events."""
        self._subscribers.append(callback)

    def step(self, current_step: int) -> dict[str, bool]:
        """Update forge readiness state.

        Returns dict of domain -> is_ready_for_forging.
        """
        # Update convergence step counters
        for gid, gov in self.governors.items():
            if gov.state == ConvergenceState.CONVERGED:
                self._converged_steps[gid] += 1
            else:
                self._converged_steps[gid] = 0

        # Check each domain
        readiness: dict[str, bool] = {}
        for domain, group_ids in self.target_groups.items():
            was_active = domain in self._activated_domains

            # Check if all target groups are stable
            all_stable = all(
                self._converged_steps.get(gid, 0) >= self.config.stability_window
                for gid in group_ids
            )

            # Check if any target group was reactivated (AWAKENED)
            any_awakened = any(
                self.governors.get(gid, None) is not None
                and self.governors[gid].state == ConvergenceState.AWAKENED
                for gid in group_ids
            )

            if any_awakened and was_active:
                # Suspend forging
                self._activated_domains.discard(domain)
                self._suspended_domains.add(domain)
                readiness[domain] = False
                self._emit_event("forge_suspended", domain, current_step)
            elif all_stable and not was_active:
                # Activate forging
                self._activated_domains.add(domain)
                self._suspended_domains.discard(domain)
                readiness[domain] = True
                self._emit_event("forge_activated", domain, current_step)
            else:
                readiness[domain] = domain in self._activated_domains

        return readiness

    def validate_delta(
        self,
        domain: str,
        quality_score: float,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Validate a forged delta. Returns True if accepted."""
        accepted = quality_score >= self.config.acceptance_threshold

        result = {
            "domain": domain,
            "quality_score": quality_score,
            "accepted": accepted,
            "threshold": self.config.acceptance_threshold,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if metadata:
            result["metadata"] = metadata
        self._delta_results.append(result)

        return accepted

    def get_feedback(self) -> dict[str, float]:
        """Get feedback signal for training budget adjustment.

        Returns dict of group_id -> budget_boost_multiplier for groups
        whose forge domains had low acceptance rates.
        """
        # Compute acceptance rate per domain
        domain_rates: dict[str, list[bool]] = {}
        for r in self._delta_results[-20:]:  # Last 20 results
            domain = r["domain"]
            if domain not in domain_rates:
                domain_rates[domain] = []
            domain_rates[domain].append(r["accepted"])

        feedback: dict[str, float] = {}
        for domain, results in domain_rates.items():
            if not results:
                continue
            acceptance_rate = sum(results) / len(results)
            if acceptance_rate < 0.5:
                # Low acceptance rate — boost budget for target groups
                for gid in self.target_groups.get(domain, []):
                    feedback[gid] = self.config.feedback_budget_boost

        return feedback

    def is_domain_ready(self, domain: str) -> bool:
        return domain in self._activated_domains

    def get_stats(self) -> dict[str, Any]:
        return {
            "activated_domains": sorted(self._activated_domains),
            "suspended_domains": sorted(self._suspended_domains),
            "converged_steps": dict(self._converged_steps),
            "total_deltas_validated": len(self._delta_results),
            "accepted": sum(1 for r in self._delta_results if r["accepted"]),
            "rejected": sum(1 for r in self._delta_results if not r["accepted"]),
        }

    def _emit_event(self, event_type: str, domain: str, step: int):
        event = {
            "type": event_type,
            "domain": domain,
            "step": step,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        for callback in self._subscribers:
            callback(event)
