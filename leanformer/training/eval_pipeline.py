"""
Change-Triggered Evaluation Pipeline

Replaces fixed-interval evaluation with change-triggered evaluation.
Subscribes to Signal<ConvergenceChange>, evaluates only metrics from
metric_dependency_map for changed groups. Budget<EvalCompute> governance.
Per-metric regression detection.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .convergence import ConvergenceSignal, ConvergenceState


@dataclass
class EvalConfig:
    """Configuration for the evaluation pipeline."""
    # Budget: max evaluations per window
    max_evals_per_window: int = 10
    eval_window_steps: int = 100

    # Metric importance weights (higher = evaluated more often)
    metric_importance: dict[str, float] = field(default_factory=lambda: {
        "perplexity": 1.0,
        "belief_injection": 0.8,
        "semantic_routing": 0.6,
        "attention_sparsity": 0.4,
        "base_weight_integrity": 1.0,
        "adaptive_depth": 0.3,
    })

    # Regression detection
    regression_threshold: float = 0.05  # 5% regression triggers alert
    regression_window: int = 3  # Must regress for N consecutive evals


class EvalPipeline:
    """Change-triggered evaluation pipeline.

    Evaluates metrics only when the parameter groups they depend on change.
    Uses Budget<EvalCompute> to limit total evaluation work.
    """

    def __init__(
        self,
        metric_dependency_map: dict[str, list[str]],
        metric_functions: dict[str, Callable[[], float]],
        config: EvalConfig | None = None,
    ):
        self.metric_dependency_map = metric_dependency_map
        self.metric_functions = metric_functions
        self.config = config or EvalConfig()

        # Tracking
        self._pending_groups: set[str] = set()  # Groups that changed since last eval
        self._metric_history: dict[str, list[float]] = {m: [] for m in metric_functions}
        self._last_eval_step: dict[str, int] = {m: 0 for m in metric_functions}
        self._evals_this_window: int = 0
        self._window_start_step: int = 0
        self._total_evals: int = 0

        # Regression tracking
        self._regression_counters: dict[str, int] = {m: 0 for m in metric_functions}
        self._regression_events: list[dict] = []

        # Signal subscribers for regression events
        self._subscribers: list[Callable] = []

    def subscribe(self, callback: Callable):
        """Subscribe to regression events."""
        self._subscribers.append(callback)

    def on_convergence_signal(self, signal: ConvergenceSignal):
        """Handle a convergence state change signal."""
        self._pending_groups.add(signal.group_id)

    def step(self, current_step: int, force_full: bool = False) -> dict[str, float]:
        """Check if evaluation should run and execute if needed.

        Args:
            current_step: Current training step.
            force_full: If True, run all metrics regardless of budget.

        Returns dict of metric_name -> value for metrics that were evaluated.
        """
        # Reset window counter
        if current_step - self._window_start_step >= self.config.eval_window_steps:
            self._evals_this_window = 0
            self._window_start_step = current_step

        if not force_full and not self._pending_groups:
            return {}

        # Determine which metrics to evaluate
        metrics_to_eval = self._select_metrics(current_step, force_full)

        if not metrics_to_eval:
            return {}

        # Run selected evaluations
        results = {}
        for metric_name in metrics_to_eval:
            if metric_name in self.metric_functions:
                value = self.metric_functions[metric_name]()
                results[metric_name] = value
                self._metric_history[metric_name].append(value)
                self._last_eval_step[metric_name] = current_step
                self._total_evals += 1
                self._evals_this_window += 1

                # Check for regression
                self._check_regression(metric_name, value, current_step)

        self._pending_groups.clear()
        return results

    def _select_metrics(
        self,
        current_step: int,
        force_full: bool,
    ) -> list[str]:
        """Select which metrics to evaluate based on changed groups and budget."""
        if force_full:
            return list(self.metric_functions.keys())

        # Find metrics affected by changed groups
        affected: set[str] = set()
        for metric_name, dep_groups in self.metric_dependency_map.items():
            if "all" in dep_groups:
                affected.add(metric_name)
            elif any(g in self._pending_groups for g in dep_groups):
                affected.add(metric_name)

        if not affected:
            return []

        # Budget check
        budget_remaining = self.config.max_evals_per_window - self._evals_this_window
        if budget_remaining <= 0:
            return []

        # Prioritize by: staleness * importance
        scored = []
        for metric_name in affected:
            if metric_name not in self.metric_functions:
                continue
            staleness = current_step - self._last_eval_step.get(metric_name, 0)
            importance = self.config.metric_importance.get(metric_name, 0.5)
            score = staleness * importance
            scored.append((score, metric_name))

        scored.sort(reverse=True)
        return [name for _, name in scored[:budget_remaining]]

    def _check_regression(self, metric_name: str, value: float, step: int):
        """Check if a metric has regressed."""
        history = self._metric_history.get(metric_name, [])
        if len(history) < 2:
            return

        # Compare against best recent value
        recent = history[-min(10, len(history)):-1]
        if not recent:
            return
        best_recent = min(recent)  # For loss metrics, lower is better

        if best_recent > 0 and (value - best_recent) / best_recent > self.config.regression_threshold:
            self._regression_counters[metric_name] += 1
        else:
            self._regression_counters[metric_name] = 0

        if self._regression_counters[metric_name] >= self.config.regression_window:
            event = {
                "type": "metric_regression",
                "metric": metric_name,
                "step": step,
                "current_value": value,
                "best_recent": best_recent,
                "regression_pct": (value - best_recent) / best_recent if best_recent > 0 else 0,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._regression_events.append(event)
            self._regression_counters[metric_name] = 0

            for callback in self._subscribers:
                callback(event)

    def get_stats(self) -> dict[str, Any]:
        """Return evaluation pipeline statistics."""
        return {
            "total_evals": self._total_evals,
            "evals_this_window": self._evals_this_window,
            "pending_groups": list(self._pending_groups),
            "regression_events": len(self._regression_events),
            "latest_values": {
                m: h[-1] if h else None
                for m, h in self._metric_history.items()
            },
        }
