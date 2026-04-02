"""
Hierarchy Activation

Coarse-to-fine parameter activation governed by convergence signals.
Training begins with only L0 parameters active. Subsequent levels
activate as prior levels converge.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .convergence import ConvergenceGovernor, ConvergenceState, ConvergenceSignal
from .param_groups import ParamGroup


@dataclass
class HierarchyConfig:
    """Configuration for hierarchy activation."""
    # Per-level LR multipliers at activation time
    # (decay to 1.0 over warmup_steps)
    lr_multipliers: dict[int, float] | None = None
    warmup_steps: int = 100

    # Emergency activation: force-activate level if training is this
    # fraction complete and the level hasn't activated yet.
    emergency_fraction: float = 0.80

    def __post_init__(self):
        if self.lr_multipliers is None:
            self.lr_multipliers = {
                0: 1.0,
                1: 1.5,
                2: 1.5,
                3: 2.0,
            }


class HierarchyManager:
    """Manages coarse-to-fine activation of parameter hierarchy levels.

    At step 0, only L0 groups have requires_grad=True.
    Level N+1 activates when ALL groups in level N reach COOLING or CONVERGED.
    Emergency activation at emergency_fraction of total steps if a level
    hasn't activated yet.
    """

    def __init__(
        self,
        groups: dict[str, ParamGroup],
        governors: dict[str, ConvergenceGovernor],
        config: HierarchyConfig | None = None,
        total_steps: int = 10000,
        log_path: Path | str | None = None,
    ):
        self.groups = groups
        self.governors = governors
        self.config = config or HierarchyConfig()
        self.total_steps = total_steps

        # Organize groups by hierarchy level
        self.levels: dict[int, list[str]] = {}
        for gid, group in groups.items():
            lvl = group.hierarchy_level
            if lvl not in self.levels:
                self.levels[lvl] = []
            self.levels[lvl].append(gid)

        self.max_level = max(self.levels.keys()) if self.levels else 0

        # Track activation state
        self.active_levels: set[int] = {0}  # L0 always active from step 0
        self.activation_steps: dict[int, int] = {0: 0}  # level -> step activated
        self.activation_reasons: dict[int, str] = {0: "initial"}

        # LR multiplier tracking (per group, decays over warmup)
        self._lr_multipliers: dict[str, float] = {}
        self._activation_step_per_group: dict[str, int] = {}

        # Initialize: freeze non-L0 groups
        for gid, group in groups.items():
            if group.hierarchy_level == 0:
                group.set_requires_grad(True)
                self._lr_multipliers[gid] = self.config.lr_multipliers.get(0, 1.0)
            else:
                group.set_requires_grad(False)
                self._lr_multipliers[gid] = 0.0

        # Logging
        self._log_path = Path(log_path) if log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # Signal subscribers
        self._subscribers: list[Callable] = []

    def subscribe(self, callback: Callable):
        """Subscribe to level activation events."""
        self._subscribers.append(callback)

    def step(self, current_step: int) -> dict[str, float]:
        """Check if any new levels should activate.

        Returns dict of group_id -> effective LR multiplier for this step.
        """
        # Check if next level should activate
        for level in range(self.max_level + 1):
            if level in self.active_levels:
                continue  # Already active

            prior_level = level - 1
            if prior_level not in self.active_levels:
                continue  # Prior level not active yet — can't skip

            # Check if prior level has converged
            if self._level_converged(prior_level):
                self._activate_level(level, current_step, "convergence")
            # Emergency activation
            elif current_step >= self.total_steps * self.config.emergency_fraction:
                self._activate_level(level, current_step, "emergency")

        # Compute effective LR multipliers with warmup decay
        return self._compute_lr_multipliers(current_step)

    def _level_converged(self, level: int) -> bool:
        """Check if ALL groups in a level have reached COOLING or CONVERGED."""
        group_ids = self.levels.get(level, [])
        if not group_ids:
            return True  # Empty level is trivially converged

        for gid in group_ids:
            if gid not in self.governors:
                continue
            state = self.governors[gid].state
            if state not in (ConvergenceState.COOLING, ConvergenceState.CONVERGED):
                return False
        return True

    def _activate_level(self, level: int, step: int, reason: str):
        """Activate a hierarchy level."""
        self.active_levels.add(level)
        self.activation_steps[level] = step
        self.activation_reasons[level] = reason

        group_ids = self.levels.get(level, [])
        for gid in group_ids:
            if gid in self.groups:
                self.groups[gid].set_requires_grad(True)
                base_mult = self.config.lr_multipliers.get(level, 1.0)
                self._lr_multipliers[gid] = base_mult
                self._activation_step_per_group[gid] = step

        # Notify subscribers
        event = {
            "type": "level_activation",
            "level": level,
            "step": step,
            "reason": reason,
            "groups": group_ids,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        for callback in self._subscribers:
            callback(event)

        # Log
        if self._log_path:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(event) + "\n")

    def _compute_lr_multipliers(self, current_step: int) -> dict[str, float]:
        """Compute per-group LR multipliers with warmup decay."""
        result = {}
        for gid in self.groups:
            group = self.groups[gid]
            if group.hierarchy_level not in self.active_levels:
                result[gid] = 0.0
                continue

            base_mult = self.config.lr_multipliers.get(group.hierarchy_level, 1.0)
            activation_step = self._activation_step_per_group.get(gid, 0)
            steps_since_activation = current_step - activation_step

            if base_mult <= 1.0 or self.config.warmup_steps <= 0:
                result[gid] = 1.0
            elif steps_since_activation >= self.config.warmup_steps:
                result[gid] = 1.0
            else:
                # Linear decay from base_mult to 1.0
                progress = steps_since_activation / self.config.warmup_steps
                result[gid] = base_mult + (1.0 - base_mult) * progress

        return result

    def is_level_active(self, level: int) -> bool:
        return level in self.active_levels

    def all_levels_active(self) -> bool:
        return all(lvl in self.active_levels for lvl in self.levels)

    def state_dict(self) -> dict:
        return {
            "active_levels": sorted(self.active_levels),
            "activation_steps": self.activation_steps,
            "activation_reasons": self.activation_reasons,
            "lr_multipliers": self._lr_multipliers,
            "activation_step_per_group": self._activation_step_per_group,
        }

    def load_state_dict(self, state: dict):
        self.active_levels = set(state["active_levels"])
        self.activation_steps = {int(k): v for k, v in state["activation_steps"].items()}
        self.activation_reasons = {int(k): v for k, v in state["activation_reasons"].items()}
        self._lr_multipliers = state["lr_multipliers"]
        self._activation_step_per_group = state.get("activation_step_per_group", {})

        # Restore requires_grad state
        for gid, group in self.groups.items():
            if group.hierarchy_level in self.active_levels:
                group.set_requires_grad(True)
            else:
                group.set_requires_grad(False)
