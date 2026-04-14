"""
Convergence Governor

Per-group convergence detection with a five-state state machine:
PENDING → ACTIVE → COOLING → CONVERGED → AWAKENED → ACTIVE

Groups start in PENDING until the hierarchy activates their level.
Each parameter group gets its own governor that tracks gradient EMA,
loss contribution, and transitions through convergence states.

The governor is phase-aware: it classifies the gradient trajectory into
COLD / WARMING / ACTIVE_LEARNING / DECLINING and enforces the
`NoCoolingFromCold` invariant — the ACTIVE → COOLING transition requires
that gradient magnitude has actually risen above the cooling threshold at
some point. This prevents cold-start gradients (B=0 initialization) from
being misread as post-learning convergence. The invariant was formally
specified in TLA+ and verified by TLC across 18.6 million states; see
`local/tla/ConvergenceGovernorPhaseAware.tla` for the specification.
"""

import enum
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch


class ConvergenceState(enum.Enum):
    PENDING = "PENDING"      # Not yet activated by hierarchy
    ACTIVE = "ACTIVE"
    COOLING = "COOLING"
    CONVERGED = "CONVERGED"
    AWAKENED = "AWAKENED"


class GradientPhase(enum.Enum):
    """Qualitative classification of the gradient trajectory.

    Two trajectories can produce the same low gradient magnitude but have
    very different meanings:

      COLD:      0 -> 0 -> 0 -> 0          (B=0, learning not started)
      DECLINING: HIGH -> MED -> LOW -> LOW  (genuine convergence)

    The governor must distinguish these. It does so by tracking whether
    the gradient has ever exceeded the cooling threshold (`peak_observed`).
    """
    COLD = "COLD"                       # never exceeded threshold, currently low
    WARMING = "WARMING"                 # never exceeded threshold, currently high
    ACTIVE_LEARNING = "ACTIVE_LEARNING" # has exceeded threshold, currently high
    DECLINING = "DECLINING"             # has exceeded threshold, currently low


@dataclass
class ConvergenceConfig:
    """Thresholds for convergence detection. Conservative defaults."""
    # ACTIVE → COOLING: gradient EMA below this for cooling_window steps
    cooling_threshold: float = 0.01
    cooling_window: int = 50

    # COOLING → CONVERGED: gradient EMA below this for confirmation_window steps
    converged_threshold: float = 0.005
    confirmation_window: int = 100

    # CONVERGED → AWAKENED: loss contribution increases by this much
    reactivation_delta: float = 0.1

    # AWAKENED → ACTIVE: warmup steps before returning to ACTIVE
    reactivation_warmup: int = 20

    # EMA decay factor for gradient norm tracking
    ema_decay: float = 0.99

    # How often to evaluate loss contribution (steps)
    eval_window: int = 50


@dataclass
class ConvergenceSignal:
    """Fired on every state transition."""
    group_id: str
    old_state: ConvergenceState
    new_state: ConvergenceState
    step: int
    gradient_ema: float
    loss_contribution: float
    gradient_phase: GradientPhase = GradientPhase.COLD
    peak_observed: bool = False
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


class ConvergenceGovernor:
    """Per-group convergence governor with five-state state machine.

    Groups start in PENDING until the hierarchy activates their level.
    Tracks gradient EMA and loss contribution for a single parameter group.
    Transitions: PENDING → ACTIVE → COOLING → CONVERGED → AWAKENED → ACTIVE.
    Toggles requires_grad on state transitions.
    """

    def __init__(
        self,
        group_id: str,
        config: ConvergenceConfig | None = None,
        initially_active: bool = True,
    ):
        self.group_id = group_id
        self.config = config or ConvergenceConfig()
        self.state = ConvergenceState.ACTIVE if initially_active else ConvergenceState.PENDING
        self.step = 0

        # Gradient EMA tracking
        self.gradient_ema: float = 1.0  # Start high so we don't trigger immediately
        self._ema_initialized: bool = False

        # Phase-aware liveness signal: has the gradient ever genuinely risen
        # above the cooling threshold? ACTIVE -> COOLING is gated on this.
        self.peak_observed: bool = False
        self.gradient_phase: GradientPhase = GradientPhase.COLD

        # Loss contribution tracking
        self.loss_contribution: float = 0.0

        # Counters for state transition windows
        self._cooling_counter: int = 0
        self._converged_counter: int = 0
        self._awakened_counter: int = 0

        # Budget multiplier per state (used by FederatedBudget)
        self._budget_multipliers = {
            ConvergenceState.PENDING: 0.0,
            ConvergenceState.ACTIVE: 1.0,
            ConvergenceState.COOLING: 0.5,
            ConvergenceState.CONVERGED: 0.05,
            ConvergenceState.AWAKENED: 1.2,
        }

        # Signal subscribers
        self._subscribers: list[Callable[[ConvergenceSignal], None]] = []

        # History for audit
        self.transition_history: list[dict] = []

    @property
    def budget_multiplier(self) -> float:
        return self._budget_multipliers[self.state]

    def subscribe(self, callback: Callable[[ConvergenceSignal], None]):
        """Subscribe to state transition signals."""
        self._subscribers.append(callback)

    def activate(self):
        """Transition from PENDING to ACTIVE. Called by hierarchy when level activates."""
        if self.state == ConvergenceState.PENDING:
            old_state = self.state
            self.state = ConvergenceState.ACTIVE
            self.gradient_ema = 1.0  # Start high to avoid immediate COOLING
            self._ema_initialized = False
            self._cooling_counter = 0
            # peak_observed intentionally NOT reset: if a group has previously
            # been learning and returns to ACTIVE (via AWAKENED), its history
            # persists. On true PENDING->ACTIVE from step 0, peak_observed is
            # already False.
            self._emit_signal(old_state, self.state)

    def _classify_phase(self, grad_norm: float) -> GradientPhase:
        """Classify the gradient trajectory.

        Uses the raw per-step grad_norm (not the EMA) for phase detection
        so a single genuine learning spike flips `peak_observed` without
        being smoothed out by the EMA.
        """
        above = grad_norm >= self.config.cooling_threshold
        if self.peak_observed:
            return GradientPhase.ACTIVE_LEARNING if above else GradientPhase.DECLINING
        return GradientPhase.WARMING if above else GradientPhase.COLD

    def update(self, grad_norm: float, loss_contribution: float | None = None) -> ConvergenceState:
        """Update governor with current step's gradient norm.

        Args:
            grad_norm: L2 norm of gradients for this group.
            loss_contribution: Fraction of total loss attributable to this group.
                Only updated when provided (computed every eval_window steps).

        Returns:
            Current state after any transitions.
        """
        # PENDING groups don't track — they haven't started training
        if self.state == ConvergenceState.PENDING:
            return self.state

        self.step += 1

        # Update gradient EMA
        if not self._ema_initialized:
            self.gradient_ema = grad_norm
            self._ema_initialized = True
        else:
            alpha = self.config.ema_decay
            self.gradient_ema = alpha * self.gradient_ema + (1 - alpha) * grad_norm

        # Phase classification. peak_observed is a monotonic liveness signal:
        # once True, it stays True for the life of the governor. Uses raw
        # grad_norm so a genuine learning spike trips the flag immediately.
        # Classify using the pre-update peak_observed so the first
        # threshold-crossing step is WARMING; subsequent above-threshold steps
        # are ACTIVE_LEARNING. This matches the TLA+ specification.
        self.gradient_phase = self._classify_phase(grad_norm)
        if grad_norm >= self.config.cooling_threshold:
            self.peak_observed = True

        # Update loss contribution when provided
        if loss_contribution is not None:
            self.loss_contribution = loss_contribution

        # State machine transitions
        old_state = self.state

        if self.state == ConvergenceState.ACTIVE:
            self._update_active()
        elif self.state == ConvergenceState.COOLING:
            self._update_cooling()
        elif self.state == ConvergenceState.CONVERGED:
            self._update_converged()
        elif self.state == ConvergenceState.AWAKENED:
            self._update_awakened()

        if self.state != old_state:
            self._emit_signal(old_state, self.state)

        return self.state

    def _update_active(self):
        """ACTIVE → COOLING if gradient EMA below cooling_threshold for cooling_window,
        AND gradient has actually risen above threshold at some point.

        The peak_observed precondition is the NoCoolingFromCold invariant: a
        governor that has never seen real gradient flow cannot interpret its
        quiet state as post-learning convergence.
        """
        if not self.peak_observed:
            # Cold-start: learning has not started. Don't accumulate.
            self._cooling_counter = 0
            return

        if self.gradient_ema < self.config.cooling_threshold:
            self._cooling_counter += 1
            if self._cooling_counter >= self.config.cooling_window:
                self.state = ConvergenceState.COOLING
                self._cooling_counter = 0
                self._converged_counter = 0
        else:
            self._cooling_counter = 0

    def _update_cooling(self):
        """COOLING → CONVERGED if below converged_threshold for confirmation_window.
        COOLING → ACTIVE if gradient EMA rises above cooling_threshold."""
        if self.gradient_ema >= self.config.cooling_threshold:
            # False alarm — back to ACTIVE
            self.state = ConvergenceState.ACTIVE
            self._converged_counter = 0
            return

        if self.gradient_ema < self.config.converged_threshold:
            self._converged_counter += 1
            if self._converged_counter >= self.config.confirmation_window:
                self.state = ConvergenceState.CONVERGED
                self._converged_counter = 0
        else:
            self._converged_counter = 0

    def _update_converged(self):
        """CONVERGED → AWAKENED if loss contribution increases significantly."""
        if self.loss_contribution > self.config.reactivation_delta:
            self.state = ConvergenceState.AWAKENED
            self._awakened_counter = 0

    def _update_awakened(self):
        """AWAKENED → ACTIVE after reactivation_warmup steps."""
        self._awakened_counter += 1
        if self._awakened_counter >= self.config.reactivation_warmup:
            self.state = ConvergenceState.ACTIVE
            self._awakened_counter = 0
            # Reset EMA to avoid immediately re-triggering cooling
            self._ema_initialized = False

    def _emit_signal(self, old_state: ConvergenceState, new_state: ConvergenceState):
        """Fire signal and record transition."""
        signal = ConvergenceSignal(
            group_id=self.group_id,
            old_state=old_state,
            new_state=new_state,
            step=self.step,
            gradient_ema=self.gradient_ema,
            loss_contribution=self.loss_contribution,
            gradient_phase=self.gradient_phase,
            peak_observed=self.peak_observed,
        )

        self.transition_history.append({
            "group_id": self.group_id,
            "old_state": old_state.value,
            "new_state": new_state.value,
            "step": self.step,
            "gradient_ema": self.gradient_ema,
            "loss_contribution": self.loss_contribution,
            "gradient_phase": self.gradient_phase.value,
            "peak_observed": self.peak_observed,
            "timestamp": signal.timestamp,
        })

        for callback in self._subscribers:
            callback(signal)

    def state_dict(self) -> dict:
        """Serialize governor state for checkpointing."""
        return {
            "group_id": self.group_id,
            "state": self.state.value,
            "step": self.step,
            "gradient_ema": self.gradient_ema,
            "loss_contribution": self.loss_contribution,
            "ema_initialized": self._ema_initialized,
            "cooling_counter": self._cooling_counter,
            "converged_counter": self._converged_counter,
            "awakened_counter": self._awakened_counter,
            "peak_observed": self.peak_observed,
            "gradient_phase": self.gradient_phase.value,
        }

    def load_state_dict(self, state: dict):
        """Restore governor state from checkpoint.

        Backwards compatible with pre-phase-aware checkpoints: missing
        `peak_observed` defaults to True for governors that are already past
        ACTIVE (since they can only have reached those states under the old
        rules after seeing gradient flow at least once), and False for
        governors still in ACTIVE or PENDING.
        """
        self.state = ConvergenceState(state["state"])
        self.step = state["step"]
        self.gradient_ema = state["gradient_ema"]
        self.loss_contribution = state["loss_contribution"]
        self._ema_initialized = state["ema_initialized"]
        self._cooling_counter = state["cooling_counter"]
        self._converged_counter = state["converged_counter"]
        self._awakened_counter = state["awakened_counter"]

        if "peak_observed" in state:
            self.peak_observed = state["peak_observed"]
            self.gradient_phase = GradientPhase(state["gradient_phase"])
        else:
            # Legacy checkpoint: infer peak_observed from state.
            self.peak_observed = self.state not in (
                ConvergenceState.PENDING, ConvergenceState.ACTIVE,
            )
            # Phase will re-classify on next update().
            self.gradient_phase = (
                GradientPhase.DECLINING if self.peak_observed else GradientPhase.COLD
            )


class GovernorManager:
    """Manages ConvergenceGovernors for all parameter groups.

    Provides a single update point that computes gradient norms and
    loss contributions, updates all governors, and applies requires_grad
    toggling.
    """

    def __init__(
        self,
        groups: dict[str, Any],  # group_id -> ParamGroup
        config: ConvergenceConfig | None = None,
        log_path: Path | str | None = None,
    ):
        self.groups = groups
        self.config = config or ConvergenceConfig()
        self.governors: dict[str, ConvergenceGovernor] = {}

        for gid in groups:
            self.governors[gid] = ConvergenceGovernor(gid, self.config)

        # Audit log (JSON-lines)
        self._log_path = Path(log_path) if log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # Global signal subscribers
        self._subscribers: list[Callable[[ConvergenceSignal], None]] = []

        # Wire per-governor signals through the manager
        for gov in self.governors.values():
            gov.subscribe(self._on_signal)

    def subscribe(self, callback: Callable[[ConvergenceSignal], None]):
        """Subscribe to all convergence state transitions."""
        self._subscribers.append(callback)

    def step(self, step_num: int | None = None) -> dict[str, ConvergenceState]:
        """Update all governors after a training step.

        Computes gradient norms from the parameter groups, updates each
        governor, and applies requires_grad toggling as needed.

        Returns dict of group_id -> current state.
        """
        states = {}
        for gid, group in self.groups.items():
            gov = self.governors[gid]
            grad_norm = group.grad_norm()
            old_state = gov.state
            new_state = gov.update(grad_norm)

            # Apply requires_grad toggling on transitions
            if new_state == ConvergenceState.CONVERGED and old_state != ConvergenceState.CONVERGED:
                group.set_requires_grad(False)
            elif old_state == ConvergenceState.CONVERGED and new_state != ConvergenceState.CONVERGED:
                group.set_requires_grad(True)

            states[gid] = new_state

        # Log per-step metrics
        if self._log_path:
            self._log_step(step_num or self.governors[next(iter(self.governors))].step, states)

        return states

    def get_gradient_emas(self) -> dict[str, float]:
        """Return current gradient EMAs for all groups."""
        return {gid: gov.gradient_ema for gid, gov in self.governors.items()}

    def get_states(self) -> dict[str, ConvergenceState]:
        """Return current states for all groups."""
        return {gid: gov.state for gid, gov in self.governors.items()}

    def get_budget_multipliers(self) -> dict[str, float]:
        """Return budget multipliers for all groups based on state."""
        return {gid: gov.budget_multiplier for gid, gov in self.governors.items()}

    def _on_signal(self, signal: ConvergenceSignal):
        """Forward signals to manager-level subscribers and log."""
        for callback in self._subscribers:
            callback(signal)

        if self._log_path:
            with open(self._log_path, "a") as f:
                record = {
                    "type": "state_transition",
                    "group_id": signal.group_id,
                    "old_state": signal.old_state.value,
                    "new_state": signal.new_state.value,
                    "step": signal.step,
                    "gradient_ema": signal.gradient_ema,
                    "loss_contribution": signal.loss_contribution,
                    "gradient_phase": signal.gradient_phase.value,
                    "peak_observed": signal.peak_observed,
                    "timestamp": signal.timestamp,
                }
                f.write(json.dumps(record) + "\n")

    def _log_step(self, step: int, states: dict[str, ConvergenceState]):
        """Log per-step gradient EMAs, states, and phase-awareness signals."""
        with open(self._log_path, "a") as f:
            record = {
                "type": "step_metrics",
                "step": step,
                "gradient_emas": {
                    gid: gov.gradient_ema for gid, gov in self.governors.items()
                },
                "states": {gid: s.value for gid, s in states.items()},
                "gradient_phases": {
                    gid: gov.gradient_phase.value for gid, gov in self.governors.items()
                },
                "peak_observed": {
                    gid: gov.peak_observed for gid, gov in self.governors.items()
                },
            }
            f.write(json.dumps(record) + "\n")

    def state_dict(self) -> dict:
        """Serialize all governor states."""
        return {gid: gov.state_dict() for gid, gov in self.governors.items()}

    def load_state_dict(self, state: dict):
        """Restore all governor states."""
        for gid, gov_state in state.items():
            if gid in self.governors:
                self.governors[gid].load_state_dict(gov_state)
