"""
Unified Audit System

SHA-256 hash-chained append-only audit log covering all pipeline stages.
Query API for step range, parameter group, sample, and convergence event.
Checkpoint-audit binding for verifiable restoration.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class AuditSink:
    """Append-only, SHA-256 hash-chained audit log.

    Each record includes the hash of the previous record, creating
    a tamper-evident chain. Records are JSON-lines for streaming reads.
    """

    def __init__(self, log_path: Path | str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._prev_hash: str = "0" * 64  # Genesis hash
        self._record_count: int = 0

        # Restore chain state if log already exists
        if self.log_path.exists():
            self._restore_chain()

    def _restore_chain(self):
        """Restore hash chain state from existing log."""
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self._prev_hash = record.get("record_hash", self._prev_hash)
                self._record_count += 1

    def append(self, record: dict[str, Any]):
        """Append a record to the audit log with hash chaining."""
        record = dict(record)  # Don't mutate caller's dict
        record["record_id"] = self._record_count
        record["prev_hash"] = self._prev_hash
        record["timestamp"] = record.get(
            "timestamp", time.strftime("%Y-%m-%dT%H:%M:%S")
        )

        # Compute hash of this record (excluding record_hash itself)
        payload = json.dumps(record, sort_keys=True, default=str)
        record_hash = hashlib.sha256(payload.encode()).hexdigest()
        record["record_hash"] = record_hash

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

        self._prev_hash = record_hash
        self._record_count += 1

    def log_step(
        self,
        step: int,
        gradient_norms: dict[str, float],
        convergence_states: dict[str, str],
        budget_allocations: dict[str, float] | None = None,
        routing_decisions: dict[str, Any] | None = None,
        loss_before: float | None = None,
        loss_after: float | None = None,
        hierarchy_active_levels: list[int] | None = None,
        **kwargs,
    ):
        """Log a training step record."""
        record: dict[str, Any] = {
            "type": "training_step",
            "step": step,
            "gradient_norms": gradient_norms,
            "convergence_states": convergence_states,
        }
        if budget_allocations is not None:
            record["budget_allocations"] = budget_allocations
        if routing_decisions is not None:
            record["routing_decisions"] = routing_decisions
        if loss_before is not None:
            record["loss_before"] = loss_before
        if loss_after is not None:
            record["loss_after"] = loss_after
        if hierarchy_active_levels is not None:
            record["hierarchy_active_levels"] = hierarchy_active_levels
        record.update(kwargs)
        self.append(record)

    def log_convergence_event(
        self,
        step: int,
        group_id: str,
        old_state: str,
        new_state: str,
        gradient_ema: float,
    ):
        """Log a convergence state transition."""
        self.append({
            "type": "convergence_event",
            "step": step,
            "group_id": group_id,
            "old_state": old_state,
            "new_state": new_state,
            "gradient_ema": gradient_ema,
        })

    def log_checkpoint(self, step: int, checkpoint_path: str):
        """Log a checkpoint with its audit binding hash."""
        self.append({
            "type": "checkpoint",
            "step": step,
            "checkpoint_path": checkpoint_path,
            "audit_hash": self._prev_hash,
        })

    @property
    def current_hash(self) -> str:
        return self._prev_hash

    @property
    def record_count(self) -> int:
        return self._record_count


class AuditQuery:
    """Query interface for the audit log."""

    def __init__(self, log_path: Path | str):
        self.log_path = Path(log_path)

    def _load_records(self) -> list[dict]:
        """Load all records from the log."""
        records = []
        if not self.log_path.exists():
            return records
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def query_by_step(
        self,
        start_step: int,
        end_step: int | None = None,
    ) -> list[dict]:
        """Get all records in a step range."""
        records = self._load_records()
        result = []
        for r in records:
            step = r.get("step")
            if step is None:
                continue
            if step < start_step:
                continue
            if end_step is not None and step > end_step:
                continue
            result.append(r)
        return result

    def query_by_group(self, group_id: str) -> list[dict]:
        """Get all records that reference a specific parameter group."""
        records = self._load_records()
        result = []
        for r in records:
            if r.get("group_id") == group_id:
                result.append(r)
            elif group_id in r.get("gradient_norms", {}):
                result.append(r)
            elif group_id in r.get("convergence_states", {}):
                result.append(r)
        return result

    def query_by_type(self, record_type: str) -> list[dict]:
        """Get all records of a specific type."""
        records = self._load_records()
        return [r for r in records if r.get("type") == record_type]

    def query_convergence_events(
        self,
        group_id: str | None = None,
    ) -> list[dict]:
        """Get convergence state transition events."""
        events = self.query_by_type("convergence_event")
        if group_id:
            events = [e for e in events if e.get("group_id") == group_id]
        return events

    def verify_chain(self) -> tuple[bool, int]:
        """Verify the hash chain integrity.

        Returns (is_valid, num_records_verified).
        """
        records = self._load_records()
        if not records:
            return True, 0

        prev_hash = "0" * 64
        for i, record in enumerate(records):
            # Check prev_hash matches
            if record.get("prev_hash") != prev_hash:
                return False, i

            # Recompute hash
            stored_hash = record.pop("record_hash", None)
            payload = json.dumps(record, sort_keys=True, default=str)
            expected_hash = hashlib.sha256(payload.encode()).hexdigest()
            record["record_hash"] = stored_hash

            if stored_hash != expected_hash:
                return False, i

            prev_hash = stored_hash

        return True, len(records)

    def get_checkpoint_hash(self, step: int) -> str | None:
        """Get the audit hash at a specific checkpoint."""
        checkpoints = self.query_by_type("checkpoint")
        for cp in checkpoints:
            if cp.get("step") == step:
                return cp.get("audit_hash")
        return None
