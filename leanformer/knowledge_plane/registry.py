"""
Delta Registry + Orthogonality Engine.

Tracks all registered deltas, enforces orthogonality at write time,
and provides capacity accounting. The single source of truth for which
subspaces are occupied and how much capacity remains.
"""

import math
import json
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pathlib import Path

from .dfs import DeltaFormatSpec


def compute_principal_angles(basis_a: torch.Tensor, basis_b: torch.Tensor) -> torch.Tensor:
    """
    Compute principal angles between two subspaces.

    Args:
        basis_a: Shape (d_model, rank_a) - orthonormal basis of subspace A
        basis_b: Shape (d_model, rank_b) - orthonormal basis of subspace B

    Returns:
        Tensor of principal angles in radians, shape (min(rank_a, rank_b),)

    The minimum principal angle indicates how close the subspaces are:
    - 0 radians = subspaces share a direction (overlap)
    - pi/2 radians = subspaces are fully orthogonal

    Method: SVD of basis_a.T @ basis_b. Singular values are cosines of
    principal angles.
    """
    # Ensure orthonormal bases (use float64 for numerical stability)
    basis_a = basis_a.double()
    basis_b = basis_b.double()
    Q_a, _ = torch.linalg.qr(basis_a)
    Q_b, _ = torch.linalg.qr(basis_b)

    # Handle NaN from rank-deficient inputs
    if not torch.isfinite(Q_a).all() or not torch.isfinite(Q_b).all():
        # Subspaces are degenerate - treat as fully overlapping
        return torch.zeros(min(Q_a.shape[1], Q_b.shape[1]))

    # Compute cosines of principal angles
    M = Q_a.T @ Q_b
    _, S, _ = torch.linalg.svd(M)

    # Clamp to valid range for arccos
    S = torch.clamp(S, -1.0, 1.0)
    angles = torch.arccos(S)
    return angles.float()


def minimum_principal_angle(basis_a: torch.Tensor, basis_b: torch.Tensor) -> float:
    """
    Returns the minimum principal angle between two subspaces in radians.
    Smaller = more overlap. pi/2 = fully orthogonal.
    """
    angles = compute_principal_angles(basis_a, basis_b)
    return angles.min().item()


def orthogonality_score(basis_a: torch.Tensor, basis_b: torch.Tensor) -> float:
    """
    Returns orthogonality score between 0 and 1.
    0 = subspaces overlap completely.
    1 = subspaces are fully orthogonal.
    Computed as: min_principal_angle / (pi/2)
    """
    min_angle = minimum_principal_angle(basis_a, basis_b)
    return min_angle / (math.pi / 2)


@dataclass
class DeltaRegistry:
    """
    Tracks all deltas known to the Knowledge Plane.
    Enforces orthogonality at write time.
    Provides capacity accounting.

    The registry is the single source of truth for:
    - Which subspaces are occupied
    - Which deltas are compatible with each other
    - How much capacity remains
    """

    base_model_hash: str
    d_model: int
    n_layers: int
    delta_rank: int = 16
    orthogonality_threshold: float = 0.3  # Minimum acceptable score (0-1)

    # State
    registered_deltas: Dict[str, DeltaFormatSpec] = field(default_factory=dict)
    # Per-layer occupied subspace bases, stacked
    # Key: layer_idx, Value: tensor of shape (d_model, total_occupied_rank)
    occupied_bases: Dict[int, torch.Tensor] = field(default_factory=dict)
    # Pairwise orthogonality scores
    compatibility_matrix: Dict[Tuple[str, str], float] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.registered_deltas)

    @property
    def theoretical_max_per_layer(self) -> int:
        """Maximum orthogonal deltas per layer = d_model / delta_rank."""
        return self.d_model // self.delta_rank

    def remaining_capacity_estimate(self) -> Dict[int, int]:
        """
        Estimate remaining orthogonal slots per layer.
        Returns {layer_idx: remaining_slots}.
        """
        result = {}
        for layer_idx in range(self.n_layers):
            if layer_idx in self.occupied_bases:
                occupied_rank = self.occupied_bases[layer_idx].shape[1]
                remaining_dims = self.d_model - occupied_rank
                result[layer_idx] = max(0, remaining_dims // self.delta_rank)
            else:
                result[layer_idx] = self.theoretical_max_per_layer
        return result

    def total_remaining_capacity(self) -> int:
        """Total remaining slots across all layers."""
        return sum(self.remaining_capacity_estimate().values())

    def check_orthogonality(self, delta: DeltaFormatSpec) -> Tuple[bool, float, List[str]]:
        """
        Check whether a proposed delta is sufficiently orthogonal to all
        existing deltas in the registry.

        Returns:
            (is_acceptable, min_score, conflicting_ids)
        """
        if delta.base_model_hash != self.base_model_hash:
            raise ValueError(
                f"Delta base_model_hash {delta.base_model_hash} "
                f"does not match registry {self.base_model_hash}"
            )

        min_score = 1.0
        conflicts = []

        for layer_idx in delta.target_layers:
            if layer_idx not in self.occupied_bases:
                continue  # Layer has no existing deltas - fully available

            existing_basis = self.occupied_bases[layer_idx]
            score = orthogonality_score(delta.subspace_basis, existing_basis)

            if score < min_score:
                min_score = score

            if score < self.orthogonality_threshold:
                # Find which specific deltas conflict at this layer
                for did, existing_delta in self.registered_deltas.items():
                    if layer_idx in existing_delta.target_layers:
                        pairwise = orthogonality_score(
                            delta.subspace_basis,
                            existing_delta.subspace_basis
                        )
                        if pairwise < self.orthogonality_threshold:
                            conflicts.append(did)

        is_acceptable = min_score >= self.orthogonality_threshold
        return is_acceptable, min_score, conflicts

    def register(self, delta: DeltaFormatSpec) -> Tuple[bool, str]:
        """
        Attempt to register a delta.

        Returns:
            (success, message)
            If rejected, message explains why and which subspaces conflict.
        """
        # Validate compatibility
        if delta.base_model_hash != self.base_model_hash:
            return False, "Incompatible base model hash"

        if delta.d_model != self.d_model or delta.n_layers != self.n_layers:
            return False, "Incompatible model dimensions"

        if delta.delta_id in self.registered_deltas:
            return False, f"Delta {delta.delta_id} already registered"

        # Check orthogonality
        is_ok, min_score, conflicts = self.check_orthogonality(delta)

        if not is_ok:
            return False, (
                f"Orthogonality check failed: min_score={min_score:.3f} "
                f"< threshold={self.orthogonality_threshold}. "
                f"Conflicts with: {conflicts}"
            )

        # Register
        self.registered_deltas[delta.delta_id] = delta

        # Update occupied bases
        for layer_idx in delta.target_layers:
            if layer_idx in self.occupied_bases:
                self.occupied_bases[layer_idx] = torch.cat(
                    [self.occupied_bases[layer_idx], delta.subspace_basis],
                    dim=1
                )
            else:
                self.occupied_bases[layer_idx] = delta.subspace_basis.clone()

        # Update compatibility matrix
        delta.orthogonality_score = min_score
        delta.conflicting_delta_ids = conflicts
        for did, existing in self.registered_deltas.items():
            if did == delta.delta_id:
                continue
            pairwise = orthogonality_score(
                delta.subspace_basis, existing.subspace_basis
            )
            self.compatibility_matrix[(delta.delta_id, did)] = pairwise
            self.compatibility_matrix[(did, delta.delta_id)] = pairwise

        return True, f"Registered delta {delta.delta_id} (score={min_score:.3f})"

    def unregister(self, delta_id: str) -> bool:
        """
        Remove a delta from the registry.
        Rebuilds occupied_bases for affected layers.
        """
        if delta_id not in self.registered_deltas:
            return False

        delta = self.registered_deltas.pop(delta_id)

        # Rebuild occupied bases for affected layers
        for layer_idx in delta.target_layers:
            bases = []
            for did, d in self.registered_deltas.items():
                if layer_idx in d.target_layers:
                    bases.append(d.subspace_basis)
            if bases:
                self.occupied_bases[layer_idx] = torch.cat(bases, dim=1)
            elif layer_idx in self.occupied_bases:
                del self.occupied_bases[layer_idx]

        # Clean compatibility matrix
        keys_to_remove = [
            k for k in self.compatibility_matrix
            if delta_id in k
        ]
        for k in keys_to_remove:
            del self.compatibility_matrix[k]

        return True

    def get_deltas_by_category(self, category: str) -> List[DeltaFormatSpec]:
        """Return all deltas in a given category."""
        return [
            d for d in self.registered_deltas.values()
            if d.category == category
        ]

    def get_compatible_deltas(self, delta_id: str,
                               min_score: float = 0.5) -> List[str]:
        """Return IDs of deltas that are well-orthogonal to the given delta."""
        results = []
        for (id_a, id_b), score in self.compatibility_matrix.items():
            if id_a == delta_id and score >= min_score:
                results.append(id_b)
        return results

    def save(self, path: str):
        """Save registry state to disk."""
        state = {
            "base_model_hash": self.base_model_hash,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "delta_rank": self.delta_rank,
            "orthogonality_threshold": self.orthogonality_threshold,
            "delta_paths": {},  # Map delta_id -> .delta file path
            "compatibility_matrix": {
                f"{k[0]}|{k[1]}": v
                for k, v in self.compatibility_matrix.items()
            },
        }
        # Save each delta to its own .delta file
        delta_dir = Path(path).parent / "deltas"
        delta_dir.mkdir(parents=True, exist_ok=True)
        for did, delta in self.registered_deltas.items():
            delta_path = str(delta_dir / f"{did}.delta")
            delta.save(delta_path)
            state["delta_paths"][did] = delta_path

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DeltaRegistry":
        """Load registry state from disk."""
        with open(path) as f:
            state = json.load(f)

        registry = cls(
            base_model_hash=state["base_model_hash"],
            d_model=state["d_model"],
            n_layers=state["n_layers"],
            delta_rank=state["delta_rank"],
            orthogonality_threshold=state["orthogonality_threshold"],
        )

        # Load deltas
        for did, delta_path in state["delta_paths"].items():
            delta = DeltaFormatSpec.load(delta_path)
            # Register without re-checking orthogonality (already validated)
            registry.registered_deltas[did] = delta
            for layer_idx in delta.target_layers:
                if layer_idx in registry.occupied_bases:
                    registry.occupied_bases[layer_idx] = torch.cat(
                        [registry.occupied_bases[layer_idx], delta.subspace_basis],
                        dim=1
                    )
                else:
                    registry.occupied_bases[layer_idx] = delta.subspace_basis.clone()

        # Restore compatibility matrix
        registry.compatibility_matrix = {
            tuple(k.split("|")): v
            for k, v in state["compatibility_matrix"].items()
        }

        return registry
