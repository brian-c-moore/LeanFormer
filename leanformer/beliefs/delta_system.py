"""
Delta-Based Belief System.

Solves catastrophic forgetting by treating it as a shared mutable state problem:
- Base weights are frozen after training (the "reasoning kernel")
- Individual beliefs are encoded as sparse, independently-addressable weight deltas
- A registry governs delta allocation to prevent overlap
- A router selects relevant deltas per query before the forward pass

Each belief stores low-rank (dA, dB) pairs per targeted layer/projection.
Forward becomes: x @ A @ B + x @ dA @ dB
A delta with rank 4 costs ~0.01% of base model parameters.

This is Orkestratum's resource management applied to neural network weights:
    Immutable base + versioned overlays (copy-on-write)
    Sparse access patterns (non-overlapping writer regions)
    Registry-governed allocation (Budget<WeightSpace>)
    Query-time routing (SlotArbitrationPass)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BeliefDelta:
    """A single belief encoded as low-rank weight deltas across targeted layers."""
    name: str
    embedding: torch.Tensor                           # semantic key for routing (d_model,)
    layer_deltas: dict[str, tuple[torch.Tensor, torch.Tensor]]  # "layer_i_target" -> (dA, dB)
    metadata: dict = field(default_factory=dict)

    def total_parameters(self) -> int:
        """Total parameters stored in this delta."""
        total = 0
        for dA, dB in self.layer_deltas.values():
            total += dA.numel() + dB.numel()
        return total

    def target_keys(self) -> set[str]:
        """Which layer/projection combinations this delta touches."""
        return set(self.layer_deltas.keys())

    def to(self, device: torch.device) -> "BeliefDelta":
        """Move all tensors to a device."""
        self.embedding = self.embedding.to(device)
        self.layer_deltas = {
            k: (dA.to(device), dB.to(device))
            for k, (dA, dB) in self.layer_deltas.items()
        }
        return self


class DeltaRegistry:
    """
    Registry of belief deltas with allocation governance.

    Tracks which weight regions each delta touches. Provides cosine similarity
    based query for routing. Enforces awareness of overlap (warns but does not
    block in the PoC — production would enforce strict non-overlap).

    This is GpuResourceRegistry for weight space.
    """

    def __init__(self):
        self.deltas: dict[str, BeliefDelta] = {}

    def register(self, delta: BeliefDelta) -> dict:
        """
        Register a delta. Returns info dict with overlap warnings if any.
        """
        info = {"name": delta.name, "overlap_warnings": []}

        # Check for overlap with existing deltas
        new_keys = delta.target_keys()
        for existing_name, existing_delta in self.deltas.items():
            overlap = new_keys & existing_delta.target_keys()
            if overlap:
                info["overlap_warnings"].append({
                    "existing": existing_name,
                    "overlapping_targets": list(overlap),
                })

        self.deltas[delta.name] = delta
        info["registered"] = True
        info["total_parameters"] = delta.total_parameters()
        return info

    def query(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 4,
        threshold: float = 0.0,
    ) -> list[BeliefDelta]:
        """
        Find the most relevant deltas for a query via cosine similarity.
        Returns up to top_k deltas whose similarity exceeds threshold.
        """
        if not self.deltas:
            return []

        # Stack all delta embeddings
        names = list(self.deltas.keys())
        embeddings = torch.stack([self.deltas[n].embedding for n in names])

        # Cosine similarity
        query_norm = F.normalize(query_embedding.unsqueeze(0), dim=-1)
        delta_norms = F.normalize(embeddings, dim=-1)
        similarities = (query_norm @ delta_norms.T).squeeze(0)  # (n_deltas,)

        # Filter by threshold and select top-k
        mask = similarities >= threshold
        if not mask.any():
            return []

        k = min(top_k, mask.sum().item())
        topk_vals, topk_idx = torch.topk(similarities * mask.float(), k)

        results = []
        for val, idx in zip(topk_vals, topk_idx):
            if val > threshold:
                results.append(self.deltas[names[idx]])

        return results

    def remove(self, name: str) -> bool:
        """Remove a belief delta by name."""
        if name in self.deltas:
            del self.deltas[name]
            return True
        return False

    def update(self, name: str, new_delta: BeliefDelta) -> dict:
        """Replace a delta with a new one (correct a belief)."""
        self.remove(name)
        return self.register(new_delta)

    def list_beliefs(self) -> list[dict]:
        """List all registered beliefs with their metadata."""
        return [
            {
                "name": d.name,
                "parameters": d.total_parameters(),
                "targets": len(d.layer_deltas),
                "metadata": d.metadata,
            }
            for d in self.deltas.values()
        ]

    def __len__(self) -> int:
        return len(self.deltas)

    def __contains__(self, name: str) -> bool:
        return name in self.deltas


class DeltaRouter(nn.Module):
    """
    Routes queries to relevant deltas before the forward pass.

    This is SlotArbitrationPass: the query embedding competes against all
    registered delta embeddings via cosine similarity. Winners are applied
    to base weights before computation.
    """

    def __init__(self, d_model: int, max_active_deltas: int = 4, threshold: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.max_active_deltas = max_active_deltas
        self.threshold = threshold

    def forward(
        self,
        hidden_states: torch.Tensor,
        registry: DeltaRegistry,
    ) -> list[BeliefDelta]:
        """
        Select relevant deltas for the current input.

        Args:
            hidden_states: (B, T, d_model) — output of embedding layer
            registry: the delta registry to query

        Returns:
            List of relevant BeliefDelta objects
        """
        if len(registry) == 0:
            return []

        # Mean-pool over batch and sequence to get a single query vector
        query_embedding = hidden_states.mean(dim=(0, 1))  # (d_model,)

        return registry.query(
            query_embedding,
            top_k=self.max_active_deltas,
            threshold=self.threshold,
        )


def build_delta_map(
    active_deltas: list[BeliefDelta],
) -> dict[int, dict[str, list[tuple[torch.Tensor, torch.Tensor]]]]:
    """
    Organize active deltas into a lookup structure for the forward pass.

    Returns: {layer_idx: {target_name: [(dA, dB), ...]}}
    """
    delta_map: dict[int, dict[str, list[tuple[torch.Tensor, torch.Tensor]]]] = {}

    for delta in active_deltas:
        for key, (dA, dB) in delta.layer_deltas.items():
            # Key format: "layer_{i}_{target}"
            parts = key.split("_", 2)
            if len(parts) != 3 or parts[0] != "layer":
                continue
            layer_idx = int(parts[1])
            target = parts[2]

            if layer_idx not in delta_map:
                delta_map[layer_idx] = {}
            if target not in delta_map[layer_idx]:
                delta_map[layer_idx][target] = []
            delta_map[layer_idx][target].append((dA, dB))

    return delta_map


def forward_with_deltas(
    module: "LowRankLinear",
    x: torch.Tensor,
    deltas: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> torch.Tensor:
    """
    Compute module(x) + sum(x @ dA @ dB for each delta).
    When deltas is None or empty, this is identical to module(x).
    """
    out = module(x)
    if deltas:
        for dA, dB in deltas:
            out = out + x @ dA @ dB
    return out
