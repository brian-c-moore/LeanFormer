"""
Output Provenance System with Graded Confidence Scoring.

Instruments the output pathway to attribute each generated output to
its knowledge source and surface a graded, architecturally-grounded
confidence score alongside each output.

The confidence score is composed of three graded components:
  1. Routing strength — cosine similarity between query and activating delta
  2. Composition coherence — how cleanly the active deltas composed
  3. Delta coverage — whether any delta exists for the query domain

This is not a softmax probability. Softmax measures distributional fit.
This measures epistemic grounding — how well the architecture's knowledge
layer covered the query. These are different things, and the difference
is the contribution.
"""

import time
import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from .registry import DeltaRegistry, orthogonality_score


@dataclass
class ProvenanceSignal:
    """
    Complete provenance record for a single inference pass.

    Carries the three graded confidence components plus the
    combined score and uncertainty flag.
    """
    # === Graded Components ===
    routing_strength: float          # Max cosine similarity in routing (0-1)
    composition_coherence: float     # How cleanly deltas composed (0-1)
    delta_coverage: float            # Whether knowledge exists for query (0-1)

    # === Combined Score ===
    confidence_score: float          # Weighted combination of components (0-1)
    uncertainty_flag: bool           # True if below threshold

    # === Attribution ===
    source_deltas: List[str]         # IDs of contributing deltas
    source_categories: List[str]     # Categories of contributing deltas
    routing_decisions: List[Tuple[str, float]]  # (delta_id, similarity) pairs

    # === Metadata ===
    timestamp: float = 0.0
    query_embedding_norm: float = 0.0


@dataclass
class ProvenanceLog:
    """Maintains a rolling log of provenance records for auditability."""
    entries: List[ProvenanceSignal] = field(default_factory=list)
    max_entries: int = 1000

    def add(self, signal: ProvenanceSignal):
        self.entries.append(signal)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def get_recent(self, n: int = 100) -> List[ProvenanceSignal]:
        return self.entries[-n:]

    def average_confidence(self, n: int = 100) -> float:
        recent = self.get_recent(n)
        if not recent:
            return 0.0
        return sum(s.confidence_score for s in recent) / len(recent)

    def uncertainty_rate(self, n: int = 100) -> float:
        recent = self.get_recent(n)
        if not recent:
            return 0.0
        return sum(1 for s in recent if s.uncertainty_flag) / len(recent)


class ConfidenceScorer:
    """
    Computes architecturally-grounded confidence scores from
    Knowledge Plane signals.

    The score derives from the knowledge retrieval mechanism itself,
    not from output token probabilities. A score of 1.0 means the
    Knowledge Plane had a strong, clean, unambiguous match for the
    query. A score of 0.0 means no knowledge was available.

    The three components are weighted and combined:
    - routing_weight: importance of query-delta similarity
    - coherence_weight: importance of clean delta composition
    - coverage_weight: importance of having knowledge at all
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.3,
        routing_weight: float = 0.4,
        coherence_weight: float = 0.3,
        coverage_weight: float = 0.3,
        min_routing_similarity: float = 0.1,
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.routing_weight = routing_weight
        self.coherence_weight = coherence_weight
        self.coverage_weight = coverage_weight
        self.min_routing_similarity = min_routing_similarity

        # Normalize weights
        total = routing_weight + coherence_weight + coverage_weight
        self.routing_weight /= total
        self.coherence_weight /= total
        self.coverage_weight /= total

    def compute_routing_strength(
        self,
        routing_decisions: List[Tuple[str, float]],
    ) -> float:
        """
        Compute routing strength from routing decisions.

        Returns the maximum cosine similarity among routed deltas,
        scaled to [0, 1]. Higher = query matched a delta closely.
        Zero if no deltas were activated.
        """
        if not routing_decisions:
            return 0.0

        max_similarity = max(score for _, score in routing_decisions)
        # Cosine similarity is already in [-1, 1]; we clamp to [0, 1]
        return max(0.0, min(1.0, max_similarity))

    def compute_composition_coherence(
        self,
        routing_decisions: List[Tuple[str, float]],
        registry: DeltaRegistry,
    ) -> float:
        """
        Compute how cleanly the active deltas composed.

        Coherence is high when:
        - Only one delta is active (no potential interference)
        - Active deltas are highly orthogonal to each other
        - All active deltas have similar routing scores (no marginal activations)

        Returns value in [0, 1]. 1.0 = perfectly coherent.
        """
        if not routing_decisions:
            return 0.0  # No deltas = no composition to evaluate

        if len(routing_decisions) == 1:
            return 1.0  # Single delta = perfectly coherent

        delta_ids = [did for did, _ in routing_decisions]
        scores = [score for _, score in routing_decisions]

        # Factor 1: Pairwise orthogonality of active deltas
        orth_scores = []
        for i, did_a in enumerate(delta_ids):
            for j, did_b in enumerate(delta_ids):
                if i >= j:
                    continue
                key = (did_a, did_b)
                if key in registry.compatibility_matrix:
                    orth_scores.append(registry.compatibility_matrix[key])
                else:
                    # Compute on the fly
                    if (did_a in registry.registered_deltas and
                            did_b in registry.registered_deltas):
                        da = registry.registered_deltas[did_a]
                        db = registry.registered_deltas[did_b]
                        score = orthogonality_score(
                            da.subspace_basis, db.subspace_basis
                        )
                        orth_scores.append(score)

        avg_orth = sum(orth_scores) / len(orth_scores) if orth_scores else 1.0

        # Factor 2: Score consistency (low variance = all deltas equally relevant)
        if len(scores) > 1:
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            # Normalize: 0 variance = 1.0 consistency, high variance = lower
            consistency = 1.0 / (1.0 + 10.0 * variance)
        else:
            consistency = 1.0

        # Combine: orthogonality is more important than consistency
        coherence = 0.7 * avg_orth + 0.3 * consistency
        return max(0.0, min(1.0, coherence))

    def compute_delta_coverage(
        self,
        routing_decisions: List[Tuple[str, float]],
        registry: DeltaRegistry,
    ) -> float:
        """
        Compute delta coverage: whether knowledge exists for the query.

        Coverage is 0 if no deltas activated (Knowledge Plane returned empty).
        Coverage scales with the number and quality of matching deltas.

        Returns value in [0, 1].
        """
        if not routing_decisions:
            return 0.0

        if registry.count == 0:
            return 0.0

        # Scale coverage by number of relevant deltas vs total available
        n_activated = len(routing_decisions)
        max_relevant = min(5, registry.count)  # Cap at 5 for normalization

        # Coverage increases with number of activated deltas and their scores
        avg_score = sum(s for _, s in routing_decisions) / n_activated
        activation_fraction = min(1.0, n_activated / max_relevant)

        # High coverage = multiple relevant deltas with good scores
        coverage = 0.6 * avg_score + 0.4 * activation_fraction
        return max(0.0, min(1.0, coverage))

    def score(
        self,
        routing_decisions: List[Tuple[str, float]],
        registry: DeltaRegistry,
        query_embedding: Optional[torch.Tensor] = None,
    ) -> ProvenanceSignal:
        """
        Compute the full provenance signal for an inference pass.

        Args:
            routing_decisions: List of (delta_id, similarity) from router
            registry: The delta registry
            query_embedding: Optional query embedding for metadata

        Returns:
            ProvenanceSignal with all components filled
        """
        routing_strength = self.compute_routing_strength(routing_decisions)
        coherence = self.compute_composition_coherence(routing_decisions, registry)
        coverage = self.compute_delta_coverage(routing_decisions, registry)

        # Weighted combination
        confidence = (
            self.routing_weight * routing_strength +
            self.coherence_weight * coherence +
            self.coverage_weight * coverage
        )

        # Extract attribution info
        source_deltas = [did for did, _ in routing_decisions]
        source_categories = []
        for did in source_deltas:
            if did in registry.registered_deltas:
                cat = registry.registered_deltas[did].category
                if cat not in source_categories:
                    source_categories.append(cat)

        emb_norm = 0.0
        if query_embedding is not None:
            emb_norm = query_embedding.norm().item()

        return ProvenanceSignal(
            routing_strength=routing_strength,
            composition_coherence=coherence,
            delta_coverage=coverage,
            confidence_score=confidence,
            uncertainty_flag=confidence < self.uncertainty_threshold,
            source_deltas=source_deltas,
            source_categories=source_categories,
            routing_decisions=routing_decisions,
            timestamp=time.time(),
            query_embedding_norm=emb_norm,
        )

    def score_base_only(self) -> ProvenanceSignal:
        """
        Produce a provenance signal for base-model-only inference
        (no Knowledge Plane activation).
        """
        return ProvenanceSignal(
            routing_strength=0.0,
            composition_coherence=0.0,
            delta_coverage=0.0,
            confidence_score=0.0,
            uncertainty_flag=True,
            source_deltas=[],
            source_categories=[],
            routing_decisions=[],
            timestamp=time.time(),
            query_embedding_norm=0.0,
        )
