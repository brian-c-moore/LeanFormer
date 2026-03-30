"""
Compositional Router: routes queries to relevant delta subsets
and composes them additively at inference time.

Key capability: multiple domain deltas can be active simultaneously
for cross-domain queries. The router selects the relevant subset,
and the compose() function additively combines them.

Because deltas are orthogonality-constrained at forge time,
additive composition is safe - the deltas modify non-overlapping
subspaces of the weight space.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from .dfs import DeltaFormatSpec
from .registry import DeltaRegistry


class KnowledgePlaneRouter:
    """
    Routes queries to relevant deltas and composes them additively.
    """

    def __init__(self, registry: DeltaRegistry, top_k: int = 5):
        self.registry = registry
        self.top_k = top_k
        self._embedding_cache = None
        self._id_order = None

    def _build_embedding_matrix(self):
        """Cache the stacked embedding matrix for fast routing."""
        deltas = list(self.registry.registered_deltas.values())
        if not deltas:
            self._embedding_cache = None
            self._id_order = []
            return

        self._id_order = [d.delta_id for d in deltas]
        self._embedding_cache = torch.stack(
            [d.embedding for d in deltas]
        )  # (n_deltas, d_model)

    def route(
        self,
        query_embedding: torch.Tensor,
        top_k: int = None,
        min_similarity: float = 0.1,
    ) -> List[Tuple[str, float]]:
        """
        Route a query to the most relevant deltas.

        Args:
            query_embedding: (d_model,) tensor - the query's embedding
            top_k: how many deltas to select (default: self.top_k)
            min_similarity: minimum cosine similarity to include

        Returns:
            List of (delta_id, similarity_score) tuples, sorted by score desc.
        """
        if top_k is None:
            top_k = self.top_k

        self._build_embedding_matrix()
        if self._embedding_cache is None:
            return []

        # Cosine similarity
        query_norm = F.normalize(query_embedding.unsqueeze(0), dim=1)
        embed_norm = F.normalize(self._embedding_cache, dim=1)
        similarities = (query_norm @ embed_norm.T).squeeze(0)  # (n_deltas,)

        # Top-k selection
        k = min(top_k, len(self._id_order))
        values, indices = torch.topk(similarities, k)

        results = []
        for val, idx in zip(values.tolist(), indices.tolist()):
            if val >= min_similarity:
                results.append((self._id_order[idx], val))

        return results

    def compose(
        self,
        active_delta_ids: List[str],
        weights: List[float] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Additively compose active deltas into per-layer weight updates.

        For each layer that any active delta targets:
            composed_weight = sum_i weight_i * (A_i @ B_i)

        The orthogonality guarantee from the registry means this sum
        is safe - deltas were verified non-interfering at forge time.

        Args:
            active_delta_ids: Which deltas to activate
            weights: Optional per-delta weights (default: uniform)

        Returns:
            {layer_idx: composed_delta_weight} where composed_delta_weight
            is shape (d_model, d_model) - the full-rank composed update
            for that layer
        """
        if weights is None:
            weights = [1.0] * len(active_delta_ids)

        composed = {}

        for did, w in zip(active_delta_ids, weights):
            if did not in self.registry.registered_deltas:
                continue
            delta = self.registry.registered_deltas[did]

            for layer_idx, (A, B) in delta.factors.items():
                update = w * (A @ B)  # (d_model, d_model)
                if layer_idx in composed:
                    composed[layer_idx] = composed[layer_idx] + update
                else:
                    composed[layer_idx] = update

        return composed

    def route_and_compose(
        self,
        query_embedding: torch.Tensor,
        top_k: int = None,
    ) -> Tuple[Dict[int, torch.Tensor], List[Tuple[str, float]]]:
        """
        Combined route + compose for inference.
        Returns (composed_updates, routing_decisions).
        """
        routing = self.route(query_embedding, top_k)
        if not routing:
            return {}, []

        ids = [r[0] for r in routing]
        scores = [r[1] for r in routing]
        composed = self.compose(ids, scores)

        return composed, routing
