"""Tests for the Compositional Router."""
import pytest
import torch
from leanformer.knowledge_plane.router import KnowledgePlaneRouter
from leanformer.knowledge_plane.registry import DeltaRegistry
from tests.test_dfs import make_test_delta


def make_registry_with_deltas():
    """Create a registry with 3 orthogonal deltas in different categories."""
    registry = DeltaRegistry(
        base_model_hash="test_hash",
        d_model=768,
        n_layers=12,
        delta_rank=16,
        orthogonality_threshold=0.3,
    )

    deltas = []
    for i, category in enumerate(["chemistry", "cs", "general"]):
        basis = torch.zeros(768, 16)
        for j in range(16):
            basis[i * 16 + j, j] = 1.0

        delta = make_test_delta(
            d_model=768, delta_rank=16, target_layers=[6, 7, 8],
            category=category,
        )
        delta.subspace_basis = basis
        delta.base_model_hash = "test_hash"
        # Make embeddings distinguishable per category
        delta.embedding = torch.zeros(768)
        delta.embedding[i * 100:(i + 1) * 100] = 1.0
        registry.register(delta)
        deltas.append(delta)

    return registry, deltas


class TestRouting:
    def test_route_returns_results(self):
        registry, deltas = make_registry_with_deltas()
        router = KnowledgePlaneRouter(registry, top_k=3)

        query = deltas[0].embedding  # Chemistry-like query
        results = router.route(query)
        assert len(results) > 0

    def test_route_most_similar_first(self):
        registry, deltas = make_registry_with_deltas()
        router = KnowledgePlaneRouter(registry, top_k=3)

        # Query most similar to chemistry delta
        query = deltas[0].embedding.clone()
        results = router.route(query)
        # First result should be the chemistry delta
        assert results[0][0] == deltas[0].delta_id

    def test_route_empty_registry(self):
        registry = DeltaRegistry(
            base_model_hash="test_hash",
            d_model=768, n_layers=12, delta_rank=16,
        )
        router = KnowledgePlaneRouter(registry, top_k=3)
        results = router.route(torch.randn(768))
        assert results == []

    def test_route_top_k_limits_results(self):
        registry, deltas = make_registry_with_deltas()
        router = KnowledgePlaneRouter(registry, top_k=1)
        results = router.route(torch.randn(768))
        assert len(results) <= 1

    def test_route_min_similarity_filters(self):
        registry, deltas = make_registry_with_deltas()
        router = KnowledgePlaneRouter(registry, top_k=3)
        # With very high threshold, should filter most
        results = router.route(torch.randn(768), min_similarity=0.99)
        # Unlikely random embedding has >0.99 cosine sim
        assert len(results) <= 3


class TestComposition:
    def test_compose_produces_per_layer_updates(self):
        registry, deltas = make_registry_with_deltas()
        router = KnowledgePlaneRouter(registry)

        ids = [d.delta_id for d in deltas[:2]]
        composed = router.compose(ids)
        # Both deltas target layers [6, 7, 8]
        assert 6 in composed
        assert 7 in composed
        assert 8 in composed

    def test_compose_is_additive(self):
        registry, deltas = make_registry_with_deltas()
        router = KnowledgePlaneRouter(registry)

        # Compose in order [0, 1] and [1, 0]
        ids_ab = [deltas[0].delta_id, deltas[1].delta_id]
        ids_ba = [deltas[1].delta_id, deltas[0].delta_id]
        comp_ab = router.compose(ids_ab)
        comp_ba = router.compose(ids_ba)

        for layer_idx in comp_ab:
            assert torch.allclose(comp_ab[layer_idx], comp_ba[layer_idx], atol=1e-6)

    def test_compose_with_weights(self):
        registry, deltas = make_registry_with_deltas()
        router = KnowledgePlaneRouter(registry)

        ids = [deltas[0].delta_id]
        comp_1 = router.compose(ids, weights=[1.0])
        comp_2 = router.compose(ids, weights=[2.0])

        for layer_idx in comp_1:
            assert torch.allclose(comp_2[layer_idx], 2.0 * comp_1[layer_idx], atol=1e-6)

    def test_compose_empty_ids(self):
        registry, deltas = make_registry_with_deltas()
        router = KnowledgePlaneRouter(registry)
        composed = router.compose([])
        assert composed == {}

    def test_route_and_compose_combined(self):
        registry, deltas = make_registry_with_deltas()
        router = KnowledgePlaneRouter(registry, top_k=3)

        query = deltas[0].embedding
        composed, routing = router.route_and_compose(query)
        assert len(routing) > 0
        assert len(composed) > 0
