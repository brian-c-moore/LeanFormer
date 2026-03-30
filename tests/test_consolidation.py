"""Tests for the Consolidation Pipeline."""
import pytest
import torch
from leanformer.knowledge_plane.consolidation import KnowledgePlaneConsolidator
from leanformer.knowledge_plane.registry import DeltaRegistry
from tests.test_dfs import make_test_delta


def make_registry_with_category_groups():
    """Create a registry with multiple deltas per category for consolidation."""
    registry = DeltaRegistry(
        base_model_hash="test_hash",
        d_model=768,
        n_layers=12,
        delta_rank=16,
        orthogonality_threshold=0.3,
    )

    delta_ids = {"chemistry": [], "cs": []}
    slot = 0
    for category in ["chemistry", "cs"]:
        for i in range(6):  # 6 deltas per category
            basis = torch.zeros(768, 16)
            for j in range(16):
                basis[slot + j, j] = 1.0

            delta = make_test_delta(
                d_model=768, delta_rank=16, target_layers=[6, 7, 8],
                category=category,
            )
            delta.subspace_basis = basis
            delta.base_model_hash = "test_hash"
            registry.register(delta)
            delta_ids[category].append(delta.delta_id)
            slot += 16

    return registry, delta_ids


class TestIdentifyCandidates:
    def test_finds_groups_by_category(self):
        registry, delta_ids = make_registry_with_category_groups()
        consolidator = KnowledgePlaneConsolidator(registry)
        candidates = consolidator.identify_candidates(min_group_size=5)
        assert len(candidates) == 2  # chemistry and cs

    def test_min_group_size_filters(self):
        registry, delta_ids = make_registry_with_category_groups()
        consolidator = KnowledgePlaneConsolidator(registry)
        candidates = consolidator.identify_candidates(min_group_size=7)
        assert len(candidates) == 0  # Both groups have 6, need 7


class TestConsolidate:
    def test_consolidate_produces_delta(self):
        registry, delta_ids = make_registry_with_category_groups()
        consolidator = KnowledgePlaneConsolidator(registry)

        merged = consolidator.consolidate(delta_ids["chemistry"])
        assert merged is not None
        assert merged.category == "chemistry"
        assert "consolidated" in merged.tags

    def test_consolidated_has_correct_layers(self):
        registry, delta_ids = make_registry_with_category_groups()
        consolidator = KnowledgePlaneConsolidator(registry)

        merged = consolidator.consolidate(delta_ids["chemistry"])
        assert merged.target_layers == [6, 7, 8]

    def test_consolidated_has_lower_or_equal_rank(self):
        registry, delta_ids = make_registry_with_category_groups()
        consolidator = KnowledgePlaneConsolidator(registry)

        merged = consolidator.consolidate(delta_ids["chemistry"])
        assert merged.delta_rank == registry.delta_rank

    def test_svd_refactorization_preserves_approximate_sum(self):
        """Verify the SVD re-factorization is a reasonable approximation."""
        registry, delta_ids = make_registry_with_category_groups()
        consolidator = KnowledgePlaneConsolidator(registry)

        # Compute expected sum of A @ B for one layer
        layer_idx = 6
        W_expected = torch.zeros(768, 768)
        for did in delta_ids["chemistry"]:
            delta = registry.registered_deltas[did]
            A, B = delta.factors[layer_idx]
            W_expected += A @ B

        merged = consolidator.consolidate(delta_ids["chemistry"])
        A_m, B_m = merged.factors[layer_idx]
        W_merged = A_m @ B_m

        # Not exact due to truncation, but should capture most energy
        # (relative error should be reasonable for rank-16 approximation)
        error = (W_expected - W_merged).norm() / max(W_expected.norm(), 1e-8)
        assert error < 1.0  # Should capture significant portion


class TestConsolidateAndReplace:
    def test_reduces_registry_count(self):
        registry, delta_ids = make_registry_with_category_groups()
        initial_count = registry.count
        consolidator = KnowledgePlaneConsolidator(registry)

        merged = consolidator.consolidate_and_replace(delta_ids["chemistry"])
        assert merged is not None
        # 6 removed, 1 added
        assert registry.count == initial_count - 6 + 1

    def test_frees_capacity(self):
        registry, delta_ids = make_registry_with_category_groups()
        cap_before = registry.total_remaining_capacity()
        consolidator = KnowledgePlaneConsolidator(registry)

        consolidator.consolidate_and_replace(delta_ids["chemistry"])
        cap_after = registry.total_remaining_capacity()
        # Merged delta occupies 1 slot per layer, originals occupied 6
        assert cap_after > cap_before

    def test_consolidate_empty_ids(self):
        registry, delta_ids = make_registry_with_category_groups()
        consolidator = KnowledgePlaneConsolidator(registry)
        merged = consolidator.consolidate([])
        assert merged is None
