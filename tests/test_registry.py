"""Tests for Delta Registry and orthogonality enforcement."""
import pytest
import torch
import math
from leanformer.knowledge_plane.registry import (
    DeltaRegistry, compute_principal_angles,
    minimum_principal_angle, orthogonality_score,
)
from leanformer.knowledge_plane.dfs import DeltaFormatSpec
from tests.test_dfs import make_test_delta


class TestPrincipalAngles:
    def test_identical_subspaces_have_zero_angle(self):
        basis = torch.eye(768, 16)
        angle = minimum_principal_angle(basis, basis)
        assert angle < 0.01  # Numerically close to 0

    def test_orthogonal_subspaces_have_pi_half(self):
        basis_a = torch.zeros(768, 16)
        basis_b = torch.zeros(768, 16)
        for i in range(16):
            basis_a[i, i] = 1.0       # First 16 dims
            basis_b[i + 16, i] = 1.0  # Next 16 dims
        angle = minimum_principal_angle(basis_a, basis_b)
        assert abs(angle - math.pi / 2) < 0.01

    def test_orthogonality_score_range(self):
        basis_a = torch.randn(768, 16)
        basis_b = torch.randn(768, 16)
        score = orthogonality_score(basis_a, basis_b)
        assert 0.0 <= score <= 1.0

    def test_identical_score_is_zero(self):
        basis = torch.eye(768, 16)
        score = orthogonality_score(basis, basis)
        assert score < 0.01

    def test_orthogonal_score_is_one(self):
        basis_a = torch.zeros(768, 16)
        basis_b = torch.zeros(768, 16)
        for i in range(16):
            basis_a[i, i] = 1.0
            basis_b[i + 16, i] = 1.0
        score = orthogonality_score(basis_a, basis_b)
        assert score > 0.99


class TestDeltaRegistry:
    def make_registry(self):
        return DeltaRegistry(
            base_model_hash="test_hash",
            d_model=768,
            n_layers=12,
            delta_rank=16,
            orthogonality_threshold=0.3,
        )

    def make_orthogonal_delta(self, registry, slot_start, category="test"):
        """Create a delta occupying a specific orthogonal slot."""
        basis = torch.zeros(768, 16)
        for i in range(16):
            basis[slot_start + i, i] = 1.0

        delta = make_test_delta(
            d_model=768, delta_rank=16, target_layers=[6, 7, 8],
            category=category,
        )
        delta.subspace_basis = basis
        delta.base_model_hash = "test_hash"
        return delta

    def test_register_first_delta(self):
        registry = self.make_registry()
        delta = self.make_orthogonal_delta(registry, 0)
        success, msg = registry.register(delta)
        assert success, msg
        assert registry.count == 1

    def test_register_orthogonal_deltas(self):
        registry = self.make_registry()
        d1 = self.make_orthogonal_delta(registry, 0)
        d2 = self.make_orthogonal_delta(registry, 16)
        d3 = self.make_orthogonal_delta(registry, 32)

        s1, _ = registry.register(d1)
        s2, _ = registry.register(d2)
        s3, _ = registry.register(d3)

        assert s1 and s2 and s3
        assert registry.count == 3

    def test_reject_overlapping_delta(self):
        registry = self.make_registry()
        d1 = self.make_orthogonal_delta(registry, 0)
        d2 = self.make_orthogonal_delta(registry, 0)  # Same slot!

        registry.register(d1)
        success, msg = registry.register(d2)
        assert not success
        assert "Orthogonality" in msg

    def test_reject_wrong_model_hash(self):
        registry = self.make_registry()
        delta = self.make_orthogonal_delta(registry, 0)
        delta.base_model_hash = "wrong_hash"
        success, msg = registry.register(delta)
        assert not success

    def test_unregister_frees_capacity(self):
        registry = self.make_registry()
        d1 = self.make_orthogonal_delta(registry, 0)
        registry.register(d1)
        cap_before = registry.total_remaining_capacity()

        registry.unregister(d1.delta_id)
        cap_after = registry.total_remaining_capacity()
        assert cap_after > cap_before
        assert registry.count == 0

    def test_capacity_accounting(self):
        registry = self.make_registry()
        max_per_layer = 768 // 16  # 48
        cap = registry.remaining_capacity_estimate()
        assert all(v == max_per_layer for v in cap.values())

    def test_compatibility_matrix_populated(self):
        registry = self.make_registry()
        d1 = self.make_orthogonal_delta(registry, 0, category="a")
        d2 = self.make_orthogonal_delta(registry, 16, category="b")
        registry.register(d1)
        registry.register(d2)

        key = (d1.delta_id, d2.delta_id)
        assert key in registry.compatibility_matrix
        assert registry.compatibility_matrix[key] > 0.9  # Should be orthogonal

    def test_get_deltas_by_category(self):
        registry = self.make_registry()
        d1 = self.make_orthogonal_delta(registry, 0, category="chemistry")
        d2 = self.make_orthogonal_delta(registry, 16, category="cs")
        d3 = self.make_orthogonal_delta(registry, 32, category="chemistry")
        registry.register(d1)
        registry.register(d2)
        registry.register(d3)

        chem = registry.get_deltas_by_category("chemistry")
        assert len(chem) == 2

    def test_save_load_roundtrip(self):
        import tempfile, os
        registry = self.make_registry()
        d1 = self.make_orthogonal_delta(registry, 0)
        d2 = self.make_orthogonal_delta(registry, 16)
        registry.register(d1)
        registry.register(d2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "registry.json")
            registry.save(path)
            loaded = DeltaRegistry.load(path)

            assert loaded.count == 2
            assert loaded.base_model_hash == "test_hash"
            assert loaded.d_model == 768
