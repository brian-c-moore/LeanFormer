"""Tests for Delta Format Specification v2.0"""
import pytest
import torch
import tempfile
import os
from leanformer.knowledge_plane.dfs import DeltaFormatSpec


def make_test_delta(d_model=768, delta_rank=16, n_layers=12,
                    target_layers=None, category="test"):
    """Helper to create a valid test delta."""
    if target_layers is None:
        target_layers = [6, 7, 8]  # Default: middle layers only

    factors = {}
    for idx in target_layers:
        A = torch.randn(d_model, delta_rank)
        B = torch.randn(delta_rank, d_model)
        factors[idx] = (A, B)

    delta = DeltaFormatSpec(
        delta_id=DeltaFormatSpec.create_id(),
        version=1,
        created_at=DeltaFormatSpec.timestamp(),
        source="test_harness",
        embedding=torch.randn(d_model),
        category=category,
        description="Test delta",
        domain_tags=["test"],
        target_layers=target_layers,
        factors=factors,
        delta_rank=delta_rank,
        param_count=0,  # Will compute
        subspace_basis=torch.randn(d_model, delta_rank),
        orthogonality_score=0.95,
        conflicting_delta_ids=[],
        base_model_hash="test_hash_abc123",
        d_model=d_model,
        n_layers=n_layers,
    )
    delta.param_count = delta.compute_param_count()
    return delta


class TestDFSCreation:
    def test_create_valid_delta(self):
        delta = make_test_delta()
        errors = delta.validate_shapes()
        assert errors == [], f"Validation errors: {errors}"

    def test_param_count_computation(self):
        delta = make_test_delta(d_model=768, delta_rank=16, target_layers=[6, 7, 8])
        expected = 3 * (768 * 16 + 16 * 768)  # 3 layers × (A + B)
        assert delta.param_count == expected

    def test_targeted_layers_reduce_params(self):
        full = make_test_delta(target_layers=list(range(12)))
        targeted = make_test_delta(target_layers=[6, 7, 8])
        assert targeted.param_count < full.param_count
        assert targeted.param_count == full.param_count * 3 / 12

    def test_shape_validation_catches_bad_embedding(self):
        delta = make_test_delta()
        delta.embedding = torch.randn(256)  # Wrong size
        errors = delta.validate_shapes()
        assert len(errors) > 0
        assert "Embedding" in errors[0]

    def test_shape_validation_catches_bad_factors(self):
        delta = make_test_delta(d_model=768, delta_rank=16, target_layers=[6])
        delta.factors[6] = (torch.randn(768, 32), torch.randn(32, 768))  # Wrong rank
        errors = delta.validate_shapes()
        assert len(errors) > 0

    def test_unique_ids(self):
        ids = {DeltaFormatSpec.create_id() for _ in range(100)}
        assert len(ids) == 100


class TestDFSSerialization:
    def test_roundtrip(self):
        delta = make_test_delta()
        with tempfile.NamedTemporaryFile(suffix=".delta", delete=False) as f:
            path = f.name
        try:
            delta.save(path)
            loaded = DeltaFormatSpec.load(path)

            assert loaded.delta_id == delta.delta_id
            assert loaded.version == delta.version
            assert loaded.category == delta.category
            assert loaded.target_layers == delta.target_layers
            assert loaded.delta_rank == delta.delta_rank
            assert loaded.param_count == delta.param_count
            assert loaded.d_model == delta.d_model
            assert loaded.format_version == "2.0"

            # Tensor equality
            assert torch.allclose(loaded.embedding, delta.embedding)
            assert torch.allclose(loaded.subspace_basis, delta.subspace_basis)
            for layer_idx in delta.target_layers:
                A_orig, B_orig = delta.factors[layer_idx]
                A_load, B_load = loaded.factors[layer_idx]
                assert torch.allclose(A_load, A_orig)
                assert torch.allclose(B_load, B_orig)
        finally:
            os.unlink(path)

    def test_roundtrip_with_metadata(self):
        delta = make_test_delta()
        delta.tags = ["important", "verified"]
        delta.composable_with = ["delta-abc"]
        delta.incompatible_with = ["delta-xyz"]
        delta.confidence = 0.87
        delta.validation_results = {"rank_before": 5000, "rank_after": 12}

        with tempfile.NamedTemporaryFile(suffix=".delta", delete=False) as f:
            path = f.name
        try:
            delta.save(path)
            loaded = DeltaFormatSpec.load(path)
            assert loaded.tags == ["important", "verified"]
            assert loaded.composable_with == ["delta-abc"]
            assert loaded.confidence == 0.87
            assert loaded.validation_results["rank_after"] == 12
        finally:
            os.unlink(path)

    def test_file_is_zip(self):
        import zipfile
        delta = make_test_delta()
        with tempfile.NamedTemporaryFile(suffix=".delta", delete=False) as f:
            path = f.name
        try:
            delta.save(path)
            assert zipfile.is_zipfile(path)
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
                assert "manifest.json" in names
                assert "embedding.pt" in names
                assert "subspace_basis.pt" in names
        finally:
            os.unlink(path)


class TestDFSConstraints:
    def test_targeted_layers_subset(self):
        """Deltas should only target a subset of layers."""
        delta = make_test_delta(target_layers=[6, 7, 8])
        assert len(delta.target_layers) == 3
        assert all(0 <= l < 12 for l in delta.target_layers)

    def test_delta_rank_smaller_than_d_model(self):
        delta = make_test_delta(d_model=768, delta_rank=16)
        assert delta.delta_rank < delta.d_model
        assert delta.delta_rank == 16
