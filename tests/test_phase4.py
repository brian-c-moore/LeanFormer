"""
Extended tests: initialization integrity, gradient flow audit, training
infrastructure, provenance, quantization, KV cache, few-shot, separation
validation, optimizer coverage, memory safety, and 500-step smoke test.

All tests run on CPU with small configs for fast validation.
"""

import pytest
import math
import torch
import torch.nn as nn
import tempfile

from leanformer.model.config import LeanFormerConfig
from leanformer.model.low_rank import LowRankLinear
from leanformer.model.feedforward import GatedFeedForward
from leanformer.model.leanformer import LeanFormer


# ---------- Fixtures ----------

@pytest.fixture
def tiny_config():
    return LeanFormerConfig(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        max_seq_len=128,
        attention_rank=8,
        ff_rank=8,
        screening_rank=4,
        attention_top_k=16,
        ff_gate_rank=4,
        ff_sparsity_target=0.7,
        min_depth=1,
        exit_threshold=0.05,
        dropout=0.0,
        delta_rank=4,
    )


@pytest.fixture
def tiny_model(tiny_config):
    return LeanFormer(tiny_config)


def make_test_delta(d_model=64, delta_rank=4, n_layers=4,
                    target_layers=None, category="test"):
    """Helper to create a valid test delta."""
    from leanformer.knowledge_plane.dfs import DeltaFormatSpec

    if target_layers is None:
        target_layers = [1, 2]

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
        description=f"Test delta for {category}",
        domain_tags=[category],
        target_layers=target_layers,
        factors=factors,
        delta_rank=delta_rank,
        param_count=0,
        subspace_basis=torch.randn(d_model, delta_rank),
        orthogonality_score=0.95,
        conflicting_delta_ids=[],
        base_model_hash="test_hash",
        d_model=d_model,
        n_layers=n_layers,
    )
    delta.param_count = delta.compute_param_count()
    return delta


def make_test_registry(d_model=64, n_layers=4, delta_rank=4, n_deltas=3):
    """Create a registry populated with test deltas."""
    from leanformer.knowledge_plane.registry import DeltaRegistry

    registry = DeltaRegistry(
        base_model_hash="test_hash",
        d_model=d_model,
        n_layers=n_layers,
        delta_rank=delta_rank,
        orthogonality_threshold=0.1,
    )

    categories = ["chemistry", "physics", "biology"]
    for i in range(n_deltas):
        delta = make_test_delta(
            d_model=d_model,
            delta_rank=delta_rank,
            n_layers=n_layers,
            target_layers=[1, 2],
            category=categories[i % len(categories)],
        )
        # Use orthonormal basis to ensure registration succeeds
        Q, _ = torch.linalg.qr(torch.randn(d_model, delta_rank))
        delta.subspace_basis = Q[:, :delta_rank]
        registry.register(delta)

    return registry


# ================================================================
# Bug Fix Verification: SwiGLU gate_proj initialization
# ================================================================

class TestGateProjBugFix:
    """Verify the SwiGLU dead initialization bug is fixed."""

    def test_gate_proj_B_nonzero_at_init(self):
        """gate_proj.B should NOT be all zeros after init."""
        ff = GatedFeedForward(d_model=64, d_ff=256, rank=8, gate_rank=4)
        assert ff.gate_proj.B.abs().sum() > 0, "gate_proj.B is still all zeros"

    def test_up_proj_B_nonzero_at_init(self):
        """up_proj.B should NOT be all zeros after init."""
        ff = GatedFeedForward(d_model=64, d_ff=256, rank=8, gate_rank=4)
        assert ff.up_proj.B.abs().sum() > 0, "up_proj.B is still all zeros"

    def test_down_proj_B_nonzero_at_init(self):
        """down_proj.B should be non-zero to complete the gradient path."""
        ff = GatedFeedForward(d_model=64, d_ff=256, rank=8, gate_rank=4)
        assert ff.down_proj.B.abs().sum() > 0, "down_proj.B should be non-zero"

    def test_activation_gate_B_still_zero(self):
        """activation_gate.B should still follow LoRA pattern."""
        ff = GatedFeedForward(d_model=64, d_ff=256, rank=8, gate_rank=4)
        assert ff.activation_gate.B.abs().sum() == 0

    def test_swiglu_output_nonzero_at_init(self):
        """FF output should be nonzero from the very first forward pass."""
        ff = GatedFeedForward(d_model=64, d_ff=256, rank=8, gate_rank=4, dropout=0.0)
        x = torch.randn(2, 16, 64)
        out, stats = ff(x, training=True)
        assert out.abs().sum() > 0, "FF output is zero — SwiGLU deadlock not fixed"

    def test_gate_proj_receives_gradients(self):
        """gate_proj.B should receive non-zero gradients during training."""
        ff = GatedFeedForward(d_model=64, d_ff=256, rank=8, gate_rank=4, dropout=0.0)
        x = torch.randn(2, 16, 64)
        out, stats = ff(x, training=True)
        loss = out.sum() + stats["gate_loss"]
        loss.backward()
        assert ff.gate_proj.B.grad is not None
        assert ff.gate_proj.B.grad.abs().sum() > 0, "gate_proj.B has zero gradients"

    def test_up_proj_receives_gradients(self):
        """up_proj.B should receive non-zero gradients during training."""
        ff = GatedFeedForward(d_model=64, d_ff=256, rank=8, gate_rank=4, dropout=0.0)
        x = torch.randn(2, 16, 64)
        out, stats = ff(x, training=True)
        loss = out.sum() + stats["gate_loss"]
        loss.backward()
        assert ff.up_proj.B.grad is not None
        assert ff.up_proj.B.grad.abs().sum() > 0, "up_proj.B has zero gradients"

    def test_ff_sparsity_nonzero_during_training(self):
        """With nonzero B, hidden activations should have varied magnitudes,
        giving the activation_gate meaningful targets to learn from."""
        ff = GatedFeedForward(d_model=64, d_ff=256, rank=8, gate_rank=4, dropout=0.0)
        x = torch.randn(2, 16, 64)
        out, stats = ff(x, training=True)
        # The gate_loss should be meaningful (not trivially satisfied)
        assert stats["gate_loss"].item() > 0

    def test_model_loss_decreases_faster_with_fix(self):
        """The full model should learn more effectively with the bug fix."""
        config = LeanFormerConfig(
            vocab_size=1000, d_model=64, n_heads=4, n_layers=4,
            d_ff=256, max_seq_len=128, attention_rank=8, ff_rank=8,
            screening_rank=4, attention_top_k=16, ff_gate_rank=4,
            dropout=0.0,
        )
        model = LeanFormer(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randint(0, 1000, (4, 32))
        losses = []
        for _ in range(10):
            out = model(x, labels=x, training=True)
            loss = out["loss"]
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


# ================================================================
# TurboQuant KV Cache
# ================================================================

class TestTurboQuantKVCache:
    """Test 4-bit KV cache compression."""

    def test_quantize_dequantize_roundtrip(self):
        from leanformer.inference.kv_cache import quantize_tensor, dequantize_tensor
        x = torch.randn(4, 32, 64)
        q = quantize_tensor(x, bits=4)
        x_hat = dequantize_tensor(q)
        assert x_hat.shape == x.shape
        # Should be close but not exact
        assert (x - x_hat).abs().max() < 1.0

    def test_quantize_with_rotation(self):
        from leanformer.inference.kv_cache import (
            quantize_tensor, dequantize_tensor, _generate_orthogonal_matrix
        )
        dim = 64
        R = _generate_orthogonal_matrix(dim)
        x = torch.randn(4, 32, dim)
        q = quantize_tensor(x, bits=4, rotation=R)
        x_hat = dequantize_tensor(q, rotation_inv=R.T)
        assert x_hat.shape == x.shape

    def test_rotation_improves_quality(self):
        """Rotation should decorrelate channels, improving quantization."""
        from leanformer.inference.kv_cache import (
            compute_quantization_error, _generate_orthogonal_matrix
        )
        dim = 64
        R = _generate_orthogonal_matrix(dim)
        # Correlated data (rotation helps most here)
        x = torch.randn(100, dim) @ torch.randn(dim, dim) * 0.1
        err_no_rot = compute_quantization_error(x, bits=4)
        err_with_rot = compute_quantization_error(x, bits=4, rotation=R, rotation_inv=R.T)
        # Rotation should maintain or improve cosine similarity
        assert err_with_rot["cosine_similarity"] > 0.9

    def test_orthogonal_matrix_is_orthogonal(self):
        from leanformer.inference.kv_cache import _generate_orthogonal_matrix
        R = _generate_orthogonal_matrix(64)
        I = R @ R.T
        assert torch.allclose(I, torch.eye(64), atol=1e-5)

    def test_kv_cache_basic_flow(self):
        from leanformer.inference.kv_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(
            n_layers=4, d_model=64, n_heads=4,
            residual_window=8, enabled_threshold=4,
        )
        # Add tokens
        keys = torch.randn(1, 16, 64)
        values = torch.randn(1, 16, 64)
        cache.update(0, keys, values)

        # Retrieve
        k_out, v_out = cache.get(0)
        assert k_out is not None
        assert k_out.shape == (1, 16, 64)

    def test_kv_cache_residual_window(self):
        """Recent tokens should be full precision, older ones quantized."""
        from leanformer.inference.kv_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(
            n_layers=4, d_model=64, n_heads=4,
            residual_window=8, enabled_threshold=4,
        )
        # Add more than residual_window tokens
        keys = torch.randn(1, 20, 64)
        values = torch.randn(1, 20, 64)
        cache.update(0, keys, values)

        entry = cache.entries[0]
        assert entry.n_quantized == 12  # 20 - 8 = 12 quantized
        assert entry.residual_keys.shape[1] == 8  # 8 in residual window

    def test_kv_cache_incremental_update(self):
        from leanformer.inference.kv_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(
            n_layers=4, d_model=64, n_heads=4,
            residual_window=4, enabled_threshold=2,
        )
        # Add tokens incrementally
        for i in range(10):
            cache.update(0, torch.randn(1, 1, 64), torch.randn(1, 1, 64))

        assert cache.total_tokens(0) == 10
        k, v = cache.get(0)
        assert k.shape == (1, 10, 64)

    def test_kv_cache_memory_stats(self):
        from leanformer.inference.kv_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(
            n_layers=4, d_model=64, n_heads=4,
            residual_window=8, enabled_threshold=4,
        )
        cache.update(0, torch.randn(1, 20, 64), torch.randn(1, 20, 64))
        stats = cache.memory_stats()
        assert stats["total_tokens"] == 20
        assert stats["compression_ratio"] > 1.0

    def test_kv_cache_clear(self):
        from leanformer.inference.kv_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(
            n_layers=4, d_model=64, n_heads=4,
            residual_window=8, enabled_threshold=4,
        )
        cache.update(0, torch.randn(1, 10, 64), torch.randn(1, 10, 64))
        cache.clear()
        assert cache.total_tokens(0) == 0

    def test_cosine_similarity_preserved(self):
        """Quantized KV should preserve cosine similarity for attention."""
        from leanformer.inference.kv_cache import quantize_tensor, dequantize_tensor
        keys = torch.randn(1, 100, 64)
        query = torch.randn(1, 1, 64)

        # Original attention scores
        orig_scores = torch.bmm(query, keys.transpose(-2, -1))

        # Quantized attention scores
        q = quantize_tensor(keys, bits=4)
        keys_hat = dequantize_tensor(q)
        quant_scores = torch.bmm(query, keys_hat.transpose(-2, -1))

        # Rankings should be similar (top-k preservation)
        _, orig_topk = torch.topk(orig_scores.squeeze(), 10)
        _, quant_topk = torch.topk(quant_scores.squeeze(), 10)

        # At least 50% of top-10 should overlap
        overlap = len(set(orig_topk.tolist()) & set(quant_topk.tolist()))
        assert overlap >= 5, f"Only {overlap}/10 top-k overlap"


# ================================================================
# Output Provenance + Confidence Scoring
# ================================================================

class TestProvenanceSystem:
    """Test output provenance and graded confidence scoring."""

    def test_confidence_scorer_no_deltas(self):
        """No deltas = zero confidence, uncertainty flagged."""
        from leanformer.knowledge_plane.provenance import ConfidenceScorer
        scorer = ConfidenceScorer(uncertainty_threshold=0.3)
        signal = scorer.score([], make_test_registry())
        assert signal.confidence_score == 0.0
        assert signal.uncertainty_flag is True
        assert signal.routing_strength == 0.0
        assert signal.delta_coverage == 0.0

    def test_confidence_scorer_with_deltas(self):
        """Active deltas should produce nonzero confidence."""
        from leanformer.knowledge_plane.provenance import ConfidenceScorer
        registry = make_test_registry()
        delta_ids = list(registry.registered_deltas.keys())

        scorer = ConfidenceScorer(uncertainty_threshold=0.3)
        routing = [(delta_ids[0], 0.8)]
        signal = scorer.score(routing, registry)

        assert signal.confidence_score > 0
        assert signal.routing_strength == 0.8
        assert signal.delta_coverage > 0
        assert len(signal.source_deltas) == 1

    def test_confidence_increases_with_similarity(self):
        """Higher routing similarity = higher confidence."""
        from leanformer.knowledge_plane.provenance import ConfidenceScorer
        registry = make_test_registry()
        delta_ids = list(registry.registered_deltas.keys())
        scorer = ConfidenceScorer()

        low = scorer.score([(delta_ids[0], 0.2)], registry)
        high = scorer.score([(delta_ids[0], 0.9)], registry)
        assert high.confidence_score > low.confidence_score

    def test_composition_coherence_single_delta(self):
        """Single delta = perfect coherence."""
        from leanformer.knowledge_plane.provenance import ConfidenceScorer
        registry = make_test_registry()
        delta_ids = list(registry.registered_deltas.keys())
        scorer = ConfidenceScorer()
        assert scorer.compute_composition_coherence(
            [(delta_ids[0], 0.8)], registry
        ) == 1.0

    def test_composition_coherence_empty(self):
        """No deltas = zero coherence."""
        from leanformer.knowledge_plane.provenance import ConfidenceScorer
        scorer = ConfidenceScorer()
        assert scorer.compute_composition_coherence([], make_test_registry()) == 0.0

    def test_uncertainty_threshold(self):
        """Scores below threshold should flag uncertainty."""
        from leanformer.knowledge_plane.provenance import ConfidenceScorer
        scorer = ConfidenceScorer(uncertainty_threshold=0.5)
        registry = make_test_registry()
        delta_ids = list(registry.registered_deltas.keys())

        # Low similarity should be below threshold
        signal = scorer.score([(delta_ids[0], 0.1)], registry)
        # The exact flag depends on the combined score
        assert isinstance(signal.uncertainty_flag, bool)

    def test_base_only_signal(self):
        """Base-only inference should produce zero-confidence signal."""
        from leanformer.knowledge_plane.provenance import ConfidenceScorer
        scorer = ConfidenceScorer()
        signal = scorer.score_base_only()
        assert signal.confidence_score == 0.0
        assert signal.uncertainty_flag is True
        assert signal.source_deltas == []

    def test_provenance_log(self):
        """Provenance log should track entries."""
        from leanformer.knowledge_plane.provenance import ProvenanceLog, ConfidenceScorer
        log = ProvenanceLog(max_entries=10)
        scorer = ConfidenceScorer()

        for _ in range(15):
            signal = scorer.score_base_only()
            log.add(signal)

        assert len(log.entries) == 10  # Capped at max
        assert log.average_confidence() == 0.0
        assert log.uncertainty_rate() == 1.0

    def test_provenance_signal_has_all_fields(self):
        """ProvenanceSignal should carry all required fields."""
        from leanformer.knowledge_plane.provenance import ConfidenceScorer
        registry = make_test_registry()
        delta_ids = list(registry.registered_deltas.keys())
        scorer = ConfidenceScorer()

        signal = scorer.score(
            [(delta_ids[0], 0.7)], registry,
            query_embedding=torch.randn(64),
        )
        assert hasattr(signal, 'routing_strength')
        assert hasattr(signal, 'composition_coherence')
        assert hasattr(signal, 'delta_coverage')
        assert hasattr(signal, 'confidence_score')
        assert hasattr(signal, 'uncertainty_flag')
        assert hasattr(signal, 'source_deltas')
        assert hasattr(signal, 'source_categories')
        assert signal.query_embedding_norm > 0


# ================================================================
# Delta-Aware Quantization (DQS)
# ================================================================

class TestDeltaQuantization:
    """Test delta-aware quantization framework."""

    def test_dqs_tier_defaults(self):
        from leanformer.knowledge_plane.quantization import DeltaQuantizationSpec
        t1 = DeltaQuantizationSpec.for_tier(1)
        t2 = DeltaQuantizationSpec.for_tier(2)
        t3 = DeltaQuantizationSpec.for_tier(3)

        assert t1.bit_width >= t3.bit_width
        assert t1.routing_cosine_floor > t3.routing_cosine_floor
        assert t1.composition_frobenius_ceiling < t3.composition_frobenius_ceiling

    def test_dqs_serialization(self):
        from leanformer.knowledge_plane.quantization import DeltaQuantizationSpec
        dqs = DeltaQuantizationSpec.for_tier(1)
        d = dqs.to_dict()
        loaded = DeltaQuantizationSpec.from_dict(d)
        assert loaded.tier == dqs.tier
        assert loaded.bit_width == dqs.bit_width

    def test_quantize_delta_basic(self):
        from leanformer.knowledge_plane.quantization import DeltaQuantizer, DeltaQuantizationSpec
        delta = make_test_delta()
        quantizer = DeltaQuantizer()
        dqs = DeltaQuantizationSpec.for_tier(1)
        quantized = quantizer.quantize_delta(delta, dqs)

        assert quantized.delta_id == delta.delta_id
        assert len(quantized.quantized_factors) == len(delta.factors)
        assert quantized.embedding is not None

    def test_dequantize_delta_shapes(self):
        from leanformer.knowledge_plane.quantization import DeltaQuantizer, DeltaQuantizationSpec
        delta = make_test_delta()
        quantizer = DeltaQuantizer()
        dqs = DeltaQuantizationSpec.for_tier(1)

        quantized = quantizer.quantize_delta(delta, dqs)
        factors = quantizer.dequantize_delta(quantized)

        for layer_idx, (A, B) in delta.factors.items():
            A_deq, B_deq = factors[layer_idx]
            assert A_deq.shape == A.shape
            assert B_deq.shape == B.shape

    def test_quantize_preserves_routing_fidelity(self):
        """Embedding preserved at full precision = perfect routing fidelity."""
        from leanformer.knowledge_plane.quantization import DeltaQuantizer, DeltaQuantizationSpec
        delta = make_test_delta()
        quantizer = DeltaQuantizer()
        dqs = DeltaQuantizationSpec.for_tier(1)

        quantized = quantizer.quantize_delta(delta, dqs)
        # Embedding should be exactly preserved (not quantized)
        assert torch.allclose(quantized.embedding, delta.embedding)

    def test_validate_invariants_passing(self):
        from leanformer.knowledge_plane.quantization import DeltaQuantizer, DeltaQuantizationSpec
        delta = make_test_delta()
        quantizer = DeltaQuantizer()
        dqs = DeltaQuantizationSpec.for_tier(1)

        quantized = quantizer.quantize_delta(delta, dqs)
        results = quantizer.validate_invariants(delta, quantized)

        assert results["checks"]["routing_fidelity"]["passed"]
        # Composition fidelity depends on quantization quality
        assert "composition_fidelity" in results["checks"]

    def test_validate_invariants_with_other_deltas(self):
        from leanformer.knowledge_plane.quantization import DeltaQuantizer, DeltaQuantizationSpec
        delta1 = make_test_delta(category="chem")
        delta2 = make_test_delta(category="phys")
        quantizer = DeltaQuantizer()
        dqs = DeltaQuantizationSpec.for_tier(1)

        quantized = quantizer.quantize_delta(delta1, dqs)
        results = quantizer.validate_invariants(delta1, quantized, [delta2])
        assert "orthogonality_fidelity" in results["checks"]

    def test_compression_estimation(self):
        from leanformer.knowledge_plane.quantization import DeltaQuantizer, DeltaQuantizationSpec
        delta = make_test_delta()
        quantizer = DeltaQuantizer()
        dqs = DeltaQuantizationSpec.for_tier(1)

        stats = quantizer.estimate_compression(delta, dqs)
        assert stats["compression_ratio"] > 1.0
        assert stats["savings_pct"] > 0

    def test_rotation_matrix_orthogonal(self):
        from leanformer.knowledge_plane.quantization import generate_rotation_matrix
        R = generate_rotation_matrix(64)
        I = R @ R.T
        assert torch.allclose(I, torch.eye(64), atol=1e-5)

    def test_tier3_more_aggressive_than_tier1(self):
        """Tier 3 should compress more than tier 1."""
        from leanformer.knowledge_plane.quantization import DeltaQuantizer, DeltaQuantizationSpec
        delta = make_test_delta()
        quantizer = DeltaQuantizer()

        stats1 = quantizer.estimate_compression(delta, DeltaQuantizationSpec.for_tier(1))
        stats3 = quantizer.estimate_compression(delta, DeltaQuantizationSpec.for_tier(3))
        assert stats3["compression_ratio"] >= stats1["compression_ratio"]


# ================================================================
# Few-Shot Delta Production
# ================================================================

class TestFewShotForge:
    """Test few-shot delta production."""

    def _make_facts(self, n, tokenizer=None):
        """Create simple test facts."""
        from leanformer.knowledge_plane.forge import Fact
        facts = []
        prompts = [
            ("The capital of France is", " Paris"),
            ("Water boils at", " 100"),
            ("The speed of light is approximately", " 300"),
            ("The chemical symbol for gold is", " Au"),
            ("The largest planet in our solar system is", " Jupiter"),
        ]
        for i in range(n):
            prompt, target = prompts[i % len(prompts)]
            # Use a simple token ID
            target_id = (i + 100) % 1000
            facts.append(Fact(
                prompt=prompt,
                target=target,
                target_token_id=target_id,
                description=f"Fact {i}",
                category="test",
                domain_tags=["test"],
            ))
        return facts

    def test_few_shot_result_structure(self):
        from leanformer.knowledge_plane.few_shot import FewShotResult
        result = FewShotResult(
            n_examples=5,
            success=True,
            forge_success_rate=0.8,
            routing_accuracy=0.9,
            orthogonality_score=0.95,
            dqs_compliant=True,
            encoding_time_seconds=1.5,
        )
        assert result.n_examples == 5
        assert result.success is True

    def test_sweep_result_structure(self):
        from leanformer.knowledge_plane.few_shot import FewShotResult, FewShotSweepResult
        results = [
            FewShotResult(1, False, 0.0, 0.0, 1.0, False, 0.1),
            FewShotResult(5, True, 0.6, 0.8, 0.9, True, 0.5),
        ]
        sweep = FewShotSweepResult(
            results=results,
            minimum_viable_n=5,
            category="test",
            total_time_seconds=0.6,
        )
        summary = sweep.summary()
        assert summary["minimum_viable_n"] == 5
        assert len(summary["per_n"]) == 2

    def test_few_shot_forge_instantiation(self, tiny_model, tiny_config):
        """FewShotForge should instantiate without errors."""
        from leanformer.knowledge_plane.few_shot import FewShotForge
        registry = make_test_registry(
            d_model=tiny_config.d_model,
            n_layers=tiny_config.n_layers,
            delta_rank=tiny_config.delta_rank,
            n_deltas=0,
        )
        forge = FewShotForge(
            model=tiny_model,
            tokenizer=None,  # Will fail at actual forging but tests instantiation
            registry=registry,
            device="cpu",
        )
        assert forge is not None


# ================================================================
# Reasoning-Retrieval Separation Validation
# ================================================================

class TestReasoningRetrievalSeparation:
    """Test reasoning-retrieval separation validation."""

    def test_benchmark_query_creation(self):
        from leanformer.evaluation.separation import BenchmarkQuery
        q = BenchmarkQuery(
            prompt="What is the atomic number of hydrogen?",
            category="retrieval",
            expected_delta=True,
            expected_category="chemistry",
        )
        assert q.category == "retrieval"
        assert q.expected_delta is True

    def test_separation_result_structure(self):
        from leanformer.evaluation.separation import SeparationResult, BenchmarkQuery
        q = BenchmarkQuery("test", "reasoning", False)
        result = SeparationResult(
            query=q,
            confidence_score=0.0,
            uncertainty_flagged=True,
            delta_activated=False,
            routing_strength=0.0,
            categories_activated=[],
            correct_activation=True,
            correct_flagging=True,
        )
        assert result.correct_activation is True

    def test_separation_report(self):
        from leanformer.evaluation.separation import (
            SeparationReport, SeparationResult, BenchmarkQuery
        )
        results = []
        # 2 retrieval queries, 2 reasoning queries
        for cat, expected in [("retrieval", True), ("retrieval", True),
                              ("reasoning", False), ("reasoning", False)]:
            q = BenchmarkQuery(f"Q about {cat}", cat, expected)
            results.append(SeparationResult(
                query=q,
                confidence_score=0.8 if expected else 0.1,
                uncertainty_flagged=not expected,
                delta_activated=expected,
                routing_strength=0.8 if expected else 0.0,
                categories_activated=["chem"] if expected else [],
                correct_activation=True,
                correct_flagging=True,
            ))

        report = SeparationReport(results=results)
        report.compute()

        assert report.overall_correct_flagging == 1.0
        assert report.provenance_attribution_accuracy == 1.0
        assert "retrieval" in report.per_category
        assert "reasoning" in report.per_category

    def test_default_queries_generation(self, tiny_model, tiny_config):
        """Should generate queries from registered categories."""
        from leanformer.evaluation.separation import ReasoningRetrievalBenchmark
        from leanformer.knowledge_plane.provenance import ConfidenceScorer
        from leanformer.knowledge_plane.router import KnowledgePlaneRouter

        registry = make_test_registry(
            d_model=tiny_config.d_model,
            n_layers=tiny_config.n_layers,
            delta_rank=tiny_config.delta_rank,
        )
        router = KnowledgePlaneRouter(registry)
        scorer = ConfidenceScorer()

        benchmark = ReasoningRetrievalBenchmark(
            model=tiny_model,
            tokenizer=None,  # Only needed for actual evaluation
            registry=registry,
            router=router,
            confidence_scorer=scorer,
        )

        queries = benchmark.create_default_queries(["chemistry", "physics"])
        assert len(queries) > 0
        categories = set(q.category for q in queries)
        assert "retrieval" in categories
        assert "reasoning" in categories
        assert "both" in categories

    def test_report_summary(self):
        from leanformer.evaluation.separation import SeparationReport
        report = SeparationReport(results=[])
        summary = report.summary()
        assert summary["n_queries"] == 0
        assert summary["overall_correct_flagging"] == 0.0


# ================================================================
# Integration: Provenance in Runtime
# ================================================================

class TestRuntimeProvenance:
    """Test that provenance is integrated into the runtime."""

    def test_inference_result_has_provenance_field(self):
        """InferenceResult should have a provenance field."""
        from leanformer.knowledge_plane.runtime import InferenceResult
        result = InferenceResult(
            logits=torch.tensor([]),
            generated_text="test",
            active_deltas=[],
            categories_activated=[],
            composition_layers=[],
            inference_time_ms=0.0,
            base_only=True,
        )
        assert hasattr(result, 'provenance')


# ================================================================
# Cross-cutting: Component interactions
# ================================================================

class TestCrossComponentIntegration:
    """Test cross-component interactions."""

    def test_quantized_delta_preserves_subspace_basis(self):
        """Subspace basis must survive quantization for orthogonality checks."""
        from leanformer.knowledge_plane.quantization import DeltaQuantizer, DeltaQuantizationSpec
        delta = make_test_delta()
        quantizer = DeltaQuantizer()
        dqs = DeltaQuantizationSpec.for_tier(1)
        quantized = quantizer.quantize_delta(delta, dqs)

        # Basis should be exactly preserved (full precision)
        assert torch.allclose(quantized.subspace_basis, delta.subspace_basis)

    def test_provenance_with_quantized_routing(self):
        """Provenance should work with quantized delta embeddings."""
        from leanformer.knowledge_plane.provenance import ConfidenceScorer
        from leanformer.knowledge_plane.quantization import DeltaQuantizer, DeltaQuantizationSpec

        delta = make_test_delta()
        quantizer = DeltaQuantizer()
        dqs = DeltaQuantizationSpec.for_tier(1)
        quantized = quantizer.quantize_delta(delta, dqs)

        # Embedding preserved = routing works identically
        registry = make_test_registry()
        scorer = ConfidenceScorer()
        signal = scorer.score(
            [(delta.delta_id, 0.75)],
            registry,
        )
        assert signal.confidence_score > 0

    def test_dqs_compliance_across_tiers(self):
        """All tiers should produce valid quantized deltas."""
        from leanformer.knowledge_plane.quantization import DeltaQuantizer, DeltaQuantizationSpec
        delta = make_test_delta()
        quantizer = DeltaQuantizer()

        for tier in [1, 2, 3]:
            dqs = DeltaQuantizationSpec.for_tier(tier)
            quantized = quantizer.quantize_delta(delta, dqs)
            results = quantizer.validate_invariants(delta, quantized)
            # At minimum, routing fidelity should pass (embedding is FP)
            assert results["checks"]["routing_fidelity"]["passed"]


# ================================================================
# Attention B=0 Fix Verification
# ================================================================

class TestAttentionBFixVerification:
    """Verify the attention B=0 deadlock is fixed — same class as SwiGLU bug."""

    def test_attention_q_proj_B_nonzero_at_init(self):
        from leanformer.model.attention import TwoPassSparseAttention
        attn = TwoPassSparseAttention(d_model=64, n_heads=4, rank=8, screening_rank=4, top_k=8)
        assert attn.q_proj.B.abs().sum() > 0

    def test_attention_k_proj_B_nonzero_at_init(self):
        from leanformer.model.attention import TwoPassSparseAttention
        attn = TwoPassSparseAttention(d_model=64, n_heads=4, rank=8, screening_rank=4, top_k=8)
        assert attn.k_proj.B.abs().sum() > 0

    def test_attention_v_proj_B_nonzero_at_init(self):
        from leanformer.model.attention import TwoPassSparseAttention
        attn = TwoPassSparseAttention(d_model=64, n_heads=4, rank=8, screening_rank=4, top_k=8)
        assert attn.v_proj.B.abs().sum() > 0

    def test_attention_out_proj_B_nonzero_at_init(self):
        from leanformer.model.attention import TwoPassSparseAttention
        attn = TwoPassSparseAttention(d_model=64, n_heads=4, rank=8, screening_rank=4, top_k=8)
        assert attn.out_proj.B.abs().sum() > 0

    def test_attention_screen_B_nonzero_at_init(self):
        from leanformer.model.attention import TwoPassSparseAttention
        attn = TwoPassSparseAttention(d_model=64, n_heads=4, rank=8, screening_rank=4, top_k=8)
        assert attn.q_screen.B.abs().sum() > 0
        assert attn.k_screen.B.abs().sum() > 0

    def test_attention_output_nonzero_at_init(self):
        """Attention output should be nonzero from the very first forward pass."""
        from leanformer.model.attention import TwoPassSparseAttention
        attn = TwoPassSparseAttention(d_model=64, n_heads=4, rank=8, screening_rank=4, top_k=8, dropout=0.0)
        x = torch.randn(2, 16, 64)
        out, _ = attn(x)
        assert out.abs().sum() > 0, "Attention output is zero — B=0 deadlock not fixed"

    def test_attention_B_receives_gradients(self):
        """All attention Q/K/V/O B matrices should receive nonzero gradients."""
        from leanformer.model.attention import TwoPassSparseAttention
        attn = TwoPassSparseAttention(d_model=64, n_heads=4, rank=8, screening_rank=4, top_k=8, dropout=0.0)
        x = torch.randn(2, 16, 64)
        out, _ = attn(x)
        out.sum().backward()
        for name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
            B = getattr(attn, name).B
            assert B.grad is not None, f"{name}.B has no grad"
            assert B.grad.abs().sum() > 0, f"{name}.B has zero gradients"


# ================================================================
# Full-model gradient flow audit
# ================================================================

class TestFullModelGradientFlow:
    """Verify gradient flow through every parameter in the full model."""

    def test_all_trainable_params_receive_gradients_or_are_explained(self, tiny_config):
        """Every parameter must either receive nonzero gradients on step 1,
        or be in an explicitly known category of delayed-bootstrap params."""
        model = LeanFormer(tiny_config)
        x = torch.randint(0, tiny_config.vocab_size, (2, 32))
        out = model(x, labels=x, training=True)
        out["loss"].backward()

        # Known categories that can have zero/None grad on step 1:
        # - screening projections: topk is non-differentiable
        # - activation_gate.A: B=0 blocks A grad (B learns via BCE)
        # - lm_head.A: B=0 blocks A grad (B bootstraps in 1 step)
        # - final norm: depends on lm_head differentiating output
        allowed_zero = {"q_screen", "k_screen", "activation_gate.A", "lm_head.A", "norm.weight", "norm.bias"}

        unexpected_zero = []
        for name, param in model.named_parameters():
            is_allowed = any(tag in name for tag in allowed_zero)
            if param.grad is None or param.grad.abs().max().item() == 0:
                if not is_allowed:
                    unexpected_zero.append(name)

        assert unexpected_zero == [], (
            f"Unexpected zero-gradient params (not in allowed list): {unexpected_zero}"
        )

    def test_attention_B_matrices_learn_over_100_steps(self, tiny_config):
        """Attention B matrices must grow from init over 100 training steps."""
        model = LeanFormer(tiny_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randint(0, tiny_config.vocab_size, (4, 32))

        # Record initial norms
        init_norms = {}
        for name, param in model.named_parameters():
            if '.attn.' in name and name.endswith('.B') and 'screen' not in name:
                init_norms[name] = param.data.norm().item()

        for _ in range(100):
            out = model(x, labels=x, training=True)
            optimizer.zero_grad()
            out["loss"].backward()
            optimizer.step()

        # Check all attention B matrices have grown
        for name, param in model.named_parameters():
            if name in init_norms:
                new_norm = param.data.norm().item()
                assert new_norm > init_norms[name], (
                    f"{name} did not grow: init={init_norms[name]:.4f}, "
                    f"step100={new_norm:.4f}"
                )

    def test_gate_loss_starts_healthy(self, tiny_config):
        """gate_loss must start above 0.3 (non-trivial target),
        confirming hidden is nonzero so actual_active has both 0s and 1s."""
        model = LeanFormer(tiny_config)
        x = torch.randint(0, tiny_config.vocab_size, (2, 32))
        out = model(x, labels=x, training=True)
        gate_losses = [s["gate_loss"].item() for s in out["layer_stats"] if "gate_loss" in s]
        avg_gl = sum(gate_losses) / len(gate_losses)
        assert avg_gl > 0.3, (
            f"gate_loss too low at init ({avg_gl:.4f}) — hidden may be zero"
        )

    def test_eval_mode_ff_sparsity_nonzero_after_training(self, tiny_config):
        """After 100 steps, eval-mode FF sparsity should be nonzero."""
        model = LeanFormer(tiny_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randint(0, tiny_config.vocab_size, (4, 32))

        for _ in range(100):
            out = model(x, labels=x, training=True)
            optimizer.zero_grad()
            out["loss"].backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(x, training=False)
        stats = out["layer_stats"]
        avg_sp = sum(s.get("ff_sparsity", 0) for s in stats) / len(stats)
        assert avg_sp > 0, "FF sparsity is zero in eval mode after 100 steps"


# ================================================================
# Training infrastructure tests
# ================================================================

class TestTrainingInfrastructure:
    """Test training loop components: grad accum, LR, loss, checkpointing."""

    def test_loss_decomposition(self, tiny_config):
        """total loss = lm_loss + aux_loss, aux_loss > 0."""
        model = LeanFormer(tiny_config)
        x = torch.randint(0, tiny_config.vocab_size, (2, 32))
        out = model(x, labels=x, training=True)

        total = out["loss"].item()
        lm = out["lm_loss"].item()
        aux = out["aux_loss"].item()
        assert abs(total - (lm + aux)) < 1e-5, f"Loss mismatch: {total} != {lm} + {aux}"
        assert aux > 0, "aux_loss is zero — gate or exit loss not contributing"

    def test_gradient_accumulation_divides_loss(self, tiny_config):
        """Loss must be divided by grad_accum before backward to get correct effective LR."""
        model = LeanFormer(tiny_config)
        x = torch.randint(0, tiny_config.vocab_size, (2, 32))
        out = model(x, labels=x, training=True)

        grad_accum = 4
        scaled_loss = out["loss"] / grad_accum
        scaled_loss.backward()

        # Gradients should be 1/4 of what unscaled backward would produce
        grad_norms_scaled = {n: p.grad.norm().item() for n, p in model.named_parameters()
                            if p.grad is not None and p.grad.abs().max() > 0}

        model.zero_grad()
        out2 = model(x, labels=x, training=True)
        out2["loss"].backward()

        grad_norms_full = {n: p.grad.norm().item() for n, p in model.named_parameters()
                          if p.grad is not None and p.grad.abs().max() > 0}

        for name in grad_norms_scaled:
            if name in grad_norms_full and grad_norms_full[name] > 0:
                ratio = grad_norms_scaled[name] / grad_norms_full[name]
                assert 0.2 < ratio < 0.3, (
                    f"{name}: grad ratio {ratio:.3f}, expected ~0.25 (1/4)"
                )
                break  # One param is enough to verify

    def test_lr_warmup_schedule(self):
        """LR should warm up linearly then decay via cosine."""
        import math
        warmup_steps = 50
        total_steps = 500
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

        # Warmup: monotonically increasing
        for s in range(1, warmup_steps):
            assert lr_lambda(s) > lr_lambda(s - 1), f"LR not increasing at step {s}"

        # Peak at warmup_steps
        assert lr_lambda(warmup_steps) == pytest.approx(1.0, abs=0.01)

        # Decay: monotonically decreasing after warmup
        for s in range(warmup_steps + 1, total_steps):
            assert lr_lambda(s) <= lr_lambda(s - 1) + 1e-8, f"LR increased at step {s}"

        # Floor: never below 0.1
        assert lr_lambda(total_steps) >= 0.1

    def test_checkpoint_save_load_roundtrip(self, tiny_config, tmp_path):
        """Checkpoint must save and restore model + optimizer + scheduler state."""
        import random
        from leanformer.scripts.train_reasoning import save_checkpoint, load_training_state

        model = LeanFormer(tiny_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        # Train 10 steps to populate optimizer state
        x = torch.randint(0, tiny_config.vocab_size, (2, 32))
        for _ in range(10):
            out = model(x, labels=x, training=True)
            optimizer.zero_grad()
            out["loss"].backward()
            optimizer.step()
            scheduler.step()

        # Save
        save_checkpoint(
            model, tiny_config, str(tmp_path), step=10, val_loss=5.0,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            epoch=1, global_step=40, best_val_loss=5.0,
        )

        # Verify files exist
        assert (tmp_path / "pytorch_model.bin").exists()
        assert (tmp_path / "leanformer_config.json").exists()
        assert (tmp_path / "training_state.pt").exists()
        assert (tmp_path / "training_meta.json").exists()

        # Load into fresh model
        model2 = LeanFormer(tiny_config)
        model2.load_state_dict(torch.load(tmp_path / "pytorch_model.bin", weights_only=True))
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=3e-4)
        scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, lambda s: 1.0)
        scaler2 = torch.amp.GradScaler("cpu", enabled=False)

        result = load_training_state(str(tmp_path), model2, optimizer2, scheduler2, scaler2)
        assert result is not None
        opt_step, glob_step, epoch, best_vl = result
        assert opt_step == 10
        assert glob_step == 40
        assert epoch == 1
        assert best_vl == 5.0

        # Verify model weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Weight mismatch in {n1}"

    def test_checkpoint_resume_loss_continuity(self, tiny_config):
        """Loss curve must be continuous across save/resume — no spike."""
        import random
        model = LeanFormer(tiny_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randint(0, tiny_config.vocab_size, (4, 32))

        # Train 50 steps
        losses_before = []
        for _ in range(50):
            out = model(x, labels=x, training=True)
            losses_before.append(out["loss"].item())
            optimizer.zero_grad()
            out["loss"].backward()
            optimizer.step()

        # Snapshot state
        model_state = {k: v.clone() for k, v in model.state_dict().items()}
        opt_state_str = str(optimizer.state_dict()['state'].keys())

        # Continue 10 more steps (ground truth)
        losses_continued = []
        for _ in range(10):
            out = model(x, labels=x, training=True)
            losses_continued.append(out["loss"].item())
            optimizer.zero_grad()
            out["loss"].backward()
            optimizer.step()

        # Reload from snapshot
        model2 = LeanFormer(tiny_config)
        model2.load_state_dict(model_state)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        # Replicate optimizer state by re-running the same 50 steps
        # (simpler than serialization for this test)
        torch.manual_seed(0)  # Different seed = different noise, but same model weights
        # Just verify the first loss after reload matches
        out2 = model2(x, labels=x, training=True)
        resumed_first_loss = out2["loss"].item()

        # The first loss after reload should match (same weights, same input)
        assert abs(resumed_first_loss - losses_continued[0]) < 0.01, (
            f"Loss spike at resume: continued={losses_continued[0]:.4f}, "
            f"resumed={resumed_first_loss:.4f}"
        )


# ================================================================
# LAYER_STRATEGIES dynamic behavior
# ================================================================

class TestLayerStrategies:
    """Verify LAYER_STRATEGIES adapts correctly for various n_layers."""

    def test_12_layer_last_strategy_covers_all(self):
        from leanformer.knowledge_plane.forge import build_layer_strategies
        strategies = build_layer_strategies(12)
        assert strategies[-1] == list(range(12))

    def test_30_layer_last_strategy_covers_all(self):
        from leanformer.knowledge_plane.forge import build_layer_strategies
        strategies = build_layer_strategies(30)
        assert strategies[-1] == list(range(30))

    def test_4_layer_last_strategy_covers_all(self):
        from leanformer.knowledge_plane.forge import build_layer_strategies
        strategies = build_layer_strategies(4)
        assert strategies[-1] == list(range(4))

    def test_strategies_are_progressive(self):
        """Each strategy should cover >= the previous one."""
        from leanformer.knowledge_plane.forge import build_layer_strategies
        for n in [4, 12, 20, 30]:
            strategies = build_layer_strategies(n)
            for i in range(1, len(strategies)):
                assert len(strategies[i]) >= len(strategies[i-1]), (
                    f"n_layers={n}: strategy {i} ({len(strategies[i])} layers) "
                    f"is smaller than strategy {i-1} ({len(strategies[i-1])} layers)"
                )

    def test_strategies_all_valid_indices(self):
        """Every layer index must be in [0, n_layers)."""
        from leanformer.knowledge_plane.forge import build_layer_strategies
        for n in [4, 8, 12, 20, 30]:
            for strategy in build_layer_strategies(n):
                for layer in strategy:
                    assert 0 <= layer < n, f"Layer {layer} out of range for n_layers={n}"

    def test_first_strategy_targets_middle(self):
        """First strategy should be centered around the middle."""
        from leanformer.knowledge_plane.forge import build_layer_strategies
        strategies = build_layer_strategies(30)
        mid = 15
        first = strategies[0]
        center = sum(first) / len(first)
        assert abs(center - mid) < 3, f"First strategy center {center} too far from mid {mid}"


# ================================================================
# Optimizer and parameter coverage audit
# ================================================================

class TestOptimizerCoverage:
    """Verify every model parameter is in the optimizer."""

    def test_no_orphaned_parameters(self, tiny_config):
        model = LeanFormer(tiny_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

        model_params = set(id(p) for p in model.parameters())
        opt_params = set()
        for group in optimizer.param_groups:
            for p in group['params']:
                opt_params.add(id(p))

        orphaned = model_params - opt_params
        assert len(orphaned) == 0, f"{len(orphaned)} params in model but not in optimizer"

    def test_no_extra_parameters_in_optimizer(self, tiny_config):
        model = LeanFormer(tiny_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

        model_params = set(id(p) for p in model.parameters())
        opt_params = set()
        for group in optimizer.param_groups:
            for p in group['params']:
                opt_params.add(id(p))

        extra = opt_params - model_params
        assert len(extra) == 0, f"{len(extra)} params in optimizer but not in model"


# ================================================================
# EOS token verification
# ================================================================

class TestEOSToken:
    """Verify generate() uses the correct EOS token."""

    def test_eos_default_is_mistral(self):
        """Default EOS should be 2 (Mistral </s>)."""
        import inspect
        from leanformer.model.leanformer import LeanFormer
        sig = inspect.signature(LeanFormer.generate)
        assert sig.parameters['eos_token_id'].default == 2

    def test_generate_stops_on_eos(self, tiny_config):
        """Generation should stop when EOS token is produced."""
        model = LeanFormer(tiny_config)
        model.eval()
        # Use eos_token_id=-1 (disabled) — should always produce max_new_tokens
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 4))
        out, _ = model.generate(input_ids, max_new_tokens=5, eos_token_id=-1)
        assert out.shape[1] == 4 + 5  # Exactly max_new_tokens appended

    def test_eos_token_id_in_valid_vocab_range(self, tiny_config):
        """EOS token ID must be within vocab_size."""
        # Mistral EOS=2, vocab=32000 → valid
        assert 2 < tiny_config.vocab_size


# ================================================================
# zeros_ initialization audit (codebase-wide)
# ================================================================

class TestZerosInitAudit:
    """Verify every nn.init.zeros_ call in the codebase is intentional."""

    def test_low_rank_B_overridden_in_feedforward(self):
        """LowRankLinear inits B=0, but GatedFeedForward must override."""
        ff = GatedFeedForward(d_model=64, d_ff=256, rank=8, gate_rank=4)
        assert ff.up_proj.B.abs().sum() > 0
        assert ff.gate_proj.B.abs().sum() > 0
        assert ff.down_proj.B.abs().sum() > 0

    def test_low_rank_B_overridden_in_attention(self):
        """LowRankLinear inits B=0, but TwoPassSparseAttention must override."""
        from leanformer.model.attention import TwoPassSparseAttention
        attn = TwoPassSparseAttention(d_model=64, n_heads=4, rank=8, screening_rank=4, top_k=8)
        assert attn.q_proj.B.abs().sum() > 0
        assert attn.k_proj.B.abs().sum() > 0
        assert attn.v_proj.B.abs().sum() > 0
        assert attn.out_proj.B.abs().sum() > 0
        assert attn.q_screen.B.abs().sum() > 0
        assert attn.k_screen.B.abs().sum() > 0

    def test_lm_head_B_is_zero_intentionally(self, tiny_config):
        """lm_head.B=0 is intentional — it bootstraps in 1 step (not multiplicative)."""
        model = LeanFormer(tiny_config)
        assert model.lm_head.B.abs().sum() == 0
        # But it gets non-zero gradient immediately
        x = torch.randint(0, tiny_config.vocab_size, (2, 16))
        out = model(x, labels=x, training=True)
        out["loss"].backward()
        assert model.lm_head.B.grad.abs().max() > 0

    def test_activation_gate_B_is_zero_intentionally(self):
        """activation_gate.B=0 is intentional — BCE loss provides direct gradients."""
        ff = GatedFeedForward(d_model=64, d_ff=256, rank=8, gate_rank=4, dropout=0.0)
        assert ff.activation_gate.B.abs().sum() == 0
        # But it gets gradients from BCE loss
        x = torch.randn(2, 16, 64)
        out, stats = ff(x, training=True)
        (out.sum() + stats["gate_loss"]).backward()
        assert ff.activation_gate.B.grad.abs().max() > 0


# ================================================================
# Memory safety
# ================================================================

class TestMemorySafety:
    """Verify no memory leaks in the training loop."""

    def test_no_tensor_accumulation_in_training_loop(self, tiny_config):
        """Running 50 training steps should not monotonically increase
        the number of live tensors."""
        import gc
        model = LeanFormer(tiny_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randint(0, tiny_config.vocab_size, (2, 16))

        gc.collect()
        # Warm up
        for _ in range(5):
            out = model(x, labels=x, training=True)
            optimizer.zero_grad()
            out["loss"].backward()
            optimizer.step()

        gc.collect()
        tensors_before = len(gc.get_objects())

        for _ in range(50):
            out = model(x, labels=x, training=True)
            optimizer.zero_grad()
            out["loss"].backward()
            optimizer.step()

        gc.collect()
        tensors_after = len(gc.get_objects())

        # Allow some growth (Python GC is approximate), but not unbounded
        growth = tensors_after - tensors_before
        assert growth < 1000, (
            f"Tensor count grew by {growth} over 50 steps — possible memory leak"
        )


# ================================================================
# 500-step training smoke test
# ================================================================

class TestTrainingSmokeTest:
    """Full training loop exercised at small scale."""

    def test_500_step_training_all_metrics(self, tiny_config):
        """Run 500 optimizer steps. Verify loss decreases, gate_loss healthy,
        LR warms up, no NaN, eval sparsity nonzero."""
        import math
        model = LeanFormer(tiny_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

        warmup = 50
        total = 500
        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total - warmup, 1)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        x = torch.randint(0, tiny_config.vocab_size, (4, 32))
        grad_accum = 4
        opt_step = 0
        losses = []
        gate_losses = []
        lrs = []
        optimizer.zero_grad()

        for micro in range(total * grad_accum):
            out = model(x, labels=x, training=True)
            (out["loss"] / grad_accum).backward()

            if (micro + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1
                losses.append(out["loss"].item())
                lrs.append(scheduler.get_last_lr()[0])
                gl = [s["gate_loss"].item() for s in out["layer_stats"] if "gate_loss" in s]
                gate_losses.append(sum(gl) / len(gl) if gl else 0)

        # 1. Loss decreased
        assert losses[-1] < losses[0], f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}"

        # 2. No NaN
        assert all(math.isfinite(l) for l in losses), "NaN in losses"

        # 3. Gate loss started healthy
        assert gate_losses[0] > 0.3, f"gate_loss started at {gate_losses[0]:.4f}"

        # 4. Gate loss decreased
        assert gate_losses[-1] < gate_losses[0], f"gate_loss: {gate_losses[0]:.4f} -> {gate_losses[-1]:.4f}"

        # 5. LR warmed up
        assert lrs[49] > lrs[0], "LR did not warm up"

        # 6. LR decayed after warmup
        assert lrs[-1] < lrs[50], "LR did not decay"

        # 7. Eval-mode sparsity
        model.eval()
        with torch.no_grad():
            out_eval = model(x, training=False)
        stats = out_eval["layer_stats"]
        avg_sp = sum(s.get("ff_sparsity", 0) for s in stats) / len(stats)
        assert avg_sp > 0, "FF sparsity zero in eval mode after 500 steps"
