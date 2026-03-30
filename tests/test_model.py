"""
Unit tests for LeanFormer model components.

These tests verify the architecture works correctly without requiring
a trained model or GPU — they run on CPU with small configs.
"""

import pytest
import torch

from leanformer.model.config import LeanFormerConfig
from leanformer.model.low_rank import LowRankLinear
from leanformer.model.attention import TwoPassSparseAttention
from leanformer.model.feedforward import GatedFeedForward
from leanformer.model.depth_controller import DepthController
from leanformer.model.leanformer import LeanFormer, LeanFormerLayer


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
        dropout=0.0,  # no dropout for deterministic tests
    )


@pytest.fixture
def tiny_model(tiny_config):
    return LeanFormer(tiny_config)


@pytest.fixture
def sample_input():
    """Batch of 2 sequences, length 32."""
    return torch.randint(0, 1000, (2, 32))


# ---------- LowRankLinear ----------

class TestLowRankLinear:
    def test_output_shape(self):
        layer = LowRankLinear(64, 128, rank=8)
        x = torch.randn(2, 32, 64)
        out = layer(x)
        assert out.shape == (2, 32, 128)

    def test_compression_ratio(self):
        layer = LowRankLinear(4096, 4096, rank=64)
        # (4096 + 4096) * 64 = 524,288 stored
        # 4096 * 4096 = 16,777,216 dense
        assert layer.compression_ratio() > 30  # should be ~32x

    def test_effective_params_less_than_dense(self):
        layer = LowRankLinear(512, 512, rank=16)
        assert layer.effective_parameters() < layer.dense_parameters()

    def test_weight_materialization(self):
        layer = LowRankLinear(64, 128, rank=8)
        W = layer.weight
        assert W.shape == (64, 128)

    def test_rank_clamped_to_min_dim(self):
        layer = LowRankLinear(32, 64, rank=100)
        assert layer.rank == 32  # clamped to min(32, 64)

    def test_no_bias(self):
        layer = LowRankLinear(64, 64, rank=8, bias=False)
        assert layer.bias is None
        x = torch.randn(2, 64)
        out = layer(x)
        assert out.shape == (2, 64)


# ---------- TwoPassSparseAttention ----------

class TestTwoPassSparseAttention:
    def test_output_shape(self):
        attn = TwoPassSparseAttention(
            d_model=64, n_heads=4, rank=8,
            screening_rank=4, top_k=8, dropout=0.0,
        )
        x = torch.randn(2, 32, 64)
        out, stats = attn(x)
        assert out.shape == (2, 32, 64)

    def test_returns_sparsity_stats(self):
        attn = TwoPassSparseAttention(
            d_model=64, n_heads=4, rank=8,
            screening_rank=4, top_k=8, dropout=0.0,
        )
        x = torch.randn(2, 32, 64)
        _, stats = attn(x)
        assert "attention_sparsity" in stats
        assert "attention_candidates" in stats
        # top_k=8 out of T=32 means 75% sparsity
        assert stats["attention_sparsity"] == pytest.approx(0.75)

    def test_with_causal_mask(self):
        attn = TwoPassSparseAttention(
            d_model=64, n_heads=4, rank=8,
            screening_rank=4, top_k=8, dropout=0.0,
        )
        T = 16
        x = torch.randn(2, T, 64)
        mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)
        out, _ = attn(x, mask=mask)
        assert out.shape == (2, T, 64)

    def test_top_k_larger_than_seq_len(self):
        """top_k > T should work gracefully."""
        attn = TwoPassSparseAttention(
            d_model=64, n_heads=4, rank=8,
            screening_rank=4, top_k=100, dropout=0.0,
        )
        x = torch.randn(2, 8, 64)
        out, stats = attn(x)
        assert out.shape == (2, 8, 64)
        assert stats["attention_sparsity"] == 0.0  # no sparsity when k >= T


# ---------- GatedFeedForward ----------

class TestGatedFeedForward:
    def test_output_shape_train(self):
        ff = GatedFeedForward(
            d_model=64, d_ff=256, rank=8, gate_rank=4, dropout=0.0,
        )
        x = torch.randn(2, 32, 64)
        out, stats = ff(x, training=True)
        assert out.shape == (2, 32, 64)

    def test_output_shape_inference(self):
        ff = GatedFeedForward(
            d_model=64, d_ff=256, rank=8, gate_rank=4, dropout=0.0,
        )
        x = torch.randn(2, 32, 64)
        out, stats = ff(x, training=False)
        assert out.shape == (2, 32, 64)

    def test_gate_loss_during_training(self):
        ff = GatedFeedForward(
            d_model=64, d_ff=256, rank=8, gate_rank=4, dropout=0.0,
        )
        x = torch.randn(2, 32, 64)
        _, stats = ff(x, training=True)
        assert "gate_loss" in stats
        assert stats["gate_loss"].item() >= 0

    def test_sparsity_during_inference(self):
        ff = GatedFeedForward(
            d_model=64, d_ff=256, rank=8, gate_rank=4,
            sparsity_target=0.8, dropout=0.0,
        )
        x = torch.randn(2, 32, 64)
        _, stats = ff(x, training=False)
        assert stats["ff_sparsity"] > 0  # some neurons should be masked


# ---------- DepthController ----------

class TestDepthController:
    def test_never_exits_during_training(self):
        dc = DepthController(d_model=64, min_depth=2, threshold=0.01)
        h = torch.randn(2, 32, 64)
        prev = torch.randn(2, 32, 64)
        should, conf = dc.should_exit(h, prev, layer_idx=3, training=True)
        assert should is False

    def test_respects_min_depth(self):
        dc = DepthController(d_model=64, min_depth=3, threshold=1.0)
        h = torch.randn(2, 32, 64)
        should, _ = dc.should_exit(h, h, layer_idx=1, training=False)
        assert should is False  # layer 1 < min_depth 3

    def test_exit_on_converged_state(self):
        dc = DepthController(d_model=64, min_depth=1, threshold=0.5)
        # When hidden == prev_hidden, residual is 0 (converged)
        h = torch.randn(2, 32, 64)
        # Force high exit probability by manipulating the exit head output
        # Just test that it doesn't crash
        should, conf = dc.should_exit(h, h, layer_idx=5, training=False)
        # conf should be some value between 0 and 1
        assert 0 <= conf <= 1


# ---------- LeanFormer (full model) ----------

class TestLeanFormer:
    def test_model_builds(self, tiny_model, tiny_config):
        """Model should instantiate without errors."""
        param_count = sum(p.numel() for p in tiny_model.parameters())
        assert param_count > 0
        assert param_count < tiny_config.dense_equivalent_estimate()

    def test_forward_pass(self, tiny_model, sample_input):
        """Forward pass should produce logits."""
        out = tiny_model(sample_input, training=False)
        assert "logits" in out
        assert out["logits"].shape == (2, 32, 1000)

    def test_forward_with_labels(self, tiny_model, sample_input):
        """Forward with labels should produce a loss."""
        out = tiny_model(sample_input, labels=sample_input, training=True)
        assert "loss" in out
        assert "lm_loss" in out
        assert "aux_loss" in out
        assert out["loss"].requires_grad

    def test_loss_is_finite(self, tiny_model, sample_input):
        out = tiny_model(sample_input, labels=sample_input, training=True)
        assert torch.isfinite(out["loss"])

    def test_loss_decreases_over_steps(self, tiny_config):
        """Loss should decrease with gradient updates — the model can learn."""
        model = LeanFormer(tiny_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randint(0, tiny_config.vocab_size, (4, 32))
        losses = []

        for _ in range(10):
            out = model(x, labels=x, training=True)
            loss = out["loss"]
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should generally decrease (allow some noise)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_reports_exit_layer(self, tiny_model, sample_input):
        out = tiny_model(sample_input, training=False)
        assert "exit_layer" in out
        assert 1 <= out["exit_layer"] <= tiny_model.config.n_layers

    def test_reports_depth_utilization(self, tiny_model, sample_input):
        out = tiny_model(sample_input, training=False)
        assert "depth_utilization" in out
        assert 0 < out["depth_utilization"] <= 1.0

    def test_reports_layer_stats(self, tiny_model, sample_input):
        out = tiny_model(sample_input, training=False)
        assert "layer_stats" in out
        assert len(out["layer_stats"]) > 0

    def test_efficiency_stats(self, tiny_model):
        stats = tiny_model.get_efficiency_stats()
        assert stats["total_parameters"] > 0
        assert stats["parameter_compression"] > 1.0  # should compress
        assert stats["num_low_rank_modules"] > 0

    def test_generate(self, tiny_model):
        """Generation should produce tokens without crashing."""
        input_ids = torch.randint(0, 1000, (1, 8))
        output_ids, step_stats = tiny_model.generate(
            input_ids, max_new_tokens=5, temperature=1.0, top_k=50,
        )
        assert output_ids.shape[1] > input_ids.shape[1]
        assert len(step_stats) > 0

    def test_parameter_compression_is_significant(self, tiny_model, tiny_config):
        """Model should be significantly smaller than a dense equivalent."""
        actual = sum(p.numel() for p in tiny_model.parameters())
        dense = tiny_config.dense_equivalent_estimate()
        ratio = dense / actual
        assert ratio > 1.5, f"Compression ratio too low: {ratio:.2f}x"


# ---------- Config ----------

class TestConfig:
    def test_save_load_roundtrip(self, tmp_path, tiny_config):
        path = tmp_path / "config.json"
        tiny_config.save(path)
        loaded = LeanFormerConfig.load(path)
        assert loaded.d_model == tiny_config.d_model
        assert loaded.n_layers == tiny_config.n_layers
        assert loaded.attention_rank == tiny_config.attention_rank

    def test_compression_ratio(self, tiny_config):
        assert tiny_config.compression_ratio() > 1.0
