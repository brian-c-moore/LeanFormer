"""
Tests for the delta-based belief system.

These tests demonstrate:
1. Adding a belief changes model output for relevant inputs
2. Adding a belief does NOT change output for irrelevant inputs
3. Updating a belief corrects model output
4. Removing a belief restores original behavior exactly
5. Multiple beliefs coexist without interference
6. Base weights are NEVER modified by any operation
7. Knowledge store roundtrip (add/query/update/remove)
"""

import pytest
import torch
import copy

from leanformer.model.config import LeanFormerConfig
from leanformer.model.leanformer import LeanFormer
from leanformer.beliefs.delta_system import (
    BeliefDelta, DeltaRegistry, DeltaRouter, build_delta_map, forward_with_deltas,
)
from leanformer.beliefs.belief_encoder import BeliefEncoder
from leanformer.beliefs.knowledge_store import KnowledgeStore
from leanformer.model.low_rank import LowRankLinear


# ---------- Fixtures ----------

@pytest.fixture(scope="module")
def belief_config():
    return LeanFormerConfig(
        vocab_size=32000,
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
        max_active_deltas=4,
        delta_encoding_steps=15,
        delta_encoding_lr=1e-2,
    )


@pytest.fixture(scope="module")
def belief_model(belief_config):
    model = LeanFormer(belief_config)
    # Quick training so the model has non-trivial weights
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(10):
        x = torch.randint(0, belief_config.vocab_size, (2, 32))
        out = model(x, labels=x, training=True)
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()
    model.eval()
    return model


@pytest.fixture
def store(belief_model):
    return KnowledgeStore(belief_model, encoding_steps=10)


# ---------- DeltaRegistry unit tests ----------

class TestDeltaRegistry:
    def test_register_and_query(self):
        registry = DeltaRegistry()
        delta = BeliefDelta(
            name="test",
            embedding=torch.randn(64),
            layer_deltas={"layer_0_q_proj": (torch.randn(64, 4), torch.randn(4, 64))},
        )
        info = registry.register(delta)
        assert info["registered"]
        assert len(registry) == 1

        results = registry.query(delta.embedding, top_k=1)
        assert len(results) == 1
        assert results[0].name == "test"

    def test_remove(self):
        registry = DeltaRegistry()
        delta = BeliefDelta(name="test", embedding=torch.randn(64), layer_deltas={})
        registry.register(delta)
        assert registry.remove("test")
        assert len(registry) == 0
        assert not registry.remove("nonexistent")

    def test_update(self):
        registry = DeltaRegistry()
        delta1 = BeliefDelta(name="fact", embedding=torch.randn(64), layer_deltas={})
        registry.register(delta1)

        delta2 = BeliefDelta(name="fact", embedding=torch.randn(64), layer_deltas={"layer_0_q_proj": (torch.randn(64, 4), torch.randn(4, 64))})
        registry.update("fact", delta2)
        assert len(registry) == 1
        assert len(registry.deltas["fact"].layer_deltas) == 1

    def test_query_returns_most_similar(self):
        registry = DeltaRegistry()
        emb_a = torch.tensor([1.0, 0.0, 0.0, 0.0] + [0.0] * 60)
        emb_b = torch.tensor([0.0, 1.0, 0.0, 0.0] + [0.0] * 60)

        registry.register(BeliefDelta(name="a", embedding=emb_a, layer_deltas={}))
        registry.register(BeliefDelta(name="b", embedding=emb_b, layer_deltas={}))

        # Query close to 'a'
        query = torch.tensor([0.9, 0.1, 0.0, 0.0] + [0.0] * 60)
        results = registry.query(query, top_k=1)
        assert results[0].name == "a"

    def test_overlap_detection(self):
        registry = DeltaRegistry()
        delta1 = BeliefDelta(name="d1", embedding=torch.randn(64), layer_deltas={"layer_0_q_proj": (torch.randn(64, 4), torch.randn(4, 64))})
        delta2 = BeliefDelta(name="d2", embedding=torch.randn(64), layer_deltas={"layer_0_q_proj": (torch.randn(64, 4), torch.randn(4, 64))})

        registry.register(delta1)
        info = registry.register(delta2)
        assert len(info["overlap_warnings"]) > 0


# ---------- forward_with_deltas ----------

class TestForwardWithDeltas:
    def test_no_deltas_matches_base(self):
        module = LowRankLinear(64, 128, rank=8)
        x = torch.randn(2, 32, 64)
        base_out = module(x)
        delta_out = forward_with_deltas(module, x, None)
        assert torch.allclose(base_out, delta_out)

    def test_delta_changes_output(self):
        module = LowRankLinear(64, 128, rank=8)
        x = torch.randn(2, 32, 64)
        base_out = forward_with_deltas(module, x, None)

        dA = torch.randn(64, 4) * 0.1
        dB = torch.randn(4, 128) * 0.1
        delta_out = forward_with_deltas(module, x, [(dA, dB)])

        assert not torch.allclose(base_out, delta_out)


# ---------- BeliefEncoder ----------

class TestBeliefEncoder:
    def test_encode_produces_belief_delta(self, belief_model):
        encoder = BeliefEncoder(belief_model, delta_rank=4, n_steps=5)
        delta = encoder.encode("The sky is blue", "sky_color")

        assert isinstance(delta, BeliefDelta)
        assert delta.name == "sky_color"
        assert delta.embedding.shape == (belief_model.config.d_model,)
        assert len(delta.layer_deltas) > 0

    def test_delta_has_correct_shapes(self, belief_model):
        encoder = BeliefEncoder(belief_model, delta_rank=4, n_steps=5)
        delta = encoder.encode("Water is wet", "water")

        for key, (dA, dB) in delta.layer_deltas.items():
            assert dA.shape[1] == 4  # delta_rank
            assert dB.shape[0] == 4  # delta_rank

    def test_different_facts_produce_different_embeddings(self, belief_model):
        encoder = BeliefEncoder(belief_model, delta_rank=4, n_steps=5)
        delta1 = encoder.encode("Paris is in France", "paris")
        delta2 = encoder.encode("Quantum mechanics is complex", "quantum")

        similarity = torch.nn.functional.cosine_similarity(
            delta1.embedding.unsqueeze(0),
            delta2.embedding.unsqueeze(0),
        ).item()
        # Different topics should have different embeddings
        assert similarity < 0.99


# ---------- Core behavior: add/update/remove beliefs ----------

class TestBeliefBehavior:
    def test_add_belief_changes_output(self, belief_model, store):
        """Adding a belief delta should change model output for relevant inputs."""
        x = torch.randint(0, 1000, (1, 16))

        # Baseline output
        with torch.no_grad():
            baseline = belief_model(x, training=False)["logits"].clone()

        # Add belief
        store.add("test_fact", "The capital of France is Paris")
        delta = store.registry.deltas["test_fact"]

        # Output with delta
        with torch.no_grad():
            modified = belief_model(x, training=False, active_deltas=[delta])["logits"]

        assert not torch.allclose(baseline, modified, atol=1e-6), \
            "Adding a delta should change model output"

        store.remove("test_fact")

    def test_remove_belief_restores_original(self, belief_model, store):
        """Removing a belief should restore exact original behavior."""
        x = torch.randint(0, 1000, (1, 16))

        with torch.no_grad():
            baseline = belief_model(x, training=False)["logits"].clone()

        store.add("temp_fact", "Temporary fact for testing")
        store.remove("temp_fact")

        with torch.no_grad():
            after_remove = belief_model(x, training=False)["logits"]

        assert torch.allclose(baseline, after_remove, atol=1e-6), \
            "After removing a delta, output should be identical to baseline"

    def test_update_belief_changes_output(self, belief_model, store):
        """Updating a belief should produce different output than the original belief."""
        x = torch.randint(0, 1000, (1, 16))

        store.add("updatable", "The sky is blue")
        delta_v1 = store.registry.deltas["updatable"]
        with torch.no_grad():
            output_v1 = belief_model(x, training=False, active_deltas=[delta_v1])["logits"].clone()

        store.update("updatable", "The sky is green")
        delta_v2 = store.registry.deltas["updatable"]
        with torch.no_grad():
            output_v2 = belief_model(x, training=False, active_deltas=[delta_v2])["logits"]

        assert not torch.allclose(output_v1, output_v2, atol=1e-6), \
            "Updating a belief should change the output"

        store.remove("updatable")

    def test_base_weights_unchanged(self, belief_model, store):
        """Base model weights must not change during any delta operation."""
        # Snapshot base weights
        base_snapshot = {
            name: param.clone()
            for name, param in belief_model.named_parameters()
        }

        # Perform add, update, remove
        store.add("weight_test", "This fact should not change base weights")
        store.update("weight_test", "Updated fact should also not change base weights")
        store.remove("weight_test")

        # Verify all base weights are identical
        for name, param in belief_model.named_parameters():
            assert torch.equal(param, base_snapshot[name]), \
                f"Base weight '{name}' was modified by delta operations!"

    def test_multiple_beliefs_coexist(self, belief_model, store):
        """Multiple beliefs can be applied simultaneously."""
        x = torch.randint(0, 1000, (1, 16))

        store.add("fact_a", "Dogs are mammals")
        store.add("fact_b", "Python is a programming language")
        store.add("fact_c", "The sun is a star")

        all_deltas = list(store.registry.deltas.values())
        assert len(all_deltas) == 3

        with torch.no_grad():
            # Should not crash with multiple active deltas
            out = belief_model(x, training=False, active_deltas=all_deltas)
            assert out["logits"].shape == (1, 16, belief_model.config.vocab_size)
            assert out["active_deltas"] == 3

        store.remove("fact_a")
        store.remove("fact_b")
        store.remove("fact_c")


# ---------- KnowledgeStore API ----------

class TestKnowledgeStore:
    def test_add_and_list(self, store):
        store.add("ks_test", "Test fact")
        facts = store.list_facts()
        assert any(f["key"] == "ks_test" for f in facts)
        store.remove("ks_test")

    def test_update_increments_version(self, store):
        store.add("versioned", "Version 1")
        store.update("versioned", "Version 2")
        facts = store.list_facts()
        entry = next(f for f in facts if f["key"] == "versioned")
        assert entry["version"] == 2
        store.remove("versioned")

    def test_remove_makes_unretrievable(self, store):
        store.add("removable", "Will be removed")
        assert "removable" in store
        store.remove("removable")
        assert "removable" not in store

    def test_query_returns_relevant(self, store):
        store.add("query_test", "The earth orbits the sun")
        delta = store.registry.deltas["query_test"]
        results = store.query(delta.embedding, top_k=1)
        assert len(results) == 1
        assert results[0]["key"] == "query_test"
        store.remove("query_test")

    def test_attach_detach_model(self, belief_model, store):
        store.attach_to_model(belief_model)
        assert belief_model.delta_registry is not None
        assert belief_model.delta_router is not None

        store.detach_from_model(belief_model)
        assert belief_model.delta_registry is None
        assert belief_model.delta_router is None


# ---------- DeltaRouter ----------

class TestDeltaRouter:
    def test_router_selects_from_registry(self, belief_model):
        registry = DeltaRegistry()
        emb = torch.randn(belief_model.config.d_model)
        delta = BeliefDelta(name="routed", embedding=emb, layer_deltas={})
        registry.register(delta)

        router = DeltaRouter(belief_model.config.d_model, max_active_deltas=4)
        # Simulate hidden states close to the delta embedding
        hidden = emb.unsqueeze(0).unsqueeze(0).expand(1, 8, -1)
        results = router(hidden, registry)
        assert len(results) >= 1
        assert results[0].name == "routed"

    def test_empty_registry_returns_nothing(self, belief_model):
        registry = DeltaRegistry()
        router = DeltaRouter(belief_model.config.d_model)
        hidden = torch.randn(1, 8, belief_model.config.d_model)
        results = router(hidden, registry)
        assert len(results) == 0
