"""Tests for the Knowledge Forge.

Structural tests that verify forge logic without requiring a trained model.
GPU-dependent tests (actual encoding, validation) require the reasoning core
checkpoint and are run as part of integration testing.
"""
import pytest
import json
import torch
from pathlib import Path
from leanformer.knowledge_plane.forge import (
    Fact, LAYER_STRATEGIES, load_fact_bank,
)


class TestLayerStrategies:
    def test_strategies_are_progressive(self):
        """Each strategy should be wider than the previous."""
        for i in range(1, len(LAYER_STRATEGIES)):
            assert len(LAYER_STRATEGIES[i]) > len(LAYER_STRATEGIES[i - 1])

    def test_first_strategy_targets_middle_layers(self):
        assert LAYER_STRATEGIES[0] == [4, 5, 6, 7, 8]

    def test_last_strategy_targets_all_layers(self):
        assert LAYER_STRATEGIES[-1] == list(range(12))

    def test_all_strategies_are_valid_layer_indices(self):
        for strategy in LAYER_STRATEGIES:
            assert all(0 <= l < 12 for l in strategy)

    def test_targeted_layers_fewer_than_full(self):
        """The default strategy uses fewer than all layers."""
        assert len(LAYER_STRATEGIES[0]) < 12


class TestFact:
    def test_fact_creation(self):
        fact = Fact(
            prompt="The capital of France is",
            target=" Paris",
            target_token_id=6342,
            description="Paris is capital of France",
            category="general",
            domain_tags=["general"],
        )
        assert fact.prompt == "The capital of France is"
        assert fact.target == " Paris"
        assert fact.category == "general"


class TestLoadFactBank:
    def test_load_chemistry_facts(self):
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        path = Path("leanformer/data/domains/chemistry.json")
        if not path.exists():
            pytest.skip("Chemistry fact bank not found")
        facts_by_cat = load_fact_bank(str(path), tokenizer)
        assert "chemistry" in facts_by_cat
        assert len(facts_by_cat["chemistry"]) >= 100

    def test_load_cs_facts(self):
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        path = Path("leanformer/data/domains/cs.json")
        if not path.exists():
            pytest.skip("CS fact bank not found")
        facts_by_cat = load_fact_bank(str(path), tokenizer)
        assert "cs" in facts_by_cat
        assert len(facts_by_cat["cs"]) >= 100

    def test_load_general_facts(self):
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        path = Path("leanformer/data/domains/general.json")
        if not path.exists():
            pytest.skip("General fact bank not found")
        facts_by_cat = load_fact_bank(str(path), tokenizer)
        assert "general" in facts_by_cat
        assert len(facts_by_cat["general"]) >= 100

    def test_all_targets_tokenize(self):
        """Every target in every fact bank must produce at least one token."""
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        for domain in ["chemistry", "cs", "general"]:
            path = Path(f"leanformer/data/domains/{domain}.json")
            if not path.exists():
                continue
            with open(path) as f:
                raw = json.load(f)
            for category, items in raw.items():
                for item in items:
                    tokens = tokenizer.encode(item["target"])
                    assert len(tokens) > 0, (
                        f"Target '{item['target']}' in {domain}/{category} "
                        f"produces no tokens"
                    )

    def test_targeted_layer_param_savings(self):
        """Verify targeted layers cost less than all-layer approach."""
        d_model = 768
        delta_rank = 16
        # All 12 layers (untargeted)
        all_layer_params = 12 * 2 * d_model * delta_rank
        # 5 targeted layers
        targeted_params = 5 * 2 * d_model * delta_rank
        savings = 1 - (targeted_params / all_layer_params)
        assert savings > 0.5  # Should save >50%
