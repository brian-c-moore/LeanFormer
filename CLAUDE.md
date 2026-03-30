# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeanFormer is an experimental transformer architecture designed using Domain Abstraction Collapse (DAC). It treats catastrophic forgetting as a shared mutable state problem: immutable base weights + orthogonality-constrained belief deltas + registry-governed allocation. The efficient transformer, delta belief system, and Knowledge Plane are all implemented and validated (119 tests passing).

See `docs/ARCHITECTURE.md` for the full technical reference and `docs/LeanFormer_Proposal.md` for the DAC experiment writeup.

## Commands

```bash
# Install
pip install -e ".[dev]"

# All tests (119 tests, ~73s)
python -m pytest tests/ -v --timeout=120

# Architecture + delta system unit tests (52 tests)
python -m pytest tests/test_model.py tests/test_beliefs.py -v

# Knowledge Plane unit tests (55 tests)
python -m pytest tests/test_dfs.py tests/test_registry.py tests/test_router.py \
  tests/test_consolidation.py tests/test_forge.py -v

# Integration tests (8 tests, requires checkpoint)
python -m pytest tests/test_integration.py -v --timeout=300

# Full pipeline
export HF_TOKEN=<token>
python -m leanformer.scripts.prepare_reasoning_data
python -m leanformer.scripts.train_reasoning
python -m leanformer.scripts.forge_all_domains --facts-per-domain 200 --max-steps 200

# Quick pipeline validation (no full training needed)
python -m leanformer.scripts.quick_train_validate --steps 500
python -m leanformer.scripts.forge_all_domains --facts-per-domain 20 --max-steps 50

# Demo (trains small model on WikiText-2, injects beliefs)
python -m leanformer.scripts.demo
```

## Key Module Relationships

```
LeanFormer (model/leanformer.py)         <- Frozen transformer architecture
  LeanFormerLayer
    TwoPassSparseAttention                  LowRankLinear Q,K,V,O + screening
    GatedFeedForward                        LowRankLinear up/gate/down + gate
  DepthController                           convergence-based early exit

KnowledgeStore (beliefs/)                      <- Delta belief system
  BeliefEncoder                                    fact -> BeliefDelta (gradient-based)
  DeltaRegistry, DeltaRouter                       store/route/govern deltas

KnowledgeForge (knowledge_plane/forge.py)      <- Knowledge Plane
  DeltaFormatSpec (knowledge_plane/dfs.py)         v2.0 contract (.delta files)
  DeltaRegistry (knowledge_plane/registry.py)      orthogonality enforcement
  KnowledgePlaneRouter (knowledge_plane/router.py) cosine similarity routing
  KnowledgeRuntime (knowledge_plane/runtime.py)    hook-based inference + provenance
  Consolidator (knowledge_plane/consolidation.py)  SVD re-factorization merging
  Server (knowledge_plane/server.py)               FastAPI inference server
```

## Important Implementation Details

- `LowRankLinear` initializes B to zero (LoRA insight). `BeliefEncoder` initializes both dA and dB with small random values (deltas need gradient flow from start).
- `forward()` takes `training: bool` and optional `active_deltas: list[BeliefDelta]`. When `active_deltas=None`, forward pass is identical to the base transformer.
- Knowledge Plane deltas target layers 4-8 by default (58% fewer params than modifying all layers). Progressive widening if validation fails.
- Orthogonality enforcement uses principal angles via SVD (float64 for stability). Threshold 0.3.
- Auxiliary losses (gate + exit) weighted at 0.01.
- Config: YAML via `from_yaml()`, JSON via `load()`/`save()`.

## DO NOT

- Do not modify model architecture files (`model/*.py`). They are frozen.
- Do not modify `tests/test_model.py` or `tests/test_beliefs.py`.
- All 119 existing tests must pass after any change.
