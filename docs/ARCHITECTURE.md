# LeanFormer Architecture

LeanFormer treats catastrophic forgetting as a shared mutable state problem. The base model weights are immutable after training. Knowledge is stored as sparse, independently-addressable belief deltas. A registry governs subspace allocation to prevent interference. The result is a model that can learn new knowledge without retraining, forget on demand, and compose knowledge from multiple domains additively.

See `LeanFormer_Proposal.md` for the theoretical foundation.

---

## System Overview

```
Inference Server (FastAPI)
    |
Knowledge Plane
    |-- Delta Registry (orthogonality enforcement, capacity accounting)
    |-- Compositional Router (cosine similarity, multi-domain activation)
    |-- Consolidation Pipeline (SVD re-factorization, delta merging)
    |-- Provenance System (confidence scoring, uncertainty flagging)
    |-- Delta Quantizer (typed compression, DQS invariant enforcement)
    |-- Few-Shot Forge (sample efficiency measurement)
    |
    |  Delta Format Specification v2.0 (.delta files)
    |
Knowledge Forge (targeted-layer encoding, validation, progressive widening)
    |
Reasoning Core (LeanFormer, frozen after training)
    |-- TurboQuant KV Cache (4-bit compression at long contexts)
    |
Evaluation
    |-- Reasoning-Retrieval Separation Benchmark
```

---

## Efficient Transformer

Four structural innovations reduce compute and storage at every layer.

### Low-Rank Weight Factorization

**File:** `leanformer/model/low_rank.py`

Every weight matrix W (d_in x d_out) is stored as two factors A (d_in x rank) and B (rank x d_out). The forward pass computes `x @ A @ B` instead of `x @ W`. B matrices in attention and feed-forward projections are initialized with small random values (`std=0.01`) to ensure gradient flow from the first training step. Only `lm_head.B` and `activation_gate.B` retain zero initialization — they have non-multiplicative gradient paths that bootstrap within one step.

At rank=96 with d_model=768, each attention projection stores 147K params instead of 590K. Per-module compression averages 5-8x.

### Two-Pass Sparse Attention

**File:** `leanformer/model/attention.py`

1. **Screening pass**: cheap low-rank projections compute approximate attention scores and select top-K candidate keys per query.
2. **Exact pass**: full Q, K, V projections compute exact attention only over the selected candidates.

At top_k=96 with seq_len=512, only 19% of key-value pairs are computed.

### Gated Sparse Feed-Forward

**File:** `leanformer/model/feedforward.py`

A small gate predictor predicts which neurons will be active before computing the expensive projections. A top-k scatter mask enforces the sparsity target (default 80%) at inference time. During training, the gate learns via auxiliary loss against actual activation magnitudes.

The main SwiGLU computation: `hidden = up_proj(x) * SiLU(gate_proj(x))`.

### Adaptive Computation Depth

**File:** `leanformer/model/depth_controller.py`

Each layer's exit classifier predicts whether the hidden state has converged. Exit requires both small residual change and high classifier confidence. Never exits during training (exit heads train via auxiliary loss only). Minimum depth is configurable.

### Model Assembly

**File:** `leanformer/model/leanformer.py`

```
LeanFormer
  token_embedding (Embedding)
  position_embedding (Embedding)
  layers[0..N] (LeanFormerLayer)
    attn (TwoPassSparseAttention)
      q_proj, k_proj, v_proj, out_proj (LowRankLinear)
      q_screen, k_screen (LowRankLinear)
    ff (GatedFeedForward)
      up_proj, gate_proj, down_proj (LowRankLinear)
      activation_gate (LowRankLinear)
    norm1, norm2 (LayerNorm)
  depth_controller (DepthController)
    exit_head (Linear -> SiLU -> Linear)
  norm (LayerNorm)
  lm_head (LowRankLinear)
```

**Configuration:** `leanformer/model/config.py` (LeanFormerConfig dataclass, YAML/JSON serialization)

---

## Delta Belief System

Adds, updates, and removes knowledge from a frozen model via low-rank weight overlays.

### Belief Encoder

**File:** `leanformer/beliefs/belief_encoder.py`

Encodes facts into low-rank weight deltas via gradient-based learning. Freezes base weights, creates trainable (dA, dB) factors, optimizes to make the target token more probable. Both dA and dB are initialized with small random values because deltas need gradient flow from the start.

### Delta System

**File:** `leanformer/beliefs/delta_system.py`

- `BeliefDelta`: per-layer (dA, dB) factors + routing embedding + metadata
- `build_delta_map()`: organizes active deltas by layer and target projection
- `forward_with_deltas(module, x, deltas)`: `base_out + sum(x @ dA @ dB)`
- `DeltaRegistry`: stores deltas, cosine-similarity routing, overlap tracking
- `DeltaRouter`: selects relevant deltas per query

### Knowledge Store

**File:** `leanformer/beliefs/knowledge_store.py`

User-facing API: `add()`, `update()`, `remove()`, `query()`. Wraps the encoder and registry with version tracking.

### How Deltas Flow Through the Model

```
LeanFormer.forward(active_deltas=[...])
  -> build_delta_map() organizes deltas by layer/target
  -> each LeanFormerLayer gets its layer_deltas
  -> _attn_with_deltas() / _ff_with_deltas() call forward_with_deltas()
  -> forward_with_deltas(module, x, deltas): base_out + sum(x @ dA @ dB)
```

When `active_deltas=None`, the forward pass is identical to the base transformer.

---

## Knowledge Plane

Governs knowledge at scale through orthogonality-constrained deltas, registry-governed allocation, and additive composition.

### Delta Format Specification v2.0

**File:** `leanformer/knowledge_plane/dfs.py`

The contract between all Knowledge Plane components. Every delta conforms to this format:

| Field Group | Contents |
|-------------|----------|
| Identity | UUID, version, timestamp, provenance source |
| Routing | Embedding vector (d_model,), category, description, domain tags |
| Content | Target layers (subset), per-layer (A, B) factors, delta rank |
| Orthogonality | Subspace basis, orthogonality score, conflicting delta IDs |
| Compatibility | Base model SHA-256 hash, d_model, n_layers |
| Validation | Confidence score, validation results, forge attempts |

Serialized as ZIP archives (`.delta` files) containing `manifest.json` and `.pt` tensor files.

### Delta Registry + Orthogonality Engine

**File:** `leanformer/knowledge_plane/registry.py`

Single source of truth for subspace allocation.

**Orthogonality computation:** principal angles between subspaces via SVD of `Q_a.T @ Q_b`. The minimum principal angle determines overlap (0 = identical, pi/2 = fully orthogonal), normalized to a 0-1 score.

**Registration:** validates base model hash, model dimensions, uniqueness, and orthogonality threshold (default 0.3). Rejected deltas never enter the registry.

**Capacity accounting:** per-layer remaining slots = `(d_model - occupied_rank) / delta_rank`.

### Knowledge Forge

**File:** `leanformer/knowledge_plane/forge.py`

Converts knowledge artifacts into validated, orthogonality-checked deltas conforming to the DFS.

Pipeline:
1. Parse fact bank (JSON)
2. Gradient-based encoding: freeze base, optimize (A, B) per target layer
3. Validation: measure target token rank improvement before vs. after
4. Registry consultation: check orthogonality against existing deltas
5. Progressive layer widening if validation fails (dynamically computed based on model depth)

Targeted layers reduce per-delta cost by 58% vs the all-layer approach.

### Compositional Router

**File:** `leanformer/knowledge_plane/router.py`

Routes queries to relevant delta subsets via cosine similarity over delta embeddings. Top-k selection with minimum similarity threshold.

For each layer, additive composition: `composed = sum_i(weight_i * A_i @ B_i)`. This is safe because orthogonality was enforced at forge time.

### Knowledge Runtime

**File:** `leanformer/knowledge_plane/runtime.py`

Inference with Knowledge Plane integration:
1. Embed query using model's own embeddings
2. Route to relevant deltas
3. Compose active deltas additively
4. Apply via PyTorch forward hooks
5. Generate response with full provenance

### Output Provenance + Confidence Scoring

**File:** `leanformer/knowledge_plane/provenance.py`

Surfaces a graded, architecturally-grounded confidence score alongside each inference result. Three components:

- **Routing strength**: max cosine similarity between query and activating delta (0-1)
- **Composition coherence**: pairwise orthogonality of active deltas + routing score consistency (0-1)
- **Delta coverage**: whether knowledge deltas exist for the query domain (0-1)

Combined via weighted sum into a normalized confidence score. Score below threshold triggers explicit uncertainty flagging. This is not softmax probability — it measures epistemic grounding from the knowledge retrieval mechanism itself.

Integrated into `KnowledgeRuntime.infer()`: every inference call returns a `ProvenanceSignal`.

### Delta-Aware Quantization

**File:** `leanformer/knowledge_plane/quantization.py`

The DQS (Delta Quantization Specification) framework defines per-delta invariant tolerances and selects compression accordingly. Three tiers:

| Tier | Purpose | Bits | Cosine Floor | Frobenius Ceiling |
|------|---------|------|-------------|-------------------|
| 1 | Routing-critical | 4 | 0.95 | 0.05 |
| 2 | Composition | 4 | 0.85 | 0.10 |
| 3 | Archive | 3 | 0.70 | 0.20 |

Quantization pipeline: random orthogonal rotation (decorrelates channels) -> per-channel affine scalar quantization -> inverse rotation at dequantization. Embedding and subspace basis preserved at full precision. Non-compliant compressed deltas are rejected at load time.

### Few-Shot Delta Production

**File:** `leanformer/knowledge_plane/few_shot.py`

Measures minimum example count for well-formed delta production. `FewShotForge.sweep()` runs controlled experiments at varying example counts. Quality metrics: forge success rate, routing accuracy, orthogonality score, DQS compliance.

### KV Cache Compression

**File:** `leanformer/inference/kv_cache.py`

4-bit KV cache compression for inference-time memory reduction. Random orthogonal rotation decorrelates channels before per-channel scalar quantization. 128-token residual window at full FP16. Activated only when context exceeds a configurable threshold.

### Consolidation Pipeline

**File:** `leanformer/knowledge_plane/consolidation.py`

Merges groups of same-category deltas via SVD re-factorization:
1. Sum all `(A_i @ B_i)` contributions per layer
2. Truncated SVD to target rank
3. Register merged delta, unregister originals
4. Frees subspace capacity

### Inference Server

**File:** `leanformer/knowledge_plane/server.py`

FastAPI server:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/infer` | POST | Knowledge-augmented inference |
| `/infer/base` | POST | Base model only (comparison) |
| `/deltas` | GET | List registered deltas |
| `/deltas/{id}` | GET | Delta details |
| `/capacity` | GET | Registry capacity report |
| `/health` | GET | Health check + hash verification |
| `/consolidate` | POST | Trigger consolidation |
| `/provenance` | GET | Audit trail |

### Reasoning-Retrieval Separation Benchmark

**File:** `leanformer/evaluation/separation.py`

Validates that reasoning and retrieval are structurally separated. Benchmark with three query categories: (a) retrieval-only, (b) reasoning-only, (c) both. Measures confidence score, uncertainty flagging, provenance attribution accuracy per category.

---

## Relationship to DAC

LeanFormer was designed using Domain Abstraction Collapse (DAC), a methodology for identifying structural isomorphisms across domain boundaries and reducing domain-specific abstractions to a minimal set of domain-agnostic primitives (see `Domain_Abstraction_Collapse.md`).

| LeanFormer Component | DAC Primitive Composition |
|---------------------|---------------------------|
| Two-pass sparse attention | CompetitiveSelection (screening) + ActuationPass (exact computation on winners) |
| Gated feed-forward | CompetitiveSelection (gate predicts active neurons) + ActuationPass (compute only winners) |
| Adaptive depth | ConvergenceGovernor (residual change detection) + Budget (depth ceiling) |
| Delta routing | CompetitiveSelection (cosine similarity over embeddings) + AllocationCut (selected subset) |
| Delta registry | ResourceRegistry (subspace-to-delta mapping) + Budget (capacity per layer) |
| Belief encoding | PropagationPass (gradient-based delta optimization) |
| Consolidation | Reduction (SVD truncation of summed contributions) |
| Provenance logging | AuditSink (append-only record of routing decisions) |

---

## Project Structure

```
leanformer/
  model/                          Transformer architecture (frozen after training)
    config.py                       LeanFormerConfig
    low_rank.py                     LowRankLinear (A @ B factorization)
    attention.py                    TwoPassSparseAttention
    feedforward.py                  GatedFeedForward
    depth_controller.py             DepthController (early exit)
    leanformer.py                   LeanFormer (top-level model)

  beliefs/                        Delta belief system
    belief_encoder.py               Gradient-based fact encoding
    delta_system.py                 BeliefDelta, DeltaRegistry, DeltaRouter
    knowledge_store.py              User-facing add/update/remove/query
    hierarchical_embedding.py       2-level k-means embedding hierarchy

  knowledge_plane/                Knowledge Plane
    dfs.py                          DeltaFormatSpec v2.0
    registry.py                     DeltaRegistry + orthogonality engine
    forge.py                        KnowledgeForge (targeted-layer encoding)
    router.py                       KnowledgePlaneRouter (cosine routing)
    runtime.py                      KnowledgeRuntime (inference + provenance)
    consolidation.py                SVD re-factorization consolidation
    server.py                       FastAPI inference server
    provenance.py                   Confidence scoring + uncertainty flagging
    quantization.py                 DQS framework + delta quantization
    few_shot.py                     Few-shot delta production measurement

  inference/
    engine.py                       LeanFormerInference (model loading + generation)
    kv_cache.py                     TurboQuant 4-bit KV cache compression

  evaluation/
    efficiency.py                   Active params, FLOPs, compression metrics
    separation.py                   Reasoning-retrieval separation benchmark

  scripts/
    prepare_reasoning_data.py       Download + tokenize training corpus
    train_reasoning.py              Full training with checkpoint resume
    quick_train_validate.py         Quick pipeline validation
    forge_all_domains.py            Forge all domain fact banks
    demo.py                         Demo with rich output

  data/domains/                   Domain fact banks (JSON)

configs/                          Model configurations (YAML)
tests/                            199 tests across all components
```

---

## Implementation Notes

- `LowRankLinear` initializes B to zero. `GatedFeedForward` and `TwoPassSparseAttention` re-initialize their B matrices with `std=0.01` to ensure gradient flow through multiplicative and attention pathways. `BeliefEncoder` initializes both dA and dB with small random values.
- `forward()` takes `training: bool` and optional `active_deltas: list[BeliefDelta]`. When `active_deltas=None`, the forward pass is identical to the base transformer.
- Auxiliary losses (gate + exit) weighted at 0.01.
- Orthogonality computation uses float64 for numerical stability.
- Config: YAML via `from_yaml()`, JSON via `load()`/`save()`.
- `KnowledgeRuntime.infer()` returns a `ProvenanceSignal` with every inference result.
- DQS invariant validation runs at delta load time; non-compliant compressed deltas are rejected.
- KV cache quantization uses MSE-only (no QJL) per community validation that softmax amplifies QJL variance.
- Checkpoint save/restore includes optimizer, scheduler, scaler, and RNG states with atomic writes.
