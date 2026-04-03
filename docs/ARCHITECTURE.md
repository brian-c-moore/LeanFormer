# LeanFormer Architecture

LeanFormer treats catastrophic forgetting as a shared mutable state problem. The base model weights are immutable after training. Knowledge is stored as sparse, independently-addressable belief deltas. A registry governs subspace allocation to prevent interference. The result is a model that can learn new knowledge without retraining, forget on demand, and compose knowledge from multiple domains additively. The training pipeline itself is governed by per-group convergence detection, coarse-to-fine hierarchy activation, federated budget allocation, and gradient routing.

See `LeanFormer_Proposal.md` for the research proposal.

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

## Governed Training Pipeline

Applies per-group convergence detection, coarse-to-fine hierarchy activation, federated budget allocation, and gradient routing to reduce training compute.

### Parameter Group Registry

**File:** `configs/parameter_groups.json`, `leanformer/training/param_groups.py`

Every parameter tensor is assigned to a named group with a hierarchy level (L0-L3). Groups map functional roles: embeddings, attention routing, gates, layer norms (L0 — structural), attention V/O, FF projections (L1 — representational), output head (L2 — refinement), exit classifier (L3 — specialization). The registry is config-independent — tensor name patterns match any model size.

### Per-Group Convergence Governors

**File:** `leanformer/training/convergence.py`

Each parameter group has its own convergence governor with a four-state machine:

```
ACTIVE → COOLING → CONVERGED → AWAKENED → ACTIVE
```

Tracks gradient EMA and loss contribution. Converged groups have `requires_grad` set to `False` (zero gradient compute). Groups reactivate if loss contribution spikes. Budget multipliers per state: ACTIVE 1.0x, COOLING 0.5x, CONVERGED 0.05x (maintenance), AWAKENED 1.2x (recovery).

### Coarse-to-Fine Hierarchy Activation

**File:** `leanformer/training/hierarchy.py`

Only L0 parameters active at step 0. Level N+1 activates when all groups in level N reach COOLING or CONVERGED. Per-level LR multipliers at activation (1.5x for L1/L2, 2.0x for L3, decaying to 1.0x). Emergency activation at 80% of total steps if convergence signals haven't fired.

### Federated Budget Allocation

**File:** `leanformer/training/budget.py`

Distributes gradient compute across groups proportional to learning need (gradient magnitude + loss contribution + convergence progress). The invariant `sum(allocations) <= master_budget` holds at every step. Converged groups release budget to active groups. Floor (2%) and ceiling (40%) prevent starvation/monopolization.

### Gradient Router

**File:** `leanformer/training/router.py`

Small MLP scores each sample against each parameter group. Top-k selection with straight-through estimator for gradient flow. Entropy regularization prevents routing collapse. Post-backward gradient masking zeros non-selected groups. Observation-only warmup for first 5% of steps.

### Governed Data Pipeline

**File:** `leanformer/training/data_pipeline.py`

Difficulty-tiered sampling (Mastered/Learning/Struggling/Failing), LSH deduplication, quality gating, periodic re-scoring. Tier fractions adjust dynamically as training progresses.

### Change-Triggered Evaluation

**File:** `leanformer/training/eval_pipeline.py`

Evaluates metrics only when the parameter groups they depend on change. Budget-governed eval frequency. Per-metric regression detection with alert system.

### Forge Readiness Gating

**File:** `leanformer/training/forge_gate.py`

Knowledge forge activates per-domain only when target parameter groups have been CONVERGED for a stability window. Suspends on group reactivation. Automated delta quality validation with accept/reject. Forge-training feedback loop boosts budget for groups with low delta acceptance.

### Unified Audit System

**File:** `leanformer/training/audit.py`

SHA-256 hash-chained append-only log covering all pipeline stages. Every training step, convergence event, and checkpoint is recorded with tamper-evident provenance. Query API by step range, parameter group, and event type. Checkpoint-audit binding for verifiable restoration.

### Deployment Profiling

**File:** `leanformer/training/deployment.py`

Profiles model at quality-tiered deployment configurations (Full, Standard, Efficient, Minimal) with latency measurements. Generates deployment manifest with per-tier quality scores, parameter counts, and active group lists.

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

  training/                       Governed training pipeline
    param_groups.py                 Parameter group registry + taxonomy
    convergence.py                  Per-group convergence governors (4-state machine)
    hierarchy.py                    Coarse-to-fine hierarchy activation
    budget.py                       Federated budget allocation
    router.py                       Gradient router (sample -> group routing)
    data_pipeline.py                Difficulty-tiered sampling, LSH dedup
    eval_pipeline.py                Change-triggered evaluation
    forge_gate.py                   Forge readiness gating
    audit.py                        SHA-256 hash-chained audit system
    deployment.py                   Deployment profiling + manifest generation
    trainer.py                      HuggingFace Trainer integration
    callbacks.py                    Training callbacks

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
    compare_training.py               Baseline vs governed training comparison
    forge_all_domains.py            Forge all domain fact banks
    demo.py                         Demo with rich output

  data/domains/                   Domain fact banks (JSON)

configs/                          Model + parameter group configurations
tests/                            307 tests across all components
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
- Parameter group registry (`configs/parameter_groups.json`) is config-independent — fnmatch patterns match any model size.
- Convergence governor four-state machine: ACTIVE (full budget) → COOLING (50%) → CONVERGED (5% maintenance, requires_grad=False) → AWAKENED (120% recovery).
- Hierarchy: L0 always active from step 0. L1-L3 activate on convergence signals. Emergency activation at 80% of training.
- Budget invariant `sum(allocations) <= master_budget` enforced at every reallocation with floor/ceiling per group.
- Gradient router uses straight-through estimator for end-to-end differentiability through discrete top-k selection.
- Audit chain: SHA-256 hash of each record includes previous record's hash. Verified on checkpoint restore.
