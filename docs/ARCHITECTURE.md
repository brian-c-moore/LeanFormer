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
    |
    |  Delta Format Specification v2.0 (.delta files)
    |
Knowledge Forge (targeted-layer encoding, validation, progressive widening)
    |
Reasoning Core (LeanFormer, frozen after training)
    66M compressed params, d_model=768, 12 layers
```

---

## Efficient Transformer

Four structural innovations reduce compute and storage at every layer.

### Low-Rank Weight Factorization

**File:** `leanformer/model/low_rank.py`

Every weight matrix W (d_in x d_out) is stored as two factors A (d_in x rank) and B (rank x d_out). The forward pass computes `x @ A @ B` instead of `x @ W`. B is initialized to zero (LoRA insight) so the model starts as identity.

At rank=96 with d_model=768, each attention projection stores 147K params instead of 590K. Per-module compression averages 5-8x.

### Two-Pass Sparse Attention

**File:** `leanformer/model/attention.py`

1. **Screening pass**: cheap low-rank projections (rank=24) compute approximate attention scores and select top-K candidate keys per query.
2. **Exact pass**: full-rank Q, K, V projections compute exact attention only over the selected candidates.

At top_k=96 with seq_len=512, only 19% of key-value pairs are computed.

### Gated Sparse Feed-Forward

**File:** `leanformer/model/feedforward.py`

A small gate predictor (rank=24) predicts which neurons will be active before computing the expensive up/down projections. A topk scatter mask enforces the sparsity target (default 80%) at inference time. During training, the gate learns via auxiliary loss against the actual activation magnitudes.

### Adaptive Computation Depth

**File:** `leanformer/model/depth_controller.py`

Each layer's exit classifier predicts whether the hidden state has converged. Exit requires both small residual change and high classifier confidence. Never exits during training (exit heads train via auxiliary loss only). Minimum depth is configurable (default 4).

### Model Assembly

**File:** `leanformer/model/leanformer.py`

```
LeanFormer
  token_embedding (Embedding)
  position_embedding (Embedding)
  layers[0..11] (LeanFormerLayer)
    attn (TwoPassSparseAttention)
      q_proj, k_proj, v_proj, out_proj (LowRankLinear)
      q_screen, k_screen (LowRankLinear)
    ff (GatedFeedForward)
      up_proj, gate_proj, down_proj (LowRankLinear)
      gate_predictor (Linear)
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

Encodes facts into low-rank weight deltas via gradient-based learning. Freezes base weights, creates trainable (dA, dB) factors, optimizes to make the target token more probable. Both dA and dB are initialized with small random values (not B-zero) because deltas need gradient flow from the start.

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

`compute_model_hash(model)` produces a SHA-256 of all parameters, locking base model identity.

### Delta Registry + Orthogonality Engine

**File:** `leanformer/knowledge_plane/registry.py`

Single source of truth for subspace allocation.

**Orthogonality computation:** principal angles between subspaces via SVD of `Q_a.T @ Q_b`. The minimum principal angle determines overlap (0 = identical, pi/2 = fully orthogonal), normalized to a 0-1 score.

**Registration:** validates base model hash, model dimensions, uniqueness, and orthogonality threshold (default 0.3). Rejected deltas never enter the registry.

**Capacity accounting:** per-layer remaining slots = `(d_model - occupied_rank) / delta_rank`. Theoretical maximum at d_model=768, rank=16: 48 orthogonal deltas per layer.

### Knowledge Forge

**File:** `leanformer/knowledge_plane/forge.py`

Converts knowledge artifacts into validated, orthogonality-checked deltas conforming to the DFS.

Pipeline:
1. Parse fact bank (JSON)
2. Gradient-based encoding: freeze base, optimize (A, B) per target layer
3. Validation: measure target token rank improvement before vs. after
4. Registry consultation: check orthogonality against existing deltas
5. Progressive layer widening if validation fails:
   - [4,5,6,7,8] (5 layers, default)
   - [3,4,5,6,7,8,9] (7 layers)
   - [2,3,4,5,6,7,8,9,10] (9 layers)
   - [0..11] (all 12, last resort)

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
4. Apply via PyTorch forward hooks (no model code modification)
5. Generate response with full provenance

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

---

## Relationship to DAC

LeanFormer was designed using Domain Abstraction Collapse (DAC), a methodology for identifying structural isomorphisms across domain boundaries and reducing domain-specific abstractions to a minimal set of domain-agnostic primitives (see `Domain_Abstraction_Collapse.md`). DAC revealed that catastrophic forgetting is structurally identical to the write-conflict problem in shared mutable state, and that the four efficiency innovations each correspond to a missing DAC primitive in the standard transformer.

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
    dfs.py                          DeltaFormatSpec v2.0 (contract)
    registry.py                     DeltaRegistry + orthogonality engine
    forge.py                        KnowledgeForge (targeted-layer encoding)
    router.py                       KnowledgePlaneRouter (cosine routing)
    runtime.py                      KnowledgeRuntime (hook-based inference)
    consolidation.py                SVD re-factorization consolidation
    server.py                       FastAPI inference server

  scripts/
    prepare_reasoning_data.py       Download + tokenize 5-source corpus
    train_reasoning.py              Full training (3 epochs + exit head tuning)
    quick_train_validate.py         Quick pipeline validation (500 steps)
    forge_all_domains.py            Forge all 3 domain fact banks
    forge_domain.py                 Single-domain forge CLI
    demo.py                         Demo with rich output

  data/
    domains/                        Domain fact banks (JSON)
      chemistry.json                  194 facts
      cs.json                         165 facts
      general.json                    173 facts

configs/
  reasoning_core.yaml               d_model=768 training config
  scale.yaml                        d_model=512 training config
  tiny.yaml                         d_model=128 PoC config

tests/                              119 tests across all components
deltas/                             Generated .delta files (gitignored)
data/                               Downloaded datasets (gitignored)
checkpoints/                        Model checkpoints (gitignored)
```

---

## Implementation Notes

- `LowRankLinear` initializes B to zero so the model starts as identity. `BeliefEncoder` initializes both dA and dB with small random values because deltas need gradient flow from the start.
- `forward()` takes `training: bool` (controls gate/exit behavior) and optional `active_deltas: list[BeliefDelta]`. When `active_deltas=None`, the forward pass is identical to the base transformer.
- Delta key format: `"layer_{i}_{target}"` where target is q_proj/k_proj/v_proj/out_proj/up_proj/gate_proj/down_proj.
- Auxiliary losses (gate + exit) weighted at 0.01.
- Orthogonality computation uses float64 for numerical stability.
- Config: YAML via `from_yaml()`, JSON via `load()`/`save()`.
