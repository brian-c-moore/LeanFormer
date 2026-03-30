# LeanFormer Project Status

**Last updated:** 2026-03-29

---

## Current State

All architecture and infrastructure code is complete. The full pipeline (data preparation, training, forging, routing, composition, consolidation, serving) is implemented and tested. **119 tests pass.** The remaining work is one long-running GPU training job (~24-36 hours), followed by full-scale forging and integration validation.

---

## Test Results

**119 tests, 0 failures** (72.84s runtime)

| Test Suite | Tests | What It Validates |
|-----------|-------|-------------------|
| test_model.py | 30 | Low-rank factorization, sparse attention, gated FF, adaptive depth, full model forward/backward, generation, compression |
| test_beliefs.py | 22 | Delta registry, forward-with-deltas, belief encoder, belief behavior (add/remove/update/coexist), knowledge store, delta router |
| test_dfs.py | 11 | Delta Format Spec creation, param counting, shape validation, serialization roundtrip, ZIP structure |
| test_registry.py | 14 | Principal angles, orthogonality scoring, registration, rejection of overlapping deltas, capacity accounting, save/load |
| test_router.py | 10 | Cosine routing, top-K selection, additive composition, order independence, weighted composition |
| test_consolidation.py | 9 | Candidate identification, SVD re-factorization, count reduction, capacity freeing |
| test_forge.py | 11 | Layer strategies, fact loading, token verification, param savings |
| test_integration.py | 8 | Single-domain knowledge, multi-domain composition, base weight integrity, provenance, latency overhead |

---

## Validated Results by Model Scale

### 7.5M Parameters (d_model=128, 4 layers)

Initial proof of concept trained on WikiText-2.

| Metric | Result |
|--------|--------|
| Per-module compression | 7.2x |
| Perplexity | 385.6 (1 epoch) |
| Belief injection | 3/3 successful |
| Rank improvement | Up to 4,238 positions ("Paris" from rank 4,272 to rank 34) |
| Base weight integrity | 142 tensors verified unchanged |

### 39M Parameters (d_model=512, 12 layers)

Scale validation trained on 500K OpenWebText samples.

| Metric | Result |
|--------|--------|
| Perplexity | 140.4 (1 epoch) |
| FF sparsity | 80% |
| Attention sparsity | 88% |
| Adaptive depth | 11.3/12 mean (2 unique depths) |
| 100 beliefs injected | 84% success rate |
| Semantic routing accuracy | 86% correct category (4.3x above chance) |
| Coexistence | 64% of 100 beliefs still improved when all active |
| Bit-for-bit restoration | Verified for all 100 beliefs (ordered and random removal) |
| Base weight integrity | 406 tensors verified unchanged |

### 66M Parameters (d_model=768, 12 layers) - Current

Knowledge Plane validated with quick-trained checkpoint (500 steps on WikiText-2).

| Metric | Result |
|--------|--------|
| Compressed params | 66,207,457 |
| Dense equivalent | 152,236,800 (2.3x compression) |
| VRAM (fwd+bwd, batch=2, fp16) | 6.29 GB on RTX 3060 |
| Training data prepared | 487,000 samples (~249M tokens) |
| Forge pipeline | 3 domains validated (chemistry, CS, general) |
| Targeted-layer delta cost | 0.19% of model (58% reduction vs all-layer) |
| Orthogonality enforcement | Correctly rejects overlapping deltas |
| Composition | Additive, order-independent, verified |
| Base weight integrity | SHA-256 verified after full routing + composition lifecycle |
| Inference latency overhead | < 2x with knowledge routing active |

---

## Training Data

| Source | Dataset | Samples | Purpose |
|--------|---------|---------|---------|
| Code (30%) | bigcode/starcoderdata (Python + JS) | 146,000 | Logic, composition, control flow |
| Math (15%) | open-web-math/open-web-math | 73,000 | Formal deduction, step-by-step reasoning |
| Science (15%) | ccdv/arxiv-summarization | 73,000 | Hypothesis-argument-conclusion structure |
| Prose (25%) | wikimedia/wikipedia | 122,000 | Grammar, coherence, narrative structure |
| Instruction (15%) | Open-Orca/SlimOrca | 73,000 | Query understanding, structured responses |
| **Total** | | **487,000** | **~249M tokens** |

The corpus is knowledge-naive by design: it emphasizes structural reasoning over factual memorization. The model should learn *how to process* information, not *what the world is like*. Factual knowledge is added post-training through the delta belief system.

---

## Domain Fact Banks

| Domain | Facts | Verified |
|--------|-------|----------|
| Chemistry | 194 | All targets tokenize correctly |
| Computer Science | 165 | All targets tokenize correctly |
| General Knowledge | 173 | All targets tokenize correctly |
| **Total** | **532** | |

---

## Forge Pipeline Validation

Tested with quick-trained checkpoint (500 steps, loss 10.84 -> 1.48):

| Domain | Batch 1 | Batch 2 | Deltas Registered |
|--------|---------|---------|-------------------|
| Chemistry | 100% improved, registered at layers [4-8] | 100% improved, rejected (orthogonality=0.244) | 1 |
| CS | 80% improved, registered at layers [4-8] | 40% improved, rejected (orthogonality=0.000) | 1 |
| General | 100% improved, registered at layers [4-8] | 40% improved, rejected (orthogonality=0.000) | 1 |

Second batches are rejected because the 500-step model produces nearly identical subspace bases across batches. The representations aren't diverse enough yet. This will resolve with full training, where layers develop distinct semantic representations that produce diverse delta subspaces.

---

## Remaining Work

### 1. Full Training (~24-36 hours GPU)

```bash
python -m leanformer.scripts.train_reasoning
```

The training script handles:
- 3 epochs with cosine LR schedule and warmup
- Mixed precision (fp16) with GradScaler
- Gradient accumulation (effective batch 64)
- Validation every 1000 steps
- Checkpointing every 5000 steps
- Exit head tuning (1000 steps, frozen base, lr=1e-3)
- Post-training validation (perplexity, generation samples, orthogonal capacity measurement)
- Model hash computation and storage

**Expected outcomes:**
- Perplexity < 50 on held-out reasoning corpus
- FF sparsity climbing toward 0.8
- Diverse per-layer representations enabling many orthogonal deltas

### 2. Full-Scale Forging (~1-2 hours GPU)

```bash
python -m leanformer.scripts.forge_all_domains --facts-per-domain 200 --max-steps 200
```

**Expected outcomes:**
- > 80% forge success rate per domain
- Most deltas at default layers [4-8]
- Registry populated with 50+ deltas across 3 domains
- Orthogonal capacity demonstrating room for hundreds more

### 3. Integration Validation

```bash
python -m pytest tests/test_integration.py -v --timeout=300
```

**Expected outcomes:**
- Single-domain: > 80% fact improvement per domain
- Multi-domain composition: > 70% per domain with two active
- Cross-domain routing: > 50% correct category
- Base weight integrity maintained through all operations
- Consolidation preserves > 90% accuracy while reducing delta count

### 4. Final Report

Complete final report with all measurements, comparisons, and analysis.

---

## Incomplete Modules

The following modules exist in the codebase but are not fully implemented. None are required to meet the current project objectives (training, forging, routing, composition, integration testing). They are scaffolding from early project structure that was superseded by the actual implementation.

### Stubs

| Module | What It Is | Status |
|--------|-----------|--------|
| `leanformer/evaluation/harness.py` | lm-eval-harness wrapper for standard benchmarks (HellaSwag, MMLU, etc.) | Stub. Raises NotImplementedError. Not needed for current objectives. Useful for future comparison against published models. |

### Cleaned Up

The following scaffolding modules were removed because they duplicated functionality that lives elsewhere in the codebase:

- `inference/budget.py` - compute budget governance, unused (inference runs without budget constraints)
- `inference/cache.py` - KV cache, unused (caching handled inline by generation loop)
- `evaluation/convergence.py` - depth distribution analysis, unused (handled by training script validation)
- `training/objectives.py` - standalone loss functions, unused (loss computed in model's forward())
- `training/schedule.py` - LR scheduling, unused (handled by training scripts via torch.optim)
