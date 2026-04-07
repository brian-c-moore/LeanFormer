# Changelog

## 0.3.0 — Governed Training Pipeline

- **Per-group convergence governors.** Five-state machine (PENDING, ACTIVE, COOLING, CONVERGED, AWAKENED) tracks gradient EMA per parameter group. Converged groups have `requires_grad` disabled and parameters detached from autograd graph for real backward savings.
- **Coarse-to-fine hierarchy.** L0 structural parameters train from step 0. L1-L3 activate progressively as prior levels converge. Emergency activation at 80% of training if convergence signals haven't fired.
- **Federated budget allocation.** Learning-need scoring distributes compute across groups. State-aware multipliers (COOLING 0.5x, CONVERGED floor-only, AWAKENED 1.2x). Invariant `sum(allocations) <= master_budget` enforced every step.
- **Gradient router.** MLP-based sample-to-group scoring with straight-through estimator and entropy regularization.
- **Governed data pipeline.** Difficulty-tiered sampling (Mastered/Learning/Struggling/Failing), LSH deduplication, quality gating.
- **Change-triggered evaluation.** Metrics evaluated only when dependent parameter groups change. Budget-governed eval frequency.
- **Forge readiness gating.** Knowledge forge activates per-domain only when target groups are CONVERGED for a stability window.
- **SHA-256 hash-chained audit.** Tamper-evident provenance logging of every training step, convergence event, and checkpoint.
- **Deployment profiling.** Quality-tiered configurations (Full, Standard, Efficient, Minimal) with latency measurement and manifest generation.
- **lm-eval-harness integration.** Adapter implementing loglikelihood, loglikelihood_rolling, and generate_until for CORE benchmark evaluation (HellaSwag, ARC-Easy, COPA, PIQA, WinoGrande, OpenBookQA, BoolQ).
- **Production validation.** 500-step GPU validation on 66M config exercising all governance transitions.

## 0.2.0 — Knowledge Plane + Extensions

- Delta Format Specification v2.0 with orthogonality enforcement
- Knowledge Forge with progressive layer widening
- Compositional Router with cosine similarity routing
- Knowledge Runtime with hook-based inference
- Output provenance with graded confidence scoring
- Delta-aware quantization (DQS framework, 3 tiers)
- Few-shot delta production measurement
- 4-bit KV cache compression with orthogonal rotation
- Consolidation pipeline (SVD re-factorization)
- FastAPI inference server
- Reasoning-retrieval separation benchmark

## 0.1.0 — Efficient Transformer + Delta Belief System

- Low-rank weight factorization (LowRankLinear)
- Two-pass sparse attention (screening + exact)
- Gated sparse feed-forward
- Adaptive computation depth (exit classifiers)
- Delta belief encoder with gradient-based learning
- Delta registry with cosine similarity routing
- Knowledge store API (add, update, remove, query)
