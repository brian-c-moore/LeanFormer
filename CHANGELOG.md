# Changelog

## 0.4.0 — Scale Validation + Formal Verification

- **204M-parameter scale validation (805M dense equivalent).** Full governed pipeline trained on reasoning corpus for 7,228 optimizer steps (one epoch) on NVIDIA L4. Zero budget violations across 722 hash-chained audit records. 18 valid governor state transitions with no skipped states. L3 activation at step 2,773 via genuine post-learning convergence. Best validation perplexity 57.6 at step 2,000. Orthogonal capacity confirmed at 53,760 dimensions (3,360 rank-16 delta slots) on two independent machines.
- **TLA+ formal verification.** 19 primitive specs + 5 LeanFormer compositions + 18 decomposition-failure specs verified by TLC across ~45.4 million states. Every primitive invariant held. Every decomposition produced a concrete counterexample, establishing operational irreducibility of the primitive set.
- **Phase-aware ConvergenceGovernor specification.** A B=0 initialization artifact in the 204M run caused premature hierarchy activation at lower levels. DAC was applied to its own failure and identified an observational degeneracy (cold-start and genuine convergence produce the same low-magnitude reading). The phase-aware fix tracks whether gradient magnitude has ever exceeded threshold; the `NoCoolingFromCold` invariant was verified in TLA+ across 18.6 million states and the original bug reproduces as a TLC counterexample in 2 states. Implementation in training code is pending.
- **Verification corrections.** Five initial specification assumptions were challenged by TLC and refined (ConvergenceGatedActivation reclassified from global invariant to precondition of `ActivateNextLevel`; FullDepthMeansUncertain boundary condition corrected; ForgingRequiresConvergence scope restricted to training phase; budget reallocation only frees the unused portion; WinnerOptimality invalidation on score update for TOCTOU).
- **DAC methodology paper in-repo.** `docs/dac/` carries a continuity copy of the DAC methodology paper and its three TLA+ / irreducibility appendices, with Figures 1-3 (training dynamics, gradient norm heatmap, governor state timeline) embedded at their reference points. The canonical home for the DAC methodology will live in its own repo.
- **204M training artifacts published.** `docs/dac/artifacts/204m_run/` contains the SHA-256 hash-chained audit log (`audit.jsonl`, 722 records), `training_log.json`, `validation_results.json`, `best_checkpoint_validation.json`, `leanformer_config.json`, `model_hash.txt`, parameter group registry, and the training config. Multi-GB binary checkpoints are not included; everything needed to independently verify zero budget violations, 18 valid governor transitions, and audit-chain integrity is.
- **Docs aligned to the current DAC paper.** README, ARCHITECTURE, and the LeanFormer proposal rewritten to frame each subsystem as a primitive composition and to report the 39M architecture results alongside the 204M governance-machinery results.

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
