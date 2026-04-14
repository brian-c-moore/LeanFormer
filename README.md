# LeanFormer

A transformer architecture and governed training pipeline designed entirely by **Domain Abstraction Collapse (DAC)** — the methodology of stripping domain vocabulary from a problem, mapping the remaining structure to a small set of abstraction primitives, and inheriting the solution from whichever domain already solved it.

LeanFormer is the generative case study for DAC. Six open problems in neural-network design (parameter inefficiency, attention cost, catastrophic forgetting, knowledge composition, confabulation, and training-process inefficiency) were each decomposed into the sixteen-primitive set and rebuilt as compositions of solved systems-engineering patterns. The result is an efficient transformer with immutable base weights, orthogonality-constrained belief deltas that can be added, composed, versioned, and removed without retraining, and a governance layer that applies per-group convergence detection, coarse-to-fine hierarchy activation, federated budget allocation, gradient routing, and SHA-256 audit to the training loop itself.

See [`docs/dac/Domain_Abstraction_Collapse.md`](docs/dac/Domain_Abstraction_Collapse.md) for the full methodology paper.

## Validation

| Scale | What Was Measured | Status |
|-------|-------------------|--------|
| 39M params (76M dense equivalent) | Architecture: 88% attention sparsity, 80% FF sparsity, 3.9x compression, 84% belief injection success, 86% semantic routing (4.3x above chance), bit-for-bit base restoration across 100 beliefs, 406 base tensors verified immutable | Validated |
| 4.8M / 300 steps | Governed-training machinery: 15 governor transitions, 7/8 groups CONVERGED, L0->L1->L2->L3 hierarchy via convergence, 0 budget violations, 300 audit records, final loss within +2.4% of baseline | Validated |
| 204M params (805M dense equivalent) / 7,228 steps / NVIDIA L4 | All sixteen primitives composing under real training: 0 budget violations across 722 audit records, 18 valid governor transitions (no skipped states), L3 activation at step 2,773 via genuine post-learning convergence, SHA-256 chain intact, best val PPL 57.6 @ step 2,000, orthogonal capacity 53,760 dims (3,360 rank-16 delta slots) confirmed on two independent machines | Validated |
| TLA+ / TLC | 19 primitive specs + 5 LeanFormer compositions + 18 decomposition-failure specs, ~45.4M states explored; every invariant held; every decomposition produced a concrete counterexample (operational irreducibility); B=0 bug reproduced in 2 states; phase-aware fix verified across 18.6M states | Verified |

The 204M run is governance-machinery validation, not a language-modeling benchmark. All invariants held through the post-step-2,000 overfit phase, confirming the structure/function separation the methodology predicts: governance correctness is independent of generalization quality.

## Why DAC

The ML community treats catastrophic forgetting, attention cost, and training inefficiency as open research problems with their own literatures. DAC strips the ML vocabulary and recognizes that each has a structural twin in a solved systems domain:

| ML Problem | Stripped Description | Structural Twin | DAC Composition |
|------------|----------------------|-----------------|-----------------|
| Parameter inefficiency | Fixed-size blocks for variable content | File-system fragmentation | `Budget<Parameters>` + low-rank factorization |
| Attention cost | Brute-force all-to-all evaluation | Pre-visibility-buffer rendering | Two-pass `CompetitiveSelection` (screen + exact) |
| Catastrophic forgetting | Writes to shared mutable state clobber prior writes | Multi-tenant write conflict | Frozen base + `ResourceRegistry`-governed deltas |
| Knowledge composition | Multiple tenants corrupting a shared space | OS/DB address-space isolation | `FederatedBudget<ParameterSubspace>` + orthogonality |
| Confabulation | No signal distinguishing retrieval from interpolation | Signal-vs-noise discrimination | `ConvergenceGovernor` on hidden-state residual |
| Training inefficiency | Brute-force gradient to every parameter from every sample | Full-table scan / broadcast-to-all | `CompetitiveSelection` + `QualityHierarchy` + `FederatedBudget` + per-group `ConvergenceGovernor` + `AuditSink` |

During the 204M run a `B=0` initialization artifact caused premature hierarchy activation at the lower levels. DAC was applied to its own failure: the pattern is an **observational degeneracy** (two different trajectories producing the same low-magnitude reading) — the same pattern solved by depth buffers in rendering, heartbeat protocols in networking, and timeouts in distributed consensus. The fix is a phase-aware `ConvergenceGovernor` that tracks whether gradient magnitude has ever exceeded threshold. The fix was specified and verified in TLA+ across 18.6 million states before any code was written, and the original bug reproduces as a TLC counterexample in 2 states.

## Architecture at a Glance

### Efficient Transformer (`leanformer/model/`)

- **Low-rank weight factorization** — every weight matrix stored as `A @ B` from initialization; 5-8x per-module compression.
- **Two-pass sparse attention** — cheap screening pass selects top-K candidates; exact attention only on winners. 88% sparsity at 39M.
- **Gated sparse feed-forward** — gate predictor identifies active neurons; 80% skipped at inference.
- **Adaptive computation depth** — exit classifiers terminate when hidden states converge.

### Delta Belief System (`leanformer/beliefs/`) + Knowledge Plane (`leanformer/knowledge_plane/`)

Base weights are frozen. Facts are encoded as low-rank deltas (`output += x @ dA @ dB`) at targeted layers (4-8 by default, 58% fewer params than modifying all layers). Each delta is independently addressable — add, update, remove without touching other deltas or the base. Removal restores bit-for-bit identical output.

- **Delta Format Specification v2.0** — the contract between all Knowledge Plane components.
- **Delta Registry** — principal-angle orthogonality via SVD (threshold 0.3), subspace capacity accounting, rejects overlapping deltas.
- **Compositional Router** — cosine similarity routing, additive composition under orthogonality guarantee.
- **Consolidation** — SVD re-factorization merges stable same-category deltas.
- **Provenance** — confidence from routing strength, composition coherence, and delta coverage; uncertainty flagging when knowledge is absent.
- **DQS Quantization** — three tiers (routing-critical, composition, archive) with typed tolerances.
- **TurboQuant KV Cache** — 4-bit with orthogonal rotation, 128-token FP16 residual window, activated above 1024 context.
- **Inference Server** — FastAPI with provenance logging and base-weight hash verification.

### Governed Training Pipeline (`leanformer/training/`)

Every stage of the training loop is governed by the same primitives that govern the architecture:

- **Parameter group taxonomy** — every tensor assigned to a named group with hierarchy level L0-L3 (fnmatch patterns; config-independent).
- **Per-group convergence governors** — four-state machine (ACTIVE → COOLING → CONVERGED → AWAKENED). Converged groups have `requires_grad=False`. Budget multipliers per state: 1.0x / 0.5x / 0.05x / 1.2x.
- **Coarse-to-fine hierarchy** — L0 active at step 0; L1-L3 activate when prior level converges; emergency activation at 80% of steps.
- **Federated budget** — compute distributed proportional to learning need. Invariant `sum(allocations) <= master_budget` at every step (floor 2%, ceiling 40%).
- **Gradient router** — MLP scores sample relevance per group; top-k with straight-through estimator and entropy regularization; 5% observation-only warmup.
- **Governed data pipeline** — difficulty-tiered sampling (Mastered/Learning/Struggling/Failing), LSH dedup, periodic re-scoring.
- **Change-triggered evaluation** — metrics evaluated only when dependent groups change.
- **Forge readiness gate** — per-domain forging activates only when target groups have been CONVERGED for a stability window.
- **SHA-256 hash-chained audit** — tamper-evident provenance for every training step, convergence event, and checkpoint.

## Project Structure

```
leanformer/
  model/              Efficient transformer (frozen after training)
  training/           Governed training pipeline (convergence, hierarchy, budget,
                      routing, data pipeline, audit, evaluation, deployment)
  beliefs/            Delta belief encoder, registry, router, knowledge store
  knowledge_plane/    Forge, registry, router, runtime, consolidation, server,
                      provenance, quantization, few-shot measurement
  inference/          Inference engine, KV cache compression
  evaluation/         Efficiency metrics, reasoning-retrieval separation benchmark
  scripts/            Training, data preparation, forging, comparison, validation
  data/domains/       Fact banks (chemistry, CS, general knowledge)
configs/              Model and parameter group configurations
tests/                307 tests
docs/
  ARCHITECTURE.md     Complete technical reference
  LeanFormer_Proposal.md  DAC-applied-to-AI research proposal
  dac/                DAC methodology paper (continuity copy; canonical home is the DAC repo)
```

## Quick Start

```bash
pip install -e ".[dev]"

# All tests (307 tests, ~180s)
python -m pytest tests/ -v --timeout=120

# Demo (trains a small model on WikiText-2, injects beliefs)
python -m leanformer.scripts.demo
```

## Full Pipeline

```bash
# 1. Prepare training data (requires HuggingFace token)
export HF_TOKEN=<your_token>
python -m leanformer.scripts.prepare_reasoning_data

# 2. Train reasoning core
python -m leanformer.scripts.train_reasoning

# 3. Evaluate on CORE benchmarks
python -m leanformer.scripts.evaluate --checkpoint checkpoints/reasoning_core

# 4. Profile deployment tiers
python -m leanformer.scripts.profile_deployment --checkpoint checkpoints/reasoning_core

# 5. Forge domain knowledge into deltas
python -m leanformer.scripts.forge_all_domains --facts-per-domain 200 --max-steps 200

# 6. Start inference server
python -m leanformer.knowledge_plane.server \
  --model-checkpoint checkpoints/reasoning_core \
  --registry-path deltas/registry.json
```

## Requirements

- Python 3.11+
- PyTorch 2.3+ with CUDA
- NVIDIA GPU with 12GB+ VRAM for the small configs (validation has been run on RTX 3060 locally and NVIDIA L4 on GCP)

## Documentation

- [Architecture Reference](docs/ARCHITECTURE.md) — complete technical documentation
- [Research Proposal](docs/LeanFormer_Proposal.md) — DAC applied to AI model design
- [Domain Abstraction Collapse (methodology paper)](docs/dac/Domain_Abstraction_Collapse.md) — the underlying methodology
  - [Appendix B: Primitive TLA+ Specifications](docs/dac/Appendix_B_TLA_Specifications.md)
  - [Appendix C: LeanFormer TLA+ Compositions](docs/dac/Appendix_C_LeanFormer_TLA_Specifications.md)
  - [Appendix D: Irreducibility Proofs](docs/dac/Appendix_D_Irreducibility_Proofs.md)
- [204M Training Artifacts](docs/dac/artifacts/204m_run/) — SHA-256 hash-chained audit log (722 records), per-step training metrics, validation results, configs, and model hash from the 204M-parameter run reported in Section 5.8 of the DAC paper. Multi-GB binary checkpoints are not included; everything needed to verify the governance claims is.

## Known Limitations

- Single-epoch 204M training produced severe overfitting after step 2,000. Multi-epoch runs with stronger regularization are the first-priority next step.
- Adaptive depth remained at 20/20 layers throughout the 204M run; the exit classifiers need either a lower threshold or explicit layer-dropping training to learn graduated depth.
- Tiered sampling scored all samples at initialization when the model could not yet evaluate difficulty; periodic re-scoring is required to activate the governed data pipeline.
- The phase-aware `ConvergenceGovernor` is specified and TLA+-verified but not yet implemented in the training code. A retrain with the fix applied is the cleanest demonstration of convergence-gated coarse-to-fine training.
- Efficiency claims (50-70% gradient-compute reduction) require a 7B+ scale run with real backward-pass skipping to validate wall-clock gains.

See Section 9.5 of the DAC paper for the full future-work list with resource estimates.

## Author

Brian Moore, M.S., CISSP, CCSP — Independent Systems Researcher

## Acknowledgement

Developed as a human-as-architect / AI-as-implementation-agent collaboration with Claude.ai and Claude Code.

## License

MIT
