# LeanFormer

An experimental transformer architecture that treats catastrophic forgetting as a shared mutable state problem — a problem solved decades ago through immutable bases, sparse overlays, and registry-governed allocation.

LeanFormer is an efficient transformer with immutable base weights, orthogonality-constrained belief deltas that can be added, composed, versioned, and removed without retraining, and a governed training pipeline that applies per-group convergence detection, coarse-to-fine hierarchy activation, federated budget allocation, and gradient routing to reduce training compute.

## Results

| Capability | Result |
|------------|--------|
| Knowledge injection without retraining | 84% success rate across 100 beliefs |
| Bit-for-bit restoration after removal | Verified for 100 beliefs |
| Base weight immutability | 406 tensors verified unchanged through full lifecycle |
| Semantic routing | 86% correct category, 4.3x above chance |
| Multi-domain composition | Additive, order-independent, orthogonality-enforced |
| Attention sparsity | 88% (top-K screening) |
| Feed-forward sparsity | 80% (gated activation) |
| Parameter compression | 2.3x vs dense equivalent |

## How It Works

### Efficient Transformer

Four structural innovations reduce compute and storage at every layer:

- **Low-rank weight factorization.** Weights stored as A x B factors from initialization, 5-8x compression per module.
- **Two-pass sparse attention.** Cheap screening pass selects top-K candidates, exact attention only on those.
- **Gated sparse feed-forward.** Small predictor identifies active neurons, 80% skipped at inference.
- **Adaptive computation depth.** Exit classifiers terminate early when hidden states converge.

### Governed Training Pipeline

The training pipeline applies per-group governance to all stages of the training process:

- **Parameter group taxonomy.** Every tensor assigned to a named group with hierarchy level (L0-L3).
- **Per-group convergence governors.** Four-state machine (ACTIVE → COOLING → CONVERGED → AWAKENED) per group. Converged groups stop consuming gradient compute.
- **Coarse-to-fine hierarchy.** Only structural parameters (L0) active at step 0. Subsequent levels activate when prior levels converge.
- **Federated budget allocation.** Gradient compute distributed proportional to learning need. Invariant: `sum(allocations) <= master_budget` at every step.
- **Gradient routing.** Small MLP routes samples to relevant parameter groups. Non-selected groups have gradients zeroed post-backward.
- **Governed data pipeline.** Difficulty-tiered sampling, LSH deduplication, periodic re-scoring.
- **Change-triggered evaluation.** Metrics evaluated only when dependent parameter groups change.
- **Forge readiness gating.** Knowledge forge activates only when target groups have converged.
- **SHA-256 hash-chained audit.** Every training step logged with tamper-evident provenance chain.

### Delta Belief System

Base weights are **frozen** after training and never modified by knowledge operations. Facts are encoded as low-rank weight deltas (`output += x @ dA @ dB`) at targeted layers. Each delta is independently addressable — add, update, remove without touching other deltas or the base. Removal restores bit-for-bit identical output. Routing via cosine similarity selects relevant deltas per query.

### Knowledge Plane

- **Knowledge Forge**: targeted-layer encoding (layers 4-8 default, 58% fewer params), validation gate, orthogonality enforcement.
- **Delta Registry**: principal angle computation, subspace capacity accounting, rejects overlapping deltas.
- **Compositional Router**: multi-domain activation, additive composition (safe under orthogonality guarantee).
- **Consolidation**: SVD re-factorization merges stable deltas to free capacity.
- **Output Provenance**: graded confidence scoring from routing strength, composition coherence, and delta coverage. Uncertainty flagging when knowledge is absent.
- **Delta Quantization**: typed compression (DQS framework) with 3 tiers preserving routing, composition, and orthogonality fidelity.
- **KV Cache Compression**: 4-bit with orthogonal rotation, ~4x memory reduction at long contexts.
- **Inference Server**: FastAPI with provenance logging, base weight integrity verification.

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
docs/                 Architecture reference, research proposal
```

## Quick Start

```bash
pip install -e ".[dev]"

# Run tests (307 tests)
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

# 3. Evaluate on standard benchmarks (CORE tasks)
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
- NVIDIA GPU with 12GB+ VRAM (tested on RTX 3060)

## Documentation

- [Architecture Reference](docs/ARCHITECTURE.md) — complete technical documentation
- [Research Proposal](docs/LeanFormer_Proposal.md) — research proposal and methodology


## Author

Brian Moore, M.S., CISSP, CCSP — Independent Systems Researcher

## Acknowledgement

Developed as a human-AI collaborative effort with Claude.ai and Claude Code.

## License

MIT
