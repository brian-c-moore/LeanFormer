# LeanFormer

An experimental transformer architecture designed using [Domain Abstraction Collapse](docs/Domain_Abstraction_Collapse.md) (DAC).

DAC's primitive analysis of the standard transformer revealed four categories of structural waste, each mapping to a known efficiency technique. Applied to catastrophic forgetting, DAC revealed it to be structurally identical to the write-conflict problem in shared mutable state — a problem solved decades ago through immutable bases, sparse overlays, and registry-governed allocation.

LeanFormer instantiates both results: an efficient transformer with immutable base weights and orthogonality-constrained belief deltas that can be added, composed, versioned, and removed without retraining.

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
  beliefs/            Delta belief encoder, registry, router, knowledge store
  knowledge_plane/    Forge, registry, router, runtime, consolidation, server,
                      provenance, quantization, few-shot measurement
  inference/          Inference engine, KV cache compression
  evaluation/         Efficiency metrics, reasoning-retrieval separation benchmark
  scripts/            Training, data preparation, forging, validation
  data/domains/       Fact banks (chemistry, CS, general knowledge)
configs/              Model configurations
tests/                199 tests
docs/                 Architecture reference, DAC methodology, research proposal
```

## Quick Start

```bash
pip install -e ".[dev]"

# Run tests (199 tests)
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

# 3. Forge domain knowledge into deltas
python -m leanformer.scripts.forge_all_domains --facts-per-domain 200 --max-steps 200

# 4. Start inference server
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
- [Research Proposal](docs/LeanFormer_Proposal.md) — applying Domain Abstraction Collapse to transformer design
- [Domain Abstraction Collapse](docs/Domain_Abstraction_Collapse.md) — the methodology


## Author

Brian Moore, M.S., CISSP, CCSP — Independent Systems Researcher

## Acknowledgement

Developed as a human-AI collaborative effort with Claude.ai and Claude Code.

## License

MIT
