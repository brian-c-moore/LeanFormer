# LeanFormer

An experimental transformer architecture designed using [Domain Abstraction Collapse](docs/Domain_Abstraction_Collapse.md) (DAC) as a problem solving and development methodology for hard problems obscured by domain-specific vocabulary.

DAC's primitive analysis of the standard transformer revealed four categories of structural waste, each mapping to a known efficiency technique. Applied to catastrophic forgetting, DAC revealed it to be structurally identical to the write-conflict problem in shared mutable state, a problem solved decades ago through immutable bases, sparse overlays, and registry-governed allocation.

LeanFormer instantiates both results: an efficient transformer with immutable base weights and orthogonality-constrained belief deltas that can be added, composed, versioned, and removed without retraining.

**This is a research experiment, not a production system.** It validates DAC's generative methodology by applying it to real architectural problems in AI, and serves as a proof of concept for the resulting design.

## Results So Far

| What We Tested | Result |
|---------------|--------|
| Knowledge injection without retraining | 84% success rate across 100 beliefs |
| Bit-for-bit restoration after removal | Verified for 100 beliefs |
| Base weight immutability | 406 tensors verified unchanged through full lifecycle |
| Semantic routing | 86% correct category, 4.3x above chance |
| Multi-domain composition | Additive, order-independent, orthogonality-enforced |
| Attention sparsity | 88% (top-K screening) |
| Feed-forward sparsity | 80% (gated activation) |
| Parameter compression | 2.3x vs dense equivalent |

See [Project Status](docs/STATUS.md) for full details and what remains.

## How It Works

### Efficient Transformer (DAC-derived)

DAC's primitive analysis identified four places where standard transformers apply uniform computation to heterogeneous resources. Each maps to a known solution:

- **Low-rank weight factorization** (Budget over parameter space). Weights stored as A x B factors from initialization, 5-8x compression per module.
- **Two-pass sparse attention** (CompetitiveSelection with hierarchical screening). Cheap pass selects top-K candidates, exact attention only on those.
- **Gated sparse feed-forward** (CompetitiveSelection over neuron space). Small predictor identifies active neurons, 80% skipped.
- **Adaptive computation depth** (ConvergenceGovernor). Exit classifiers terminate early when hidden states converge.

### Delta Belief System (DAC-derived)

DAC revealed catastrophic forgetting as the write-conflict problem. The solution: immutable base + sparse overlays + registry governance.

- Base weights are **frozen** after training and never modified by knowledge operations.
- Facts are encoded as low-rank weight deltas: `output += x @ dA @ dB` at targeted layers.
- Each delta is **independently addressable**: add, update, remove without touching other deltas or the base.
- Removal restores **bit-for-bit identical** output because the base was never touched.
- Routing via cosine similarity selects relevant deltas per query.

### Knowledge Plane

Production-grade governance for knowledge at scale:

- **Knowledge Forge**: targeted-layer encoding (layers 4-8 default, 58% fewer params), validation gate, orthogonality enforcement.
- **Delta Registry**: principal angle computation, subspace capacity accounting, rejects overlapping deltas.
- **Compositional Router**: multi-domain activation, additive composition (safe under orthogonality guarantee).
- **Consolidation**: SVD re-factorization merges stable deltas to free capacity.
- **Inference Server**: FastAPI with provenance logging, base weight integrity verification.

## Project Structure

```
leanformer/
  model/              Efficient transformer (frozen after training)
  beliefs/            Delta belief encoder, registry, router, knowledge store
  knowledge_plane/    Knowledge Plane: forge, orthogonality registry, router,
                      runtime, consolidation, inference server
  scripts/        Training, data preparation, forging, validation
  data/domains/   Fact banks (chemistry, CS, general knowledge)
configs/          Model configurations
tests/            119 tests
docs/             Architecture, DAC paper, proposal, status
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run all tests (119 tests)
python -m pytest tests/ -v --timeout=120

# Demo (trains a small model on WikiText-2, injects beliefs)
python -m leanformer.scripts.demo
```

## Full Pipeline

```bash
# 1. Prepare training data (requires HuggingFace token for starcoderdata)
export HF_TOKEN=<your_token>
python -m leanformer.scripts.prepare_reasoning_data

# 2. Train reasoning core (~24-36h on RTX 3060)
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

- [Architecture Reference](docs/ARCHITECTURE.md) - complete technical documentation
- [DAC Experiment](docs/LeanFormer_Proposal.md) - applying Domain Abstraction Collapse to AI model design
- [Domain Abstraction Collapse](docs/Domain_Abstraction_Collapse.md) - the methodology that produced this architecture
- [Project Status](docs/STATUS.md) - current results and remaining work

## Author

Brian Moore, M.S., CISSP, CCSP - Independent Systems Researcher

## Acknowledgement

This project is an experimental proof of concept developed as a human-AI collaborative effort with Claude.ai and Claude Code.

## License

MIT
