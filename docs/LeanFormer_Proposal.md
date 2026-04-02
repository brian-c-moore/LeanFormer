# LeanFormer: Applying Domain Abstraction Collapse to AI Architecture

## Abstract

Domain Abstraction Collapse (DAC) claims that hard problems in one domain are often solved problems in another, obscured by domain-specific vocabulary. This paper tests that claim by applying DAC to the design of a transformer-based language model. The methodology (strip domain vocabulary, map structural patterns to the abstraction primitive set, search for solved isomorphisms) produces two results. First, primitive analysis of standard transformer architectures identifies structural waste in four areas, each mapping to a known efficiency technique: low-rank factorization (Budget over parameter space), two-pass sparse attention (CompetitiveSelection with hierarchical screening), gated activation sparsity (CompetitiveSelection over neuron space), and adaptive computation depth (ConvergenceGovernor). Second, DAC reveals that catastrophic forgetting is structurally identical to the write-conflict problem in shared mutable state, a problem solved decades ago through immutable bases, sparse overlays, and registry-governed allocation. The resulting architecture, LeanFormer, can acquire, compose, version, and discard knowledge without retraining, with bit-for-bit reversibility and verified base weight immutability. The architecture was designed and implemented to proof-of-concept in 24 hours.

---

## 1. Motivation: Testing DAC on AI Model Design

DAC identifies sixteen irreducible abstraction primitives sufficient to express computational patterns across twelve engineering domains. The methodology claims not only analytical power (explaining existing systems) but generative power (designing new ones by recognizing structural isomorphisms with solved problems).

To test the generative claim, we apply DAC to the design of a transformer-based language model, a domain where both efficiency and adaptability are active research problems. Two questions:

1. **The efficiency problem**: standard transformers apply uniform computation to heterogeneous inputs. Can DAC's primitive analysis identify the structural waste and map each category to a known solution from resource-constrained engineering?

2. **The forgetting problem**: when a neural network learns new information, it overwrites what it previously knew. The machine learning community has invested significant effort in mitigation (elastic weight consolidation, progressive neural networks, replay buffers) with partial success but no architectural solution. Can DAC reveal a structural isomorphism with a solved problem?

The DAC test: strip the AI vocabulary, map the problems to the primitive set, and check whether someone in another domain has already solved them.

---

## 2. Applying DAC: The Abstraction Collapse

### 2.1 Strip Domain Vocabulary

Remove "neural network," "weights," "gradient descent," "catastrophic forgetting," "fine-tuning." What remains?

A **shared mutable substrate**. Multiple **writers** modify this substrate by encoding information into its state. When a new writer modifies the substrate, it disturbs the changes made by previous writers. The more writers, the more interference. The substrate has no mechanism for isolating one writer's changes from another's.

### 2.2 Search the Primitive Set

This is the write-conflict problem in shared mutable state. It is one of the oldest and most precisely understood problems in computer science. Every operating system, every database, every concurrent system has solved it.

The standard solutions, expressed in DAC primitives:

| Systems Solution | DAC Primitive | Mechanism |
|-----------------|---------------|-----------|
| Immutable base + versioned overlays | ResourceRegistry + AllocationCut | Never modify shared state; apply changes as independent overlays |
| Sparse access patterns | Budget (address space) | Minimize overlap between writers through governed allocation |
| Copy-on-write | ActuationPass (on overlay, not base) | New writer creates new overlay, doesn't touch existing |
| Registry-governed allocation | ResourceRegistry + Budget | Central authority enforces non-overlapping address ranges |
| Convergence-governed depth | ConvergenceGovernor | Allocate computation proportional to input complexity |

### 2.3 The Structural Isomorphism

The mapping is exact:

| Catastrophic Forgetting Concept | Shared Mutable State Concept |
|-------------------------------|------------------------------|
| Weight matrix | Shared mutable memory |
| Training on new data | Writing to shared memory |
| Forgetting old knowledge | Write conflict (overwrite) |
| Regularization (EWC, SI) | Asking programmers to be careful |
| Replay buffers | Logging and replaying previous writes |
| Base model weights | Immutable shared base |
| Per-belief weight delta | Copy-on-write overlay |
| Delta routing | Registry-governed address lookup |
| Orthogonality enforcement | Non-overlapping address allocation |

The AI community has been treating forgetting as an optimization problem, trying to minimize interference through training technique. The DAC analysis reveals it as a data structure problem: the storage format conflates retrieval index with stored content and uses dense rather than sparse representation.

### 2.4 The Fundamental Tension

Dense weights produce generalization: similar concepts end up nearby in weight space because compression forces shared structure. Full localization eliminates generalization because independent storage prevents knowledge transfer.

The resolution, which DAC reveals by analogy to copy-on-write filesystems: **structured overlap**. A shared base encodes general reasoning capability (immutable after training). Sparse per-belief deltas encode specific content (independently addressable, updateable). The base provides generalization. The deltas provide isolation.

---

## 3. DAC Applied to Transformer Efficiency

Applying DAC's primitive analysis to the standard transformer forward pass reveals structural waste in five areas, all sharing a single root cause: uniform treatment of heterogeneous resources. Each maps to a DAC primitive that provides the solution.

### 3.1 Dense Weight Matrices

Trained weight matrices are empirically low-rank. Most variance is captured by 10-50 dimensions out of 4096. LoRA demonstrated that weight updates can be expressed as low-rank factors with near-identical quality. But LoRA is post-hoc. The correct approach: **initialize and train in factored form from the start**. Storage scales with effective information content, not layer dimension.

### 3.2 Fixed Computation Depth

Adjacent layers learn near-identical representations. 20-30% of layers can be skipped with minimal quality loss. Simple inputs converge early; complex inputs need more layers. This is a ConvergenceGovernor problem: detect when further computation contributes negligible refinement and exit early.

### 3.3 Dense Attention Computation

Standard attention computes O(n^2) scores, though most are near-zero. Attention patterns have exploitable structure: local connections are dense, long-range connections are sparse. This is CompetitiveSelection applied in two passes: **a cheap screening pass selects candidates, an exact pass computes weights only for candidates**.

### 3.4 Universal Neuron Activation

80-95% of feed-forward neurons produce near-zero activations per forward pass. A CompetitiveSelection gate (a small predictor that identifies active neurons before the expensive computation) eliminates this waste.

### 3.5 Knowledge Fused with Reasoning

Current models store reasoning capability and factual knowledge in the same weights. This conflates two fundamentally different functions. Reasoning is stable and expensive to train. Knowledge is volatile and cheap to update. The correct architecture separates them: a compact reasoning core (ResourceRegistry of capabilities) with knowledge stored as independently addressable overlays.

---

## 4. The LeanFormer Architecture

Instantiating the DAC-derived solution produces a four-layer architecture.

### 4.1 Efficient Transformer Core

Four innovations, each derived from applying a DAC primitive to an identified waste:

1. **Low-rank weight factorization**: weights as A x B factors from initialization, not post-hoc. Rank is a Budget constraint on per-layer capacity.

2. **Two-pass sparse attention**: CompetitiveSelection in two stages. Cheap screening (low-rank projections) selects top-K candidates, exact attention computed only for winners.

3. **Gated sparse feed-forward**: CompetitiveSelection gate predicts active neurons. ActuationPass computes only for winners. 80% of neurons skipped.

4. **Adaptive computation depth**: ConvergenceGovernor detects when hidden states converge. Budget constrains minimum depth. Simple inputs exit early.

### 4.2 Delta Belief System

The write-conflict solution instantiated as weight overlays:

- **Immutable base**: model weights frozen after training. Never modified by knowledge operations.
- **Belief encoder**: gradient-based optimization producing (dA, dB) low-rank factor pairs. Base frozen; only overlay factors are trainable.
- **Delta application**: `output += x @ dA @ dB` at targeted layers. Additive, independently removable.
- **Routing**: CompetitiveSelection over learned embeddings selects relevant deltas per query.
- **Reversibility**: removing a delta restores exact original output. Bit-for-bit identical. The base was never touched.

### 4.3 Knowledge Plane

Production-grade governance for knowledge at scale, mapping directly to DAC's governance primitives:

- **Delta Format Specification**: standardized contract for all deltas (identity, routing embedding, target layers, factors, orthogonality record, base model hash, validation results).
- **Delta Registry** (ResourceRegistry + Budget): tracks occupied subspaces, enforces orthogonality threshold via principal angle computation, per-layer capacity accounting, rejects deltas that would interfere with existing allocations.
- **Knowledge Forge**: targeted-layer encoding (Budget constrains which layers, progressive widening on validation failure), validation gate (delta must improve target token rank to pass).
- **Compositional Router** (CompetitiveSelection + ActuationPass): cosine similarity routing selects relevant deltas, additive composition applies them. Safe because orthogonality was enforced at registration.
- **Consolidation** (Reduction): SVD re-factorization merges stable same-category deltas, freeing subspace capacity.
- **Provenance** (AuditSink): every inference records which deltas were active, routing scores, composition layers, and timing.

### 4.4 Computational Flow

**Inference:**
```
Query -> Embed -> Route (CompetitiveSelection over delta embeddings)
     -> Compose (additive, order-independent)
     -> Apply via hooks (ActuationPass on allocated deltas only)
     -> Forward pass (ConvergenceGovernor controls depth)
     -> Response + provenance (AuditSink)
```

**Knowledge acquisition:**
```
Fact -> Encode (PropagationPass: gradient-based, targeted layers)
     -> Validate (CompetitiveSelection: rank improvement required)
     -> Check orthogonality (Budget: subspace capacity check)
     -> Register (ResourceRegistry: allocate address range)
```

**Knowledge removal:**
```
Delta ID -> Unregister (ResourceRegistry: free address range)
         -> Rebuild occupied bases -> Capacity freed
         -> Base weights untouched, all other deltas untouched
```

---

## 5. Evaluation of DAC's Generative Claim

### 5.1 Did DAC Work?

The test had two parts:

**Efficiency**: DAC's primitive analysis of the transformer forward pass identified four categories of structural waste, each mapping to a known solution from resource-constrained systems engineering. All four were implemented and validated: low-rank factorization (2.3x compression), sparse attention (88% sparsity), gated feed-forward (80% neuron skip), and adaptive depth (convergence-based early exit). These are not novel inventions; each technique exists independently in the literature. DAC's contribution was identifying them systematically through primitive analysis rather than through domain-specific intuition.

**Forgetting**: The vocabulary-stripping step revealed that catastrophic forgetting is the write-conflict problem. The primitive mapping step identified immutable base + sparse overlays + registry governance as the known solution. The instantiation step produced a working architecture: 84% belief injection success rate, 86% routing accuracy, bit-for-bit restoration verified for 100 beliefs, base weight immutability proven across 406 tensors.

Both results were achieved in a single architecture, designed and implemented to proof-of-concept in 24 hours.

### 5.2 What DAC Contributed

For efficiency: DAC provided a systematic search strategy. Rather than surveying the ML literature for efficiency techniques, we analyzed the forward pass through DAC primitives and identified where Budget, CompetitiveSelection, and ConvergenceGovernor were missing. Each gap pointed to a specific technique.

For forgetting: DAC redirected the search entirely. Without DAC, the natural approach would have been to search the machine learning literature for continual learning solutions. DAC redirected to systems engineering, where the write-conflict problem has mature solutions. The architectural insight (separate the immutable reasoning base from independently addressable knowledge overlays) came directly from recognizing the isomorphism.

### 5.3 What DAC Did Not Contribute

DAC identified the architecture but did not solve the implementation details:
- How to encode facts as low-rank deltas efficiently (gradient-based optimization)
- How to enforce orthogonality at scale (principal angle computation via SVD)
- How to target specific layers for specific knowledge (progressive widening strategy)
- How to compose multiple deltas safely (additive composition under orthogonality guarantee)

These are engineering problems within the architecture that DAC derived. DAC's value was architectural: it pointed at the right design space. The solutions within that space required standard machine learning and numerical linear algebra engineering.

---

## 6. Conclusion

Applying DAC to AI model design produced two results:

First, primitive analysis of the transformer forward pass identified four efficiency innovations (low-rank factorization, sparse attention, gated activation, and adaptive depth), each derived by recognizing where Budget, CompetitiveSelection, and ConvergenceGovernor primitives were absent in the standard architecture.

Second, vocabulary-stripping revealed that catastrophic forgetting is the write-conflict problem in shared mutable state. The solution (immutable bases, sparse independently-addressable overlays, registry-governed allocation) has been standard practice in systems engineering for decades.

LeanFormer instantiates both results as a single architecture. The efficient transformer core applies budget-governed resource allocation at every layer. The delta belief system applies the write-conflict solution to knowledge management. The Knowledge Plane provides production governance through orthogonality enforcement, capacity accounting, compositional routing, and provenance logging.

This serves as empirical evidence for DAC's generative claim: when a problem resists solution in its native domain vocabulary, strip the vocabulary, map the structure to the abstraction primitive set, and check whether someone in another domain has already solved it. For catastrophic forgetting, they had, decades ago. For transformer efficiency, the solutions existed as isolated techniques that DAC's systematic analysis unified into a coherent architectural design.
