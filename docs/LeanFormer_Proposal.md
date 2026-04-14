# LeanFormer: Applying Domain Abstraction Collapse to AI Architecture

## Abstract

Domain Abstraction Collapse (DAC) claims that hard problems in one domain are often solved problems in another, obscured by domain-specific vocabulary. This paper tests that claim by applying DAC to the design of a transformer-based language model and to the training pipeline that produces it. The methodology (strip domain vocabulary, map structural patterns to the sixteen-primitive set, search for solved isomorphisms) produces three results.

**First**, primitive analysis of the standard transformer forward pass identifies four categories of structural waste, each mapping to a known efficiency technique from resource-constrained systems engineering: low-rank factorization (`Budget<Parameters>`), two-pass sparse attention (`CompetitiveSelection` with hierarchical screening), gated activation sparsity (`CompetitiveSelection` over neuron space), and adaptive computation depth (`ConvergenceGovernor`).

**Second**, DAC reveals that catastrophic forgetting is structurally identical to the write-conflict problem in shared mutable state, a problem solved decades ago through immutable bases, sparse overlays, and registry-governed allocation. The resulting architecture, LeanFormer, can acquire, compose, version, and discard knowledge without retraining, with bit-for-bit reversibility and verified base-weight immutability (100 beliefs, 406 tensors).

**Third**, DAC reveals that the standard training loop is the only governed computational system in the DAC collapse table operating without selectivity, quality hierarchy, federated budget, or per-group convergence detection. Adding these primitives — the same primitives every other resource-governed domain uses — produces a governed training pipeline whose invariants have been specified in TLA+ and empirically confirmed at 204M parameters across 7,228 training steps with zero governance violations.

The initial proof-of-concept was designed and implemented in 24 hours. Architecture validation at 39M parameters (76M dense equivalent) confirmed the architectural thesis. Scale validation at 204M parameters (805M dense equivalent) on NVIDIA L4 confirmed that all sixteen primitives compose correctly under real training conditions. A B=0 initialization artifact encountered during the 204M run was diagnosed by applying DAC to its own failure, identified as an observational degeneracy, and fixed with a phase-aware `ConvergenceGovernor` verified in TLA+ across 18.6 million states before any code was written.

---

## 1. Motivation: Testing DAC on AI Model Design

DAC identifies sixteen abstraction primitives that express computational patterns across twelve engineering domains, with governance semantics preserved across composition. The methodology claims not only analytical power (explaining existing systems) but generative power (designing new ones by recognizing structural isomorphisms with solved problems) and implementation power (turning a described computation into a primitive-composition build plan).

To test the generative claim, we apply DAC to six open problems in transformer design:

1. **Parameter inefficiency** — dense weight matrices waste capacity.
2. **Attention cost** — self-attention is O(n²) in sequence length.
3. **Catastrophic forgetting** — new training overwrites old knowledge.
4. **Knowledge composition** — fine-tuning on multiple domains causes interference.
5. **Confabulation** — models produce confident text with no retrieval-vs-interpolation signal.
6. **Training-process inefficiency** — gradient compute is allocated uniformly regardless of learning need.

The DAC test for each: strip the ML vocabulary, map the stripped problem to the primitive set, and check whether someone in another domain has already solved it.

---

## 2. The Six Decompositions

### 2.1 Parameter Inefficiency → File-System Fragmentation

**ML framing:** dense weight matrices allocate full dimensional capacity even when most weights contribute minimally.

**Stripped:** a storage system allocates fixed-size blocks for variable-size records. Most blocks are mostly empty.

**Structural twin:** file-system fragmentation, solved by variable-size allocation under a capacity budget.

**Composition:** `Budget<Parameters>` + low-rank factorization. Every weight matrix is stored as `A @ B` from initialization. Rank is the budget knob. The domain function (what the matrix computes) is unchanged; the governance (how many parameters it uses) is now explicit and tunable.

### 2.2 Attention Cost → Brute-Force Rendering

**ML framing:** self-attention evaluates every query against every key, costing O(n²).

**Stripped:** a selection system evaluates every candidate against every output position, even when most candidates are irrelevant to most positions.

**Structural twin:** brute-force rendering before the visibility buffer. Rendering solved it with two passes: cheap coarse culling followed by fine evaluation of survivors.

**Composition:** `CompetitiveSelection` (ranked) for screening, then `CompetitiveSelection` (soft) for exact attention over the survivors. The gated feed-forward is the same pattern applied to MLP layers: a cheap gate identifies active neurons, and only active neurons are computed.

### 2.3 Catastrophic Forgetting → Write-Conflict

**ML framing:** training on new data overwrites previously learned information because the same parameters encode both.

**Stripped:** a shared mutable storage system where writes destroy existing content because the storage conflates retrieval index with stored content and uses dense encoding.

**Structural twin:** the write-conflict problem in shared mutable state, solved by immutable bases, sparse overlays, and registry-governed allocation.

**Composition:**

| LeanFormer Component | Primitive Composition |
|----------------------|----------------------|
| Frozen base weights | Immutable foundation (analogous to kernel state) |
| Belief delta | `Budget<Parameters>` + `Transaction` (atomic, bounded, reversible) |
| Delta registry | `ResourceRegistry<BeliefID, DeltaWeights>` with non-overlap enforcement |
| Routing network | `CompetitiveSelection` (ranked): query embedding → relevant deltas |
| Belief injection | `Transaction`: atomic addition of delta to registry |
| Belief removal | `Transaction`: atomic removal restoring pre-injection state |

### 2.4 Knowledge Composition → Multi-Tenant Isolation

**ML framing:** deltas for different domains interfere when loaded together.

**Stripped:** multiple tenants writing to shared resource without isolation corrupt each other's data.

**Structural twin:** multi-tenancy isolation in databases and operating systems, solved by address-space partitioning with enforced non-overlap.

**Composition:** extend the delta registry with `FederatedBudget<ParameterSubspace>`. The master parameter space is subdivided into non-overlapping regions; each domain's deltas are constrained to their region via orthogonality enforcement (principal angles via SVD, threshold 0.3). Composition becomes additive: disjoint subspaces load simultaneously without interference.

### 2.5 Confabulation → Signal-vs-Noise Discrimination

**ML framing:** language models produce confident-sounding text that is factually wrong because they have no mechanism to distinguish retrieval from interpolation.

**Stripped:** a system produces output with no confidence signal and no mechanism to separate cached retrieval from plausible-continuation generation.

**Structural twin:** the error-detection problem in signal processing, solved by separating the data path from the confidence path.

**Composition:** `ConvergenceGovernor` on hidden-state residual. Low residual after delta routing means the answer came from stored knowledge. High residual means the answer was generated from base model patterns with no grounding in injected knowledge. This does not eliminate confabulation; it makes it architecturally detectable. Confidence scoring combines routing strength, composition coherence, and delta coverage.

### 2.6 Training-Process Inefficiency → Every Other Governed Domain

**ML framing:** training is expensive because gradient computation scales with model size, dataset size, and epoch count. The ML community has produced mixed-precision training, gradient accumulation, parallelism strategies, curriculum learning, progressive training, layer freezing, sparse training, and importance sampling. Each addresses one dimension of the cost problem; none compose into a unified governed system.

**Stripped:** a process iteratively modifies a shared mutable state by computing error signals from sampled inputs and propagating corrections globally across the entire state on every iteration, regardless of which regions are relevant. The process has no selectivity (every parameter receives gradient from every sample), no quality hierarchy (all parameters trained at the same fidelity from step 0), no per-region convergence detection (only global), no budget governance (uniform rather than proportional), and no audit provenance.

**Structural twin:** this is the brute-force rendering problem applied to gradient computation, the full-table scan applied to parameter updates, the broadcast-to-all-nodes applied to learning signal. Every other governed computational system in the DAC collapse table uses `CompetitiveSelection` gating, `QualityHierarchy` traversal, `FederatedBudget` allocation, and per-group `ConvergenceGovernor`. Training uses none of them.

**Composition:**

| Training Component | Primitive Composition |
|-------------------|----------------------|
| Current loop | Sampler + ActuationPass + Reduction + PropagationPass + ConvergenceGovernor (global) |
| Targeted pipeline | Sampler + `CompetitiveSelection` (ranked) + `FederatedBudget<GradientCompute>` + `QualityHierarchy` + `TraversalEngine` + ActuationPass + Reduction + PropagationPass + `ConvergenceGovernor` (per-group) + `AuditSink` |
| Governed data pipeline | `QualityHierarchy<SampleDifficulty>` + `CompetitiveSelection` (ranked) + `Budget<SamplesPerStep>` + `AuditSink` |
| Change-triggered eval | `Signal<ConvergenceChange>` + `CompetitiveSelection` (ranked) + `Budget<EvalCompute>` + ActuationPass + Reduction + `AuditSink` |
| Forge gate | `Signal<GroupConverged>` + `ConvergenceGovernor` + `CompetitiveSelection` (ranked) + `Budget<ForgeCompute>` + ActuationPass + Reduction + `AuditSink` |

The ML community has independently invented fragments of this composition — Mixture of Experts (`CompetitiveSelection` at inference, not training), LoRA (static `Budget<Parameters>`, not sample-adaptive), curriculum learning (`QualityHierarchy` over data, not parameters), layer freezing (binary `ConvergenceGovernor` without graduated states), GradNorm (partial `FederatedBudget` at task level) — each in isolation, each in ML vocabulary that prevented recognizing the unified structural pattern.

---

## 3. Results

Architecture results are from the 39M-parameter model (76M dense equivalent) trained on 500K OpenWebText samples for one epoch. Training-governance results include both the 4.8M preliminary validation (300 steps, WikiText-2) and the 204M scale validation (7,228 steps, reasoning corpus, NVIDIA L4).

| Mechanism | Metric | Result | Status |
|-----------|--------|--------|--------|
| Low-rank compression | Compression ratio | 3.9x at 39M; 3.94x at 204M | Validated |
| Sparse attention | Attention sparsity | 88% | Validated (39M) |
| Gated feed-forward | FF sparsity | 80% | Validated (39M) |
| Belief injection | Success rate | 84% across 100 beliefs | Validated (39M) |
| Semantic routing | Routing accuracy | 86% (4.3x above chance) | Validated (39M) |
| Belief coexistence | Simultaneous improvement | 64% of 100 beliefs | Validated (39M) |
| Base weight restoration | Bit-for-bit fidelity | Exact across all tensors | Validated (39M) |
| Base weight immutability | Tensor integrity | 406 tensors verified | Validated (39M) |
| Adaptive depth | Mean exit depth | 11.3/12 (39M); 20/20 at 204M (not activated) | Partial |
| Governed training | Budget invariant | 0 violations / 300 steps (4.8M); 0 / 722 records (204M) | Validated |
| Audit chain integrity | SHA-256 chain | 300 records (4.8M); 722 records (204M) | Validated |
| Convergence governors | State transitions | 15 (4.8M, 7/8 converged); 18 (204M, all valid) | Validated |
| Hierarchy activation | Coarse-to-fine | L0 → L1 → L2 → L3 via convergence | Validated (4.8M, 204M) |
| L3 genuine convergence | Activation step | 2,773 (non-round, after 2,300+ steps of real gradient flow) | Validated (204M) |
| L1/L2 timing | Activation steps | 200 / 400 (B=0 initialization artifact, disclosed) | Disclosed |
| B=0 diagnosis | DAC applied to own failure | Observational degeneracy identified; phase-aware fix verified in TLA+ | Validated |
| Budget reallocation | Dynamic budget shifts | Converged groups drop to 0.012, active up to 0.40 | Validated (204M) |
| Language modeling | Best val PPL | 57.6 @ step 2,000 | Measured (204M) |
| Language modeling | Final val PPL | 1,463.9 (overfit, invariants held throughout) | Expected (204M) |
| Orthogonal capacity | Available dims | 53,760 (3,360 rank-16 deltas), two machines | Measured (204M) |
| Training time | Wall clock | 140.9h on NVIDIA L4 | Measured (204M) |

The 204M run is governance-machinery validation at a scale where parameter group ratios are representative (L0 ≈ 41.5% of parameters, compared with ≈86% at 4.8M where the embedding table dominates). The language-modeling perplexity is reported for completeness; the claim is that all sixteen primitives compose correctly under real training conditions, which they did across the entire 7,228-step run including the post-overfit phase.

---

## 4. DAC Applied to Its Own Failure

The most instructive result from the 204M run was unplanned. Low-rank layers initialize `B` matrices to zero, producing near-zero gradient flow regardless of whether training signal has been meaningful. The convergence governors correctly detected low gradient EMA and transitioned ACTIVE → COOLING as specified, producing hierarchy activations at steps 200 (L1) and 400 (L2) — both at round-number intervals aligned with the cooling window. Loss remained flat at 10.388 through both activations. Actual training progress began only when the output head activated at L2 and introduced real gradient flow.

Applying DAC's vocabulary-stripping process to this failure reveals it as an **observational degeneracy**: two qualitatively different trajectories (cold start and genuine convergence) produce the same low-magnitude reading. The collapse table documents this pattern repeatedly: depth buffers disambiguate zero-color pixels in rendering, heartbeats disambiguate silent nodes in networking, timeouts disambiguate non-responsive voters in distributed consensus. The solution is always the same primitive: a second `Signal<T>` that breaks the degeneracy.

The fix follows mechanically: a phase-aware `ConvergenceGovernor` that tracks whether gradient magnitude has ever exceeded threshold, classifying trajectories into COLD, WARMING, ACTIVE_LEARNING, DECLINING. The ACTIVE → COOLING transition requires the phase to be ACTIVE_LEARNING or DECLINING, never COLD. The `NoCoolingFromCold` invariant was specified and verified in TLA+ across 18.6 million states. A specification without phase awareness produces a TLC counterexample in 2 states matching the exact failure observed.

This episode demonstrates three methodology properties simultaneously:

1. **Structure/function separation holds.** The governance machinery was correct (the governor followed its spec). The domain function was miscalibrated (the cooling threshold applied to B=0-initialized parameters). The primitive did not change; one precondition was added.
2. **Generative mode works in real time.** The fix was not invented; it was recognized as a solved problem from the collapse table.
3. **Formal verification pays off.** The fix was verified before any code was written, and the original failure was reproduced as a concrete counterexample.

A `COOLING → ACTIVE` regression in the `attention_output` group further validated the four-state machine: the group was prematurely cooled, then reactivated when real gradient flow pushed its EMA above threshold. The hysteresis behavior is exactly what the state machine was designed to provide.

---

## 5. What DAC Did Not Provide

DAC did not design domain functions. The specific choice of low-rank factorization for deltas, the cosine similarity metric for routing, the exit-threshold tuning for adaptive depth, the loss function, the optimizer, the scoring function for the gradient router, the convergence thresholds for per-group governors, the budget allocation policy, the hierarchy level boundaries, and the sample difficulty thresholds are all domain functions requiring ML engineering judgment.

DAC provided the structural skeleton. Domain knowledge filled in the scoring functions, the loss formulations, the training recipes, and the governance thresholds. This is the structure/function separation the methodology predicts: the abstraction primitives provide structure (data flow, resource allocation, convergence detection, audit). The domain provides function (what computation to apply at each step). Neither replaces the other.

---

## 6. Open Questions and Next Steps

1. **Phase-aware `ConvergenceGovernor` implementation + retrain.** The TLA+ spec exists and passes. Adding `gradient_phase` tracking and a `peak_observed` flag to the training code, then retraining 204M, would produce clean hierarchy activations free of the B=0 artifact — the first unambiguous empirical demonstration of convergence-gated coarse-to-fine training.
2. **Multi-epoch training with proper regularization.** The current 204M run used a single epoch with dropout 0.1 as the only regularization; a 3-5 epoch run with a stronger regularization sweep would test generalization and exercise AWAKENED state transitions triggered by distribution shifts.
3. **Scale validation at 7B+ parameters.** The governance invariants are formally verified and empirically confirmed scale-independent, but the efficiency claims (50-70% gradient-compute reduction) require a 7B+ run with real backward-pass skipping for converged groups and comparison against an ungoverned baseline.
4. **Knowledge Plane validation at 204M.** Repeat the 39M belief injection / routing / coexistence measurements at 204M against the step-2,000 checkpoint.

See Section 9.5 of `dac/Domain_Abstraction_Collapse.md` for the full future-work list with cost estimates.

---

## 7. Conclusion

Applying DAC to AI model design produced three validated results. Primitive analysis of the transformer forward pass identified four efficiency innovations by recognizing where `Budget`, `CompetitiveSelection`, and `ConvergenceGovernor` were absent from the standard architecture. Vocabulary-stripping revealed that catastrophic forgetting is the write-conflict problem, with a solution that has been standard practice in systems engineering for decades. The same process revealed that training itself is the only governed computational system in the DAC collapse table operating without the selectivity, quality-hierarchy, federated-budget, and per-group-convergence primitives that every other resource-governed domain uses.

LeanFormer instantiates all three results as a single architecture and a single training pipeline. The entire development arc from initial DAC decomposition to 204M-parameter validated results took approximately three weeks, one architect, and an AI implementation agent. The speed is not incidental. It is the point: when the methodology reveals that an "unsolved" problem is a solved problem wearing unfamiliar vocabulary, the path from recognition to implementation is short.
