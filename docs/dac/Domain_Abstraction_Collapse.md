# Domain Abstraction Collapse DRAFT

## A Methodology for Recognizing Solved Problems Across Domain Boundaries

Brian Moore, CISSP, CCSP
Independent Systems Researcher

---

## Abstract

The vocabulary used to describe a problem shapes which solutions feel reachable. When two engineering domains use different language for what is structurally the same operation, each domain tends to reinvent its own solution, often without noticing that the work has already been done somewhere else. This paper describes Domain Abstraction Collapse (DAC): a process for stripping domain-specific vocabulary from computational patterns, looking for structural similarity across domain boundaries, and reducing domain-specific abstractions to a candidate set of domain-agnostic abstraction primitives from which domain patterns can be reconstructed through composition.

The starting point was practical. I was working across three automation domains — ETL, SOAR, and configuration management — that the relevant communities already knew were expressible as workloads over a streaming DAG, and I wanted to build an orthogonal kernel with a composable module system that could serve all three from one execution engine. That meant deliberately looking for the structural operations the three genuinely shared, separated from the domain-specific framing each tool puts on those operations. As I worked across nine more engineering domains — real-time rendering, physics simulation, audio spatialization, network replication, container orchestration, CI/CD pipelines, distributed consensus, economic simulation, and machine learning — the same shapes kept appearing. A set of sixteen primitives covered the patterns I found across all twelve. The claim is bounded. Sixteen primitives sufficed for these twelve domains. The set is not proven minimal, I am not particularly invested in whether it is, and other domains may require additional primitives. The value of the methodology does not depend on the exact count.

DAC works in three modes. As an analytical tool, it decomposes existing systems and surfaces structural similarities that domain vocabulary tends to conceal. As a generative methodology, it offers a search strategy: when a problem looks hard in its domain, strip the vocabulary, see whether the structure matches a primitive or composition that another domain has already solved. As an implementation methodology, it converts a computation described in domain notation into a build plan composed of known primitives.

I demonstrate the generative and implementation modes through LeanFormer, a transformer architecture where DAC was applied to six open problems in neural network design: parameter inefficiency, attention cost, catastrophic forgetting, knowledge composition, confabulation, and training process inefficiency. LeanFormer was also a deliberate test of the methodology on a domain I do not have research-level expertise in. My background is security and Linux systems engineering and automation, not machine learning. The hypothesis was whether DAC, applied carefully by a non-expert, could produce a workable architecture for problems the ML literature treats as open. If the methodology required deep ML expertise to apply, that would suggest DAC is dressed-up domain expertise rather than an independent process worth naming. In each case, stripping the ML vocabulary produced a structural description that matched a problem from systems engineering. The initial proof-of-concept architecture was designed and built in twenty-four hours. Subsequent validation at 39M parameters (76M dense equivalent) covered 119 tests with the following measurements: 84% belief injection success across 100 beliefs, 86% semantic routing accuracy, bit-for-bit base weight restoration across 100 beliefs, and 88% attention sparsity. A 204M-parameter scale run (805M dense equivalent, 7,228 optimizer steps) confirmed that the sixteen primitives compose under real training conditions: zero budget violations across 722 hash-chained audit records, eighteen valid governor state transitions with no skipped states, and hierarchy activation in the predicted coarse-to-fine order. Best validation perplexity reached 57.6 at step 2,000. This run is not competitive as a language model. The claim is governance machinery, not language modeling performance.

During the 204M run, a B=0 initialization artifact caused two parameter groups to be misclassified as converged before they had received meaningful gradient signal. Stripping the ML vocabulary described the failure as an observational degeneracy: two qualitatively different gradient trajectories (cold start and post-learning convergence) produced the same low-magnitude reading. The same pattern shows up elsewhere in the collapse table: depth buffers in rendering, heartbeats in networking, timeouts in distributed consensus. The fix follows the same shape across all of them: add a second signal that disambiguates the measurement. A phase-aware ConvergenceGovernor that records whether gradient magnitude has ever exceeded the threshold was specified in TLA+ and verified by TLC across approximately 18.6 million states. The original failure reproduces as a 2-state TLC counterexample without the phase guard. The fix itself is a standard latching pattern; the value of DAC here was recognizing that the bug class was already familiar.

To show the primitive set is not tied to a specific language or codebase, Appendix A provides implementations of all sixteen primitives in Rust, Python, Go, and TypeScript across multiple domains. The structure is identical across languages; only the type annotations and domain functions change.

Twenty-one TLA+ specifications were verified by TLC across approximately 44.6 million states. The counting expands as follows: sixteen conceptual primitives, eighteen spec modules (because CompetitiveSelection is split into hard, soft, and ranked variants), and twenty-one specs total once two cross-primitive composition theorems and the phase-aware ConvergenceGovernor variant are included. Five LeanFormer compositions were verified separately across approximately 178,000 additional states. Eighteen decomposition-failure specifications produced concrete TLC counterexamples in which decomposing a primitive into separable check-then-act steps introduces a TOCTOU window where the primitive's invariant is violated. This shows that each primitive's invariant requires atomicity at the primitive boundary. It is the systematic demonstration of this property across all sixteen primitives that is novel; the property itself (that atomicity-dependent invariants fail when atomicity is removed) is a known consequence of how atomic operations behave in concurrent systems. The result is weaker than algebraic minimality (which would require showing that no primitive can be expressed as a composition of the others). It is stronger than the paper would be without formal verification. The verification process also produced five corrections that refined imprecise specification choices.

The point of the paper is straightforward. Across twelve engineering domains I examined, structurally similar operations appear under different vocabularies. A set of sixteen primitives covered the patterns. The vocabulary made the structures look different. They were not.

---

## 1. Introduction

### 1.1 Patterns Hidden by Vocabulary

Fred Brooks distinguished between essential complexity (complexity inherent to the problem) and accidental complexity (complexity introduced by the tools and methods used to solve it) [1]. There is a third category worth naming: vocabulary-induced complexity, where the domain-specific language used to describe a problem creates the impression that the problem itself is domain-specific.

Consider the Entity Component System (ECS), a common architectural pattern in game engine development since the late 1990s. An entity is a unique identifier. A component is a typed data record associated with an entity. A system is a function that queries entities by their component composition and applies transformations. The pattern shows up in major game engines under different names: GameObjects and MonoBehaviours in Unity, Actors and Components in Unreal, Entities and Components and Systems in Bevy.

Stripped of game-specific vocabulary, an ECS is structurally a relational table with column-oriented storage. Entities are rows. Components are columns. Systems are queries with side effects. The cache-friendly memory layout (structure-of-arrays organization) is a real engineering contribution. The structural pattern itself (typed records queried by composition and transformed by functions) is well-trodden ground in the database community, which has spent decades developing query planning, transaction semantics, and access control for exactly that shape. The observation that ECS resembles a database is not original to me; data-oriented design literature and several practitioners (notably Tim Sweeney in public talks) have made versions of it. What is worth noting is the scale of the parallel reinvention: many game engine teams have built ECS infrastructure independently, each within game-engine vocabulary, without much cross-pollination from the database side.

This is one example of a recurring pattern. Across the engineering domains I examined, systems that appear to solve domain-specific problems with domain-specific architectures often turn out, when the vocabulary is stripped, to be compositions of a small number of computational patterns that have been solved repeatedly under different names.

### 1.2 The Methodology: Domain Abstraction Collapse

In plain terms, DAC is a habit. When you learn a new technology and ask "how does this work?", you get an answer that is technically correct but full of domain jargon you have to look up. So you ask the second question: "OK, but how does this *really* work?" That second question, asked deliberately and across enough domains to start noticing patterns, is DAC. It is pattern matching against solutions you already know, with the domain vocabulary stripped off so the matching is possible. The five steps below are the systematic version of that habit.

Domain Abstraction Collapse (DAC) is a five-step process:

1. Enumerate domain-specific abstractions. List every named concept, pattern, data structure, and algorithm used within a domain. Accept the domain's own vocabulary.

2. Strip domain vocabulary. For each abstraction, remove every word that is specific to the domain. Replace domain nouns with generic descriptions of what the abstraction actually does structurally.

3. Identify cross-domain similarities. Compare the stripped descriptions across domains. When two abstractions from different domains reduce to the same structural description, they are candidates for the same operation under different names.

4. Reduce to abstraction primitives. Continue stripping until further decomposition would either lose governance properties (budget invariants, convergence detection, audit completeness) or descend to an implementation level (register operations, logic gates) where domain patterns require unbounded composition counts. The operations that survive this process are the abstraction primitives.

5. Reconstruct domains as compositions. Verify that every domain-specific abstraction from step 1 can be expressed as a composition of the abstraction primitives from step 4, instantiated with domain-specific data and functions.

If step 5 succeeds with no residual (every domain pattern is expressible, and no domain pattern requires a primitive not in the set), the collapse is complete for that domain. The domain-specific abstractions were vocabulary, not structure.

These five steps describe DAC as an analytical methodology. The same process works in reverse as an implementation methodology. Given any computation described in domain-specific notation (a mathematical formula, a protocol specification, a biological pathway diagram), strip the notation's vocabulary and map the computation's structure to the primitive set. The result is a build plan: a composition of primitives that implements the computation. Section 5 develops this generative and implementation application.

### 1.3 Abstraction Primitives: Why the Decomposition Stops Here

The level at which decomposition stops is the part of the claim that needs care, because if it is set too low the result trivially reduces to NAND gates and the methodology has no operational value, and if it is set too high domain vocabulary creeps back in.

An abstraction primitive is an operation that cannot be further decomposed without one of two consequences:

(a) Loss of governance semantics. The primitive carries formal properties (budget invariants, convergence detection, audit completeness, transaction atomicity) that make cross-domain composition meaningful. Decompose below this level and those properties have to be reimplemented per domain, which is exactly the redundancy DAC eliminates.

(b) Explosion of composition count. At lower levels of abstraction (register operations, logic gates, individual arithmetic instructions), expressing a single domain pattern requires hundreds or thousands of composed operations, and the compositions become unwieldy enough to lose their explanatory and constructive value.

The abstraction primitives sit at what I will call the governance boundary: the thinnest layer of operations that still carries formal properties. Below this boundary are implementation details that vary by hardware and runtime. Above it is domain vocabulary that prevents cross-domain reuse. NAND gates are primitives but not abstraction primitives, because they carry no governance semantics. Domain-specific patterns like "visibility buffer" or "playbook" are not primitives at all, because they decompose into compositions of the abstraction primitive set.

A note on counting. The paper refers to sixteen primitives in conceptual terms. The TLA+ specifications expand this to eighteen spec modules because CompetitiveSelection is split into hard, soft, and ranked variants for verification clarity. Two cross-primitive composition theorems (TraversalBudgetComposition, SelectThenActuate) and a phase-aware variant of ConvergenceGovernor (introduced in Section 5.8.1) bring the verified spec count to twenty-one. Throughout the paper, "sixteen primitives" refers to the conceptual set; specific section text will clarify when the spec count is meant.

A second note on what is and is not claimed. Each primitive's invariant has been formally checked under the assumption that primitive operations are atomic. Section 9.4 describes systematic decompositions of each primitive into separable check-then-act steps and the TOCTOU patterns that result. This shows that the primitives, as specified, depend on atomicity at the primitive boundary. It is consistent with how atomic operations behave in any concurrent system. It does not show that the primitive set is algebraically minimal in the sense that no primitive can be expressed as a composition of the others. Algebraic minimality is open and is discussed in Section 9.1.

### 1.4 Related Work and Intellectual Lineage

DAC sits in a tradition of work that strips domain vocabulary to find shared structure, without introducing new mathematical formalism beyond what the engineering audience already uses.

Brooks' essential/accidental complexity distinction [1] is the closest philosophical ancestor. The contribution here is to point at vocabulary specifically as a source of accidental complexity that operates at the level of the conceptual framing of the problem, not just the tools used to solve it.

Pattern languages, including Alexander's original work in architecture [a-1] and Gamma et al.'s software design patterns [a-2], identify recurring structural motifs across many designs and give them names. DAC has the same instinct but pushes it further in two ways: the patterns are stripped of all domain-specific framing rather than kept at the design-pattern level, and the catalog is closed (the claim is that the sixteen primitives suffice for the twelve domains examined, not that they are one entry in an open list).

Information hiding (Parnas) [a-3] and abstract data types (Liskov) [a-4] articulated the principle that interfaces should hide implementation details that vary across cases. The primitives here can be read as abstract data types whose interfaces are the governance invariants and whose implementations vary per domain.

Coordination languages (Linda [a-5], Reo) and process algebras (CSP [a-6], the actor model [a-7], pi-calculus) formalized concurrent and distributed computation. Several of the sixteen primitives have shape similar to operations in these formalisms (RateLimit resembles a token bucket from queueing theory; PropagationPass resembles message-passing semantics). DAC is engineering-flavored rather than algebraic; the relationship is one of family resemblance, not derivation.

Wing's work on computational thinking [a-8] argues that recognizing the computational structure underneath domain problems is a general intellectual skill. DAC is a specific procedure for doing that for a class of resource-governed execution systems.

Workflow management formalisms (van der Aalst's workflow patterns [a-9]) cataloged control-flow patterns across many process-management systems. The DAG workload collapse in Section 7 overlaps with that work.

Category theory provides a formal language for structure-preserving mappings between domains. Some of the cross-domain identifications in this paper could be formalized as functors or as adjunctions; the paper does not develop that formalization because the engineering audience for whom the methodology is most useful does not typically reach for it. The categorical formalization is a worthwhile direction for future work.

The contribution of this paper, given that lineage, is a specific catalog of sixteen primitives, formal specifications and verification of each, and a worked example (LeanFormer) demonstrating that the catalog is sufficient to design a non-trivial system end-to-end.

### 1.5 The Turing Tarpit Question

A reasonable objection to any claim of cross-domain structural similarity is the Turing Tarpit argument: because all general computation is Turing-equivalent, of course any computational pattern can be mapped to any other, and the mapping is operationally useless if pursued far enough. Reduce far enough and everything is NAND gates.

The defense here is empirical rather than theoretical. The sixteen primitives were not designed top-down from a theory of computation. The starting point was three automation domains — ETL, SOAR, and configuration management — known in their respective communities to be expressible as workloads over a streaming DAG. The original goal was practical: build an orthogonal kernel with a composable module system that could serve all three from one execution engine. That required identifying the operations the three genuinely shared at the structural level. As I worked through additional domains, asking how each one worked underneath its vocabulary, the same shapes kept appearing. Rendering came in via orchestration: once I noticed that orchestration is substrate-agnostic — a game engine orchestrating rendering, physics, and audio subsystems at per-tick granularity is doing the same thing as a Kubernetes scheduler placing pods or a hypervisor placing VMs — rendering was pulled into the analysis as another orchestration domain operating under tighter constraints. The level-of-detail (LOD) system in rendering shared shape with OSPF priority propagation in routing, both being budget-bounded decisions about what to compute or synchronize given a partial view of the system. That parallel forced the addition of QualityHierarchy, TraversalEngine, and CompetitiveSelection — primitives the original three automation domains did not require. Each subsequent domain either mapped onto existing primitives or forced new primitives to be added when the existing set was genuinely insufficient.

If the primitives were too low-level (register operations, logic gates), the collapse would have produced hundreds of primitives per domain pattern, and the compositions would be unwieldy. If they were too high-level (domain-specific abstractions), no shared primitives would have emerged across domains. The fact that sixteen primitives sufficed for twelve domains, discovered through incremental domain analysis rather than designed to fit a target count, is the evidence that the decomposition level is in roughly the right place. It is not proof of optimality.

### 1.6 Contributions

The paper makes the following contributions:

1. A description of Domain Abstraction Collapse as a named, repeatable design methodology with defined steps, a working definition of what counts as an abstraction primitive, and a sufficiency criterion.

2. A case study of the methodology applied to twelve engineering domains, with the resulting primitive set documented and the genuine vs. trivial collapses called out honestly.

3. Identification of the Competitive Selection family as parameterized rather than monolithic, resolving an overcounting issue in the original primitive set where seventeen domain patterns were collapsed into a single primitive.

4. LeanFormer, a transformer architecture designed end-to-end through DAC, with measurements at 39M parameters across 119 tests and a 204M-parameter governance-machinery validation across 7,228 training steps.

5. An articulation of DAC as an implementation methodology that converts domain-notation computations into build plans composed of primitives.

6. A demonstration that the same primitive set can govern not just a system's architecture but its training, evaluation, knowledge management, and deployment, validated at 204M parameters with all governance invariants holding throughout the run.

7. TLA+ specifications for all twenty-one primitive specs, eighteen decomposition-failure specs, and five LeanFormer compositions, verified by TLC across approximately 44.6M states for the universal primitive verification plus approximately 178K additional states for the LeanFormer case study. The verification establishes that primitive invariants hold under all reachable states at the tested bounds, that compositions preserve those invariants, and that decomposition into separable check-then-act steps systematically introduces TOCTOU windows that violate the invariants. A B=0 initialization bug observed during the 204M run was reproduced as a 2-state TLC counterexample, and a phase-aware fix was verified across approximately 18.6 million states.

---

## 2. The Collapse: Twelve Domains, Sixteen Primitives

### 2.1 The Domains Examined

The following twelve engineering domains were subjected to abstraction collapse. They were not selected from an a priori list. They emerged as the application domains encountered during the design of a general-purpose execution framework, beginning with SOAR automation and expanding as each new domain pulled in adjacent material.

| # | Domain | Entry Point |
|---|--------|-------------|
| 1 | Security Orchestration (SOAR) | Original motivation: frustration with existing SOAR platforms |
| 2 | CI/CD Pipelines | Recognized as DAG workloads with the same shape as SOAR playbooks |
| 3 | ETL / Data Pipelines | Recognized as DAG workloads with different I/O stages |
| 4 | Configuration Management | Recognized as declarative state convergence |
| 5 | Container Orchestration | Recognized as budget-constrained resource allocation |
| 6 | Real-Time Rendering | First GPU domain: tested whether primitives held under real-time constraints |
| 7 | Physics Simulation | Multi-stage pipeline decomposed to selection plus propagation |
| 8 | Audio Spatialization | Budget-constrained propagation over spatial graph |
| 9 | Network Replication | Spatial budget allocation in bandwidth units |
| 10 | Economic Simulation | Competitive selection plus propagation plus transactions |
| 11 | Distributed Consensus | Fixed-point iteration over a cluster graph (RAFT) |
| 12 | Machine Learning | Complete AI lifecycle decomposed to primitives |

A scope caveat worth stating up front: these twelve domains are all infrastructure and systems domains with a strong bias toward "things that allocate resources under constraints." Domains outside this family (functional reactive programming, formal theorem proving, constraint logic programming, bioinformatics sequence alignment) have not been tested and may not collapse as cleanly. The claim is bounded to the twelve domains examined.

### 2.2 The Abstraction Primitive Set

After abstraction collapse across all twelve domains, sixteen primitives covered the patterns I found: operations that could not be further decomposed without losing governance semantics, and that appeared in multiple domains under different vocabulary.

Data Primitives (how data is structured and related):

| Primitive | Single Responsibility |
|-----------|----------------------|
| Budget\<U\> | A pool with a capacity and an invariant: consumed <= capacity |
| FederatedBudget\<U\> | A master pool subdivided into sub-pools: sum(sub) <= master |
| QualityHierarchy | A tree of representations at multiple fidelity levels |
| AllocationSnapshot | An immutable record of a resource allocation decision |
| RelationshipGraph | A typed graph linking entities |
| ResourceRegistry\<K, V\> | A unique key to value mapping |

Control Primitives (how computation is organized and orchestrated):

| Primitive | Single Responsibility |
|-----------|----------------------|
| TraversalEngine | Budget-bounded depth-first traversal of a graph |
| PropagationPass | Iterative message-passing over a graph toward a consistent state |
| CompetitiveSelection | Allocate candidates to seats under a scoring function (three modes: hard, soft, ranked) |
| ActuationPass | Apply a side-effecting operation to a set of items |
| Reduction | Aggregate a stream of values to a smaller summary |
| Sampler | Probabilistically select items from a distribution |

Coordination Primitives (how the system observes and governs itself):

| Primitive | Single Responsibility |
|-----------|----------------------|
| Signal\<T\> | A typed event channel from producer to consumer |
| RateLimit | Bound the rate of an operation to a maximum per window |
| ConvergenceGovernor | Detect fixed-point convergence and govern iteration |
| AuditSink | A hash-chained append-only log of mutations |

A note on atomicity. Several of the compositions in Sections 2.6 and 8 require that a sequence of primitive operations execute atomically (e.g., "atomic write to destination," "atomic exchange between two Budget instances"). Atomicity is treated here as a runtime requirement that the primitive set assumes, not as an additional primitive in the set. The TLA+ specifications model atomicity by treating each primitive operation as a single transition; in implementation, ensuring atomicity is the responsibility of the runtime (transaction manager, software transactional memory, or equivalent). I considered adding an explicit Transaction primitive but decided against it because every primitive's invariant already implicitly requires atomicity at the primitive boundary, and adding Transaction as a primitive would either be redundant with that or would expand into the territory of distributed transaction protocols that are out of scope for this paper.

### 2.3 The Primitive Trait Signature

The structural skeleton of a primitive, expressed in Rust trait syntax for concreteness:

```rust
trait AbstractionPrimitive {
    type Input;
    type Output;
    type Invariant;

    fn apply(&self, input: Self::Input) -> Result<Self::Output, InvariantViolation>;
    fn invariant(&self) -> Self::Invariant;
}
```

The shape is uniform: a primitive takes a typed input, produces a typed output (or an invariant violation), and exposes its governance invariant. The trait is realized differently per primitive (Budget's invariant is a numeric ceiling; ConvergenceGovernor's is a state-machine property; AuditSink's is a hash-chain property). The point is structural: each primitive is a self-contained unit that can be composed with others without leaking its governance to its caller.

### 2.4 Domain Patterns Reduced to Primitive Compositions

The following tables document the reduction of common domain patterns to primitive compositions. They are not exhaustive. They are representative of the kind of mapping the methodology produces. I have noted which collapses are genuinely informative and which are largely renaming.

Real-Time Rendering:

| Domain Abstraction | What It Actually Is |
|---|---|
| Visibility buffer | CompetitiveSelection (hard): triangles compete for pixel ownership, depth is the scoring function |
| Deferred shading | ActuationPass over the AllocationSnapshot from the visibility CompetitiveSelection |
| Level of detail (LOD) | QualityHierarchy + TraversalEngine + AllocationSnapshot under Budget\<Bytes\> |
| Frustum culling | Reduction over RelationshipGraph (spatial partition) (largely a renaming) |
| Global illumination | PropagationPass + ConvergenceGovernor under Budget\<Microseconds\> |
| Temporal amortization | Budget\<TimeSlot\> + RateLimit + ConvergenceGovernor (feedback) |

Physics Simulation:

| Domain Abstraction | What It Actually Is |
|---|---|
| Broad-phase collision | Reduction over RelationshipGraph (spatial partition) — see Section 4.4 for why this is not CompetitiveSelection |
| Narrow-phase collision | CompetitiveSelection (hard): contact pairs compete for resolution priority |
| Constraint solver | PropagationPass + ConvergenceGovernor: iterative relaxation of constraint forces |
| Integration step | ActuationPass: apply force-derived velocity changes to position |

Audio Spatialization:

| Domain Abstraction | What It Actually Is |
|---|---|
| Spatial audio mixer | PropagationPass + Budget\<Voices\>: sound propagates over scene graph under voice budget |
| Audio LOD / culling | QualityHierarchy + TraversalEngine: distant sources at lower fidelity |
| Reverb buses | RelationshipGraph + PropagationPass: signal flows over a routing graph |
| Voice priority | CompetitiveSelection (ranked): voices compete for the K available channels |

Network Replication:

| Domain Abstraction | What It Actually Is |
|---|---|
| Interest management | CompetitiveSelection (ranked): entities compete for relevance to each viewer |
| Bandwidth budget | FederatedBudget\<BytesPerSecond\>: master bandwidth subdivided per viewer |
| Delta compression | Reduction\<(Old, New) → ChangeRecord\>: aggregate state diffs |
| Reliable channels | RateLimit + ConvergenceGovernor: bound retransmits, converge on acknowledgment |

Container Orchestration:

| Domain Abstraction | What It Actually Is |
|---|---|
| Pod scheduling | CompetitiveSelection (hard): nodes compete for pod placement, fit is the scoring function |
| Resource quotas | FederatedBudget\<CPU/Memory\>: cluster master subdivided per namespace |
| Liveness probes | Signal\<Healthy\> + RateLimit: periodic check at governed rate |
| Reconciliation loop | ConvergenceGovernor + ActuationPass: detect drift, drive toward desired state |

CI/CD Pipelines:

| Domain Abstraction | What It Actually Is |
|---|---|
| Pipeline DAG | RelationshipGraph\<Stage, Dependency\> + TraversalEngine |
| Artifact caching | ResourceRegistry\<BuildInput, BuildOutput\> + Budget\<CacheCapacity\> with content-hash invalidation |
| Parallel test execution | Budget\<Compute\> + Sampler\<TestPartition\>: budget-constrained parallel work |

ETL / Data Pipelines:

| Domain Abstraction | What It Actually Is |
|---|---|
| Extract stage | ActuationPass: I/O-gated data retrieval from a ResourceRegistry (largely a renaming) |
| Transform stage | ActuationPass: map function over data (largely a renaming) |
| Load stage | ActuationPass + Budget\<DestinationCapacity\> + ResourceRegistry\<Key, Record\> with atomic write boundary (atomicity is a runtime requirement, not a primitive) |
| Backfill | TraversalEngine over a temporal QualityHierarchy: budget-constrained reprocessing |

Configuration Management:

| Domain Abstraction | What It Actually Is |
|---|---|
| Desired state convergence | ConvergenceGovernor: detect current vs. desired state delta, iterate until converged |
| Idempotent operation | ActuationPass with Reduction\<(Old, New) → ChangeRecord\>: only actuate if delta is non-zero |
| Role/playbook | RelationshipGraph\<Task, Dependency\> + TraversalEngine: a DAG workload |
| Inventory | ResourceRegistry\<Host, Configuration\> (largely a renaming) |

SOAR (Security Orchestration, Automation, and Response):

| Domain Abstraction | What It Actually Is |
|---|---|
| Playbook | RelationshipGraph\<Action, Dependency\> + TraversalEngine: a DAG workload with capability-gated I/O, structurally the same shape as CI/CD pipelines, ETL workflows, and configuration management playbooks |
| Alert triage | CompetitiveSelection (ranked): alerts compete for analyst attention seats, prioritized by severity score |
| Enrichment | ActuationPass + ResourceRegistry: look up context from external sources (largely a renaming) |
| Response action | ActuationPass with capability-based access control |

Distributed Consensus (RAFT):

| Domain Abstraction | What It Actually Is |
|---|---|
| Leader election | CompetitiveSelection (hard): nodes compete for the leader seat |
| Log replication | PropagationPass: leader propagates log entries to followers until majority acknowledgment (convergence) |
| Heartbeat | Signal\<LeaderAlive\> + RateLimit: periodic notification at governed rate |
| Commit | Reduction\<Acknowledgment, Count\> + Budget\<CommitSlot\> + ResourceRegistry\<TxnID, Record\> with atomic commit boundary (atomicity is a runtime requirement) |

Economic Simulation:

| Domain Abstraction | What It Actually Is |
|---|---|
| Market order matching | CompetitiveSelection (hard): buy orders compete for sell-order seats, price is the scoring function |
| Price discovery | PropagationPass + ConvergenceGovernor: iterative relaxation toward equilibrium price |
| Inventory management | Budget\<ItemSlots\> + ResourceRegistry\<ItemID, Slot\>: multi-resource updates under capacity constraints with atomic boundary |
| Trade | Budget\<U_source\> + Budget\<U_target\>: atomic exchange between two Budget instances (atomicity is a runtime requirement) |

Machine Learning (extended discussion in Section 6):

| Domain Abstraction | What It Actually Is |
|---|---|
| Attention mechanism | CompetitiveSelection (soft): tokens compete for attention weight, similarity is the scoring function, softmax produces weighted combination [2] |
| Backpropagation | PropagationPass (single-pass variant): reverse message passing on the computation graph (see Section 6.1 for a precision caveat) |
| Speculative decoding | Two-level CompetitiveSelection: fast model generates candidates (coarse), slow model verifies (fine) [8] |
| Beam search | CompetitiveSelection (ranked): candidate continuations compete for K beam seats |
| KV cache | ResourceRegistry\<SequencePosition, (Key, Value)\> + Budget\<CacheCapacity\> with position-based invalidation |
| Learning rate scheduling | Budget\<TimeSlot\> + RateLimit + ConvergenceGovernor (feedback): PID-like control governing a resource (learning rate) over time |
| Early stopping | ConvergenceGovernor: detect when validation loss improvement falls below threshold |
| LoRA / Adapters | Budget\<Parameters\>: constrain trainable parameters to a fraction of total [6] |
| Dropout | Sampler\<ActivationMask\>: probabilistic selection of active neurons |
| Loss computation | Reduction\<Prediction, Scalar\>: aggregate prediction-ground-truth distances |
| Uniform gradient computation | Brute-force evaluation: every parameter receives gradient from every sample regardless of relevance. Structurally the same shape as evaluating every triangle against every pixel before visibility buffers. The absence of CompetitiveSelection gating in the gradient path |
| Fixed-interval evaluation | Brute-force assessment: every metric evaluated at every checkpoint regardless of what parameters changed. The absence of Signal\<ConvergenceChange\> + change-impact analysis |
| Flat parameter training | All parameters trained at the same fidelity from step 0. The absence of QualityHierarchy + TraversalEngine in the gradient path |
| Uniform gradient budget | All parameter groups receive equal gradient compute regardless of learning need. The absence of FederatedBudget\<GradientCompute\> |
| Layer freezing | Binary ConvergenceGovernor (per-group) without graduated states, budget reallocation, or reactivation |
| Curriculum learning | QualityHierarchy over data only, without hierarchy over parameters or budget governance |

---

## 3. Why the Pattern Is Hard to See

### 3.1 Vocabulary as Cognitive Lens

The domain-specific vocabulary that makes abstraction collapse difficult to detect is not arbitrary. It evolved because the people solving each problem came from that domain. Graphics engineers built rendering systems and named their concepts in rendering vocabulary. Network engineers built routing protocols and named their concepts in networking vocabulary. Machine learning researchers built training frameworks and named their concepts in statistical learning vocabulary.

Each vocabulary is internally coherent and useful within its domain. The side effect is that vocabulary creates cognitive boundaries. A graphics engineer who thinks in terms of "visibility buffers" and "deferred shading" does not necessarily recognize that these are instances of the same evaluate-then-actuate pattern that governs Kubernetes pod scheduling. A machine learning researcher who thinks in terms of "attention" and "softmax" does not necessarily recognize that the attention mechanism shares structure with the competitive selection pass that determines which triangle owns each pixel in the visibility buffer.

The vocabulary focuses attention on domain-specific details and tends to hide the structural similarity to other domains. Recognizing the similarity requires deliberately stripping the domain words and looking at what remains.

### 3.2 The Specialization Trap

Software engineering culture rewards specialization. "Rendering engineer" and "systems engineer" and "ML engineer" are different job titles, different conference communities, different publication venues. Solutions published at SIGGRAPH use rendering vocabulary. Solutions published at NSDI use networking vocabulary. Solutions published at NeurIPS use ML vocabulary. Cross-domain references happen, but they tend to happen through analogy ("attention is like a soft dictionary lookup") rather than through explicit identification ("attention shares structure with competitive selection under a specific scoring function").

Analogy preserves the domain boundary. Explicit identification reduces it. The latter is operationally more useful but is less common in practice, because the incentives of specialized publication venues do not reward it.

### 3.3 The Path I Took

DAC was not developed by a rendering engineer, a networking engineer, or an ML researcher. My background is security and Linux systems engineering and automation. The methodology grew out of a practical problem: I was working across three automation domains (ETL, SOAR, configuration management) that the relevant communities already knew were expressible as workloads over a streaming DAG, and I wanted to build a single execution engine for all three rather than maintain three vocabularies and three plugin ecosystems. That meant deliberately looking for the structural operations all three shared, separated from the domain-specific framing each tool puts on those operations. The intent from the start was to identify automation primitives that could compose into an orthogonal kernel.

The attempt to build a better execution engine began with modeling it after an OS kernel (a well-understood solution to resource governance), an OSI stack (a well-understood solution to layered abstraction), and Unix pipes (a well-understood solution to composable data flow). These were not novel intellectual ingredients. They were proven patterns from systems engineering applied to a problem I was already trying to solve.

The methodology grew from there. The first major expansion came from orchestration. As I worked through the orchestration problem more carefully, I noticed that orchestration is substrate-agnostic. A scheduler placing containers on nodes, a hypervisor placing VMs on hosts, and a game engine orchestrating its rendering, physics, and audio subsystems at per-tick granularity are all doing the same thing structurally: allocating bounded resources to consumers under priority and capacity constraints, on a clock. The substrate (container, VM, subsystem) is not the operation. That recognition pulled rendering into the analysis, because game engines run orchestration internally at very high frequency and very tight constraints, which made them a useful stress test of the same primitives.

Once rendering was on the table, the cross-domain parallels came quickly. The level-of-detail (LOD) system in rendering — deciding which geometry to draw at full fidelity, which to draw at reduced fidelity, and which to skip entirely based on what the camera can see — looked structurally similar to OSPF's link-state propagation, where each node decides which neighbors to prioritize having synchronized state with based on topology and route value. Both are answering "what should I prioritize knowing or computing right now, given a bounded budget and a partial view of the system?" That parallel forced the addition of QualityHierarchy (the LOD tree, the OSPF area hierarchy), TraversalEngine (the budget-bounded walk that visits the prioritized subset), and CompetitiveSelection (which triangle owns each pixel; which path wins each route; structurally the same shape).

That sequence — orchestration → rendering → routing — is also where the methodology stopped feeling like ad hoc cross-domain analogy and started feeling like a process worth naming. I kept asking the same question of each new domain ("how does this *really* work?") and kept getting answers that mapped onto operations I had already named. At some point the abstractions had grown enough teeth that they were worth writing down. That is what produced this paper. DAC is the name I gave, retroactively, to a process I had been doing for some time without naming it.

The path that led to DAC was not "an outsider noticed something insiders missed." Many of the individual cross-domain observations in this paper have been made before by people in those domains. ECS-as-database has been observed in game-development circles. Attention-as-kernel-lookup is in the original Transformer paper's framing. Backpropagation-as-message-passing is the standard view in graph-based autodiff. Configuration management as fixed-point convergence has been formalized in academic work on Puppet and Chef. What I did was apply the same vocabulary-stripping process across enough domains that the pattern of patterns became visible, and then specify and verify the resulting primitive set formally. The contribution is the synthesis and the verification, not the individual observations.

The reason I was the one to do this synthesis, rather than someone with deeper domain expertise in any single area, is contingent. I started with a goal (a single kernel for three automation domains) that required finding shared structure, and I kept asking how each new domain I encountered actually worked rather than accepting its vocabulary at face value. That made the cross-domain similarities easier to notice, but a domain expert with the same goal would likely have arrived at a similar set. I do not claim that domain expertise prevented others from making this synthesis; I only describe the path I took.

---

## 4. The Competitive Selection Family

### 4.1 The Overcounting Problem

In the original formulation of the abstraction primitive set, a single primitive called "SlotArbitrationPass" was mapped to over seventeen domain patterns: visibility buffers, attention, pod scheduling, market order matching, alert triage, beam search, leader election, broad-phase collision, BGP path selection, speculative decoding, audio priority, and more. When one primitive absorbs that many structurally different operations, it raises a legitimate concern: is the primitive defined so broadly ("anything where something competes for something") that it approaches the Turing Tarpit the methodology claims to avoid?

The answer is that there is genuine shared structure here, but it is a family of related primitives rather than a single primitive. The shared structure is: given a set of output positions (seats) and a set of candidates, evaluate each candidate against each seat using a scoring function, and allocate candidates to seats based on the scores. What differs across the family members, and what matters operationally, is the selection mechanism.

### 4.2 Three Selection Modes

The Competitive Selection family decomposes into three distinct selection modes that share the scoring interface but differ in their allocation semantics:

Hard Selection (argmax): exactly one winner per seat. The candidate with the highest score takes the seat exclusively. All other candidates receive nothing. This is the mode used in pixel ownership (visibility buffer), leader election (RAFT), pod scheduling, market order matching, and BGP path selection. The key property is mutual exclusion: a seat is owned by exactly one candidate.

Soft Selection (softmax/weighted): every candidate contributes to every seat in proportion to its score. There is no single winner. The output for each seat is a weighted combination of all candidates. This is the mode used in transformer attention [2]. The key property is proportional allocation: candidates share seats continuously rather than claiming them exclusively.

Ranked Selection (top-k): the top K candidates by score are all allocated. This is the mode used in beam search, audio channel priority, network update priority, and alert triage. The key property is bounded multiplicity: multiple winners are allowed, but the count is capped.

### 4.3 What Is Actually Shared

All three modes require a scoring function that evaluates candidate-seat affinity. All three produce an AllocationRecord that maps seats to their allocated candidates. All three compose with ActuationPass (which operates on the allocation result) and with Budget\<U\> (which constrains the total allocation). All three can be governed by the same audit and observability infrastructure.

The differences (hard vs. soft vs. ranked) are parameterizations of the selection mechanism. This is the same kind of parameterization seen in other primitive families: a sorting algorithm parameterized by its comparison function is one algorithm, not several. There is a defensible objection here, however, which is that argmax and softmax are not just different parameterizations but mathematically different operations (one is non-differentiable and produces a point; the other is differentiable and produces a distribution). The decision to treat them as modes of the same family rather than as distinct primitives reflects a judgment that the shared scoring interface and shared allocation-record output are the operationally important features, while the differentiability and output type are implementation properties of specific scoring functions. A reviewer could reasonably take the opposite view.

The trait signature in Section 2.3 accommodates all three modes through the AllocationRecord return type: hard selection returns a one-to-one map, soft selection returns a weighted distribution, and ranked selection returns a one-to-many map with a bounded fan-out.

### 4.4 What Is Not Shared

Two of the patterns originally mapped to this family deserve separate scrutiny.

Anomaly detection (described in some formulations as "not winning the normal slot") is a stretch. Anomaly detection is better characterized as a Reduction (compute a distance metric from a reference distribution) followed by a threshold comparison. Forcing it into the competitive selection frame adds vocabulary without adding structural insight.

Broad-phase collision detection is often described as "AABB overlaps competing for pair-slots," but it is more precisely a spatial partitioning and filtering operation. The competitive selection mapping is defensible (candidate pairs do compete for a limited processing budget in a real-time physics pipeline), but it is a weaker match than pixel ownership or attention, where the competitive structure is intrinsic rather than imposed by the resource constraint. For this reason, the collapse table in Section 2.4 classifies broad-phase collision as Reduction + spatial partitioning rather than CompetitiveSelection.

The revised count: CompetitiveSelection with its three modes accounts for roughly fifteen of the seventeen original mappings, with two entries better classified as compositions of other primitives.

---

## 5. DAC as a Generative and Implementation Methodology

### 5.1 Beyond Analysis

The methodology described in Section 1.2 is framed as analytical: take existing domains, strip vocabulary, find similarities. The same process works as a generative methodology (a tool for engineering solutions to problems that resist solution in their native vocabulary) and as an implementation methodology (a tool for turning any described computation into a build plan).

The generative application works as follows. When confronted with a problem that appears hard in its domain:

1. Strip the domain vocabulary from the problem statement. Describe what the problem actually requires structurally, without using any domain-specific terms.

2. Map the stripped problem to the abstraction primitive set. Does the structure match any known primitive or composition of primitives?

3. If the mapping succeeds, the problem may inherit a solution from whichever domain has already worked through it. The "hard problem" was hard at least partly because its domain vocabulary was hiding the structural similarity to a problem that had been solved.

4. If the mapping fails, the structure is genuinely novel. That is also useful: it identifies where original work is needed, and rules out the hypothesis that the difficulty was vocabulary-induced.

The implementation application extends this. Any computation described in domain-specific notation (a mathematical formula, a protocol diagram, an algorithm in pseudocode) can be decomposed into a build plan by stripping the notation and mapping each step to the primitive set. The notation describes the relationship between inputs and outputs. The DAC decomposition describes the computational steps a machine actually executes, which are some combination of traversal, transformation, reduction, propagation, budgeting, sampling, and gating.

### 5.2 Case Study: LeanFormer

LeanFormer is a transformer architecture designed through DAC. It was also a deliberate test of the methodology on a domain I do not have research-level expertise in. My background is security and Linux systems engineering and automation; I am not an ML researcher and do not have research-level training in transformer architecture or model training. LeanFormer was the first project where I applied DAC to a domain I did not already understand at the implementation level. The hypothesis being tested was simple: if DAC is doing real structural work, a non-expert applying it carefully should be able to produce a workable architecture for problems the ML literature treats as open. If DAC requires deep domain expertise to apply usefully, then the methodology is just dressed-up domain expertise and is less interesting as an independent process.

Rather than starting from the ML literature and making incremental improvements to existing architectures, the design started by stripping the ML vocabulary from six open problems in neural network design and mapping each to the abstraction primitive set. In every case, the stripped problem turned out to share structure with a problem from systems engineering. The first five problems address the model architecture. The sixth addresses the training process itself.

The arc from initial DAC decomposition to 204M-parameter measured results took approximately three weeks. The initial proof-of-concept (7.5M parameters, 4 layers) was designed and implemented in 24 hours using DAC as the design methodology and Claude Code as the implementation agent. The architect provided the DAC decomposition, the cross-domain mappings, and the verification discipline. Claude Code produced all implementation code. No deep ML implementation experience was required on either side of this collaboration; the structural solutions were systems engineering solutions identified through vocabulary stripping. The domain functions (loss formulation, optimizer choice, scoring functions) still required ML knowledge, and those parts of the work involved more iteration and external reference than the structural decomposition did.

Subsequent scale-up to 39M parameters (76M dense equivalent, trained on 500K OpenWebText samples) covered 119 tests. A 204M parameter model (805M dense equivalent) was trained with the full governed pipeline on a reasoning corpus for 7,228 optimizer steps (one epoch) on an NVIDIA L4 GPU. The architectural results in Sections 5.3-5.7 are from the 39M run. The training governance results in Section 5.8 include both the 4.8M preliminary run and the 204M scale run. The 204M run is the first measurement of all sixteen primitives composing under real training conditions at a scale where parameter group ratios are representative (L0 at approximately 41.5% of parameters, compared to approximately 86% at 4.8M where the embedding table dominates).

LeanFormer is not presented as a competitive language model. It is presented as a worked example of what DAC produces when applied to a domain, and as the result of the experiment described above (a non-expert applying DAC to an unfamiliar domain). The 204M run has best validation perplexity of 57.6, which is well off what comparable-size models reach on similar corpora. The measurement on offer is governance machinery: whether the sixteen primitives compose without invariant violations through a real training run, not whether the resulting model fits language well. The current implementation has known limitations: modest scale, single-epoch training, no comparison to an ungoverned baseline, two subsystems (adaptive depth and tiered sampling) that did not produce useful signal at this scale. These are identified as concrete next steps in Section 9.5.

A candid note on what the LeanFormer experiment does and does not show. The structural decomposition produced an architecture that runs end-to-end with all governance invariants holding through a real training run. That is consistent with DAC doing real structural work. The model's poor language modeling performance reflects, among other things, my limited ML expertise in the domain functions (loss, optimizer, regularization, training recipe, data curation), which DAC does not supply. The honest summary is: structural skeleton, yes; competitive language model, not yet. Whether DAC is actually doing the structural work I attribute to it, or whether the same architecture would have been arrived at by reading enough papers without the methodology, is a question the experiment as run cannot definitively answer. What can be said is that the route I took — vocabulary stripping, primitive mapping, composition — produced an architecture that does not violate any of its specified invariants under load, and that I do not believe I would have arrived at the same architecture in three weeks without the methodology.

### 5.3 Problem 1: Parameter Inefficiency

In ML vocabulary: dense weight matrices waste parameters because most weights contribute minimally to the output. Pruning, quantization, and low-rank approximation are active research areas with large bodies of literature.

Stripped of vocabulary: a storage system allocates fixed-size blocks for every record regardless of the record's actual content size. Most blocks are mostly empty. This is the fragmentation problem in file systems, with a long-standing solution: variable-size allocation under a budget constraint.

DAC decomposition: replace dense matrices with low-rank factorizations. Each weight matrix W becomes a product of two smaller matrices (down-projection and up-projection) constrained by Budget\<Parameters\> to use only a fraction of the original parameter count. The rank becomes the budget knob. The domain function (what the weight matrix computes) is unchanged. The governance (how many parameters it uses) is now explicit and tunable.

Result at 39M parameters: the model has 3.9x fewer parameters than a dense model with the same hidden dimensions. This is a parameter ratio. It is not a claim that the low-rank model has equivalent representational capacity to a 76M dense model; representational capacity differs between the two architectures and would require separate measurement to compare. All four efficiency mechanisms (low-rank weights, sparse attention, gated feed-forward, adaptive depth) operate simultaneously without mutual interference.

### 5.4 Problem 2: Attention Cost

In ML vocabulary: self-attention is O(n squared) in sequence length, making long-context inference expensive. Flash attention, sparse attention, and linear attention are competing approaches with different tradeoffs.

Stripped of vocabulary: a selection system evaluates every candidate against every output position, even when most candidates are irrelevant to most positions. This is the brute-force rendering problem: evaluating every triangle against every pixel. The rendering community solved it with a two-pass architecture. A cheap coarse pass identifies which candidates are relevant. An expensive fine pass evaluates only the relevant ones.

DAC decomposition: replace single-pass dense attention with a two-pass sparse attention pipeline. The first pass is a lightweight scoring pass (CompetitiveSelection in ranked mode) that identifies the top-k relevant key-value pairs per query. The second pass computes full attention (CompetitiveSelection in soft mode) over only the selected candidates. The structure is the same shape as the visibility buffer pipeline in rendering: coarse culling followed by fine evaluation.

Result at 39M parameters: 88% attention sparsity. The model evaluates 12% of key-value interactions per query position, with the sparse selection pass routing attention to the relevant subset. The gated feed-forward network achieves 80% sparsity through the same principle applied to the MLP layers: a cheap gate determines which neurons fire, and only active neurons are computed.

### 5.5 Problem 3: Catastrophic Forgetting

In ML vocabulary: when a neural network learns new information, it overwrites previously learned information because the same parameters encode both old and new knowledge. This is an open research problem with a large literature on continual learning, elastic weight consolidation, and progressive networks.

Stripped of vocabulary: a shared mutable storage system where writes to encode new content destroy existing content, because the storage addressing conflates the retrieval index with the stored content and uses dense rather than sparse encoding. This is the write-conflict problem in shared mutable state, addressed in systems engineering by separating immutable content from mutable indices.

DAC decomposition: separate the retrieval index from the stored content. Use a fixed compact base for general computation (the frozen base model weights). Encode individual records as sparse, independently-addressable deltas over the base (low-rank delta matrices, similar in shape to LoRA [6] adapters but dynamically loaded and unloaded rather than statically merged). Govern delta allocation with a ResourceRegistry that enforces non-overlapping address ranges. Route queries to relevant deltas using a cheap CompetitiveSelection (ranked) pass before the expensive computation.

The primitive composition:

| LeanFormer Component | Abstraction Primitive Composition |
|---|---|
| Frozen base weights | The immutable foundation |
| Belief delta | Budget\<Parameters\> + ResourceRegistry\<BeliefID, DeltaWeights\> with atomic write boundary (atomicity is a runtime requirement) |
| Routing layer | CompetitiveSelection (ranked) over belief deltas |
| Combined inference | Base computation + active deltas applied additively |
| Belief removal | Delete from ResourceRegistry, restore base weights bit-for-bit |

Result at 39M parameters: 84% belief injection success across 100 beliefs. Bit-for-bit base weight restoration on belief removal across all 100 beliefs and 406 base tensors. Within the LeanFormer setup, the deltas are non-overlapping by construction, and removing a delta restores the prior state exactly. This addresses forgetting of the base weights specifically; if multiple deltas were trained jointly with shared parameters, interference between deltas could still occur. The architecture as built avoids that case by enforcing non-overlapping delta address ranges through the ResourceRegistry. A baseline comparison against full fine-tuning, vanilla LoRA, and RAG-based knowledge stores has not been performed and is identified as future work in Section 9.5.

### 5.6 Problem 4: Knowledge Composition

In ML vocabulary: how does a model combine multiple pieces of knowledge to produce a coherent answer? Mechanistic interpretability work [12] suggests that knowledge is distributed across attention heads in ways that are difficult to compose deliberately.

Stripped of vocabulary: a routing system needs to combine information from multiple address ranges into a single output, where the relevance of each range to the query is determined by a similarity comparison.

DAC decomposition: a semantic routing layer that uses CompetitiveSelection (ranked) to identify which belief deltas are relevant to a given input, scored by cosine similarity between the input embedding and each delta's address vector. Selected deltas are activated for the forward pass. Unselected deltas remain dormant.

Result at 39M parameters: 86% routing accuracy on a held-out test set (chance baseline approximately 20% on the evaluation setup, which clusters beliefs into 5 categories rather than treating each of 100 beliefs as a separate routing target; the 4.3x ratio reflects this clustering). 64% of beliefs improve simultaneously without interfering with each other.

### 5.7 Problem 5: Confabulation

In ML vocabulary: language models produce outputs that look fluent and confident but are factually wrong, and the model has no way to signal that it is uncertain.

Stripped of vocabulary: a process that produces an output without converging requires a way to signal that it has not converged, separately from the output itself.

DAC decomposition: an adaptive depth mechanism with a per-layer ConvergenceGovernor. At each layer, the governor evaluates whether further computation is likely to change the output. If converged, exit early. If full depth is reached without convergence, output a confidence flag indicating non-convergence. The flag is the structural correlate of "I don't know."

Result at 39M parameters: mean exit depth of 11.3/12, with two distinct exit depths used. This is a partial result; the model is not yet using the adaptive depth mechanism aggressively, which suggests the exit threshold needs tuning or that explicit layer-dropping training is needed to teach the model to use early exits. At 204M, the mechanism did not activate at all (20/20 throughout). The architectural primitive composition is in place; the domain function (the convergence criterion) needs further work.

### 5.8 Problem 6: Training Process Inefficiency

In ML vocabulary: standard training is uniform across parameters, samples, and time. Every parameter receives gradient signal from every sample. Every metric is evaluated at fixed intervals. This is computationally expensive and treats all parameters as equally valuable to update at all times.

Stripped of vocabulary: a resource-allocation system that distributes a finite budget (compute) uniformly across consumers (parameter groups) regardless of which consumers are still actively producing useful work. This is precisely what Kubernetes pod scheduling, Linux process scheduling, and rendering LOD systems address: resource governance with variable allocation based on observed need.

DAC decomposition: apply the same primitive set that governs the model's architecture to govern the training process itself. The full composition includes:

- FederatedBudget\<GradientCompute\>: a master gradient compute budget subdivided per parameter group, with per-group allocation that adapts to observed convergence state.
- QualityHierarchy over parameters: parameter groups organized into levels (L0 structural, L1 representational, L2 refinement, L3 specialization). Coarse levels train first; finer levels activate via convergence signal.
- ConvergenceGovernor (per-group): each parameter group has its own four-state governor (PENDING → ACTIVE → COOLING → CONVERGED, with AWAKENED for reactivation after perturbation).
- CompetitiveSelection (ranked) over gradients: the gradient router scores each (sample, parameter group) pair and gates the gradient signal to only the most relevant pairs.
- Signal\<GroupConverged\>: when a group converges, downstream subsystems (evaluation, knowledge forging) receive the signal and act.
- AuditSink: every governance decision (budget allocation, state transition, hierarchy activation, gradient routing update) is logged to a hash-chained audit log with SHA-256 integrity.

Preliminary result at 4.8M parameters (300-step validation run): all governance components activated and composed correctly. Per-group convergence governors produced 15 state transitions across 8 parameter groups, with 7 reaching CONVERGED state. The quality hierarchy activated in the predicted order: L0 (structural) then L1 (representational) then L2 (refinement) then L3 (specialization), all via convergence signal with no emergency activation required. The FederatedBudget invariant (sum of allocations <= master budget) held for all 300 steps with zero violations. The gradient router achieved 15.4% selectivity post-warmup. The SHA-256-chained audit log verified across all 300 records. The change-triggered evaluation pipeline fired 7 targeted evaluations on convergence signals. Final loss was within +2.4% of baseline, with the gap narrowing throughout training (from +3.17 at step 50 to +0.15 at step 299). At 4.8M parameters, the embedding table dominates (~86% of parameters), so this run primarily tested whether the governance machinery could be wired up correctly, not whether it would produce different behavior at scale.

Scale validation at 204M parameters (805M dense equivalent, 7,228 optimizer steps, one epoch on a reasoning corpus, NVIDIA L4 GPU, 140.9 hours wall clock): the question being answered here is "do the governance primitives compose correctly at a scale where parameter group ratios are representative, and do all invariants hold across thousands of training steps?" The language modeling metrics are reported for completeness but are not the subject of evaluation. The governance results:

The FederatedBudget invariant (sum of allocations <= 1.0) held for all 722 audit records with zero violations. Budget adapted dynamically throughout training: L0 groups started at 0.25 each, converged groups dropped to as low as 0.012, and active groups received up to 0.40 of the total budget.

The convergence governor four-state machine produced 18 total state transitions across 8 parameter groups, all valid: PENDING to ACTIVE (3 transitions), ACTIVE to COOLING (8), COOLING to CONVERGED (4), and COOLING to ACTIVE (1 regression). No states were skipped. Final states: embeddings COOLING, attention_routing CONVERGED, gates CONVERGED, layer_norms CONVERGED, attention_output COOLING, ff_projections COOLING, output_head COOLING, exit_classifier CONVERGED.

The SHA-256 hash-chained audit log maintained integrity across all 722 records from step 10 to step 7,220, with zero chain breaks.

Best validation perplexity reached 57.6 at step 2,000, with train loss declining from 10.39 to 1.64 across the full run. Severe overfitting occurred after step 2,000 (validation perplexity rose from 57.6 to 1,463.9 by step 7,228), expected behavior from single-epoch training with limited regularization (dropout 0.1 only). All governance invariants held throughout the overfit phase: the budget was never exceeded, the hierarchy ordering was maintained, the convergence governor never skipped a state, and the audit chain was never broken. Governance correctness held independently of generalization quality.

Orthogonal capacity measurement confirmed 53,760 available dimensions across 20 layers (2,688 per layer), with a theoretical maximum of 3,360 rank-16 knowledge deltas. This was measured on two independent machines (NVIDIA L4 on GCP, RTX 3060 locally) with identical results.

Two governed subsystems did not produce meaningful signal at this scale and configuration. Adaptive depth remained at 20/20 (all layers used) throughout the entire run, meaning the exit classifiers never learned to route samples to early exits. Tiered sampling scored all samples as "Failing" at initialization and was never re-scored, rendering it effectively random. Both require further work in future training runs.

A reframe of what this run shows. The "primitives compose correctly" claim requires care. What was directly measured is that each primitive's local invariant held throughout the run (budget never exceeded, governor never skipped a state, audit chain never broke). This is consistent with the primitives composing correctly, but it does not by itself establish emergent system-level guarantees that go beyond the per-primitive invariants. The composition-level invariants verified in TLA+ (Appendix C, particularly the GovernedTrainingPipeline specification's thirteen simultaneous invariants) do address composition properties; the empirical run confirms those invariants are not violated in practice at 204M parameters and 7,228 steps. The honest framing is "every primitive's invariants held empirically, and the TLA+ verification covered the composition-level invariants formally," not "the run by itself proves correct composition."

### 5.8.1 The B=0 Initialization Episode

The most informative result from the 204M training run was a failure mode that DAC's vocabulary-stripping process turned out to apply to.

LeanFormer's low-rank layers initialize B matrices to zero (following standard LoRA practice), which means all parameter groups begin with near-zero gradient flow regardless of whether they have received meaningful training signal. The convergence governors correctly detected low gradient EMA and transitioned through the state machine as specified: ACTIVE to COOLING after the configured cooling window. This produced hierarchy activations at steps 200 (L1) and 400 (L2), both at round-number intervals aligned with the cooling window configuration. Loss remained flat at 10.388 through both activations. Actual training progress began only when the output head activated at L2 and introduced significant gradient flow through the network.

The L3 activation at step 2,773 was qualitatively different. It occurred at a non-round step number, after the attention_output and ff_projections groups (L1 parameters) had received 2,300+ steps of real gradient flow following the output head's activation at step 400. The convergence governor's decision to activate L3 was based on actual post-learning convergence in those groups, not on a calibration artifact from initialization.

The COOLING to ACTIVE regression observed in the attention_output group illustrates the existing state machine handling part of this case. This group was prematurely cooled by the B=0 artifact, then reactivated when real gradient flow from the output head pushed its EMA above the cooling threshold. The four-state machine self-corrected for this group via the COOLING-to-ACTIVE transition path. The hierarchy activations at L1 and L2, however, were not self-corrected; once those levels activated, they stayed activated, even though the activation was driven by initialization rather than real convergence. So the existing state machine handled the bug for one parameter group but not for the hierarchy activation overall.

Applying the vocabulary-stripping process to this failure described it as an observational degeneracy: two qualitatively different gradient trajectories (cold start and post-learning convergence) produced the same low-magnitude reading, and the governor could not distinguish them from the magnitude alone. The same shape shows up elsewhere in the collapse table. In rendering, a pixel with zero color could be background or a black surface; the depth buffer adds a signal to break the degeneracy. In networking, a silent node could be idle or crashed; heartbeat protocols add a liveness signal. In distributed consensus, a node that has not voted could be slow or partitioned; timeouts add a temporal boundary. In each case, the fix is the same shape: add a second signal that disambiguates the measurement.

The fix here follows the same shape: a phase-aware ConvergenceGovernor that tracks whether gradient magnitude has ever exceeded the threshold, classifying gradient trajectories into qualitative phases (COLD, WARMING, ACTIVE_LEARNING, DECLINING). The ACTIVE-to-COOLING transition requires the gradient phase to be ACTIVE_LEARNING or DECLINING, never COLD. This NoCoolingFromCold invariant was specified in TLA+ and verified by TLC across approximately 18.6 million states. A specification without phase awareness produces a counterexample matching the exact failure observed in the training run in just 2 states.

Two honest observations about this episode. First, the underlying bug class (zero-as-ambiguous-signal, also known as the sentinel-value problem) is well-understood, and the fix (a latching phase tracker) is a standard hysteresis pattern. The contribution of DAC here is not the discovery of a new engineering technique. It is that the vocabulary-stripping process surfaced the connection to the cross-domain pattern, which gave a direct route from "this is the kind of bug it is" to "this is the kind of fix that works." A skilled control engineer or ML researcher could have arrived at the same fix without the methodology; the methodology made the recognition faster for someone whose training was not in either of those areas.

Second, the narrative structure of "the methodology applied to its own failure" is satisfying but not by itself evidence of generative power. The vocabulary stripping happened after the bug was understood; there is no controlled comparison showing DAC produced the diagnosis faster than direct ML reasoning would have. What can be claimed is that the diagnosis route was available, was used, produced a fix, and the fix has been formally verified.

This is a machinery validation, not a scale validation. The efficiency gains from governed training (estimated 50-70% gradient compute reduction from hierarchical activation and gradient routing) depend on model size and training duration. At 204M parameters, the governance machinery correctly identified which parameter groups to stop training and dynamically reallocated budget to groups still learning. However, the compute savings were not realized as wall-clock improvement in this run because the implementation zeros gradients for converged groups after computation rather than skipping the backward pass entirely. Implementing actual compute skipping for governed groups (setting requires_grad=False with proper autograd graph pruning) is an engineering optimization for future runs.

### 5.9 Summary of Results

Architecture results are from the 39M parameter model (76M dense equivalent) trained on 500K OpenWebText samples for one epoch. Training governance results include both the 4.8M preliminary run (300 steps, WikiText-2) and the 204M scale run (7,228 steps, reasoning corpus, NVIDIA L4).

Result columns are labeled "Measured" when the value is observed but no baseline comparison was performed, and "Validated" only where a specific invariant was checked or a comparison was made.

| Mechanism | Metric | Result | Status |
|-----------|--------|--------|--------|
| Low-rank parameter ratio | Parameter ratio vs. dense equivalent | 3.9x (39M); 3.94x (204M) | Measured (no capacity-equivalence baseline) |
| Sparse attention | Attention sparsity | 88% | Measured (39M) |
| Gated feed-forward | FF sparsity | 80% | Measured (39M) |
| Belief injection | Success rate | 84% across 100 beliefs | Measured (39M, no baseline) |
| Semantic routing | Routing accuracy | 86%, 5-way clustering, chance ~20% | Measured (39M) |
| Belief coexistence | Simultaneous improvement | 64% of 100 beliefs | Measured (39M) |
| Base weight restoration | Bit-for-bit fidelity | Exact across all tensors | Validated (39M) |
| Base weight immutability | Tensor integrity | 406 tensors verified | Validated (39M) |
| Adaptive depth | Mean exit depth | 11.3/12, 2 unique depths (39M); 20/20 not activated (204M) | Partial |
| Governed training | Budget invariant | 0 violations / 300 steps (4.8M); 0 violations / 722 records (204M) | Validated |
| Governed training | Audit chain integrity | 300 records verified (4.8M); 722 records, SHA-256 chain intact (204M) | Validated |
| Convergence governors | State transitions | 15 transitions, 7/8 converged (4.8M); 18 transitions, all valid, no skips (204M) | Validated |
| Hierarchy activation | Coarse-to-fine ordering | L0 then L1 then L2 then L3 via convergence signal | Validated (4.8M, 204M) |
| Hierarchy activation | L3 genuine convergence | Step 2,773 (non-round, after 2,300+ steps of real gradient flow) | Measured (204M) |
| Hierarchy activation | L1/L2 timing | Steps 200/400 (B=0 initialization artifact, disclosed) | Disclosed artifact |
| B=0 diagnosis | DAC applied to own failure | Observational degeneracy identified, phase-aware fix formally verified | Measured (no controlled comparison) |
| Budget reallocation | Dynamic budget shifts | Converged groups drop to 0.012, active get up to 0.40 | Measured (204M) |
| Gradient routing | Gate activation | 0.34-0.69 gate density (non-degenerate) | Measured (204M) |
| Language modeling | Best val PPL | 57.6 at step 2,000 (204M) | Measured (not LM-competitive) |
| Language modeling | Final val PPL | 1,463.9 (overfit, single epoch, governance invariants held throughout) | Expected |
| Orthogonal capacity | Available dims | 53,760 (3,360 max rank-16 deltas), confirmed on two machines | Measured (204M) |
| Tiered sampling | Tier distribution | All samples scored as Failing (scored pre-training only) | Not functional |
| Training time | Wall clock | 140.9h on NVIDIA L4 | Measured (204M) |

The two partial results (adaptive depth and tiered sampling) are not architectural failures. Adaptive depth requires either a lower exit threshold or explicit layer-dropping training to learn meaningful early-exit behavior. Tiered sampling scored all samples at initialization when the model could not yet evaluate difficulty; periodic re-scoring during training would enable the governed data pipeline.

The training governance results show that the sixteen primitives' invariants held empirically at 204M parameters across 7,228 training steps, including through the severe overfitting phase after step 2,000. The B=0 observational degeneracy was diagnosed using DAC's vocabulary-stripping process and the fix was formally verified in TLA+. The language modeling perplexity is not competitive at this scale and training duration; the claim is governance machinery, not language modeling performance. Baseline comparisons against ungoverned training, vanilla LoRA, full fine-tuning, and RAG-style knowledge stores have not been performed. They are identified as future work in Section 9.5.

### 5.10 What DAC Did Not Provide

DAC does not design domain functions. The specific choice of low-rank factorization for the deltas, the cosine similarity metric for the routing network, the exit-threshold tuning for adaptive depth, the choice of loss function and optimizer: these are domain-specific engineering decisions that require ML expertise. The same applies to the training governance system: the specific scoring function for the gradient router, the convergence thresholds for per-group governors, the budget allocation policy, the hierarchy level boundaries, and the sample difficulty thresholds are all domain functions that DAC's structural skeleton does not supply. DAC provided the structural skeleton. Domain knowledge filled in the scoring functions, the loss formulations, the training recipes, and the governance thresholds.

This is the same structure/function separation described throughout the paper. The abstraction primitives provide structure (how data flows, how resources are allocated, how convergence is detected, how mutations are audited). The domain provides function (what computation to apply at each step). Neither replaces the other.

---

## 6. Extended Case Study: The AI Domain

### 6.1 The Mapping to AI

The most far-reaching application of DAC in this paper is to machine learning. The transformer architecture [2], the foundation of every modern large language model, can be expressed as a composition of primitives that the methodology had identified before the AI domain was examined.

Attention is soft competitive selection. The transformer's multi-head attention mechanism computes, for each query token, a weighted combination of value vectors where the weights are determined by similarity between the query and key vectors. This is CompetitiveSelection in soft mode: the dot-product similarity is the scoring function, softmax produces the weighted allocation, and each attention head is one selection pass. Multi-head attention is parallel selection passes over the same candidates with different scoring functions.

Backpropagation is a single-pass variant of PropagationPass. The forward pass builds a directed acyclic computation graph. The backward pass propagates gradient values from the loss node backward through the graph in reverse topological order, applying the chain rule at each node to compute local gradients.

A precision caveat is necessary here. Backpropagation is a single reverse pass on a DAG. It does not require iterative relaxation to a fixed point the way Bellman-Ford does on graphs with cycles [3]. The mapping is not "backpropagation equals Bellman-Ford." The mapping is that both backpropagation and Bellman-Ford are instances of PropagationPass: the shared structural primitive of message passing over a graph toward a consistent state. The instantiation parameters differ: graph topology (DAG vs. cyclic), traversal order (single reverse pass vs. iterative relaxation), message function (Jacobians vs. edge weights), and termination condition (one pass vs. convergence). The primitive is the message-passing structure. The domain determines the graph, the messages, and the termination condition.

This mapping is the weakest of the three major AI mappings. The shared structure (messages flowing through a graph) is real but broad enough that calling both instances of the same primitive carries some risk of overclaiming. The honest framing is that PropagationPass is a family of operations parameterized by topology and termination, and backpropagation is one member of that family. The same caveat applies to attention and CompetitiveSelection (soft): argmax and softmax are mathematically different operations, and treating them as modes of the same primitive is a judgment call that can be defensibly made either way (Section 4.3).

Speculative decoding is the two-level fidelity architecture. A fast small model generates candidate tokens (coarse pass) [8]. A large model verifies them (fine selection). Accepted tokens are actuated. Rejected tokens are discarded. This is the same shape as the rendering fidelity pipeline: coarse traversal reduces the candidate set, fine selection determines winners, actuation evaluates only winners.

### 6.2 Implications

These are structural mappings, with the precision caveats noted above. The implication is that any optimization of CompetitiveSelection, discovered in any domain, is a candidate for transfer to attention. Any optimization of PropagationPass, discovered in any domain, is a candidate for transfer to gradient computation. The shared primitive vocabulary creates a channel for cross-domain optimization transfer that is harder to see when each domain maintains its own vocabulary.

Linear attention [10], sparse attention, and flash attention can be read as optimizations of CompetitiveSelection in soft mode. Convergence governors and temporal amortization, developed for real-time lighting, are candidates for application to training loop optimization. The primitive vocabulary makes these connections visible. The domain vocabulary tends to hide them.

A candid assessment of validation status: cross-domain optimization transfer is the most useful claim DAC makes, and the evidence so far runs in one direction (systems engineering optima → ML). The LeanFormer architecture (Sections 5.3-5.7) shows five systems engineering solutions applied to model design problems. The targeted training system (Section 5.8) shows the same primitive set applied to the training process itself, validated at 204M parameters across 7,228 training steps. The B=0 observational degeneracy diagnosis (Section 5.8.1) shows DAC's vocabulary-stripping process applied to a failure in a DAC-governed system. All three demonstrations move structural insights from systems engineering into ML. The methodology would be more robustly demonstrated by a transfer in the opposite direction: an optimization that originated in ML, applied to (say) rendering or consensus, with measurable improvement. That has not yet been done and is identified as future work in Section 9.5.

---

## 7. The DAG Workload Pattern

### 7.1 Four Domains, One Pattern

The most practically actionable cross-domain mapping in this work is the identification that CI/CD pipelines, ETL workflows, configuration management playbooks, and SOAR automation playbooks share a common shape. All four are:

A directed acyclic graph of tasks with typed dependencies, executed under budget constraints (compute, time, or both), with capability-gated I/O at task boundaries, converging toward a desired end state, and audit-logged for observability and compliance.

If this mapping is correct in practice, the consequence is that a single execution engine could replace Ansible, Jenkins, Airflow, and Splunk SOAR. Not by reimplementing four separate systems, but by recognizing that the underlying execution pattern is shared. The "domain-specific" part is the task function, which belongs in a sandboxed execution boundary, not in the kernel. Workflow management research [a-9] has cataloged similar patterns under different names; this section's contribution is identifying that the DAC primitive set covers the four named domains specifically.

### 7.2 Why This Pattern Was Hard to Notice

These four domains are served by different industries, different conferences, different vendor ecosystems, and different job titles. A CI/CD engineer uses Jenkins or GitHub Actions. An ETL engineer uses Airflow or dbt. A configuration management engineer uses Ansible or Puppet. A security engineer uses Splunk SOAR or Palo Alto XSOAR. Each tool has its own vocabulary, its own plugin ecosystem, its own certification program.

The vocabulary creates the market. The market creates the specialization. The specialization makes it less likely that a single practitioner sees all four tools as instances of the same shape. The DAC process here just asks: what does each tool actually compute? The answer, in every case, is: it traverses a DAG under constraints and executes tasks at each node. The task function is the part that varies.

---

## 8. Composition Patterns

### 8.1 How Domains Are Reconstructed

The point of identifying primitives is constructive: once they are named, every domain pattern can be expressed as a composition. The following table documents the reconstruction of common computational patterns from the primitive set. This is a catalog of compositions, not a formal algebra (no operators, laws, or normal forms are defined).

| Pattern | Core Primitives | Governance |
|---------|----------------|------------|
| LOD / Quality scaling | QualityHierarchy + TraversalEngine + AllocationSnapshot | Budget\<Bytes\> |
| Global illumination | PropagationPass + ConvergenceGovernor | Budget\<Microseconds\> + Budget\<TimeSlot\> + RateLimit + ConvergenceGovernor (feedback) |
| Physics simulation | Reduction + CompetitiveSelection (hard) + PropagationPass + ActuationPass | Budget\<Microseconds\> + ConvergenceGovernor |
| Network sync | PropagationPass + TraversalEngine + Reduction\<(Old, New) → ChangeRecord\> | FederatedBudget\<BytesPerSecond\> |
| Container orchestration | QualityHierarchy + TraversalEngine + ActuationPass | Budget\<CPU/Memory\> + ConvergenceGovernor |
| CI/CD pipeline | RelationshipGraph + TraversalEngine + ActuationPass | Budget\<Compute\> |
| ETL pipeline | RelationshipGraph + ActuationPass + Budget\<DestinationCapacity\> + ResourceRegistry\<Key, Record\> with atomic write boundary | Budget\<Microseconds\> |
| Configuration management | Reduction\<(Old, New) → ChangeRecord\> + ConvergenceGovernor + ActuationPass | (none) |
| SOAR playbook | RelationshipGraph + TraversalEngine + ActuationPass | Capability-gated I/O |
| Market simulation | CompetitiveSelection (hard) + PropagationPass + Budget\<U_source\> + Budget\<U_target\> + ResourceRegistry\<OrderID, Order\> with atomic transaction boundary | FederatedBudget + ConvergenceGovernor |
| ML training loop | ActuationPass + Reduction + PropagationPass | ConvergenceGovernor + Budget\<TimeSlot\> + RateLimit (feedback) |
| ML inference | ActuationPass + CompetitiveSelection (soft) + ResourceRegistry\<Key, Value\> + Budget\<CacheCapacity\> | Budget\<FLOPs\> |
| Belief delta system | Budget\<Parameters\> + ResourceRegistry + CompetitiveSelection (ranked) with atomic write boundary | AuditSink |
| Targeted training pipeline | CompetitiveSelection (ranked) + FederatedBudget\<GradientCompute\> + QualityHierarchy + ActuationPass + Signal | ConvergenceGovernorPhaseAware (per-group) + AuditSink |
| Governed data pipeline | QualityHierarchy\<SampleDifficulty\> + CompetitiveSelection (ranked) + Sampler | Budget\<SamplesPerStep\> + AuditSink |
| Change-triggered evaluation | Signal\<ConvergenceChange\> + CompetitiveSelection (ranked) + ActuationPass + Reduction | Budget\<EvalCompute\> + ConvergenceGovernor (per-metric) + AuditSink |
| Readiness-gated knowledge forge | Signal\<GroupConverged\> + ConvergenceGovernor + CompetitiveSelection (ranked) + ActuationPass + Reduction | Budget\<ForgeCompute\> + AuditSink |

In every composition above, the primitives provide the structure (how data flows, how resources are allocated, how convergence is detected, how mutations are audited). The domain provides the function (what computation to apply at each step). Several compositions require atomicity at a boundary (write boundary, transaction boundary). Atomicity is treated as a runtime requirement, not as an additional primitive (Section 2.2).

### 8.2 Structure vs. Function

The boundary between structure and function is the part of this work that requires the most care. The rendering equation is a domain function. Newton's laws are a domain function. The RAFT consensus protocol's log replication rule is a domain function. The machine learning loss function is a domain function. The attention scoring function (dot-product similarity) is a domain function.

Where the boundary gets fuzzy: the softmax inside CompetitiveSelection (soft) is a specific mathematical operation that is not part of CompetitiveSelection (hard). Treating softmax as part of the primitive's "structure" rather than as a domain-supplied function is a judgment that the differentiability and distribution-output properties are operationally important features of the soft-selection mode. That is a defensible judgment but not the only defensible one. Section 4.3 discusses the alternative.

This separation is what makes the primitive set simultaneously lean and broadly applicable. It contains no domain knowledge. It contains the execution model that recurred across the domains examined.

---

## 9. Methodology Validation

### 9.1 How to Know the Mapping Is Real

A cross-domain mapping is meaningful (not a renaming exercise) if and only if:

1. Sufficiency: Every domain pattern from step 1 can be expressed as a composition of the collapsed primitives. No residual domain-specific primitives are required.

2. Minimality: No primitive in the collapsed set can be expressed as a composition of the others. Removing any primitive leaves at least one domain pattern unexpressible.

3. Operational equivalence: A system built from the collapsed primitives produces the same outputs as the domain-specific system it replaces, under the same inputs and constraints.

4. Cross-domain transfer: An optimization discovered in one domain, when applied to the shared primitive, produces measurable improvement in other domains that use the same primitive.

Sufficiency (criterion 1) is satisfied for all twelve domains examined, with the caveat that "satisfied" means "I was able to express every domain pattern I considered as a composition," which is dependent on which patterns were considered. Operational equivalence (criterion 3) is satisfied for the domains where working implementations exist (LeanFormer demonstrates this for the ML composition; other domains have not been independently implemented from primitives in this paper). Cross-domain transfer (criterion 4) is partially demonstrated: the LeanFormer architecture (Sections 5.3-5.7), the targeted training system (Section 5.8), and the B=0 diagnosis (Section 5.8.1) all transfer structural insights from systems engineering into ML. Transfer in the opposite direction has not been demonstrated.

Minimality (criterion 2) is open. The claim that no primitive can be expressed as a composition of the others has not been formally proven. Specific candidates worth investigating: Can RateLimit be expressed as Budget\<Operations\> with periodic reset? Can Signal\<T\> be derived from AuditSink with a predicate filter? These might be legitimate standalone primitives, or they might be compositions that should be eliminated from the set. If they are compositions, the primitive count drops below sixteen. The methodology's validity does not depend on the exact count; sufficiency is the load-bearing criterion. Proving minimality (or disproving it for specific primitives) would require showing that removing each primitive creates at least one domain pattern that becomes unexpressible. That work has not been completed.

A weaker but formally established property: each primitive's invariant requires atomicity at the primitive boundary. Section 9.4 documents the systematic decomposition of each primitive into separable check-then-act steps and the TOCTOU patterns that result. This is a property of how atomic operations behave in concurrent systems generally, not a unique property of these primitives. The systematic demonstration across all sixteen primitives in the set is what's documented for the first time in this work.

### 9.2 Threats to Validity

The most significant methodological risk is confirmation bias. All twelve domains were collapsed by the same individual, and once a primitive vocabulary exists, there is a cognitive pull to force every new domain into it rather than honestly admitting when the existing primitives are insufficient.

Three properties of the development process partially mitigate this risk. First, the primitives were arrived at iteratively rather than designed to fit a target count. The starting point was three deliberately chosen automation domains (ETL, SOAR, configuration management) where I was looking for shared operations to support a single execution engine. Rendering, added later, forced the addition of QualityHierarchy, TraversalEngine, and CompetitiveSelection — primitives that the original three did not require. Each subsequent domain either mapped onto existing primitives or forced additions when the existing set was genuinely insufficient. The primitive count grew over the course of twelve domains. If confirmation bias were the dominant force in the process, the count would not have grown the way it did, and primitives like the three forced by rendering would not have been added.

Second, the LeanFormer implementation produces working measurements. Bit-for-bit weight restoration after belief removal, 119 passing tests, 84% injection success rates, and 86% routing accuracy are objective evidence that the compositions are operationally correct (in the sense that the local invariants hold and the system runs end-to-end), not just descriptively plausible. The 4.8M and 204M training runs provide independent additional evidence that the governance machinery composes without invariant violations.

Third, the TLA+ formal verification process (Section 9.4) subjected every primitive specification and every composition to model checking. The model checker does not share the author's assumptions, but it does check whatever specification the author writes. If the specification is wrong in a way the author did not notice, the verification will faithfully verify the wrong specification. The five corrections produced during verification (Section 9.4) demonstrate that the verification was not trivial; if every initial specification had been confirmed without correction, that would be evidence of a bias-confirming exercise, not a bias-checking one.

The residual risk remains. Domains not yet examined may require primitives outside the current set. All twelve domains examined have a strong bias toward resource-governed execution. Independent replication by other researchers applying the methodology to domains outside the current set is necessary to establish that the primitive set generalizes. Negative results (domains where the primitive set is genuinely insufficient) would be at least as valuable as positive ones.

### 9.3 Limitations

DAC does not claim that domain expertise is unnecessary. The domain function (the rendering equation, Newton's laws, the attention scoring function) requires domain expertise to design. The claim is that the execution infrastructure around the domain function is generic and need not be redesigned per domain. Domain-specific vocabulary is accidental complexity for the execution infrastructure. It is essential complexity for the domain function. The boundary between the two is real, and the paper draws it at the governance layer: the primitives provide structure and governance; the domain provides the computation that executes within that governance.

DAC also does not claim that sixteen primitives are the final, minimal set. Future domains may reveal operations that genuinely cannot be expressed as compositions of the current set, requiring the addition of new primitives. The claim is sufficiency for the twelve domains examined.

The twelve domains examined share a bias toward resource-governed execution. Domains with fundamentally different computational characters (constraint logic programming, probabilistic programming, formal verification, bioinformatics) may stress the primitive set in ways that reveal gaps. Testing against these domains is identified as future work.

All twelve domains were collapsed by the same individual. Independent replication by other researchers applying DAC to domains outside the current twelve is necessary to establish that the methodology is reproducible and that the primitive set generalizes beyond a single analytical perspective. The incremental discovery process, the formal verification, and the verification protocols mitigate confirmation bias but do not eliminate it.

The LeanFormer empirical work has a significant scale limitation that must be stated directly. The architectural results (39M parameters) are proof-of-concept measurements without baseline comparisons. The 204M parameter training run measured that all sixteen primitives' invariants held under real training conditions across 7,228 optimizer steps. However, 204M is modest by current standards, and the efficiency mechanisms that DAC predicts (gradient compute reduction from hierarchical activation, budget-governed routing of gradient signal to relevant parameter groups) were not realized as wall-clock improvement in this run. The governance machinery correctly identified which parameters to stop training and reallocated budget, but the training loop did not skip computation for converged groups.

The 204M run also suffered from training configuration limitations that prevent strong claims about language modeling quality. Single-epoch training with minimal regularization (dropout 0.1 only) produced severe overfitting after step 2,000, with validation perplexity rising from 57.6 to 1,463.9. The overfitting is expected behavior given the configuration; it means the model's generalization quality should be evaluated from the best checkpoint (step 2,000), not the final checkpoint. Multi-epoch training with proper regularization is necessary to show that the architecture can produce competitive language models, and is identified as future work.

The critical open question is whether DAC's governance primitives continue to compose correctly and produce measurable efficiency gains at the scales where modern language models operate (7B+ parameters). Emergent training dynamics, optimization instabilities, and loss landscape characteristics at the billion-parameter scale may interact with the governance mechanisms in ways that the current measurements cannot anticipate. A specific concern is governor thrashing: at scales where gradient magnitudes spike randomly, the convergence governor could oscillate rapidly between states. The existing design mitigates this structurally — state transitions require sustained EMA trends over a configurable cooling window, not single-step readings, and the AWAKENED state is explicitly designed to handle perturbation events without re-traversing the full state sequence — but the thresholds will require recalibration for frontier-scale gradient distributions. The phase-aware governor (Section 5.8.1) adds a further structural guard: a group in COLD phase cannot transition to COOLING regardless of gradient magnitude, preventing the initialization artifact class of false transitions entirely. A full-scale measurement at 7B parameters or above, with proper compute skipping for governed groups and a comparison against an ungoverned baseline, is the most important next step for this research. Until that work is complete, the LeanFormer results should be read as architectural proof-of-concept and governance machinery validation, not as demonstrated production-scale efficiency gains.

A data preservation gap from the 204M run warrants mention as a reproducibility concern. The tokenized training data (32K vocabulary) was stored only on the GCP VM and in a cloud storage bucket, both of which were deleted when the VM was terminated. The local copy of the raw corpus was tokenized with a different vocabulary size (50K), making it incompatible with the trained model for validation purposes. The inline EVAL measurements from the training log remain the authoritative validation numbers, as they were computed against the correctly-tokenized data during training. The lesson is straightforward: the governance machinery (audit chain, hash-chained provenance) preserved every training decision with tamper-evident integrity, but the training data itself was not under governance. Future runs will archive the tokenized data alongside checkpoints. This gap is an unforced error and is acknowledged as such.

The TLA+ formal verification is bounded model checking, not unbounded proof. The TLC model checker exhaustively explores all reachable states within finite bounds (Budget capacity of 4, three-node hierarchies, two parameter groups). Structural bugs and invariant violations are reliably caught at these bounds because the primitive structures do not change with scale. However, bounded verification does not constitute a mathematical proof that the invariants hold for all possible values of the constants. It provides strong evidence, not certainty.

The governance invariants are scale-independent by construction, and this was confirmed empirically at 204M parameters. Budget\<U\> enforces `consumed <= capacity` whether capacity is 4 or 4 billion. The ConvergenceGovernor's four-state machine has the same transitions whether it monitors 8 parameter groups or 8,000. HierarchyOrdering holds whether there are 2 levels or 20. The TLA+ specifications are parameterized by constants, and the invariants hold for any value of those constants because the logic does not reference the constants' magnitudes. The 204M training run confirmed this empirically: zero budget violations across 722 audit records, all 18 governor transitions valid, hierarchy ordering maintained throughout. These are structural properties of the state machines, not empirical properties of any particular training run. What remains empirical, and what requires the 7B+ work, is whether the governance produces efficient training outcomes at that scale: whether the convergence thresholds, budget allocation policies, and gradient routing scores (all domain functions, not primitives) produce the predicted efficiency gains when operating on real gradient distributions at frontier scale.

The formal verification also does not verify implementations. The TLA+ specifications define what the primitives must do. The Python, Rust, Go, and TypeScript implementations in Appendix A are intended to be faithful to those specifications, but the verification that each implementation correctly implements the specification is a separate concern (refinement checking) that has not been performed. A bug in the Python implementation of Budget\<U\> would not be caught by the TLA+ verification of the Budget specification.

### 9.4 Formal Verification: What Was Done and What It Shows

All twenty-one primitive specifications, eighteen decomposition-failure specifications, and five LeanFormer compositions were formalized in TLA+ [14] and verified by the TLC model checker. The twenty-one primitive specifications comprise eighteen primitive modules (the sixteen primitives, with CompetitiveSelection split into three modules for hard, soft, and ranked variants), two cross-primitive composition theorems (TraversalBudgetComposition, SelectThenActuate), and the phase-aware ConvergenceGovernor. The universal primitive verification explored approximately 44.6 million states across 39 specifications (1,002,222 distinct states across the 21 primitives plus 1,051 distinct states across the 18 irreducibility specifications), with the two ConvergenceGovernor specifications (basic and phase-aware) together accounting for 37 million generated states across their full space of possible delta sequences and gradient phase transitions. The five LeanFormer case-study compositions were verified separately across approximately 178,000 additional states (7,512 distinct). The complete specifications, configuration files, and TLC output logs are available in the project repository.

The verification produced three categories of results.

First, all twenty-one primitive specifications passed: their safety invariants held across all reachable states at the tested bounds. Budget never exceeded capacity. The ConvergenceGovernor never skipped a state. CompetitiveSelection (hard) always produced the highest-scoring winner. The hash chain in AuditSink was never broken. These results are bounded verification, not unbounded proof, but TLC's exhaustive exploration of the finite state space provides evidence that the invariants hold in general for the structural reasons described above.

Second, all five LeanFormer composition specifications passed, confirming that primitive invariants are preserved under composition and that the composed system produces composition-level invariants drawn from the constituent primitives. The GovernedTrainingPipeline specification (Appendix C.3) verified thirteen simultaneous invariants across all interleavings of training steps, phase-aware governor updates, hierarchy activations, signal-gated transitions, ranked routing updates, and audit log appends. The composition embodies seven DAC primitives (FederatedBudget, QualityHierarchy, ConvergenceGovernorPhaseAware, CompetitiveSelectionRanked, ActuationPass, Signal, AuditSink) using the canonical B.21 pattern with `peak_observed` as a separate variable, with the new B.21 invariant `NoCoolingFromCold` holding under composition, embodying the B=0 fix at the case-study level. The B=0 fix from Section 5.8.1 is therefore verified at the composition level, not only standalone in the primitive specification. The BeliefDeltaLifecycle specification confirmed base weight immutability, parameter non-overlap, and exact restoration across all possible sequences of belief injection and removal. SemanticRouting, AdaptiveDepth, and LeanFormerLifecycle similarly verify composition-level invariants drawn from their constituent primitives. Across the five compositions, eleven spec modules are embodied across ten conceptual primitives, with B.15 and B.21 being two variants of ConvergenceGovernor (B.15 the basic 4-state machine in C.4 AdaptiveDepth and C.5 LeanFormerLifecycle, B.21 the phase-aware refinement in C.3 GovernedTrainingPipeline). The full list: B.1 Budget, B.2 FederatedBudget, B.3 QualityHierarchy, B.6 ResourceRegistry, B.11 CompetitiveSelectionRanked, B.12 ActuationPass, B.15 ConvergenceGovernor (in C.4 and C.5), B.16 Signal, B.17 RateLimit, B.18 AuditSink, and B.21 ConvergenceGovernorPhaseAware (in C.3).

Third, all eighteen decomposition-failure specifications produced concrete counterexamples in which decomposing a primitive into separable check-then-act steps introduces a TOCTOU window where the primitive's invariant is violated. The counterexamples are not hypothetical. They are state traces that TLC discovered through exhaustive exploration. Some violations were found in as few as two states (QualityHierarchy: adding a child without checking the level constraint; the basic ConvergenceGovernor: a single zero-delta step causing premature COOLING, reproducing the exact B=0 bug from the training run). Others required longer traces (Budget: a concurrent allocation between check and act, at seven states).

The honest framing of what these decomposition results show. The decompositions all factor an atomic primitive operation into separate check and act steps and introduce a concurrent action between them. The TOCTOU violations that result are a known consequence of how atomicity works in concurrent systems: any atomic operation, in any system, will fail its atomicity-dependent invariant if you remove its atomicity in a context with concurrent actors. This is not a discovery about these particular sixteen primitives. What the verification provides is the systematic application of this technique to every primitive in the set, with concrete counterexamples produced by an unbiased model checker rather than imagined ones. The result establishes that each primitive's invariant requires atomicity at the primitive boundary. It does not establish algebraic minimality (that no primitive can be expressed as a composition of others in the set). The two properties are different. The first is what was verified; the second remains open (Section 9.1).

The verification process also challenged and corrected five initial assumptions about the formal specifications. These corrections are worth documenting because they show that the verification was not a confirmation exercise.

| Specification | Initial Assumption | TLC Finding | Correction |
|---|---|---|---|
| ConvergenceGatedActivation | Global invariant (always holds) | L0 can regress to ACTIVE after L1 activates (AWAKENED mechanism) | Reclassified as precondition of ActivateNextLevel |
| FullDepthMeansUncertain | Full depth implies no convergence | Convergence at the final layer is valid | Corrected to: full depth with no convergence at any layer implies uncertainty |
| ForgingRequiresConvergence | Holds across all phases | Converged group could revert to ACTIVE during forging | Restricted governor updates to training phase |
| Budget reallocation | Free half of converged group's budget | Group may have consumed more than half, violating SubPoolInvariant | Free only the unused portion |
| WinnerOptimality | Holds between evaluations | Score update between evaluations creates stale winner | Invalidate allocation when scores change (TOCTOU pattern) |

None of these corrections invalidated the underlying primitives or their invariants. They refined imprecise classifications: the difference between a global invariant and a precondition, the boundary conditions of a convergence definition, the interaction between budget usage and budget reallocation, and the scope of governance operations across lifecycle phases. In every case, the corrected specification is more precise than the original, not more permissive. These are the kinds of issues that prose specifications miss and that formal verification catches.

### 9.5 Future Work

The following are the most important open questions and necessary next steps, ordered by priority and with estimated resource requirements where applicable.

1. Phase-aware convergence governor implementation and retraining. The TLA+ specification exists and passes (ConvergenceGovernorPhaseAware with NoCoolingFromCold invariant). The implementation requires adding gradient_phase tracking to the convergence governor code, adding a peak_observed flag, and enforcing the NoCoolingFromCold precondition on the ACTIVE-to-COOLING transition. A retrain of the 204M model with the fixed governor would produce clean hierarchy activations free of the B=0 artifact, providing the first unambiguous empirical demonstration of convergence-gated coarse-to-fine training. Estimated cost: approximately $150 (one 6-day L4 GPU run).

2. Multi-epoch training with proper regularization and ungoverned baseline comparison. The current 204M run used a single epoch with dropout 0.1 as the only regularization, producing severe overfitting after step 2,000, and there was no comparison against an ungoverned baseline. A 3-5 epoch run with stronger regularization (dropout 0.15-0.2, weight decay sweep) plus a parallel ungoverned baseline would address both the overfitting and the missing comparison. Estimated cost: approximately $900-1,500.

3. Knowledge Plane validation at 204M with baseline comparison. The 39M model showed 84% belief injection success and 86% semantic routing accuracy, but without baselines (vanilla LoRA, full fine-tuning, RAG). Repeating this validation at 204M against the best checkpoint (step 2,000) with parallel runs of those baselines would show whether the belief delta system offers measurable advantages over existing approaches. Estimated cost: approximately $200 (a few hours of GPU time on each baseline plus the LeanFormer measurement).

4. Scale validation of LeanFormer at 7B+ parameters. This is the most important validation that the current work lacks. It would include: gradient compute reduction measurements with actual backward-pass skipping for converged groups, wall-clock training time comparisons against an ungoverned baseline, Knowledge Plane validation with orthogonality enforcement at a scale where the delta address space is practically significant, and convergence ordering analysis at a scale where the parameter group ratios are representative of production models. The governance invariants are established as scale-independent (formally verified and empirically confirmed at 204M), but the efficiency claims require empirical validation at frontier scale. Estimated cost: approximately $5,000-15,000 depending on GPU tier and training duration.

5. Cross-domain optimization transfer in the opposite direction. The methodology's strongest theoretical claim is that optimizations transfer across domains via shared primitives. So far, all three demonstrations move structural insights from systems engineering into ML. A demonstration in the opposite direction (e.g., applying an ML-derived attention optimization to a non-ML CompetitiveSelection use case in rendering or networking, with measurable improvement) would substantially strengthen the central claim.

6. Independent replication of DAC by other researchers applying the methodology to domains outside the current twelve. Positive results would confirm the methodology's generality. Negative results (domains where the primitive set is genuinely insufficient) would be equally valuable, identifying the boundaries of the current primitive set and potentially revealing new primitives.

7. Adversarial domain testing against domains with fundamentally different computational characters than the resource-governed execution domains in the current set. Constraint logic programming, probabilistic programming, formal verification, and bioinformatics sequence alignment are the highest-priority candidates.

8. Independent implementation of two or more non-ML domains from the collapse table (candidates: real-time rendering pipeline, network routing system) to provide stronger evidence for cross-domain primitive transfer. This is engineering work requiring months of effort but no GPU cost.

9. Refinement checking between TLA+ specifications and implementations, verifying that the Python, Rust, Go, and TypeScript code in Appendix A correctly implements the formal specifications in Appendix B.

10. Investigation of specific minimality candidates: whether RateLimit can be expressed as Budget\<Operations\> with periodic reset, and whether Signal\<T\> can be derived from AuditSink with a predicate filter. If either decomposition succeeds, the primitive count drops below sixteen and the relevant primitives should be removed from the set or reclassified as standard compositions.

11. Categorical formalization of the cross-domain mappings as functors or adjunctions, for the audience that does prefer that level of formality.

---

## 10. Conclusion

Sixteen abstraction primitives sufficed for twelve engineering domains. The Competitive Selection family accounts for a large fraction of the cross-domain mappings and decomposes honestly into three selection modes (hard, soft, ranked) that share a scoring interface but differ in allocation semantics. Some mappings in the table are genuinely informative (attention as soft competitive selection, the four-domain DAG workload pattern, training as the only governed system without selectivity primitives in standard form). Others are largely renamings (ETL extract/transform stages as ActuationPass; inventory as ResourceRegistry). Both categories are documented honestly.

LeanFormer is a worked example of what DAC produces when applied to a domain, and a deliberate test of the methodology on a domain I do not have research-level expertise in. Six open problems in neural network design (parameter inefficiency, attention cost, catastrophic forgetting, knowledge composition, confabulation, and training process inefficiency), each framed as a hard ML research problem with years of dedicated literature, each mapped to a composition of solved systems engineering primitives once the ML vocabulary was stripped away. The architecture composes four efficiency mechanisms simultaneously, addresses catastrophic forgetting of the base weights through architectural separation (bit-for-bit base weight restoration across 100 beliefs), makes confabulation architecturally detectable, and governs the entire training lifecycle with the same primitives that govern the model's architecture. The arc from initial DAC decomposition to 204M-parameter measurements took approximately three weeks. At 204M parameters, all sixteen primitives' invariants held empirically across 7,228 training steps with zero governance violations. The model is not competitive as a language model, which reflects, among other things, the limits of my ML expertise in the domain functions DAC does not supply (loss formulation, training recipe, regularization, data curation). The structural skeleton DAC produced runs end-to-end without violating its specified invariants.

The B=0 observational degeneracy episode during the 204M run shows DAC's vocabulary-stripping process applied to a failure in a DAC-governed system. The bug class (zero-as-ambiguous-signal) is well-known; the fix (a latching phase tracker) is a standard hysteresis pattern. The contribution of DAC here is not the discovery of a new technique but the recognition route: stripping the ML vocabulary from the failure produced a description that matched a pattern already solved across the collapse table (depth buffers, heartbeats, timeouts). The fix was specified in TLA+ and verified by TLC before any code was written.

LeanFormer has clear limitations: modest scale, single-epoch training, two subsystems that did not activate, and no comparison against an ungoverned baseline. These are not presented as completed work. They are the next steps. The governance machinery's invariants held empirically. The primitives composed without violations. The architectural insights (catastrophic forgetting as write-conflict, training as the only ungoverned system in the collapse table) hold regardless of scale. What remains is to push the implementation further: multi-epoch training to demonstrate generalization, the phase-aware governor fix to produce clean hierarchy activations, baseline comparisons, and scale validation at 7B+ parameters where the efficiency predictions become practically significant. The risk profile is low, the costs are bounded (Section 9.5), and the success criteria are clear.

The extension from model architecture to model lifecycle is the strongest evidence that the primitive set is sufficient across the examined domains. The same sixteen primitives that express how attention works also express how training should be governed, how evaluation should be triggered, and how knowledge should be forged. The vocabulary was different at every stage. The structure was the same.

The primitives are not artifacts of any particular programming language or codebase. Appendix A demonstrates all sixteen primitives implemented in Rust, Python, Go, and TypeScript across multiple domains. The same loop, the same invariant, the same state machine appears in every language. Only the type annotations and domain functions change.

The primitives are formally specified mathematical structures. Forty-four TLA+ specifications (twenty-one primitive specifications, eighteen decomposition-failure specifications, and five LeanFormer compositions) were verified by the TLC model checker across approximately 44.6 million states for the universal primitive verification plus approximately 178,000 additional states for the LeanFormer case study. Every primitive's safety invariant held. Every decomposition into separable check-then-act steps produced a concrete TOCTOU counterexample, demonstrating that each primitive's invariant requires atomicity at the primitive boundary (a property of atomic operations in concurrent systems generally; the contribution is the systematic demonstration across all sixteen primitives in the set). A real training bug (B=0 initialization causing premature convergence detection) was reproduced as a TLC counterexample in 2 states and fixed with a phase-aware governor verified across approximately 18.6 million states. The verification process challenged five initial assumptions about the specifications and produced corrections that refined imprecise classifications without weakening any falsified claim.

The point is straightforward. I noticed that several engineering domains use different vocabularies for what often turn out to be structurally similar operations. I cataloged those operations into a candidate set of sixteen primitives, formally specified them in TLA+, and built a system end-to-end using them. The catalog is sufficient for the twelve domains examined. It is not proven minimal, and it may not be sufficient for domains outside that set. The vocabulary made the structures look different. Often, they were not.

---

## Acknowledgments

The LeanFormer architecture and targeted training system were implemented using Claude Code as an AI implementation agent, with the architect maintaining design authority, formal specification, and verification discipline. The human-as-architect / AI-as-implementation-agent development model produced a working, tested system governed by the DAC primitive set. The verification discipline that enabled this: requiring log evidence rather than diff evidence, enforcing strict scope boundaries per fix, audit protocols that re-read source rather than trust summaries, and a firm policy against accepting "looks right" as a verification outcome.

---

## References

[1] Brooks, F.P. (1987). No Silver Bullet: Essence and Accident in Software Engineering. IEEE Computer, 20(4), 10-19.

[2] Vaswani, A. et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems, 30.

[3] Bellman, R. (1958). On a Routing Problem. Quarterly of Applied Mathematics, 16(1), 87-90.

[4] Codd, E.F. (1970). A Relational Model of Data for Large Shared Data Banks. Communications of the ACM, 13(6), 377-387.

[5] Lampson, B.W. (1974). Protection. ACM SIGOPS Operating Systems Review, 8(1), 18-24.

[6] Hu, E.J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

[7] Saltzer, J.H., Reed, D.P., and Clark, D.D. (1984). End-to-End Arguments in System Design. ACM Transactions on Computer Systems, 2(4), 277-288.

[8] Leviathan, Y. et al. (2023). Fast Inference from Transformers via Speculative Decoding. ICML 2023.

[9] Fedus, W. et al. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. JMLR, 23(120), 1-39.

[10] Katharopoulos, A. et al. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020.

[11] Schuster, T. et al. (2022). Confident Adaptive Language Modeling. NeurIPS 2022.

[12] Meng, K. et al. (2022). Locating and Editing Factual Associations in GPT. NeurIPS 2022.

[13] Karis, B. et al. (2021). Nanite: A Scalable Mesh Representation. SIGGRAPH Advances in Real-Time Rendering course.

[14] Lamport, L. (2002). Specifying Systems: The TLA+ Language and Tools for Hardware and Software Engineers. Addison-Wesley.

[a-1] Alexander, C. et al. (1977). A Pattern Language: Towns, Buildings, Construction. Oxford University Press.

[a-2] Gamma, E., Helm, R., Johnson, R., and Vlissides, J. (1994). Design Patterns: Elements of Reusable Object-Oriented Software. Addison-Wesley.

[a-3] Parnas, D.L. (1972). On the Criteria To Be Used in Decomposing Systems into Modules. Communications of the ACM, 15(12), 1053-1058.

[a-4] Liskov, B. and Zilles, S. (1974). Programming with Abstract Data Types. SIGPLAN Notices, 9(4), 50-59.

[a-5] Gelernter, D. (1985). Generative Communication in Linda. ACM Transactions on Programming Languages and Systems, 7(1), 80-112.

[a-6] Hoare, C.A.R. (1978). Communicating Sequential Processes. Communications of the ACM, 21(8), 666-677.

[a-7] Hewitt, C., Bishop, P., and Steiger, R. (1973). A Universal Modular ACTOR Formalism for Artificial Intelligence. IJCAI 1973.

[a-8] Wing, J.M. (2006). Computational Thinking. Communications of the ACM, 49(3), 33-35.

[a-9] van der Aalst, W.M.P. et al. (2003). Workflow Patterns. Distributed and Parallel Databases, 14(1), 5-51.

---

## Appendix A: The Sixteen Primitives in Code

### Cross-Domain, Cross-Language Implementation Examples

Each primitive is shown in multiple languages and multiple domains to demonstrate that the structure is identical and only the domain-specific types and functions change. The primitive does not know which domain it serves. The domain provides the types and the scoring function. The primitive provides the governance.

---

### Data Primitives

---

#### A.1 Budget\<U\>

A pool with a capacity and an invariant: consumed <= capacity. There is no force_allocate.

Rust (VRAM bytes):

```rust
pub struct Budget {
    capacity: u64,
    allocated: u64,
}

impl Budget {
    pub fn try_allocate(&mut self, amount: u64) -> bool {
        if self.allocated + amount > self.capacity { return false; }
        self.allocated += amount;
        true
    }

    pub fn release(&mut self, amount: u64) {
        self.allocated = self.allocated.saturating_sub(amount);
    }
}
```

Python (gradient compute FLOPs):

```python
class Budget:
    def __init__(self, capacity):
        self.capacity = capacity
        self.allocated = 0

    def try_allocate(self, amount):
        if self.allocated + amount > self.capacity:
            return False
        self.allocated += amount
        return True

    def release(self, amount):
        self.allocated -= amount
```

Go (Kubernetes CPU millicores):

```go
type Budget struct {
    capacity  int64
    allocated int64
}

func (b *Budget) TryAllocate(amount int64) bool {
    if b.allocated+amount > b.capacity { return false }
    b.allocated += amount
    return true
}

func (b *Budget) Release(amount int64) { b.allocated -= amount }
```

The invariant is identical. The language is different. The domain (VRAM, gradient compute, CPU) is different. The four lines that enforce the ceiling do not change.

---

The remaining fourteen primitives (FederatedBudget, QualityHierarchy, AllocationSnapshot, RelationshipGraph, ResourceRegistry, TraversalEngine, PropagationPass, CompetitiveSelection in all three modes, ActuationPass, Reduction, Sampler, Signal, RateLimit, and AuditSink) follow the same pattern: identical structure across languages, with only the domain-specific types and functions changing. The complete set of cross-language implementations is available in the project repository. One additional example is included here because of its centrality to Section 5.8.1.

---

#### A.15 ConvergenceGovernor

Detect fixed-point convergence and govern iteration count. The four-state machine: ACTIVE (still changing), COOLING (change rate declining), CONVERGED (stable), AWAKENED (reconverged after perturbation).

Rust (physics constraint solver):

```rust
pub enum ConvergenceState { Active, Cooling, Converged, Awakened }

pub struct ConvergenceGovernor {
    state: ConvergenceState,
    delta_history: VecDeque<f64>,
    threshold: f64,
}

impl ConvergenceGovernor {
    pub fn update(&mut self, delta: f64) -> ConvergenceState {
        self.delta_history.push_back(delta);
        if self.delta_history.len() > 5 { self.delta_history.pop_front(); }

        let avg_delta: f64 = self.delta_history.iter().sum::<f64>()
            / self.delta_history.len() as f64;

        self.state = match self.state {
            ConvergenceState::Active if avg_delta < self.threshold * 2.0
                => ConvergenceState::Cooling,
            ConvergenceState::Cooling if avg_delta < self.threshold
                => ConvergenceState::Converged,
            ConvergenceState::Converged if avg_delta > self.threshold * 3.0
                => ConvergenceState::Awakened,
            ConvergenceState::Awakened if avg_delta < self.threshold
                => ConvergenceState::Converged,
            other => other,
        };
        self.state
    }
}
```

Python (per-group training convergence):

```python
class ConvergenceGovernor:
    ACTIVE, COOLING, CONVERGED, AWAKENED = range(4)

    def __init__(self, threshold, window=5):
        self.state = self.ACTIVE
        self.threshold = threshold
        self.delta_history = deque(maxlen=window)

    def update(self, delta):
        self.delta_history.append(delta)
        avg = sum(self.delta_history) / len(self.delta_history)

        if self.state == self.ACTIVE and avg < self.threshold * 2:
            self.state = self.COOLING
        elif self.state == self.COOLING and avg < self.threshold:
            self.state = self.CONVERGED
        elif self.state == self.CONVERGED and avg > self.threshold * 3:
            self.state = self.AWAKENED
        elif self.state == self.AWAKENED and avg < self.threshold:
            self.state = self.CONVERGED
        return self.state
```

Go (RAFT term stability detection):

```go
type ConvergenceState int
const (
    Active ConvergenceState = iota
    Cooling
    Converged
    Awakened
)

type ConvergenceGovernor struct {
    State     ConvergenceState
    Threshold float64
    History   []float64
    Window    int
}

func (g *ConvergenceGovernor) Update(delta float64) ConvergenceState {
    g.History = append(g.History, delta)
    if len(g.History) > g.Window {
        g.History = g.History[1:]
    }
    avg := sum(g.History) / float64(len(g.History))

    switch g.State {
    case Active:
        if avg < g.Threshold*2 { g.State = Cooling }
    case Cooling:
        if avg < g.Threshold { g.State = Converged }
    case Converged:
        if avg > g.Threshold*3 { g.State = Awakened }
    case Awakened:
        if avg < g.Threshold { g.State = Converged }
    }
    return g.State
}
```

The four-state machine is identical. The delta (constraint residual, gradient magnitude, term change rate) is the domain.

### The Pattern

Every example in this appendix shows the same property: the domain provides the types, the data, and the scoring/aggregation/message functions. The primitive provides the structure, the governance, and the invariants. Swap the domain function and the primitive serves a different domain without modification.

The vocabulary was different. The language was different. The code was the same.
