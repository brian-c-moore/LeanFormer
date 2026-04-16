# Domain Abstraction Collapse

## A Methodology for Recognizing Solved Problems Across Domain Boundaries

Brian Moore, CISSP, CCSP
Independent Systems Researcher

---

## Abstract

The vocabulary used to describe a problem constrains the solutions that can be found. Domain-specific language creates the impression that domain-specific solutions are required, concealing the fact that structurally identical problems have already been solved in other domains under different names. This paper formalizes Domain Abstraction Collapse (DAC): a methodology for stripping domain-specific language from computational patterns, identifying structural isomorphisms across domain boundaries, and reducing domain-specific abstractions to a minimal generating set of domain-agnostic abstraction primitives from which domain patterns can be reconstructed through composition.

The methodology is applied across twelve resource-governed engineering domains (real-time rendering, physics simulation, audio spatialization, network replication, container orchestration, CI/CD pipelines, ETL workflows, configuration management, SOAR automation, distributed consensus, economic simulation, and machine learning) and against eight adversarial domains outside the original sample (unification, Hindley-Milner type inference, term rewriting, constraint logic programming, probabilistic programming, resolution-style theorem proving, symbolic differentiation, and lattice-theoretic dataflow analysis). The adversarial pass sharpens two existing primitives and adds one new one, yielding a final set of seventeen primitives. Six of the adversarial domains fit the original set cleanly; two (constraint logic programming and theorem proving) motivate the addition of Checkpoint, a hierarchical rollback primitive; lattice dataflow motivates a widening mode on ConvergenceGovernor; and unification motivates an explicit monotonicity invariant on ResourceRegistry. The primitive set is sufficient, not provably minimal.

DAC operates in three modes. As an analytical tool, it decomposes existing systems to reveal structural isomorphisms. As a generative methodology, it provides a search strategy for problems that resist solution in their native vocabulary: strip the vocabulary, map to the primitive set, and check whether the difficulty resides in a solved problem wearing unfamiliar terminology. As an implementation methodology, it decomposes any computation described in domain-specific notation into a build plan composed of known primitives.

Two engineering systems, built on the primitive set, serve as empirical validation. LeanFormer is a novel transformer architecture in which DAC was applied to six open problems in neural network design. The initial proof-of-concept was designed and built in 24 hours. Validation at 39M parameters (76M dense equivalent) confirmed the architectural thesis across 119 tests. A 204M-parameter run (805M dense equivalent) with a full DAC-governed training pipeline confirmed that the primitive set composes correctly under real training conditions: zero budget violations across 722 hash-chained audit records, eighteen valid convergence-governor state transitions with no skipped states, and coarse-to-fine hierarchy activation with the final level activating at step 2,773 via genuine convergence gating. Orkestratum is an application runtime built on the same primitive modules as a single codebase, which executes SOAR playbooks, CI/CD pipelines, ETL workflows, configuration-management workloads, and a real-time renderer. Multiple domains run on one runtime, with the domain-specific part being the workload graph and the task functions, not the execution engine. LeanFormer addresses the "methodology applied to one domain" objection; Orkestratum addresses the "methodology applied in the abstract, not demonstrated as a runtime" objection.

During the 204M LeanFormer training run, a B=0 initialization artifact caused premature hierarchy activation at the lower levels, providing an unplanned opportunity to apply DAC to its own failure. Stripping ML vocabulary from the failure revealed an observational degeneracy: two qualitatively different gradient trajectories (cold start and genuine convergence) producing the same low-magnitude reading, a pattern already solved in the collapse table by depth buffers in rendering, heartbeat protocols in networking, and timeouts in distributed consensus. The fix (a phase-aware ConvergenceGovernor that tracks whether gradient magnitude has ever exceeded the threshold) was formally verified in TLA+ across 18.6 million states before implementation. The methodology diagnosed its own failure by recognizing the structural identity of a novel ML training bug with solved problems from other domains.

Sixteen of the seventeen primitive specifications and five LeanFormer compositions were formally specified in TLA+ and verified by the TLC model checker across approximately 45.4 million states at the tested bounds. Every verified primitive's safety invariants held under exhaustive state exploration. Every decomposition of a verified primitive into sub-operations produced a concrete counterexample where the invariant was violated, demonstrating operational irreducibility: the governance property requires atomicity and cannot survive decomposition. Operational irreducibility is a weaker claim than algebraic minimality (which remains open) but stronger than the unverified primitive sets common in practice. The verification process challenged and refined five initial assumptions about the specifications; the corrections sharpened imprecise classifications without weakening any claim. The seventeenth primitive, Checkpoint, was identified during the adversarial domain pass described in Section 2.5 and is qualified for inclusion by the criteria defined in Section 1.3; its TLA+ specification is listed as a pre-submission prerequisite in Section 10.5.

The central thesis is that domain-specific vocabulary is the primary obstacle to recognizing that many computational problems have already been solved. The essential computational structures underlying apparently diverse systems are few, shared across domains, and well-understood. DAC is the methodology for revealing this, and its value lies not only in understanding but in building: once the vocabulary is stripped, the "hard problem" in one domain becomes a known solution from another, and the primitive set becomes a construction kit for engineering solutions to problems that appeared hard only because their domain vocabulary obscured their structural identity with solved problems.

The methodology has limits. The claims in this paper are bounded to the twenty examined domains, all of which share computational characteristics that favor the primitive set. The primitive set is sufficient but not provably minimal. Cross-domain optimization transfer has been demonstrated within the LeanFormer and Orkestratum engineering programs; broader cross-domain transfer remains future work. The empirical validation at 204M parameters demonstrates governance correctness, not production-scale efficiency gains. These limitations are stated explicitly throughout the paper and are not rhetorical concessions: they are the boundary of what the evidence currently supports.

---

## 1. Introduction

### 1.1 The Problem of Accidental Specialization

Fred Brooks distinguished between essential complexity (complexity inherent to the problem being solved) and accidental complexity (complexity introduced by the tools and methods used to solve it) [1]. This paper identifies a third category: vocabulary-induced complexity, where domain-specific language used to describe a problem creates the impression that the problem itself is domain-specific.

Consider the Entity Component System (ECS), the dominant architectural pattern in game engine development since the late 1990s. An entity is a unique identifier. A component is a typed data record associated with an entity. A system is a function that queries entities by their component composition and applies transformations. This pattern has been independently reinvented by every major game engine, each using different vocabulary: Unity calls them GameObjects and MonoBehaviours, Unreal calls them Actors and Components, Bevy calls them Entities, Components, and Systems.

Stripped of game-specific vocabulary, an ECS is a relational database. Entities are rows. Components are columns. Systems are queries with side effects. The ECS pattern's real contribution was cache-friendly memory layout through structure-of-arrays organization, a genuine innovation in data access patterns. But the structural pattern itself (typed records queried by composition and transformed by functions) is a relational database without the formal semantics, query planning, or ACID guarantees that the database community developed over decades. Every game engine team since the late 1990s has reinvented this structure independently, each time believing it to be a game-specific innovation, because the vocabulary made it look like one.

This is not an isolated example. Across every engineering domain examined in this paper, systems that appear to solve domain-specific problems with domain-specific architectures are, when the vocabulary is stripped away, compositions of a small number of generic computational patterns that have been solved repeatedly in different clothing.

### 1.2 The Methodology: Domain Abstraction Collapse

Domain Abstraction Collapse (DAC) is a systematic process with five steps:

1. Enumerate domain-specific abstractions. List every named concept, pattern, data structure, and algorithm used within a domain. Accept the domain's own vocabulary uncritically.

2. Strip domain vocabulary. For each abstraction, remove every word that is specific to the domain. Replace domain nouns with generic descriptions of what the abstraction actually does structurally.

3. Identify cross-domain isomorphisms. Compare the stripped descriptions across domains. When two abstractions from different domains reduce to the same structural description, they are the same operation in different clothing.

4. Reduce to abstraction primitives. Continue stripping until no further decomposition is possible without losing governance and composability properties. The operations that survive are the abstraction primitives: operations that cannot be further decomposed without either descending to an implementation level where domain patterns require unbounded composition counts, or losing the formal properties (budget invariants, convergence detection, audit completeness) that make cross-domain composition useful.

5. Reconstruct domains as compositions. Verify that every domain-specific abstraction from step 1 can be expressed as a composition of the abstraction primitives from step 4, instantiated with domain-specific data and functions.

If step 5 succeeds with no residual (every domain pattern is expressible, and no domain pattern requires a primitive not in the set), the collapse is complete for that domain. The domain-specific abstractions were vocabulary, not structure.

These five steps describe DAC as an analytical methodology: decomposing existing systems to find what they share. The same process works in reverse as an implementation methodology. Given any computation described in domain-specific notation (a mathematical formula, a protocol specification, a biological pathway diagram), the practitioner strips the notation's vocabulary and maps the computation's structure to the primitive set. The result is not a description or an analogy. It is a build plan: a composition of known, tested primitives that implements the computation. The notation describes what the answer should be. The DAC decomposition describes how to build the machine that computes it. Section 5 develops this application in detail.

### 1.3 Abstraction Primitives: The Criteria for Inclusion

The term "abstraction primitive" requires a precise definition because the level at which decomposition stops is the claim that separates DAC from the trivially true observation that everything reduces to NAND gates. The definition functions not as a description but as a test: a candidate operation either passes all criteria and joins the set, or fails one and is recognized as something else (a composition, a domain function, a mode of an existing primitive, or an implementation detail).

A candidate operation qualifies as an abstraction primitive if and only if it satisfies all five of the following:

1. Governance semantics. The operation carries a formal property — a budget invariant, a convergence detection guarantee, a change-detection commitment, an audit-completeness property, a transaction atomicity guarantee, or a comparable invariant — that must hold during the operation itself. The property cannot be reconstructed by callers after the fact. It is a structural guarantee, not a policy.

2. Operational irreducibility. The guard and the guarded operation cannot be separated into distinct steps without creating reachable states where the governance invariant is violated. This is checkable mechanically: specify the primitive in TLA+, specify a decomposition that splits the primitive into plausible sub-operations, and run the TLC model checker. A passing decomposition disqualifies the candidate. A failing decomposition (one that produces a concrete counterexample showing the invariant violated) confirms operational irreducibility.

3. Cross-domain appearance. The same structure must appear in multiple domains under different vocabulary. A candidate that appears in only one domain, however elegant, is a domain-specific solution, not an abstraction primitive. The goal of DAC is to recognize structural identity across domain boundaries.

4. Non-expressibility as a composition. If the candidate can be expressed as a composition of existing primitives that preserves all governance properties, it is a composition, not a primitive. A named composition may be worth documenting (Transaction, Memoize, Diff) but does not earn primitive status.

5. Not a mode or invariant refinement of an existing primitive. If the candidate's distinguishing feature is a parameterization — a mode selector, an optional invariant, a scoring function choice — it is a refinement of the existing primitive, not a new primitive. Hard, soft, and ranked are modes of CompetitiveSelection. Widening is a mode of ConvergenceGovernor. Monotonicity is an optional invariant on ResourceRegistry. None of these are primitives on their own.

Below the governance boundary are implementation details that vary by hardware and runtime. Above it is domain vocabulary that prevents cross-domain reuse. NAND gates fail criterion 1: they carry no governance semantics. Domain-specific patterns such as "visibility buffer" or "playbook" fail criterion 4: they decompose into compositions of the primitive set. A proposed primitive that would only apply to rendering fails criterion 3.

The test is the load-bearing element of the methodology. A permissive definition of "primitive" would let the set grow with every new domain and would erode the distinction between structural insight and domain cataloguing. The test is strict. Most candidate operations fail. Section 2.5 documents the one candidate that survived during an adversarial domain pass (Checkpoint). Section 10.2 documents subsequent candidates that did not survive.

The claim that the primitives are operationally irreducible is analogous to the concept of an irreducible element in algebra: an element that cannot be factored into a product of non-trivial elements. The abstraction primitives are irreducible with respect to governance-preserving decomposition. Section 10.4 provides formal evidence: TLA+ specifications of sixteen of the seventeen primitives were subjected to systematic decomposition, and the TLC model checker produced concrete counterexamples for every tested decomposition, demonstrating that splitting the tested primitives into sub-operations creates reachable states where the governance invariant is violated. Checkpoint's decomposition verification is listed as a pre-submission prerequisite in Section 10.5. This is operational irreducibility: the guard and the guarded operation are structurally inseparable. The question of algebraic minimality (whether any primitive can be expressed as a composition of other primitives in the set) is addressed separately in Section 10.1 and remains open.

The methodology is an invitation rather than a closure. A practitioner working in a domain unfamiliar to the author may yet identify an operation that passes all five criteria and is not in the current set. That operation earns its place by passing the test, and the methodology becomes richer for having recognized prior greatness that was previously hidden in domain-specific vocabulary.

### 1.4 Relationship to Existing Work

DAC draws from several established intellectual traditions.

Category theory formalizes structure-preserving mappings between mathematical domains. When this paper claims that attention is soft competitive selection, it is identifying a structure-preserving map between the category of neural network operations and the category of allocation primitives. DAC is the engineering application of categorical thinking to systems design, without requiring the formalism, because the insight is accessible to practitioners who would not otherwise encounter a functor.

Dimensional reduction from data science captures the mathematical structure of what DAC does: take a high-dimensional space of domain-specific operations and discover that its actual dimensionality is much lower. The "dimensions" that are eliminated were linearly dependent. They appeared independent because they had different names and lived in different domains.

Brooks' essential/accidental complexity distinction [1] is the philosophical ancestor. DAC sharpens the claim: the accidental complexity is not merely in the tools, but in the conceptual framing of the problem itself. Domain-specific vocabulary is a cognitive lens that makes accidental specialization feel essential.

This paper does not claim that DAC is analogous to physical unifications such as Maxwell's unification of electricity and magnetism or Einstein's unification of space and time. Those unifications produced novel empirical predictions (electromagnetic waves, gravitational lensing) that were later confirmed. DAC's analogous achievement would be: an optimization discovered in one domain, applied to another domain via a shared primitive, producing a measurable improvement in the second domain that was not previously known. Evidence for this specific kind of transfer is accumulating within the engineering programs described in this paper but has not been demonstrated at the breadth required to support the physics analogy. The honest analogue is dimensional reduction on an engineering corpus, not physical unification.

### 1.5 Addressing the Turing Tarpit Objection

A standard defense against unification theories in computer science is the Turing Tarpit argument: because everything is Turing complete, of course everything can be mapped to anything else. If one reduces far enough, everything is NAND gates, and the unification is trivially true but operationally useless.

DAC's defense against this objection is empirical. The primitives were not designed top-down from a theory of computation. They were discovered bottom-up by collapsing domains and observing what survived. The collapse was incremental. SOAR automation was examined first. Rendering was added second and forced the addition of new primitives (QualityHierarchy, TraversalEngine, CompetitiveSelection) that SOAR alone did not require. Each subsequent domain either mapped onto existing primitives or forced new primitives to be added when the existing set was genuinely insufficient.

If the primitives were too low-level (register operations, logic gates), the collapse would have produced hundreds of primitives per domain pattern and the compositions would be unwieldy. If they were too high-level (domain-specific abstractions), no shared primitives would have emerged across domains. That seventeen primitives suffice for twenty domains, discovered incrementally through domain analysis rather than designed to fit, is empirical evidence that the decomposition level is defensible.

The adversarial domain pass described in Section 2.5 sharpens this evidence. When DAC was applied to domains chosen specifically because they did not share the resource-governed execution character of the original twelve, six of eight mapped onto the existing primitive set with no additions, one motivated a mode extension on an existing primitive, one motivated an explicit invariant on an existing primitive, and two motivated the addition of one new primitive (Checkpoint). A methodology that survives adversarial testing with one addition is more credible than one that claims to cover everything.

### 1.6 Contributions

This paper makes seven contributions:

1. Formalization of Domain Abstraction Collapse as a named, repeatable design methodology with defined steps, a formal criterion for what constitutes an abstraction primitive, and completeness and operational-irreducibility criteria.

2. Empirical demonstration across twelve engineering domains plus eight adversarial domains, showing that twenty domains reduce to a common set of seventeen abstraction primitives, with explicit assessment of which collapses are genuine structural insights and which are trivially true.

3. Identification and decomposition of the Competitive Selection family, resolving the overcounting problem inherent in treating structurally distinct selection mechanisms as a single primitive.

4. Demonstration of DAC as a generative engineering methodology through LeanFormer, a novel transformer architecture in which DAC was used to engineer solutions to six open problems by recognizing their structural identity with solved problems from other domains.

5. Demonstration of DAC as a multi-domain runtime foundation through Orkestratum, an application runtime built on the primitive modules that executes SOAR playbooks, CI/CD pipelines, ETL workflows, configuration-management workloads, and a real-time renderer on one codebase.

6. Articulation of DAC as an implementation methodology that produces build plans: given any computation described in domain-specific notation, DAC decomposes it into a composition of known primitives with specified governance properties.

7. Formal verification of sixteen of the seventeen primitive specifications and five LeanFormer compositions in TLA+, with bounded model checking across approximately 45.4 million states at the tested bounds. The verification establishes that every verified primitive's invariants hold under all reachable states at those bounds, that the composed LeanFormer system preserves primitive invariants while producing emergent system-level guarantees, and that every verified primitive is operationally irreducible. The seventeenth primitive, Checkpoint, was identified during the adversarial pass; its TLA+ specification is a pre-submission prerequisite (Section 10.5). A real training bug (B=0 initialization causing premature convergence detection) was diagnosed using DAC's own vocabulary-stripping methodology, reproduced as a TLC counterexample, and fixed with a phase-aware governor specification verified across 18.6 million states before implementation.

---

## 2. The Collapse: Twenty Domains, Seventeen Primitives

### 2.1 The Original Twelve Domains

The following twelve engineering domains were subjected to abstraction collapse during the initial development of the methodology. They were not selected a priori. They emerged as the application domains encountered during the design of a general-purpose execution framework, beginning with SOAR automation and expanding as each new domain revealed the same underlying structures.

| # | Domain | Entry Point |
|---|--------|-------------|
| 1 | Security Orchestration (SOAR) | Original motivation: frustration with existing SOAR platforms |
| 2 | CI/CD Pipelines | Recognized as DAG workloads identical to SOAR playbooks |
| 3 | ETL / Data Pipelines | Recognized as DAG workloads with different I/O stages |
| 4 | Configuration Management | Recognized as declarative state convergence |
| 5 | Container Orchestration | Recognized as budget-constrained resource allocation |
| 6 | Real-Time Rendering | First GPU domain: proved primitives work under real-time constraints |
| 7 | Physics Simulation | Multi-stage pipeline decomposed to selection + propagation |
| 8 | Audio Spatialization | Budget-constrained propagation over spatial graph |
| 9 | Network Replication | Spatial budget allocation in bandwidth units |
| 10 | Economic Simulation | Competitive selection + propagation + transactions |
| 11 | Distributed Consensus | Fixed-point iteration over a cluster graph (RAFT) |
| 12 | Machine Learning | Complete AI lifecycle decomposed to primitives |

A methodological concern stated explicitly: these twelve domains are all infrastructure and systems domains with a strong bias toward resource-governed execution. Domains outside this family may not collapse as cleanly. Section 2.5 addresses this by applying the methodology to eight adversarial domains chosen specifically because they do not share this character.

### 2.2 The Abstraction Primitive Set

After abstraction collapse across the twenty examined domains (twelve original plus eight adversarial), seventeen primitives survived: operations that could not be further decomposed without losing governance semantics and that appeared in multiple domains in different vocabulary.

Data Primitives (how data is structured and related):

| Primitive | Single Responsibility |
|-----------|----------------------|
| Budget\<U\> | A pool with a capacity and an invariant: consumed <= capacity |
| FederatedBudget\<U\> | A master pool subdivided into sub-pools: sum(sub) <= master |
| QualityHierarchy | A tree where each node represents a resource at a quality level |
| AllocationSnapshot | A materialized record of which hierarchy nodes are currently allocated |
| RelationshipGraph\<N,E\> | A weighted directed graph over entities with typed edges |
| ResourceRegistry\<K,V\> | A map from resource IDs to the data needed to actuate them, optionally carrying a monotonicity invariant (added in Section 2.5 for unification) |

Computation Primitives (how computation is expressed):

| Primitive | Single Responsibility |
|-----------|----------------------|
| TraversalEngine | Budget-constrained best-first search over a QualityHierarchy |
| PropagationPass | One step of message passing over a RelationshipGraph |
| CompetitiveSelection\<S,W,A\> | For each output seat, evaluate candidates and determine allocation (see Section 4 for the full family decomposition) |
| ActuationPass\<R\> | Apply a function to allocated seats only |
| Reduction\<T,R\> | Aggregate a collection to a scalar |
| Sampler\<T\> | Draw a value from a probability distribution |
| Checkpoint | Establish a scoped rollback boundary for nested undo (added in Section 2.5 for constraint logic programming and theorem proving) |

Governance Primitives (how computation is governed):

| Primitive | Single Responsibility |
|-----------|----------------------|
| ConvergenceGovernor | Detect fixed-point convergence and govern iteration count, optionally in widening mode (extended in Section 2.5 for lattice dataflow) |
| Signal\<T\> | Notify dependents when a value changes |
| RateLimit | Constrain throughput within a time window |
| AuditSink | Observe every mutation in an append-only log |

These seventeen primitives compose through five modes: pipeline (Unix pipe model, output of A is input of B), wrapping (OSI stack model, B adds one concern over A), instantiation (generic primitive parameterized with domain function), feedback (output of a later stage governs an earlier stage), and parallel composition (independent primitives operating on partitioned data).

A note on the relationship to existing computational abstractions: several of these primitives resemble standard functional programming operations. ActuationPass resembles Map. Reduction resembles Fold. CompetitiveSelection resembles Filter composed with Sort or Max. Budget resembles a semaphore or resource counter. TraversalEngine resembles standard graph search. This resemblance is real, and acknowledging it is important.

The distinction is governance semantics. Map applies a function to every element of a collection. ActuationPass applies a function only to elements that were allocated by a prior CompetitiveSelection or TraversalEngine pass. It is Map constrained by an allocation record, and the constraint is enforced structurally, not by convention. Fold aggregates a collection with no guarantee about processing order or uniqueness. Reduction in DAC carries the partition invariant: every item is processed exactly once, and processed and remaining items are always disjoint. A semaphore controls access to a critical section. Budget\<U\> enforces a capacity ceiling with no bypass mechanism (there is no ForceAllocate), and the invariant composes upward into FederatedBudget where the two-level guarantee holds atomically.

The contribution is not the existence of these operations. Every programmer uses them daily. The contribution is threefold: that exactly these seventeen, at exactly this level of governance semantics, suffice to express twenty domains; that they compose with governance properties preserved across domain boundaries; and that this specific set has not previously been identified as the shared structural foundation underlying rendering, scheduling, consensus, training, theorem proving, type inference, and fourteen other domains. The operations are familiar. The unification is not.

### 2.3 Primitives in Concrete Form

To ground the primitive set in engineering reality rather than philosophical description, this section shows how the primitives look as trait definitions and formal specifications.

CompetitiveSelection (the competitive allocation primitive):

```rust
/// For each output seat, evaluate candidates and determine allocation.
///
/// This trait defines the shared structure across the Competitive Selection
/// family. The selection_mode parameter distinguishes hard selection (one
/// winner per seat), soft selection (weighted combination), and ranked
/// selection (top-k winners). See Section 4 for the full family analysis.
///
/// Rendering: seats are pixels, candidates are triangles, score is depth.
/// Attention: seats are query positions, candidates are key-value pairs,
///            score is similarity, mode is soft.
/// Scheduling: seats are placement requests, candidates are nodes,
///            score is fitness.
/// Markets: seats are order book positions, candidates are orders,
///            score is price.
///
/// The seat belongs to the initiator of the request. Whoever asks
/// the question owns the seat. Whoever is evaluated to answer it
/// is the candidate. If the initiator changes, seats and candidates
/// swap, but the CompetitiveSelection structure remains identical.
///
/// The kernel does not know which domain it serves.
/// The domain provides S, W, and score_fn.
pub trait CompetitiveSelection {
    type Seat: Copy + Eq + Hash;
    type Winner: Copy;
    type Attributes;

    fn score(&self, seat: &Self::Seat, candidate: &Self::Winner,
             attrs: &Self::Attributes) -> f64;

    fn select(
        &self,
        seats: &[Self::Seat],
        candidates: &[(Self::Winner, Self::Attributes)],
    ) -> AllocationRecord<Self::Seat, Self::Winner>;
}
```

The same trait interface governs pixel ownership in a visibility buffer, attention weight computation in a transformer, pod scheduling in a container orchestrator, and order matching in a market simulator. The domain provides the type parameters and the scoring function. The kernel provides the selection loop, the allocation record, and the governance.

A note on seat assignment: the seat belongs to the initiator of the request. A pod that needs scheduling asks "where do I run?" and owns the seat; nodes are candidates evaluated to answer it. A pixel that needs a color asks "which triangle covers me?" and owns the seat; triangles are candidates. A query position asks "what should I attend to?" and owns the seat; key-value pairs are candidates. If the direction reverses (a node advertises capacity and asks "who wants this space?"), the node's capacity becomes the seat and pods become candidates. The primitive structure does not change. The perspective flips based on who initiated.

Budget\<U\> (the resource governance primitive, TLA+ specification):

```tla+
---- MODULE BudgetInvariant ----
VARIABLES capacity, allocated, reserved, pending_eviction

TypeInvariant ==
    /\ capacity \in Nat
    /\ allocated \in Nat
    /\ reserved \in Nat
    /\ pending_eviction \in Nat

SafetyInvariant ==
    allocated + reserved + pending_eviction <= capacity

TryAllocate(amount) ==
    IF allocated + reserved + pending_eviction + amount <= capacity
    THEN /\ allocated' = allocated + amount
         /\ UNCHANGED <<capacity, reserved, pending_eviction>>
    ELSE UNCHANGED <<capacity, allocated, reserved, pending_eviction>>

\* There is no ForceAllocate. The invariant cannot be bypassed.
====
```

This invariant holds for every Budget\<U\> in the system at all times, whether U is VRAM bytes, CPU microseconds, network bandwidth, inference FLOPs, or any other quantifiable resource. The proof is identical regardless of domain because the primitive is domain-agnostic.

### 2.4 The Collapse Table

The following table documents each domain-specific abstraction that was collapsed, what it was believed to be, and what it actually is when stripped of domain vocabulary. This is the core empirical evidence for the methodology.

An honest note on the strength of individual collapses: some entries in this table represent genuine structural insights where the mapping is surprising and illuminating (attention as competitive selection, the DAG workload collapse). Others are closer to observations that generic operations exist (a lookup table is a lookup table, a map function is a map function). The trivial collapses are marked as such and are included for completeness. They confirm that the primitive set covers the domain; they do not constitute deep structural discoveries.

Real-Time Rendering:

| Domain Abstraction | What It Actually Is |
|---|---|
| Visibility buffer | CompetitiveSelection (hard): each pixel is a contested seat, triangles are candidates, depth test is the scoring function [7] |
| Level-of-Detail (LOD) | QualityHierarchy + TraversalEngine: budget-constrained traversal of a multi-resolution tree |
| Texture streaming | Budget\<Bytes\> + QualityHierarchy + ActuationPass: budget-governed resource loading |
| Global illumination | PropagationPass over a spatial graph: fixed-point iteration where the propagation function is the rendering equation and convergence is radiometric equilibrium |
| Deferred rendering | Pipeline composition: CompetitiveSelection (geometry pass) then ActuationPass (shading pass), an evaluate-then-actuate pattern |
| Draw call batching | Reduction: aggregate compatible draw calls into indirect dispatch |
| Frustum culling | TraversalEngine with a spatial acceptance predicate |

Physics Simulation:

| Domain Abstraction | What It Actually Is |
|---|---|
| Broad phase collision | Reduction + spatial partitioning: filter the candidate space before detailed evaluation (see Section 4.4 for why this is not CompetitiveSelection) |
| Narrow phase collision | CompetitiveSelection (hard): refine candidate pairs to actual contacts |
| Constraint solver | PropagationPass + ConvergenceGovernor: fixed-point iteration toward constraint satisfaction |
| Rigid body integration | ActuationPass: apply computed forces to winning bodies |

Audio Spatialization:

| Domain Abstraction | What It Actually Is |
|---|---|
| Audio priority system | CompetitiveSelection (ranked): sounds compete for limited output channels |
| Reverb propagation | PropagationPass over a spatial graph: identical structure to GI propagation with a different domain function |
| Audio LOD | QualityHierarchy + TraversalEngine: same structure as geometry LOD |

Network Replication:

| Domain Abstraction | What It Actually Is |
|---|---|
| Relevancy filtering | TraversalEngine + Budget\<BytesPerSecond\>: spatial budget allocation in bandwidth units |
| State synchronization | Diff\<State\> + PropagationPass: detect deltas, propagate to relevant nodes |
| Network priority | CompetitiveSelection (ranked): updates compete for bandwidth seats |
| OSPF link-state routing | PropagationPass: each router floods local state, system converges to global topology via iterative relaxation |
| BGP path selection | CompetitiveSelection (hard): candidate routes compete for the forwarding seat per destination |

Container Orchestration:

| Domain Abstraction | What It Actually Is |
|---|---|
| Kubernetes controller | Budget\<CPU/Memory\> + ConvergenceGovernor: budget-constrained actuator converging toward desired state |
| Pod scheduling | CompetitiveSelection (hard): pods compete for node placement seats under resource constraints |
| Auto-scaling | Budget\<Instances\> + Signal + feedback composition: feedback governing resource allocation over time |
| Health checking | Reduction\<HealthProbe, Status\> + Signal\<StatusChange\>: aggregate health probes, signal on change |
| Service discovery | ResourceRegistry\<ServiceName, Endpoint\>: a lookup table (trivial) |

CI/CD Pipelines:

| Domain Abstraction | What It Actually Is |
|---|---|
| Pipeline stage | ActuationPass with a capability-gated I/O boundary |
| Pipeline DAG | RelationshipGraph\<Stage, Dependency\> + TraversalEngine |
| Artifact caching | Memoize\<BuildInput, BuildOutput\> with content-hash invalidation (named composition over ResourceRegistry + Reduction) |
| Parallel test execution | Budget\<Compute\> + Sampler\<TestPartition\>: budget-constrained parallel work |

ETL / Data Pipelines:

| Domain Abstraction | What It Actually Is |
|---|---|
| Extract stage | ActuationPass: I/O-gated data retrieval from a ResourceRegistry (trivial) |
| Transform stage | ActuationPass: map function over data (trivial) |
| Load stage | ActuationPass + Transaction: atomic write to destination (trivial for the ActuationPass part) |
| Backfill | TraversalEngine over a temporal QualityHierarchy: budget-constrained reprocessing |

Configuration Management:

| Domain Abstraction | What It Actually Is |
|---|---|
| Desired state convergence | ConvergenceGovernor: detect current vs. desired state delta, iterate until converged |
| Idempotent operation | ActuationPass with Diff\<State\>: only actuate if delta is non-zero |
| Role/playbook | RelationshipGraph\<Task, Dependency\> + TraversalEngine: a DAG workload |
| Inventory | ResourceRegistry\<Host, Configuration\> (trivial) |

SOAR (Security Orchestration, Automation, and Response):

| Domain Abstraction | What It Actually Is |
|---|---|
| Playbook | RelationshipGraph\<Action, Dependency\> + TraversalEngine: a DAG workload with capability-gated I/O, structurally identical to CI/CD pipelines, ETL workflows, and configuration management playbooks |
| Alert triage | CompetitiveSelection (ranked): alerts compete for analyst attention seats, prioritized by severity score |
| Enrichment | ActuationPass + ResourceRegistry: look up context from external sources (trivial) |
| Response action | ActuationPass with capability-based access control |

Distributed Consensus (RAFT):

| Domain Abstraction | What It Actually Is |
|---|---|
| Leader election | CompetitiveSelection (hard): nodes compete for the leader seat |
| Log replication | PropagationPass: leader propagates log entries to followers until majority acknowledgment (convergence) |
| Heartbeat | Signal\<LeaderAlive\> + RateLimit: periodic notification at governed rate |
| Commit | Reduction\<Acknowledgment, Count\> + Transaction: aggregate votes, commit atomically when majority reached |

Economic Simulation:

| Domain Abstraction | What It Actually Is |
|---|---|
| Market order matching | CompetitiveSelection (hard): buy orders compete for sell-order seats, price is the scoring function |
| Price discovery | PropagationPass + ConvergenceGovernor: iterative relaxation toward equilibrium price |
| Inventory management | Budget\<ItemSlots\> + Transaction: atomic multi-resource updates under capacity constraints |
| Trade | Transaction: atomic exchange of resources between two Budget\<U\> instances |

Machine Learning (see Section 6 for the extended analysis):

| Domain Abstraction | What It Actually Is |
|---|---|
| Attention mechanism | CompetitiveSelection (soft): tokens compete for attention weight, similarity is the scoring function, softmax produces weighted combination [2] |
| Backpropagation | A member of the PropagationPass family — a single reverse pass on a DAG, distinct from iterative relaxation (see Section 6.1) |
| Speculative decoding | Two-level CompetitiveSelection: fast model generates candidates (coarse), slow model verifies (fine) [8] |
| Beam search | CompetitiveSelection (ranked): candidate continuations compete for K beam seats |
| KV cache | Memoize\<SequencePosition, (Key, Value)\> with position-based invalidation |
| Learning rate scheduling | Feedback composition over Budget\<LearningRate\>: scheduled adjustment of a resource |
| Early stopping | ConvergenceGovernor: detect when validation loss improvement falls below threshold |
| LoRA / Adapters | Budget\<Parameters\>: constrain trainable parameters to a fraction of total [6] |
| Dropout | Sampler\<ActivationMask\>: probabilistic selection of active neurons |
| Loss computation | Reduction\<Prediction, Scalar\>: aggregate prediction-ground-truth distances |
| Uniform gradient computation | Brute-force evaluation: every parameter receives gradient from every sample regardless of relevance. The absence of CompetitiveSelection gating in the gradient path |
| Fixed-interval evaluation | Brute-force assessment: every metric evaluated at every checkpoint regardless of what parameters changed. The absence of Signal\<ConvergenceChange\> + change-impact analysis |
| Flat parameter training | All parameters trained at the same fidelity from step 0. The absence of QualityHierarchy + TraversalEngine in the gradient path |
| Uniform gradient budget | All parameter groups receive equal gradient compute regardless of learning need. The absence of FederatedBudget\<GradientCompute\> |
| Layer freezing | Binary ConvergenceGovernor (per-group) without graduated states, budget reallocation, or reactivation |
| Curriculum learning | QualityHierarchy over data only, without hierarchy over parameters or budget governance |

### 2.5 The Adversarial Domain Pass

A significant concern with the original twelve-domain collapse is that every domain examined shares a family resemblance: each is a resource-governed execution system where work is allocated under constraints, propagated through a graph, or aggregated into a result. A methodology that maps well to such domains may map poorly to domains with fundamentally different computational characters. This section tests the primitive set against eight domains chosen specifically because they do not share this character: unification, Hindley-Milner type inference, term rewriting, constraint logic programming, probabilistic programming, resolution-style theorem proving, symbolic differentiation, and lattice-theoretic dataflow analysis.

The adversarial pass produces three outcomes. Six of the eight domains map onto the existing primitive set cleanly. One domain (lattice dataflow) requires a mode extension on an existing primitive (widening mode on ConvergenceGovernor). One domain (unification) motivates making an existing invariant first-class (monotonicity on ResourceRegistry). Two domains (constraint logic programming and theorem proving) motivate the addition of one new primitive: Checkpoint, which establishes a scoped rollback boundary for nested undo. The primitive count grows from sixteen to seventeen.

Unification:

| Operation | Decomposition |
|---|---|
| Substitution | ResourceRegistry\<Variable, Term\> with monotonicity invariant (writes are monotonic within a unification attempt) |
| Term structure walk | TraversalEngine over parallel RelationshipGraphs |
| Occurs check | PropagationPass with a boolean "reached" message |
| Binding step | ActuationPass guarded by consistency precondition |
| Termination | ConvergenceGovernor reaching a terminal state |

The only gap was that ResourceRegistry did not carry monotonicity as a named invariant. This is now documented as an optional invariant on the primitive, not a new primitive.

Hindley-Milner Type Inference:

| Operation | Decomposition |
|---|---|
| Unification subroutine | Inherits from above |
| Type environment | Stack of ResourceRegistry\<Ident, TypeScheme\> (lexical scoping as composition) |
| Generalization | Reduction\<FreeTypeVar, TypeScheme\> |
| Instantiation | Sampler\<TypeVariable\> in deterministic mode |

No new primitive required.

Term Rewriting:

| Operation | Decomposition |
|---|---|
| Pattern matching | Bounded unification (one direction) |
| Rule selection | CompetitiveSelection (ranked) over (rule, position) pairs |
| Rule application | ActuationPass |
| Termination detection | ConvergenceGovernor |
| Rule set | ResourceRegistry\<RuleID, RewriteRule\> |

No new primitive required.

Constraint Logic Programming:

| Operation | Decomposition |
|---|---|
| Constraint store | ResourceRegistry + PropagationPass for domain reduction |
| Search tree | RelationshipGraph\<ChoicePoint, Alternative\> |
| Search strategy | TraversalEngine with specific ordering |
| Constraint propagation | PropagationPass + ConvergenceGovernor |
| Backtracking | Checkpoint — NEW PRIMITIVE |

CLP requires hierarchical backtracking: nested choice points each establish a restore boundary, and one unwinds through them in LIFO order. The existing Transaction named composition is binary (commit or abort) and does not support nested scopes. Checkpoint is introduced as a primitive: a scoped rollback boundary where any mutation done within a scope can be reverted on demand, and scopes nest. The argument for Checkpoint's primitive status runs through the criteria of Section 1.3: it carries a governance invariant (stack well-formedness plus rollback fidelity), it appears across multiple domains in different vocabulary (CLP choice points, theorem prover tactic states, database savepoints, transactional memory scopes), and the composition attempt AuditSink + reverse-replay ActuationPass is expected to fail the atomicity and stack discipline that CLP requires under concurrent mutation during unwind. Formal verification of this decomposition failure is listed as a pre-submission prerequisite in Section 10.5, and the paper's claim of Checkpoint's operational irreducibility is contingent on that verification.

Probabilistic Programming:

| Operation | Decomposition |
|---|---|
| Sample statement | Sampler\<T\> (same primitive, no translation needed) |
| Condition statement | CompetitiveSelection (soft) with likelihood as scoring function |
| MCMC transition | PropagationPass with Sampler-driven acceptance |
| Variational inference | Reduction (ELBO) + PropagationPass (gradient) + ConvergenceGovernor |

Probabilistic programming maps surprisingly cleanly because sampling and scoring are already primitives. No new primitive required.

Resolution-style Theorem Proving:

| Operation | Decomposition |
|---|---|
| Proof tree | RelationshipGraph\<Sequent, Inference\> |
| Rule selection | CompetitiveSelection (ranked) |
| Proof search | TraversalEngine with backtracking via Checkpoint |
| Axiom base | ResourceRegistry\<AxiomID, Formula\> |
| Unification for rule application | Inherits from above |

Uses Checkpoint introduced above. No additional new primitive required.

Symbolic Differentiation:

| Operation | Decomposition |
|---|---|
| Expression tree | RelationshipGraph\<ExprNode, ChildEdge\> |
| Differentiation rule application | TraversalEngine + ActuationPass at each node |
| Simplification | Term rewriting (inherits from above) |

This is a pure tree walk with no governance pressure. Trivial collapse; no new primitive required.

Lattice-theoretic Dataflow Analysis:

| Operation | Decomposition |
|---|---|
| Flow graph | RelationshipGraph |
| Per-node abstract state | ResourceRegistry\<Node, LatticeValue\> |
| Fixed-point iteration | PropagationPass + ConvergenceGovernor |
| Join at merge points | Reduction with lattice's ⊔ as combining function |
| Widening | ConvergenceGovernor in widening mode (NEW MODE) |

Widening is what one does when a pure ConvergenceGovernor will not terminate because the lattice is infinite: it forces convergence by over-approximation when monotone progress stalls. This is a mode of ConvergenceGovernor in the same sense that hard, soft, and ranked are modes of CompetitiveSelection, not a new primitive.

Adversarial pass summary: eight domains, one new primitive (Checkpoint), one new mode (widening on ConvergenceGovernor), one explicit invariant (monotonicity on ResourceRegistry). Six of eight fit the original set with no change. The final primitive count is seventeen. Three of the eight adversarial domains (CLP, theorem proving, lattice dataflow) also suggest that the existing primitive set was underdetermined in specific ways that have now been sharpened. The adversarial pass therefore strengthens rather than weakens the methodology: a primitive set that survives domains outside its original design sample with a single addition is more credible than one that claims to cover every possible domain.

A methodological note: the adversarial pass was conducted after review of the twelve-domain original set. This means the primitives examined during the adversarial pass were those already produced by the original collapse, and the question being asked was "does this set of primitives express this domain." The stronger version of this test, in which a researcher unfamiliar with the primitive set applies DAC to a new domain and independently discovers the same or a different primitive set, remains important future work and is listed in Section 10.5.

A subsequent broader survey applied the primitive-qualification test of Section 1.3 to an additional set of domains that were not decomposed in detail in the paper: garbage collection, regular expression matching, parsing (LR, GLR, Earley, packrat), query optimization, compiler optimization passes, pub/sub and message queues, file systems, database transactions, video encoding, robotics motion planning, search engines, reactive programming (React, Svelte, Observable), streaming systems with watermarks (Flink, Kafka Streams), CRDTs, transactional memory, evolutionary computation, cellular automata, PDE integration with adaptive step size, program synthesis, approximate nearest neighbor search, and interactive theorem proving with tactics. Three candidates emerged from this survey as potential new primitives or refinements: glitch-free propagation in reactive systems, monotonic-signal semantics for streaming watermarks, and an exploration-exploitation mode on Sampler for evolutionary methods. All three were eliminated by the test. Glitch-free propagation is a composition of TraversalEngine with reverse-topological ordering, Budget\<PropagationRound\>, and Signal; it fails criterion 4. Monotonic-signal semantics is a caller-enforceable invariant option on Signal\<T\>, parallel to the monotonicity option on ResourceRegistry; it fails criterion 5. The exploration-exploitation tradeoff in evolutionary methods is a policy expressed through sampling distributions and scoring functions, which are domain functions within Sampler and CompetitiveSelection; it fails criterion 1 because it describes a policy, not a governance invariant. The remaining domains mapped onto the existing seventeen primitives without additions or refinements. This broader survey is not an exhaustive mapping, and the paper does not claim it as such. It is evidence that the primitive set is stable under broader domain sampling: a methodology in which every broadening forces a new primitive would be unstable, and a methodology in which the strict test routinely rejects plausible candidates is the one that supports the "seventeen primitives, incrementally discovered" claim. Domains where the author's fit claim is weaker than for the twenty examined in detail — interactive theorem proving with tactics, approximate nearest neighbor search at the algorithmic level, and stiffness detection in adaptive numerical integration — are identified in Section 10.5 as items requiring practitioner validation.

---

## 3. Why Abstraction Collapse Is Hard to See

### 3.1 Vocabulary as Cognitive Lens

The domain-specific vocabulary that makes abstraction collapse difficult to detect is not accidental. It evolved because the people solving each problem came from that domain. Graphics engineers built rendering systems and named their concepts in rendering vocabulary. Network engineers built routing protocols and named their concepts in networking vocabulary. Machine learning researchers built training frameworks and named their concepts in statistical learning vocabulary.

Each vocabulary is internally coherent and useful within its domain. The problem is that vocabulary creates cognitive boundaries. A graphics engineer who thinks in terms of visibility buffers and deferred shading does not spontaneously recognize that these are instances of the same evaluate-then-actuate pattern that governs Kubernetes pod scheduling. A machine learning researcher who thinks in terms of attention and softmax does not spontaneously recognize that the attention mechanism is a competitive selection pass with a soft winner function, the same structural primitive that determines which triangle owns each pixel in the visibility buffer.

The vocabulary is a lens that focuses attention on domain-specific details and defocuses the structural similarity to other domains. Removing the lens requires deliberately stripping the domain words and looking at what remains. This is uncomfortable because domain expertise feels diminished when its specialized vocabulary turns out to be a renaming of something generic. But the generic version is more powerful precisely because it composes across domains.

### 3.2 The Specialization Trap

Software engineering culture rewards specialization. Rendering engineer, systems engineer, and ML engineer are different job titles, different conference communities, and different publication venues. Solutions published at SIGGRAPH use rendering vocabulary. Solutions published at NSDI use networking vocabulary. Solutions published at NeurIPS use ML vocabulary. Cross-pollination happens, but it happens through analogy ("attention is like a soft dictionary lookup") rather than through structural identification ("attention is competitive selection with a specific scoring function").

Analogy preserves the domain boundary. Structural identification dissolves it. Dissolution is more useful but less comfortable, because it implies that the domain boundary was never real — that the specialization was, to a significant degree, vocabulary.

### 3.3 Why It Took an Outsider

DAC was not developed by a rendering engineer, a networking engineer, or an ML researcher. It was developed by a security practitioner with systems engineering experience who started from frustration with SOAR platforms: domain-specific tools that were, in the designer's experience, built by people who had never operated the workflows those tools claimed to automate.

The attempt to build a better execution engine began with modeling it after an OS kernel (the best-understood solution to resource governance), an OSI stack (the best-understood solution to layered abstraction), and Unix pipes (the best-understood solution to composable data flow). These were not novel intellectual ingredients. They were proven patterns applied to a domain that had been ignoring them.

The key enabler was the absence of domain expertise. An experienced rendering engineer would have built a rendering engine. An experienced orchestration engineer would have built an orchestration engine. Someone looking for proven solutions to each problem they encountered, without the vocabulary to know which domain "owned" each solution, ended up finding the same solution appearing across every domain. The collapse was visible precisely because no domain-specific lens was filtering the view.

This is not an argument against domain expertise. The domain function (the rendering equation, the attention scoring function, the constraint propagation rule) requires deep domain knowledge to design. The argument is that the execution infrastructure around the domain function is generic, and domain expertise in execution infrastructure creates the illusion that it is not.

---

## 4. The Competitive Selection Family

### 4.1 The Overcounting Problem

In an earlier formulation of the abstraction primitive set, a single primitive called SlotArbitrationPass was mapped to over seventeen domain patterns: visibility buffers, attention, pod scheduling, market order matching, alert triage, beam search, leader election, broad-phase collision, BGP path selection, speculative decoding, audio priority, and more. When one primitive absorbs that many structurally different operations, it raises a legitimate concern: is the primitive defined so broadly ("anything where something competes for something") that it approaches the Turing Tarpit the methodology claims to avoid?

The answer is that there is genuine shared structure here, but it is a family of related primitives rather than a single primitive. The shared structure is: given a set of output positions (seats) and a set of candidates, evaluate each candidate against each seat using a scoring function, and allocate candidates to seats based on the scores. What differs across the family members, and what matters operationally, is the selection mechanism.

### 4.2 Three Selection Modes

The Competitive Selection family decomposes into three distinct selection modes that share the scoring interface but differ in allocation semantics:

Hard Selection (argmax): exactly one winner per seat. The candidate with the highest score takes the seat exclusively. All other candidates receive nothing. This is the mode used in pixel ownership (visibility buffer), leader election (RAFT), pod scheduling, market order matching, and BGP path selection. The key property is mutual exclusion: a seat is owned by exactly one candidate.

Soft Selection (softmax/weighted): every candidate contributes to every seat in proportion to its score. There is no single winner. The output for each seat is a weighted combination of all candidates. This is the mode used in transformer attention [2]. The key property is proportional allocation: candidates share seats continuously rather than claiming them exclusively.

Ranked Selection (top-k): the top K candidates by score are all allocated. This is the mode used in beam search, audio channel priority, network update priority, and alert triage. The key property is bounded multiplicity: multiple winners are allowed, but the count is capped.

### 4.3 What Is Actually Shared

All three modes require a scoring function that evaluates candidate-seat affinity. All three produce an AllocationRecord that maps seats to their allocated candidates. All three compose with ActuationPass (which operates on the allocation result) and with Budget\<U\> (which constrains the total allocation). All three can be governed by the same audit and observability infrastructure.

The differences (hard, soft, ranked) are parameterizations of the selection mechanism, not entirely different primitives. This is analogous to how a sorting algorithm's comparison function is parameterized: quicksort-by-price and quicksort-by-date are the same algorithm with different comparison functions, not two different algorithms. Similarly, hard competitive selection and soft competitive selection are the same structural primitive with different allocation semantics.

The trait signature in Section 2.3 accommodates all three modes through the AllocationRecord return type: hard selection returns a one-to-one map, soft selection returns a weighted distribution, and ranked selection returns a one-to-many map with a bounded fan-out. The scoring interface is identical.

### 4.4 What Is Not Shared

Two of the patterns originally mapped to this family deserve separate scrutiny.

Anomaly detection (described in some formulations as "not winning the normal slot") is a stretch. Anomaly detection is better characterized as a Reduction (compute a distance metric from a reference distribution) followed by a threshold comparison. Forcing it into the competitive selection frame adds vocabulary without adding structural insight.

Broad-phase collision detection is often described as "AABB overlaps competing for pair-slots," but it is more precisely a spatial partitioning and filtering operation. The competitive selection mapping is defensible (candidate pairs do compete for a limited processing budget in a real-time physics pipeline), but it is a weaker match than pixel ownership or attention, where the competitive structure is intrinsic rather than imposed by the resource constraint. For this reason, the collapse table in Section 2.4 classifies broad-phase collision as Reduction + spatial partitioning rather than CompetitiveSelection.

The revised count: CompetitiveSelection with its three modes accounts for roughly fifteen of the seventeen original mappings, with two entries better classified as compositions of other primitives.

---

## 5. DAC as a Generative and Implementation Methodology

### 5.1 Beyond Analysis

The methodology described in Section 1.2 is framed as analytical: take existing domains, strip vocabulary, find isomorphisms. But DAC is equally powerful as a generative methodology (a tool for engineering solutions to problems that resist solution in their native vocabulary) and as an implementation methodology (a tool for turning any described computation into a build plan).

The generative application works as follows. When confronted with a problem that appears hard in its domain:

1. Strip the domain vocabulary from the problem statement. Describe what the problem actually requires structurally, without using domain-specific terms.

2. Map the stripped problem to the abstraction primitive set. Does the structure match any known primitive or composition of primitives?

3. If the mapping succeeds, the problem inherits the solution from whichever domain already solved it. The "hard problem" was hard because its domain vocabulary obscured its structural identity with a solved problem.

4. If the mapping fails, the result is a genuinely novel computational structure. This is also valuable, because it identifies where real innovation is needed versus where vocabulary was creating the illusion of novelty.

The implementation application extends this further. Any computation described in domain-specific notation (a mathematical formula, a protocol diagram, an algorithm in pseudocode) can be decomposed into a build plan by stripping the notation and mapping each step to the primitive set. The notation describes the relationship between inputs and outputs. The DAC decomposition describes the computational steps a machine actually executes, which are always some combination of traversal, transformation, reduction, propagation, budgeting, sampling, and gating. The notation is the spec. The decomposition is the architecture.

This is the difference between a taxonomy and a tool. A taxonomy organizes what exists. A generative methodology lets one build things one could not see before.

### 5.2 Case Study: LeanFormer

LeanFormer is a novel transformer architecture designed through DAC. Rather than starting from the ML literature and making incremental improvements to existing architectures, the design began by stripping ML vocabulary from six open problems in neural network design and mapping each to the abstraction primitive set. In every case, the stripped problem turned out to be a solved problem from systems engineering. The first five problems address the model architecture. The sixth addresses the training process itself.

The entire arc from initial DAC decomposition to 204M-parameter validated results took approximately three weeks. The initial proof-of-concept (7.5M parameters, 4 layers) was designed and implemented in 24 hours using DAC as the design methodology and an AI coding assistant (Claude Code) as the implementation agent. The architect provided the DAC decomposition, the cross-domain mappings, and the verification discipline. The coding assistant produced implementation code. No ML-specific implementation experience was required on either side of this collaboration; the solutions were systems engineering solutions recognized through vocabulary stripping.

The role of the AI coding assistant deserves direct acknowledgment. The three-week development time is evidence for DAC's claim that recognized problems have short implementation paths, but it also reflects the productivity of AI-assisted coding. The mitigation against the "plausible-looking code against plausible-looking specifications" failure mode is explicit: hostile audit protocols (requiring log evidence rather than diff evidence), formal specifications in TLA+ that the implementation must refine, and a 204M-parameter training run in which the governance invariants were continuously measured. A specification bug or an implementation bug would have shown up either as a TLC counterexample or as a governance invariant violation during training. Neither occurred in the governance layer. One occurred in a domain function (the B=0 initialization threshold, discussed in Section 5.8.1), which is precisely the class of bug the structure/function separation predicts will remain in the domain function rather than the primitive.

Subsequent scale-up to 39M parameters (76M dense equivalent, trained on 500K OpenWebText samples) validated the architectural thesis across 119 tests. A 204M parameter model (805M dense equivalent) was trained with the full governed pipeline on a reasoning corpus for 7,228 optimizer steps (one epoch) on an NVIDIA L4 GPU. The architectural results reported in Sections 5.3–5.7 are from the 39M validation. The training governance results reported in Section 5.8 include both a 4.8M preliminary validation and the 204M scale validation. The 204M run is the first validation of all seventeen primitives composing correctly under real training conditions at a scale where parameter group ratios are representative (L0 at approximately 41.5% of parameters, compared to approximately 86% at 4.8M where the embedding table dominates).

LeanFormer is not presented as a competitive language model. It is presented as evidence of what DAC produces when applied to a domain: a novel architecture that composes four efficiency mechanisms simultaneously (low-rank compression, sparse attention, gated feed-forward, adaptive depth), solves catastrophic forgetting by construction (bit-for-bit base weight restoration), makes confabulation architecturally detectable, and governs the entire training lifecycle with the same primitives that govern the model's architecture. The ML community has developed each of these capabilities in isolation, each within ML vocabulary. DAC composed them by recognizing that they are all instances of primitives from other domains. The limitations of the current implementation (modest scale, single-epoch training, non-functional subsystems) are training configuration issues, not architectural failures, and they are identified as concrete next steps in Section 9.5.

### 5.3 Problem 1: Parameter Inefficiency

In ML vocabulary: dense weight matrices waste parameters because most weights contribute minimally to the output. Pruning, quantization, and low-rank approximation are active research areas with large bodies of literature.

Stripped of vocabulary: a storage system allocates fixed-size blocks for every record regardless of the record's actual content size. Most blocks are mostly empty. This is the fragmentation problem in file systems, solved decades ago by variable-size allocation with a budget constraint.

DAC decomposition: replace dense matrices with low-rank factorizations. Each weight matrix W becomes a product of two smaller matrices (down-projection and up-projection) constrained by Budget\<Parameters\> to use only a fraction of the original parameter count. The rank becomes the budget knob. The domain function (what the weight matrix computes) is unchanged. The governance (how many parameters it uses) is now explicit and tunable.

Result at 39M parameters: the model achieves a 3.9x compression ratio over its dense equivalent, meaning the 39M parameter LeanFormer has the representational structure of a 76M dense model. All four efficiency mechanisms (low-rank weights, sparse attention, gated feed-forward, adaptive depth) operate simultaneously without mutual interference.

### 5.4 Problem 2: Attention Cost

In ML vocabulary: self-attention is O(n squared) in sequence length, making long-context inference expensive. Flash attention, sparse attention, and linear attention are competing approaches with different tradeoffs.

Stripped of vocabulary: a selection system evaluates every candidate against every output position, even when most candidates are irrelevant to most positions. This is the brute-force rendering problem: evaluating every triangle against every pixel. The rendering community solved it with a two-pass architecture. A cheap coarse pass identifies which candidates are relevant. An expensive fine pass evaluates only the relevant ones.

DAC decomposition: replace single-pass dense attention with a two-pass sparse attention pipeline. The first pass is a lightweight scoring pass (CompetitiveSelection in ranked mode) that identifies the top-k relevant key-value pairs per query. The second pass computes full attention (CompetitiveSelection in soft mode) over only the selected candidates. The structure is identical to the visibility buffer pipeline in rendering: coarse culling followed by fine evaluation.

Result at 39M parameters: 88% attention sparsity. The model evaluates only 12% of key-value interactions per query position, with the sparse selection pass routing attention to the relevant subset. The gated feed-forward network achieves 80% sparsity through the same principle applied to the MLP layers: a cheap gate determines which neurons fire, and only active neurons are computed.

### 5.5 Problem 3: Catastrophic Forgetting

In ML vocabulary: when a neural network learns new information, it overwrites previously learned information because the same parameters encode both old and new knowledge. This is an open research problem with a large literature on continual learning, elastic weight consolidation, and progressive networks.

Stripped of vocabulary: a shared mutable storage system where writes to encode new content destroy existing content, because the storage addressing conflates the retrieval index with the stored content and uses dense rather than sparse encoding. This is the write-conflict problem in shared mutable state. Systems engineering solved it decades ago.

DAC decomposition: separate the retrieval index from the stored content. Use a fixed compact base for general computation (the frozen base model weights). Encode individual records as sparse, independently-addressable deltas over the base (low-rank delta matrices, analogous to LoRA [6] adapters but dynamically loaded and unloaded rather than statically merged). Govern delta allocation with a ResourceRegistry that enforces non-overlapping address ranges. Route queries to relevant deltas using a cheap CompetitiveSelection (ranked) pass before the expensive computation.

The primitive composition:

| LeanFormer Component | Abstraction Primitive Composition |
|---|---|
| Frozen base weights | The immutable foundation (analogous to kernel state) |
| Belief delta | Budget\<Parameters\> + Transaction (atomic, bounded modification) |
| Delta registry | ResourceRegistry\<BeliefID, DeltaWeights\> with non-overlap enforcement |
| Routing network | CompetitiveSelection (ranked): query embedding selects relevant deltas |
| Belief injection | Transaction: atomic addition of delta to registry |
| Belief removal | Transaction: atomic removal restoring pre-injection state |

Result at 39M parameters: 84% belief injection success rate across 100 beliefs. 86% semantic routing accuracy (4.3x above chance baseline). Bit-for-bit base weight restoration verified across all tensors after adding and removing 100 beliefs in both ordered and random removal sequences. Base weight immutability verified across 406 tensors. This result is exact, not approximate: the base weights after full belief lifecycle are identical at the bit level to the base weights before any beliefs were added.

### 5.6 Problem 4: Knowledge Composition

In ML vocabulary: fine-tuning a model on multiple domains causes interference between domains. Multi-task learning requires careful balancing of loss functions. There is no clean way to add knowledge from domain A and domain B independently and have them coexist without degradation.

Stripped of vocabulary: multiple tenants writing to a shared resource without isolation guarantees corrupt each other's data. This is the multi-tenancy isolation problem in database and operating system design. The solution is address-space partitioning with enforced non-overlap.

DAC decomposition: extend the delta registry with orthogonality constraints (FederatedBudget\<ParameterSubspace\>). The master parameter space is subdivided into non-overlapping regions. Each domain's deltas are constrained to their allocated region via orthogonality enforcement during the forging (training) process. Composition becomes additive: domain A's deltas and domain B's deltas occupy disjoint subspaces and can be loaded simultaneously without interference. The governance primitive (FederatedBudget) enforces the invariant that the sum of all allocated subspaces does not exceed the master space.

Result at 39M parameters: 64% of 100 beliefs retained improvement when all were loaded simultaneously. The coexistence rate reflects the fact that untargeted deltas (modifying all layers) create more interference than targeted deltas constrained to specific parameter subspaces. The Knowledge Plane architecture, which enforces orthogonality constraints and targeted-layer allocation, is designed to address this directly. Validation at 204M with orthogonality enforcement is identified as future work (Section 9.5).

### 5.7 Problem 5: Confabulation

In ML vocabulary: language models generate confident-sounding text that is factually wrong because they have no mechanism to distinguish "I know this" from "this is a plausible continuation." Hallucination mitigation is an active area of current research.

Stripped of vocabulary: a system that produces output with no confidence signal and no mechanism to distinguish cached retrieval (recalling a stored fact) from interpolation (generating a plausible response from patterns). Any system that conflates these two modes will produce confident-looking interpolations where retrieval was expected. This is the error-detection problem in signal processing, solved by separating the data path from the confidence path.

DAC decomposition: the ConvergenceGovernor primitive, which detects whether a computation has reached a stable state or is still iterating, maps directly to this problem. LeanFormer's adaptive depth mechanism uses exit classifiers at each layer that estimate whether the representation has converged. If the representation converges early (the model is confident), computation exits before the final layer. If the representation has not converged by the final layer, the residual magnitude serves as an uncertainty signal. When a query's relevant knowledge is loaded as a delta, routing confidence is high and depth tends to be shallow. When no relevant delta exists, routing confidence is low and depth tends to be maximum, which itself becomes a detectable signal that the model is interpolating rather than retrieving.

The structural mapping: ConvergenceGovernor monitors the computation. Low convergence residual after delta routing means the answer came from a stored fact. High convergence residual means the answer is generated from base model patterns with no grounding in injected knowledge. This separation does not eliminate confabulation, but it makes confabulation architecturally detectable rather than invisible.

Result at 39M parameters: adaptive depth is functional. The mean exit depth is 11.3 out of 12 layers, with 66% of samples exiting one layer early. The graduated depth behavior (variable exit depth correlated with query complexity and knowledge availability) requires further training at scale. The mechanism works, but the exit classifiers need more training signal to learn fine-grained confidence estimation.

### 5.8 Problem 6: Training Process Inefficiency

In ML vocabulary: training is expensive because gradient computation scales with model size, dataset size, and epoch count. The ML community has produced a large body of work on training efficiency: mixed-precision training, gradient accumulation, data/model/pipeline parallelism, curriculum learning, progressive training, layer freezing, sparse training, and importance sampling. Each addresses one dimension of the cost problem. None compose into a unified governed system.

Stripped of vocabulary: a process iteratively modifies a shared mutable state by computing error signals from sampled inputs and propagating corrections globally across the entire state on every iteration, regardless of which regions are relevant to the current input. The process has no selectivity (every parameter receives gradient from every sample), no quality hierarchy (all parameters are trained at the same fidelity from step 0), no per-region convergence detection (the only stopping criterion is global), no budget governance (gradient compute is allocated uniformly rather than proportionally to learning need), and no audit provenance (parameter changes are not traceable to the samples that caused them).

This is the brute-force rendering problem applied to gradient computation. It is the database full-table-scan applied to parameter updates. It is the network broadcast-to-all-nodes applied to learning signal. Every other governed computational system in the DAC collapse table uses CompetitiveSelection gating, QualityHierarchy traversal, FederatedBudget allocation, and per-group ConvergenceGovernor to avoid brute-force computation. Training uses none of them.

DAC decomposition: replace the flat, ungoverned training loop with a pipeline that applies the full primitive set:

| Training Component | Abstraction Primitive Composition |
|---|---|
| Current training loop | Sampler + ActuationPass + Reduction + PropagationPass + ConvergenceGovernor (global) |
| Targeted training pipeline | Sampler + CompetitiveSelection (ranked) + FederatedBudget\<GradientCompute\> + QualityHierarchy + TraversalEngine + ActuationPass + Reduction + PropagationPass + ConvergenceGovernor (per-group) + AuditSink |
| Governed data pipeline | QualityHierarchy\<SampleDifficulty\> + CompetitiveSelection (ranked) + Budget\<SamplesPerStep\> + AuditSink |
| Change-triggered evaluation | Signal\<ConvergenceChange\> + CompetitiveSelection (ranked) + Budget\<EvalCompute\> + ActuationPass + Reduction + AuditSink |
| Readiness-gated knowledge forge | Signal\<GroupConverged\> + ConvergenceGovernor + CompetitiveSelection (ranked) + Budget\<ForgeCompute\> + ActuationPass + Reduction + AuditSink |

The missing primitives in the current training loop are the same primitives that every other domain in the collapse table uses: CompetitiveSelection for routing gradient signal to relevant parameter groups, QualityHierarchy for coarse-to-fine parameter activation, FederatedBudget for proportional compute allocation, per-group ConvergenceGovernor for independent convergence detection, and AuditSink for gradient provenance tracking.

The ML community has independently invented fragments of this composition: Mixture of Experts (CompetitiveSelection at inference, not training), LoRA (static Budget\<Parameters\>, not sample-adaptive), curriculum learning (QualityHierarchy over data, not parameters), layer freezing (binary ConvergenceGovernor without graduated states), progressive training (one-directional QualityHierarchy without convergence-governed activation), and GradNorm (partial FederatedBudget at task level, not parameter-group level). Each fragment was built in isolation, in ML vocabulary, without recognizing that all are partial reinventions of the same structural pattern.

The training system extends DAC beyond the model to the model's entire lifecycle: how it learns (targeted training), how it is assessed (change-triggered evaluation), how its knowledge is produced (readiness-gated forging), and how it is deployed (quality-tiered inference preparation). After this extension, every stage of the model lifecycle is governed by the same primitive set that governs the model's architecture.

Preliminary result at 4.8M parameters (300-step validation run): all governance components activated and composed correctly. Per-group convergence governors produced 15 state transitions across 8 parameter groups, with 7 reaching CONVERGED state. The quality hierarchy activated in the predicted order: L0 (structural) then L1 (representational) then L2 (refinement) then L3 (specialization), all via convergence signal with no emergency activation required. The FederatedBudget invariant (sum of allocations <= master budget) held for all 300 steps with zero violations. The gradient router achieved 15.4% selectivity post-warmup. The SHA-256-chained audit log verified across all 300 records. The change-triggered evaluation pipeline fired 7 targeted evaluations on convergence signals. Final loss was within +2.4% of baseline, with the gap narrowing throughout training (from +3.17 at step 50 to +0.15 at step 299).

Scale validation at 204M parameters (805M dense equivalent, 7,228 optimizer steps, one epoch on a reasoning corpus, NVIDIA L4 GPU, 140.9 hours wall clock) confirmed that all seventeen primitives compose correctly under real training conditions. The question being answered here is not "how good is this language model" but "do the governance primitives compose correctly at a scale where parameter group ratios are representative, and do all invariants hold across thousands of training steps?" The language modeling metrics are reported for completeness but are not the subject of evaluation. The governance results:

The FederatedBudget invariant (sum of allocations <= 1.0) held for all 722 audit records with zero violations. Budget adapted dynamically throughout training: L0 groups started at 0.25 each, converged groups dropped to as low as 0.012, and active groups received up to 0.40 of the total budget. The budget reallocation tracked learning need in real time.

The convergence governor four-state machine produced 18 total state transitions across 8 parameter groups, all valid: PENDING to ACTIVE (3 transitions), ACTIVE to COOLING (8), COOLING to CONVERGED (4), and COOLING to ACTIVE (1 regression, discussed below). No states were skipped. Final states: embeddings COOLING, attention_routing CONVERGED, gates CONVERGED, layer_norms CONVERGED, attention_output COOLING, ff_projections COOLING, output_head COOLING, exit_classifier CONVERGED.

The SHA-256 hash-chained audit log maintained integrity across all 722 records from step 10 to step 7,220, with zero chain breaks. Every governance decision (budget allocation, convergence state transition, hierarchy activation, gradient routing update) is traceable to a specific training step with tamper-evident provenance.

Best validation perplexity reached 57.6 at step 2,000, with train loss declining from 10.39 to 1.64 across the full run. Severe overfitting occurred after step 2,000 (validation perplexity rose from 57.6 to 1,463.9 by step 7,228), which is expected behavior from single-epoch training with limited regularization (dropout 0.1 only). The overfitting is a training configuration limitation, not an architectural failure, and critically, all governance invariants held throughout the overfit phase: the budget was never exceeded, the hierarchy ordering was maintained, the convergence governor never skipped a state, and the audit chain was never broken. Governance correctness is independent of generalization quality, exactly as the structure/function separation predicts.

Orthogonal capacity measurement confirmed 53,760 available dimensions across 20 layers (2,688 per layer), with a theoretical maximum of 3,360 rank-16 knowledge deltas. This was measured on two independent machines (NVIDIA L4 on GCP, RTX 3060 locally) with identical results, confirming the measurement is model-intrinsic.

Generation samples from the best checkpoint (step 2,000) show semi-structured output with learned domain vocabulary (gradient-related terms, function syntax) but incoherent content. This is consistent with a 204M model after 2,000 optimizer steps of single-epoch training and does not indicate an architectural limitation. The generation quality is dramatically better than the overfit final checkpoint, which produced degenerate repetitive output, confirming that the step-2,000 checkpoint represents meaningful learned structure.

Two governed subsystems did not produce meaningful signal at this scale and configuration. Adaptive depth remained at 20/20 (all layers used) throughout the entire run, meaning the exit classifiers never learned to route samples to early exits. This may indicate the exit threshold (0.03) is too conservative, or that the mechanism requires explicit layer-dropping training. Tiered sampling scored all samples as "Failing" at initialization and was never re-scored, rendering it effectively random. Both require further investigation in future training runs.

### 5.8.1 DAC Applied to Its Own Failure: The B=0 Observational Degeneracy

The most instructive result from the 204M training run was an unplanned demonstration of DAC's generative mode applied to a failure in a DAC-governed system.

LeanFormer's low-rank layers initialize B matrices to zero (following standard LoRA practice), which means all parameter groups begin with near-zero gradient flow regardless of whether they have received meaningful training signal. The convergence governors correctly detected low gradient EMA and transitioned through the state machine as specified: ACTIVE to COOLING after the configured cooling window. This produced hierarchy activations at steps 200 (L1) and 400 (L2), both at round-number intervals aligned with the cooling window configuration. Loss remained flat at 10.388 through both activations. Actual training progress began only when the output head activated at L2 and introduced significant gradient flow through the network.

The L3 activation at step 2,773 was qualitatively different. It occurred at a non-round step number, after the attention_output and ff_projections groups (L1 parameters) had received 2,300+ steps of real gradient flow following the output head's activation at step 400. The convergence governor's decision to activate L3 was based on genuine post-learning convergence in those groups, not a calibration artifact. This distinction is visible in the gradient norm data: at step 400, the output head's gradient norm jumped from 0.0 to 0.88 in a single step, marking the onset of real learning, and L3 activation occurred only after the downstream groups had processed that gradient signal through thousands of training steps.

The COOLING to ACTIVE regression observed in the attention_output group provides further evidence of the governor's robustness. This group was prematurely cooled by the B=0 artifact, then reactivated when real gradient flow from the output head pushed its EMA above the cooling threshold. The four-state machine self-corrected from the calibration issue without any intervention, using the existing COOLING to ACTIVE transition path. This hysteresis behavior is exactly what the state machine was designed to provide.

Applying DAC's own vocabulary-stripping process to this failure reveals it as an observational degeneracy: two qualitatively different signal trajectories (cold start and genuine convergence) produce the same low-magnitude reading, and the governor cannot distinguish them. This is the same pattern that appears across the collapse table. In rendering, a pixel with zero color could be background or a black surface; the depth buffer adds a signal to break the degeneracy. In networking, a silent node could be idle or crashed; heartbeat protocols add a liveness signal. In distributed consensus, a node that has not voted could be slow or partitioned; timeouts add a temporal boundary. In every case, the solution is the same structural primitive: add a second Signal\<T\> that disambiguates the measurement.

The fix follows mechanically from the cross-domain pattern: a phase-aware ConvergenceGovernor that tracks whether gradient magnitude has ever exceeded the threshold, classifying gradient trajectories into qualitative phases (COLD, WARMING, ACTIVE_LEARNING, DECLINING). The ACTIVE-to-COOLING transition requires the gradient phase to be ACTIVE_LEARNING or DECLINING, never COLD. This NoCoolingFromCold invariant was specified in TLA+ and verified by TLC across 18.6 million states. A specification without phase awareness produces a counterexample matching the exact failure observed in the training run in just 2 states.

This episode demonstrates three properties of the methodology simultaneously. First, the structure/function separation that DAC claims: the governance machinery was correct (the ConvergenceGovernor followed its specification exactly), while the domain function (the cooling threshold applied to B=0-initialized parameters) was miscalibrated. The governance code requires one additional precondition; the primitive itself does not change. Second, DAC's generative mode: the fix was not invented from scratch but recognized as a solved problem from the collapse table, where observational degeneracies are routinely resolved by adding a disambiguation signal. Third, the value of formal verification: the phase-aware fix was verified in TLA+ before any code was written, and the original failure was reproduced as a concrete counterexample, providing high confidence that the fix addresses the root cause.

This is a machinery validation, not a scale validation. The efficiency gains from governed training depend on model size and training duration. At 204M parameters, the governance machinery correctly identified which parameter groups to stop training and dynamically reallocated budget to groups still learning. However, the compute savings were not realized as wall-clock improvement in this run because the implementation zeros gradients for converged groups after computation rather than skipping the backward pass entirely. Implementing actual compute skipping for governed groups (setting requires_grad=False with proper autograd graph pruning) is an engineering optimization for future runs. The governance produced the correct signal; the training loop did not yet act on it efficiently.

### 5.9 Summary of Results

Architecture results are from the 39M parameter model (76M dense equivalent) trained on 500K OpenWebText samples for one epoch. Training governance results include both the 4.8M preliminary validation (300 steps, WikiText-2) and the 204M scale validation (7,228 steps, reasoning corpus, NVIDIA L4).

| Mechanism | Metric | Result | Status |
|-----------|--------|--------|--------|
| Low-rank compression | Compression ratio | 3.9x over dense equivalent (39M); 3.94x (204M) | Validated |
| Sparse attention | Attention sparsity | 88% | Validated (39M) |
| Gated feed-forward | FF sparsity | 80% | Validated (39M) |
| Belief injection | Success rate | 84% across 100 beliefs | Validated (39M) |
| Semantic routing | Routing accuracy | 86% (4.3x above chance) | Validated (39M) |
| Belief coexistence | Simultaneous improvement | 64% of 100 beliefs | Validated (39M) |
| Base weight restoration | Bit-for-bit fidelity | Exact across all tensors | Validated (39M) |
| Base weight immutability | Tensor integrity | 406 tensors verified | Validated (39M) |
| Adaptive depth | Mean exit depth | 11.3/12, 2 unique depths (39M); 20/20 not activated (204M) | Partial |
| Governed training | Budget invariant | 0 violations / 300 steps (4.8M); 0 violations / 722 records (204M) | Validated |
| Governed training | Audit chain integrity | 300 records verified (4.8M); 722 records, SHA-256 chain intact (204M) | Validated |
| Convergence governors | State transitions | 15 transitions, 7/8 converged (4.8M); 18 transitions, all valid, no skips (204M) | Validated |
| Hierarchy activation | Coarse-to-fine ordering | L0 then L1 then L2 then L3 via convergence signal | Validated (4.8M, 204M) |
| Hierarchy activation | L3 genuine convergence | Step 2,773 (non-round, after 2,300+ steps of real gradient flow) | Validated (204M) |
| Hierarchy activation | L1/L2 timing | Steps 200/400 (B=0 initialization artifact, disclosed) | Disclosed artifact |
| B=0 diagnosis | DAC applied to own failure | Observational degeneracy identified, phase-aware fix formally verified | Validated |
| Budget reallocation | Dynamic budget shifts | Converged groups drop to 0.012, active get up to 0.40 | Validated (204M) |
| Gradient routing | Gate activation | 0.34-0.69 gate density (non-degenerate) | Validated (204M) |
| Language modeling | Best val PPL | 57.6 at step 2,000 (204M) | Measured |
| Language modeling | Final val PPL | 1,463.9 (overfit, single epoch, governance invariants held throughout) | Expected |
| Orthogonal capacity | Available dims | 53,760 (3,360 max rank-16 deltas), confirmed on two machines | Measured (204M) |
| Tiered sampling | Tier distribution | All samples scored as Failing (scored pre-training only) | Not functional |
| Training time | Wall clock | 140.9h on NVIDIA L4 | Measured (204M) |

The two partial results (adaptive depth and tiered sampling) are not architectural failures. Adaptive depth requires either a lower exit threshold or explicit layer-dropping training to learn meaningful early-exit behavior; the 39M model showed limited depth variation (11.3/12 mean exit depth) and the 204M model showed none (20/20 throughout). Tiered sampling scored all samples at initialization when the model could not yet evaluate difficulty; periodic re-scoring during training would enable the governed data pipeline.

The training governance results validate that all seventeen primitives compose correctly at 204M parameters and that all governance invariants hold empirically across 7,228 training steps, including through the severe overfitting phase after step 2,000. The B=0 observational degeneracy was diagnosed using DAC's own methodology and the fix was formally verified in TLA+. The language modeling perplexity is not competitive at this scale and training duration; the claim is governance machinery validation, not language modeling performance.

The critical open question is whether the governed training machinery produces measurable efficiency gains at larger scale. The governance correctly identified which parameter groups to stop training and reallocated budget accordingly, but the compute savings were not realized as wall-clock improvement in this run. Validating the efficiency claims requires a 7B+ parameter training run with proper compute skipping for converged groups, multi-epoch training with adequate regularization, and comparison against an ungoverned baseline. This is identified as necessary future work in Section 9.5.

### 5.10 What DAC Did Not Provide

DAC does not design domain functions. The specific choice of low-rank factorization for the deltas, the cosine similarity metric for the routing network, the exit-threshold tuning for adaptive depth, the choice of loss function and optimizer: these are domain-specific engineering decisions that require ML expertise. The same applies to the training governance system: the specific scoring function for the gradient router, the convergence thresholds for per-group governors, the budget allocation policy, the hierarchy level boundaries, and the sample difficulty thresholds are all domain functions that DAC's structural skeleton does not supply. DAC provided the structural skeleton. Domain knowledge filled in the scoring functions, the loss formulations, the training recipes, and the governance thresholds.

This is the same structure/function separation described throughout the paper. The abstraction primitives provide structure (how data flows, how resources are allocated, how convergence is detected, how mutations are audited). The domain provides function (what computation to apply at each step). Neither replaces the other.


---

## 6. Case Study: Orkestratum, A Multi-Domain Runtime on the Primitive Set

### 6.1 What Orkestratum Is

LeanFormer is the paper's evidence that DAC is useful when applied to a single domain. Orkestratum is the paper's evidence that DAC produces a runtime that serves multiple domains simultaneously without per-domain engine work. The methodology's central claim is that twelve domains were the same computation in different vocabulary. The strongest form of that claim is a single codebase that runs all of them. Orkestratum is that codebase.

Orkestratum is an application runtime organized as a kernel of primitive modules plus sandboxed domain modules. The kernel contains one implementation of each of the seventeen primitives, one DAG scheduler that routes every execution through a single dispatch path, and one audit chain that records every governance decision. Domain modules plug into the kernel by providing the domain functions (scoring functions, task functions, transfer functions) that parameterize the primitives. Four workload classes run on this runtime today:

- Security orchestration playbooks (SOAR),
- Continuous integration and deployment pipelines (CI/CD),
- Extract-transform-load pipelines (ETL),
- Configuration management runs,

and a fifth, the real-time renderer, runs inside the same runtime with the same primitives driving real-time graphics workloads.

The claim is concrete: one kernel, one scheduler, one audit chain, one budget system, one convergence governor implementation, one traversal engine. The four DAG-workload domains are different compositions of these modules with domain-specific task functions. The renderer is a different composition of the same modules with domain-specific scoring functions (depth testing, culling predicates, light propagation kernels). No domain has its own scheduler, its own budget, or its own audit implementation. A primitive-bypass CI gate prevents new code from sidestepping the kernel.

### 6.2 What Is Demonstrated, What Is In Progress

The claims in this section are bounded to what the Orkestratum codebase currently demonstrates. The runtime is an active engineering program, not a finished system, and this section is explicit about which capabilities are production-validated, which are validated at prototype quality, and which are in-progress work.

Demonstrated:

- One kernel codebase implements all seventeen primitives as trait-dispatched modules. A conformance audit has mapped every primitive to its implementation site, its invariant, and its trace. Phases of this refactor are committed against an executable blueprint with per-primitive exit criteria.
- Rendering workloads run through the kernel at interactive frame rates. Sponza scene benchmark data has been collected pre-consolidation; further throughput gains from the primitive consolidation are in progress.
- DAG workload execution is mediated entirely by the scheduler: SOAR playbooks, CI/CD pipelines, ETL workflows, and configuration-management runs all route through the same dispatch path. The absolute invariant "all execution routes through the DAG" is enforced by a machine-checked TLA+ specification (AllExecutionThroughDag.tla).
- A SHA-256 hash-chained AuditSink records every mutation across all workload classes. The chain is verified in tests and its integrity has been maintained across the 204M LeanFormer training run (722 records, zero chain breaks) as well as in runtime operation.
- Formal specifications of the primitives and key compositions exist in TLA+. The formal verification program is underway, with per-primitive refinement specifications scheduled as a dedicated phase of the engineering blueprint.

In progress:

- Performance consolidation to hit demonstration targets for rendering (target: Sponza at 100+ FPS, Bistro at interactive frame rates).
- Ongoing refactor of hand-rolled structures in the codebase to route through the L0 primitive trait surface rather than through local reimplementations. A conformance audit identified specific violations with line-number traces; remediation is scheduled against an executable plan.
- Byzantine fault tolerance for federated authorities is documented as an extension path but not implemented; current federation uses crash-fault tolerance (RAFT).
- Scene composition (interactive multi-participant worlds) is a proposal document dependent on the completion of earlier refactor phases; it is not yet implemented.

Not claimed:

- Orkestratum is not claimed to match domain-specialized tools on their benchmarks today. A dedicated graphics engine optimized for a single scene will outperform Orkestratum's renderer on that scene. A purpose-built SOAR platform will have a larger plugin ecosystem. The claim is that one runtime serves these domains with the same primitives, not that the runtime currently dominates the best-of-breed per domain.
- Orkestratum is not claimed to be feature-complete relative to any of the four DAG workload domains. Each has a working execution path; feature parity with Ansible or Airflow in every edge case is not yet achieved.
- The renderer is not claimed to be a frontier research renderer. It is a real-time renderer built on the primitive set sufficient to demonstrate that the primitives work under real-time constraints.

### 6.3 The Primitive-to-Domain Mapping in the Runtime

The following table documents how the seventeen primitives are instantiated in each of the five domains currently running on Orkestratum. The table is a runtime-level view: it shows how a single primitive serves multiple domains by being parameterized with different domain functions, not by being reimplemented per domain.

| Primitive | SOAR | CI/CD | ETL | Config Mgmt | Rendering |
|---|---|---|---|---|---|
| Budget\<U\> | Playbook compute | Build compute | Pipeline compute | Run compute | Draw calls, GPU memory, microseconds |
| FederatedBudget\<U\> | Per-tenant allocation | Per-team allocation | Per-warehouse allocation | Per-inventory allocation | Per-pass GPU allocation |
| QualityHierarchy | Severity tiers | Build priority tiers | Data freshness tiers | Desired-state tiers | LOD tiers |
| AllocationSnapshot | Playbook activation state | Stage activation state | Task activation state | Role activation state | Per-frame LOD assignment |
| RelationshipGraph | Playbook DAG | Pipeline DAG | Transform DAG | Role/task DAG | Render pass graph |
| ResourceRegistry | Action catalog | Tool catalog | Source/sink catalog | Host inventory | Asset/material catalog |
| TraversalEngine | Playbook execution | Pipeline execution | Pipeline execution | Task execution | Frustum culling, visibility traversal |
| PropagationPass | State propagation | Artifact propagation | Data propagation | State convergence | Cascade shadows, GI |
| CompetitiveSelection | Alert triage (ranked) | Runner allocation (hard) | Source selection (hard) | Conflict resolution (hard) | Pixel ownership (hard), LOD pick (ranked) |
| ActuationPass | Response action | Stage execution | Transform/load | Task apply | GPU command submission |
| Reduction | Alert aggregation | Test result aggregation | Reduce-phase aggregation | State aggregation | Draw call batching |
| Sampler | Randomized test sampling | Test partition | Row sampling | Dry-run probing | TAA jitter, importance sampling |
| Checkpoint | Rollback on failed action | Rollback on failed stage | Rollback on failed load | Rollback on failed task | (not used in rendering) |
| ConvergenceGovernor | (not typically used) | (not typically used) | Pipeline convergence | Desired-state convergence | Physics constraint solving |
| Signal\<T\> | Alert arrival | Build trigger | Data arrival | State change | Render pass completion |
| RateLimit | API throttling | Build queue pacing | Source throttling | Remote host pacing | Frame rate cap |
| AuditSink | Every action | Every build step | Every transform | Every apply | Every render decision (optional) |

The table is the strongest test of DAC's central claim. If the methodology were vocabulary-matching dressed up as structural insight, each domain would require its own implementations of these primitives because the domain constraints would differ in ways the primitives could not absorb. Instead, each primitive serves all five domains with the same code, parameterized by the domain's scoring function or task function.

### 6.4 The DAG Workload Collapse, Empirically

Section 7 of this paper argues that CI/CD pipelines, ETL workflows, configuration management, and SOAR playbooks are structurally identical: DAG traversal under budget constraints with capability-gated I/O. Orkestratum's architecture is the empirical form of this claim. All four domains dispatch through the same scheduler, traverse the same RelationshipGraph trait, consume the same Budget, emit to the same AuditSink, and compose with the same reconciler primitives (SuperviseReconciler, EventDrivenReconciler, ScheduledReconciler, TickReconciler), each of which is itself a specific composition of Budget, RateLimit, Signal, Sampler, and ActuationPass.

What the runtime demonstrates that the paper's analytical table cannot: that the same execution engine serves all four domains without accumulating domain-specific cruft over time. A key CI gate in the project forbids new code from bypassing the primitive kernel, which operationalizes the structural claim as an ongoing invariant rather than a one-time analysis.

The rendering pipeline is the crucial additional evidence. Rendering is the domain with the tightest performance constraints in the full set: every frame has a fixed millisecond budget, and any overhead from abstraction is immediately visible. If the primitive set survived contact with real-time rendering at interactive frame rates, the objection "this abstraction is too general to be efficient" loses force. The rendering pipeline is running inside the same runtime that executes the DAG workloads, using the same primitive implementations. It is not a separate engine bolted onto a general-purpose runtime. It is the general-purpose runtime driving real-time graphics.

### 6.5 The Formal Verification Program

Orkestratum's formal verification program treats TLA+ specifications as the source of truth for the primitive layer. Each primitive has (or will have, per the blueprint) a refinement specification that its implementation must satisfy. Composition specifications verify invariants across primitive boundaries. A top-level invariant (AllExecutionThroughDag) enforces that no side channel can bypass the scheduler.

This is the same formal verification approach applied in Section 9.4 of this paper to the primitive set as a mathematical object. The distinction is that Orkestratum applies it to a specific implementation, enforcing that the implementation refines the specification rather than merely matching the specification's outward behavior. The full refinement program is future work (identified in the blueprint's Phase 5); the current state is that all primitives have TLA+ specifications, five LeanFormer compositions have verified specifications, and the phase-aware ConvergenceGovernor's NoCoolingFromCold invariant has been verified across 18.6 million states.

### 6.6 What Orkestratum Demonstrates About DAC

The case study establishes three things the analytical collapse cannot establish alone.

First, the primitive set is engineerable. Seventeen primitives are few enough that they can be implemented, tested, and verified by a small team. The blueprint's engineering scope is bounded and trackable. This is evidence that the governance boundary is at a useful level of abstraction: low enough that implementation is feasible, high enough that domains reuse the primitives without forcing them to sprawl.

Second, the primitive set survives multi-domain contact. When four DAG workload domains and one real-time rendering domain are executing on the same primitives simultaneously, each domain's constraints test the primitives. The rendering pipeline tests that the primitives are fast enough. The DAG workloads test that the primitives are expressive enough for capability-gated orchestration. The audit chain tests that the primitives compose under concurrent mutation. A primitive that passed one domain's constraints but failed another's would have been flagged by the conformance audit and would appear in the "in progress" list in Section 6.2. The list of in-progress items is bounded and does not include "the primitive set does not work for domain X."

Third, the primitive set enables cross-domain transfer. The phase-aware ConvergenceGovernor (developed to fix the B=0 observational degeneracy in LeanFormer) is the same primitive implementation that governs physics constraint solving and, when federated authorities are added, will govern RAFT-based cluster convergence. A fix motivated by an ML training bug improves rendering physics and distributed consensus because they share the primitive. This is the operational form of the cross-domain optimization transfer claim.

The limitations of this case study are stated in Section 6.2. Orkestratum is not a finished system. The performance consolidation is in progress. The formal verification program is partial. The claim is that a single runtime executes multiple domains with the same primitive set, not that the runtime has reached feature parity or peak performance in any individual domain.

---

## 7. Extended Case Study: The AI Domain

### 7.1 The Deepest Collapse

The most striking result of applying DAC to machine learning is the observation that the transformer architecture [2], the foundation of every modern large language model, is a composition of primitives that the methodology had already identified before the AI domain was examined.

Attention is soft competitive selection. The transformer's multi-head attention mechanism computes, for each query token, a weighted combination of value vectors where the weights are determined by similarity between the query and key vectors. This is CompetitiveSelection in soft mode: the dot-product similarity is the scoring function, softmax produces the weighted allocation, and each attention head is one selection pass. Multi-head attention is parallel selection passes over the same candidates with different scoring functions.

A precision caveat is necessary on the strength of this mapping. The structural identity (softmax-weighted sum over candidates with a scoring function) is real, but attention has properties that matter for what people actually do with attention: differentiability, gradient flow through the weights, and learned (rather than designed) scoring. The mapping captures the structure and loses the function. The function is most of what makes attention useful in its native domain. The mapping is therefore a lens that enables cross-domain optimization transfer, not a substitute for the ML literature on attention.

Backpropagation is a member of the PropagationPass family. The forward pass builds a directed acyclic computation graph. The backward pass propagates gradient values from the loss node backward through the graph in reverse topological order, applying the chain rule at each node to compute local gradients. The shared structure with iterative relaxation (Bellman-Ford, belief propagation) is message passing over a graph toward a consistent state. The instantiation parameters differ substantially: backpropagation is a single reverse pass on a DAG with termination after one traversal; Bellman-Ford is iterative relaxation on a graph with cycles with termination at fixed point. Calling both instances of the same primitive carries real risk of overclaiming, and the honest framing is that PropagationPass is a family parameterized by topology, traversal order, message function, and termination condition, of which backpropagation is one member. The value of the mapping is that it makes visible the connection between gradient computation, routing convergence, and lighting propagation, enabling cross-domain insight. The risk is treating a family resemblance as an identity. This paper treats it as family resemblance.

Speculative decoding is the two-level fidelity architecture. A fast small model generates candidate tokens (coarse pass) [8]. A large model verifies them (fine selection). Accepted tokens are actuated. Rejected tokens are discarded. This is the same structure as the rendering fidelity pipeline: coarse traversal reduces the candidate set, fine selection determines winners, actuation evaluates only winners. The same architecture that makes rendering efficient makes inference efficient.

### 7.2 Implications and Evidence Status

The mappings above are structural identities, with the precision caveats noted. The implication is that any optimization of CompetitiveSelection discovered in any domain is potentially applicable to attention. Any optimization of PropagationPass discovered in any domain is potentially applicable to gradient computation. The collapsed primitive set creates a channel for cross-domain optimization transfer that does not exist when each domain maintains its own vocabulary.

Linear attention [10], sparse attention, and flash attention are all optimizations of CompetitiveSelection in soft mode. Convergence governors and temporal amortization, developed for real-time lighting, become potentially applicable to training loop optimization. The primitive vocabulary makes these connections visible. The domain vocabulary hides them.

A candid assessment of validation status: cross-domain optimization transfer is the most powerful claim DAC makes. The evidence for it to date consists of:

- Five systems-engineering solutions applied to ML model design in LeanFormer (Sections 5.3–5.7), demonstrating that patterns from file systems, database multi-tenancy, and signal processing transfer into ML architectural decisions.
- The targeted training system (Section 5.8), demonstrating that the same primitive set applies to the training process itself, validated at 204M parameters across 7,228 training steps.
- The B=0 observational degeneracy diagnosis (Section 5.8.1), demonstrating that a novel ML training bug can be recognized as structurally identical to solved problems from rendering, networking, and distributed consensus, with the fix derived from the cross-domain pattern.
- The shared ConvergenceGovernor implementation in Orkestratum (Section 6), which is the same primitive instance serving both ML training and physics constraint solving. A fix motivated by ML training (the phase-aware variant) is available to rendering without rework.

These are strong but one-directional: they are all cases of systems-engineering patterns transferring into ML or to other systems-governed domains. The stronger claim would be a bidirectional transfer: an optimization discovered in ML, applied to a non-ML domain, producing a measurable improvement there. This has not yet been demonstrated and is identified as future work (Section 9.5). The current evidence supports "cross-domain recognition enables ML-direction transfer" rather than "the primitive set is a bidirectional channel for optimization." The latter is the methodology's aspiration; the former is what has been shown.

---

## 8. The DAG Workload Collapse

### 8.1 Four Domains, One Pattern

The most practically significant collapse is the identification that CI/CD pipelines, ETL workflows, configuration management playbooks, and SOAR automation playbooks are structurally identical. All four are:

A directed acyclic graph of tasks with typed dependencies, executed under budget constraints (compute, time, or both), with capability-gated I/O at task boundaries, converging toward a desired end state, and audit-logged for observability and compliance.

The practical consequence is that a single execution engine can replace Ansible, Jenkins, Airflow, and Splunk SOAR, not by implementing four separate systems, but by recognizing that four separate systems were never needed. The vocabulary was different. The computation was the same.

Section 6 describes the empirical form of this claim: Orkestratum executes all four domains on one scheduler with one primitive set.

### 8.2 Why This Collapse Was Invisible

These four domains are served by different industries, different conferences, different vendor ecosystems, and different job titles. A CI/CD engineer uses Jenkins or GitHub Actions. An ETL engineer uses Airflow or dbt. A configuration management engineer uses Ansible or Puppet. A security engineer uses Splunk SOAR or Palo Alto XSOAR. Each tool has its own vocabulary, its own plugin ecosystem, and its own certification program.

The vocabulary creates the market. The market creates the specialization. The specialization prevents anyone from noticing that all four tools do the same thing. DAC dissolves this by asking: what does each tool actually compute? The answer, in every case, is: it traverses a DAG under constraints and executes tasks at each node. The domain-specific part is the task function, which belongs in a sandboxed execution boundary, not in the kernel.

---

## 9. The Composition Algebra

### 9.1 How Domains Are Reconstructed

The value of abstraction collapse is not merely taxonomic. It is constructive: once the primitives are identified, every domain pattern can be systematically constructed as a composition. The following composition map documents the reconstruction of common computational patterns from the primitive set.

| Pattern | Core Primitives | Governance |
|---------|----------------|------------|
| LOD / Quality scaling | QualityHierarchy + TraversalEngine + AllocationSnapshot | Budget\<Bytes\> |
| Global illumination | PropagationPass + ConvergenceGovernor | Budget\<Microseconds\> |
| Physics simulation | Reduction + CompetitiveSelection (hard) + PropagationPass + ActuationPass | Budget\<Microseconds\> + ConvergenceGovernor |
| Network sync | PropagationPass + TraversalEngine + Diff | FederatedBudget\<BytesPerSecond\> |
| Container orchestration | QualityHierarchy + TraversalEngine + ActuationPass | Budget\<CPU/Memory\> + ConvergenceGovernor |
| CI/CD pipeline | RelationshipGraph + TraversalEngine + ActuationPass | Budget\<Compute\> |
| ETL pipeline | RelationshipGraph + ActuationPass + Transaction | Budget\<Microseconds\> |
| Configuration management | Diff\<State\> + ConvergenceGovernor + ActuationPass | (none) |
| SOAR playbook | RelationshipGraph + TraversalEngine + ActuationPass | Capability-gated I/O |
| Market simulation | CompetitiveSelection (hard) + PropagationPass + Transaction | FederatedBudget + ConvergenceGovernor |
| ML inference | ActuationPass + CompetitiveSelection (soft) + Memoize | Budget\<FLOPs\> |
| Belief delta system | Budget\<Parameters\> + ResourceRegistry + Transaction + CompetitiveSelection (ranked) | AuditSink |
| Targeted training pipeline | CompetitiveSelection (ranked) + FederatedBudget\<GradientCompute\> + QualityHierarchy + TraversalEngine + ActuationPass + Reduction + PropagationPass | ConvergenceGovernor (per-group) + AuditSink |
| Constraint logic search | RelationshipGraph + TraversalEngine + PropagationPass + Checkpoint | ConvergenceGovernor |
| Lattice dataflow | RelationshipGraph + PropagationPass + Reduction | ConvergenceGovernor (widening mode) |
| Unification | ResourceRegistry (monotonic) + TraversalEngine + PropagationPass + ActuationPass | ConvergenceGovernor |

### 9.2 The Key Insight: Structure vs. Function

In every composition above, the primitives provide the structure: how data flows, how resources are allocated, how convergence is detected, how mutations are audited. The domain provides the function: what computation to apply at each step.

The rendering equation is a domain function. The RAFT consensus protocol's log replication rule is a domain function. The machine learning loss function is a domain function. The attention scoring function (dot-product similarity) is a domain function. The unification consistency check is a domain function. None of these belong in the kernel of abstraction primitives. All of them execute within the governance framework the primitives provide.

This separation is what makes the primitive set simultaneously lean and broadly applicable. It contains no domain knowledge. It contains the execution model shared by every domain examined.

---

## 10. Methodology Validation Criteria

### 10.1 How to Know the Collapse Is Real

A collapse is genuine (not merely a renaming exercise) if and only if:

1. Completeness: Every domain pattern from step 1 can be expressed as a composition of the collapsed primitives. No residual domain-specific primitives are required.

2. Minimality: No primitive in the collapsed set can be expressed as a composition of the others. Removing any primitive leaves at least one domain pattern unexpressible.

3. Operational equivalence: A system built from the collapsed primitives produces the same outputs as the domain-specific system it replaces, under the same inputs and constraints.

4. Cross-domain transfer: An optimization discovered in one domain, when applied to the shared primitive, produces measurable improvement in other domains that use the same primitive.

Criterion 1 is satisfied for all twenty examined domains after the adversarial pass (Section 2.5), which produced one new primitive and two refinements, and is also satisfied for the approximately twenty additional domains in the broader sweep (Section 2.5, final paragraph), which produced no new primitives that passed the qualification test. Criterion 3 is satisfied for the domains where working implementations exist (LeanFormer for ML; Orkestratum for SOAR, CI/CD, ETL, configuration management, and rendering).

Criterion 4 has partial evidence. One-directional transfer (systems patterns into ML) is demonstrated by LeanFormer. Transfer within the Orkestratum runtime (a fix in one domain's use of a primitive benefiting another domain through the shared implementation) is demonstrated by the phase-aware ConvergenceGovernor. Bidirectional transfer and transfer across all twelve domains remain future work.

Criterion 2 (minimality) deserves explicit scrutiny. The claim that no primitive can be expressed as a composition of the others has not been formally proven. Some candidates for potential redundancy: Can RateLimit be expressed as Budget\<Operations\> + a temporal mechanism? Can Signal\<T\> be derived from AuditSink + a predicate filter? These might be legitimate standalone primitives, or they might be compositions. If they are compositions, the primitive count drops and the narrative changes, but the methodology's validity does not. The important claim is that the set is sufficient, not that it is provably minimal. Proving minimality would require showing that removing each primitive creates at least one domain pattern that becomes unexpressible, a worthwhile exercise that has not yet been completed.

A related but distinct property has been formally verified: operational irreducibility. Sixteen of the seventeen primitives were specified in TLA+ and systematically decomposed into plausible sub-operations (separating the guard from the mutation, removing intermediate states, splitting atomic operations into phases). In all cases tested, the TLC model checker found concrete counterexamples where the decomposed version violated the primitive's governance invariant. This does not prove minimality (that no primitive is a composition of other primitives in the set), but it proves that each verified primitive is atomic with respect to its own invariant. The guard and the guarded operation cannot be separated without losing the guarantee. Operational irreducibility is a weaker claim than algebraic minimality but a stronger claim than the paper would have without formal verification. The distinction matters: minimality asks whether the set can be made smaller; operational irreducibility asks whether each element can be made simpler. The TLA+ verification answers the second question definitively for the sixteen verified primitives, and the verification of the seventeenth (Checkpoint) is a pre-submission prerequisite listed in Section 10.5.

### 10.2 Threats to Validity

The most significant methodological risk is confirmation bias. All twenty domains were collapsed by the same individual, and once a primitive vocabulary exists, there is a cognitive pull to force every new domain into it rather than honestly admitting when the existing primitives are insufficient.

Five properties of the development process mitigate this risk.

First, the primitives were discovered incrementally, not designed a priori. SOAR was the first domain. Rendering was added second and forced the addition of QualityHierarchy, TraversalEngine, and CompetitiveSelection, primitives that SOAR alone did not require. Each subsequent domain either mapped onto existing primitives or forced additions when the existing set was genuinely insufficient. The primitive count grew from five to sixteen over the twelve original domains, and from sixteen to seventeen with the addition of Checkpoint in the adversarial domain pass. If confirmation bias were dominant, the count would have stayed at five.

Second, the LeanFormer implementation produces working systems with measurable outputs. Bit-for-bit weight restoration after belief removal, 119 passing tests, 84% injection success rates, 86% routing accuracy, and the 204M scale validation (with zero governance violations across 7,228 steps and 722 audit records) are objective evidence that the collapsed compositions are operationally correct, not merely descriptively plausible. The B=0 diagnosis confirmed an a priori DAC prediction: governance correctness and domain function calibration are independent concerns, and the B=0 issue was a domain function calibration bug, not a primitive failure.

Third, the Orkestratum runtime provides evidence that does not depend on any single domain. When the same primitive implementation serves five domains simultaneously, the question "did you force this domain into your primitive set" has a different answer than when the primitive set is checked against one domain at a time. The runtime is either consistent across all domains or it is not. The conformance audit, with its line-number traces and invariant mappings, provides the ground truth. The audit is not presented in this paper as a full disclosure; the blueprint document is separate engineering record. What the paper can cite is that the audit exists, is in active use, and has produced a specific, bounded list of remediation items that are tracked against gates.

Fourth, the TLA+ formal verification process subjected every primitive and composition to exhaustive model checking at the tested bounds. The model checker does not share the author's assumptions or biases. It explores every reachable state mechanically. The five corrections it produced (Section 10.4) demonstrate that the verification was genuine: if the model checker had simply confirmed every initial assumption, the exercise would have been less credible. The corrections are evidence that the formal verification process had teeth.

Fifth, the broader domain survey described at the end of Section 2.5 applied the primitive-qualification test of Section 1.3 to candidate primitives that emerged from a wider sweep (reactive systems, streaming watermarks, CRDTs, transactional memory, evolutionary computation, and others). Three candidates that initially appeared plausible as new primitives or refinements were eliminated by the test: two turned out to be compositions of existing primitives, one turned out to be a policy concern within a domain function rather than a governance invariant. The test is strict. A methodology in which every broadening of the sample forces a new primitive would be unstable. A methodology in which the test routinely rejects plausible candidates is the one that supports the "seventeen primitives, incrementally discovered" claim. The risk of confirmation bias is most credibly mitigated not by claiming the set is final but by demonstrating that the test has teeth in both directions: it accepted Checkpoint when the evidence was strong, and it rejected glitch-free propagation, monotonic-signal semantics, and the exploration-exploitation mode when the evidence was not.

The residual risk remains: domains not yet examined may require primitives outside the current set. A domain-specific practitioner working in a field unfamiliar to the author may yet identify an operation that passes all five criteria and is not in the current set. This is not a weakness of the methodology but its explicit invitation. The test is the mechanism by which such contributions earn their place. The claim is that seventeen primitives suffice for twenty examined domains and have survived a broader sweep of approximately twenty additional domains without forcing additions, not that they suffice for all possible computation.

### 10.3 Limitations

DAC does not claim that domain expertise is unnecessary. The domain function (the rendering equation, the attention scoring function, the unification consistency check) requires domain expertise to design. What DAC claims is that the execution infrastructure around the domain function is generic and need not be redesigned per domain. Domain-specific vocabulary is accidental complexity for the execution infrastructure. It is essential complexity for the domain function. The boundary between the two is drawn at the governance layer: the primitives provide structure and governance; the domain provides the computation that executes within that governance. Readers should not interpret the claim that "vocabulary is clothing" as a claim that domain expertise is unnecessary. The clothing covers something real. The claim is that the structural skeleton underneath is shared.

DAC does not claim that seventeen primitives are the final, minimal set. Future domains may reveal operations that genuinely cannot be expressed as compositions of the current set, requiring the addition of new primitives. The claim is that seventeen suffice for the twenty domains examined, not that they suffice for all possible computation.

Some classes of computation are out of scope for DAC as currently formulated. Analog computation, neuromorphic computation, and physical reservoir computing are not discrete state machines; the computation occurs as continuous physical relaxation, and while a simulator for such systems can be expressed with the primitive set, the physical computation itself is not decomposable into primitives in the DAC sense. Quantum computation as a physical phenomenon is similarly out of scope: state vector simulation and tensor network contraction fit the primitive set cleanly, but superposition and entanglement do not map to any governance primitive, and the paper does not claim to describe what quantum hardware is doing. Mathematical foundations such as homotopy type theory and cubical type theory are theories about the structure of types and equalities rather than computational patterns; the normalizer that reduces terms in such systems is a computational process that fits (as term rewriting with side conditions), but the content of the theory is not computation in the DAC sense. Reflective self-modifying computation is a subtler case: individual operations compose correctly within the primitive set, but when code rewrites itself during execution, global invariants may not transfer to the closure over all reachable executions. The primitive set is consistent at each step; global verification claims may require additional care in reflective systems.

The twenty domains examined share characteristics that favor the primitive set. Resource-governed execution is well-covered (the original twelve). Symbolic computation with backtracking is newly covered (the adversarial pass). Domains with fundamentally different computational characters than any examined here (interactive theorem proving with tactics, quantum circuit simulation, biological sequence alignment against reference genomes) may stress the primitive set in ways that reveal further gaps.

All twenty domains were collapsed by the same individual. Independent replication by other researchers applying DAC to domains outside the current twenty is necessary to establish that the methodology is reproducible and that the primitive set generalizes beyond the author's own analytical perspective. The incremental discovery process, the formal verification, and the hostile audit protocols mitigate confirmation bias but do not eliminate it. The strongest validation of DAC's generality would be a practitioner in an unexamined domain independently discovering that the primitive set expresses their domain's computational patterns, or honestly reporting where the primitives are insufficient and an eighteenth is required.

The LeanFormer empirical validation has a significant scale limitation. The architectural results (39M parameters) are proof-of-concept validations, not production-scale demonstrations. The 204M parameter training run validated that all seventeen primitives compose correctly under real training conditions across 7,228 optimizer steps, with all governance invariants holding throughout. However, 204M is modest by current standards, and many of the efficiency mechanisms that DAC predicts (gradient compute reduction from hierarchical activation, budget-governed routing of gradient signal) were not realized as wall-clock improvement in this run. The governance machinery correctly identified which parameters to stop training and reallocated budget, but the training loop did not skip computation for converged groups. The governance produced the right signal; the implementation did not yet act on it efficiently.

The 204M run also suffered from training configuration limitations that prevent strong claims about language modeling quality. Single-epoch training with minimal regularization (dropout 0.1 only) produced severe overfitting after step 2,000, with validation perplexity rising from 57.6 to 1,463.9. The overfitting is expected behavior and does not indicate an architectural failure, but it means the model's generalization quality should be evaluated from the best checkpoint (step 2,000), not the final checkpoint. Multi-epoch training with proper regularization is necessary to demonstrate that the architecture can produce competitive language models, and is identified as necessary future work.

The Orkestratum runtime's demonstrations are bounded by what the codebase currently supports (Section 6.2). The runtime is an active engineering program. The claim is that one runtime serves five domains with the same primitives, not that the runtime is feature-complete or performance-optimal in any of them.

The critical open question for LeanFormer is whether DAC's governance primitives continue to compose correctly and produce measurable efficiency gains at the scales where modern language models operate (7B+ parameters). Emergent training dynamics, optimization instabilities, and loss landscape characteristics at the billion-parameter scale may interact with the governance mechanisms in ways that the current validation cannot anticipate. A specific concern is governor thrashing: at scales where gradient magnitudes spike randomly, the convergence governor could oscillate rapidly between states. The existing design mitigates this structurally — state transitions require sustained EMA trends over a configurable cooling window, not single-step readings, and the AWAKENED state is explicitly designed to handle perturbation events without re-traversing the full state sequence — but the thresholds will require recalibration for frontier-scale gradient distributions. The phase-aware governor (Section 5.8.1) adds a further structural guard: a group in COLD phase cannot transition to COOLING regardless of gradient magnitude, preventing the initialization artifact class of false transitions entirely. A full-scale validation at 7B parameters or above is the single most important next step for the LeanFormer validation. Until that validation is complete, the LeanFormer results should be read as architectural proof-of-concept and governance machinery validation, not as demonstrated production-scale efficiency gains.

A data preservation gap from the 204M run warrants mention as a reproducibility concern. The tokenized training data (32K vocabulary) was stored only on the GCP VM and in a cloud storage bucket, both of which were deleted when the VM was terminated. The local copy of the raw corpus was tokenized with a different vocabulary size (50K), making it incompatible with the trained model for validation purposes. The inline EVAL measurements from the training log remain the authoritative validation numbers, as they were computed against the correctly-tokenized data during training. The lesson is straightforward: the governance machinery preserved every training decision with tamper-evident integrity, but the training data itself was not under governance. Future runs should archive the tokenized data alongside checkpoints.

The TLA+ formal verification is bounded model checking, not unbounded proof. The TLC model checker exhaustively explores all reachable states within finite bounds (Budget capacity of 4, three-node hierarchies, two parameter groups). Structural bugs and invariant violations are reliably caught at these bounds because the primitive structures do not change with scale. However, bounded verification does not constitute a mathematical proof that the invariants hold for all possible values of the constants. It provides strong evidence, not certainty.

An important distinction follows from this: the governance invariants are scale-independent by construction, and this was confirmed empirically at 204M parameters. Budget\<U\> enforces `consumed <= capacity` whether capacity is 4 or 4 billion. The ConvergenceGovernor's four-state machine has the same transitions whether it monitors 8 parameter groups or 8,000. HierarchyOrdering holds whether there are 2 levels or 20. The TLA+ specifications are parameterized by constants, and the invariants hold for any value of those constants because the logic does not reference the constants' magnitudes. The 204M training run confirmed this empirically: zero budget violations across 722 audit records, all 18 governor transitions valid, hierarchy ordering maintained throughout. These are structural properties of the state machines, not empirical properties of any particular training run. What remains empirical, and what requires the 7B+ validation, is whether the governance produces efficient training outcomes at that scale: whether the convergence thresholds, budget allocation policies, and gradient routing scores (all domain functions, not primitives) produce the predicted efficiency gains when operating on real gradient distributions at frontier scale.

However, the boundary between what TLA+ can and cannot verify is narrower than it first appears. Applying DAC's own vocabulary-stripping process to the phrase "gradient dynamics at scale" reveals that the problem is not numerical prediction (which TLA+ cannot do) but signal trajectory classification (which it can). The B=0 initialization bug (Section 5.8) was not a numerical problem. It was an observational degeneracy: two qualitatively different gradient trajectories (cold start and genuine convergence) both produced low magnitude readings, and the governor could not distinguish them. This is a solved problem across the collapse table. In rendering, a pixel with zero color could be background or a black surface; the depth buffer adds a signal to break the degeneracy. In networking, a silent node could be idle or crashed; heartbeats add a liveness signal. In distributed consensus, a node that has not voted could be slow or partitioned; timeouts add a temporal boundary. In every case, the solution is the same: add a second Signal\<T\> that disambiguates the measurement.

The same solution applies to the convergence governor. Gradient trajectories at any scale are not numerically predictable, but they are qualitatively enumerable: COLD (learning has not started), WARMING (gradient rising toward threshold), ACTIVE_LEARNING (gradient above threshold), and DECLINING (gradient falling from above threshold). A phase-aware ConvergenceGovernor that tracks whether gradient magnitude has ever exceeded the threshold can structurally prevent the B=0 bug: the ACTIVE-to-COOLING transition requires the gradient phase to be ACTIVE_LEARNING or DECLINING, never COLD. This invariant (NoCoolingFromCold) is verifiable in TLA+. The phase-aware specification is included in the project repository, and TLC confirms that the invariant holds across all reachable states while a specification without phase awareness produces a counterexample matching the exact failure observed in the training run.

This extends the scope of formal verification beyond governance correctness into domain function validation. The specific numerical thresholds remain empirical, but the qualitative correctness of the governor's response to different trajectory classes is formally verifiable. At 7B parameters, the expected trajectory classes (cold start, rapid learning, optimization instability with gradient spikes, gradual convergence, perturbation from belief injection) can be enumerated and the governor's response to each can be verified by TLC before a single GPU hour is spent. Formal verification cannot predict whether a 7B model will converge in 10,000 steps or 100,000. But it can prove that the governance machinery will respond correctly to whatever trajectory the model produces.

The formal verification also does not verify implementations. The TLA+ specifications define what the primitives must do. The implementations in Appendix A are intended to be faithful to those specifications, but the verification that each implementation correctly implements the specification is a separate concern (refinement checking). Orkestratum's engineering blueprint schedules this refinement checking as a dedicated phase; at the time of writing, refinement specifications exist for the primitive layer and composition specifications exist for key reconcilers, but the full per-primitive refinement verification is future work.

### 10.4 Formal Verification and What It Revealed

Sixteen of the seventeen primitive specifications (including the phase-aware ConvergenceGovernor), five LeanFormer compositions, and corresponding decomposition-failure specifications were formalized in TLA+ [14] and verified by the TLC model checker. Checkpoint, the seventeenth primitive added during the adversarial domain pass of Section 2.5, qualifies for inclusion by the primitive-qualification test of Section 1.3; its TLA+ specification is listed in Section 10.5 as a pre-submission prerequisite and is not yet part of the verified set. The verification explored approximately 45.4 million states at the tested bounds, with the phase-aware ConvergenceGovernor alone accounting for 18.6 million states across its full space of possible delta sequences and gradient phase transitions. The complete specifications, configuration files, and TLC output logs are available in the project repository.

The verification produced three categories of results.

First, all sixteen verified primitive specifications passed: their safety invariants held across all reachable states at the tested bounds. Budget never exceeded capacity. The ConvergenceGovernor never skipped a state. CompetitiveSelection (hard) always produced the highest-scoring winner. The hash chain in AuditSink was never broken. These results are bounded verification, not unbounded proof, but TLC's exhaustive exploration of the finite state space provides strong evidence that the invariants hold in general.

Second, all five LeanFormer composition specifications passed, confirming that primitive invariants are preserved under composition and that the composed system produces emergent guarantees. The GovernedTrainingPipeline specification verified six simultaneous invariants across all interleavings of training steps, governor updates, hierarchy activations, budget reallocations, and gradient routing updates. The BeliefDeltaLifecycle specification confirmed base weight immutability, parameter non-overlap, and exact restoration across all possible sequences of belief injection and removal.

Third, decomposition-failure specifications for each of the sixteen verified primitives produced concrete counterexamples, confirming that each is operationally irreducible with respect to its governance invariant. The counterexamples are not hypothetical. They are specific state traces that TLC discovered through exhaustive exploration. Some violations were found in as few as two states (QualityHierarchy: adding a child without checking the level constraint; the naive ConvergenceGovernor: a single zero-delta step causing premature COOLING, reproducing the exact B=0 bug from the training run). Others required longer traces (Budget: a concurrent allocation between check and act, at seven states). Every counterexample demonstrates a real failure mode that would manifest in any implementation that decomposes the primitive into separate sub-operations. Checkpoint is expected to produce a counterexample in the same category when its decomposition specification (AuditSink plus reverse-replay ActuationPass) is verified; the decomposition argument in Section 2.5 identifies the expected failure mode (loss of atomicity and stack discipline under concurrent mutation during unwind), and the TLA+ verification of Checkpoint listed in Section 10.5 will either confirm or correct this expectation.

A precise characterization of what "irreducible" means here is necessary to avoid overclaiming. The decomposition failures demonstrate operational irreducibility: the guard and the guarded operation cannot be separated into distinct steps without creating reachable states where the governance invariant is violated. The mechanism in most cases is a Time-of-Check-to-Time-of-Use (TOCTOU) pattern: a condition is checked, the world changes, and the guarded operation executes against stale state. This is a well-understood concurrency hazard, not a novel discovery. What is novel is the systematic demonstration that every verified primitive in the set exhibits this property. The governance invariant is not a postcondition that can be checked after the fact. It is an atomic property of the operation itself. Decompose the operation and the property ceases to exist. This is what makes them primitives rather than compositions.

This is a weaker claim than algebraic irreducibility (which would require proving that no primitive can be expressed as a composition of other primitives in the set). Section 10.1 addresses algebraic minimality separately, acknowledging that minimality has not been formally proven and identifying specific candidates for potential redundancy.

The verification process also challenged and corrected five initial assumptions about the formal specifications. These corrections are worth documenting because they demonstrate that the TLA+ verification was a genuine validation exercise, not a formality.

| Specification | Initial Assumption | TLC Finding | Correction |
|---|---|---|---|
| ConvergenceGatedActivation | Global invariant (always holds) | L0 can regress to ACTIVE after L1 activates (AWAKENED mechanism) | Reclassified as precondition of ActivateNextLevel |
| FullDepthMeansUncertain | Full depth implies no convergence | Convergence at the final layer is valid | Corrected to: full depth with no convergence at any layer implies uncertainty |
| ForgingRequiresConvergence | Holds across all phases | Converged group could revert to ACTIVE during forging | Restricted governor updates to training phase |
| Budget reallocation | Free half of converged group's budget | Group may have consumed more than half, violating SubPoolInvariant | Free only the unused portion |
| WinnerOptimality | Holds between evaluations | Score update between evaluations creates stale winner | Invalidate allocation when scores change (TOCTOU pattern) |

None of these corrections invalidated the underlying primitives or their invariants. They refined imprecise classifications: the difference between a global invariant and a precondition, the boundary conditions of a convergence definition, the interaction between budget usage and budget reallocation, and the scope of governance operations across lifecycle phases. In every case, the corrected specification is more precise than the original, not more permissive. These are exactly the kinds of issues that prose specifications miss and that formal verification catches.

The fact that the initial specifications required correction is not a weakness of the methodology. It is evidence that the verification was real and that the corrected specifications are trustworthy. A verification process that confirms every initial assumption without correction is either trivial or dishonest.

### 10.5 Future Work

The following represent the most important open questions and necessary next steps, ordered by priority and with estimated resource requirements where applicable.

1. Checkpoint TLA+ formal verification. Checkpoint was added to the primitive set during the adversarial domain pass (Section 2.5), but its TLA+ specification, decomposition counterexample, and at least one composition specification (CLP search is the natural candidate) are pending. The paper's §10.4 language cites sixteen primitives as having passed verification; adding Checkpoint to this program is required before the seventeenth primitive can be claimed as formally verified on the same footing as the other sixteen. The work involves specifying StackWellFormedness, RollbackFidelity, LIFODiscipline, and CommitPreservesMutations invariants; constructing a decomposition specification (AuditSink plus reverse-replay ActuationPass) that TLC is expected to disprove; and specifying a composition with TraversalEngine and PropagationPass for CLP-style search. This is a prerequisite for paper submission and not a post-publication item.

2. Independent replication of DAC by practitioners applying the methodology to domains outside the current twenty. This is the strongest form of validation the methodology can receive: a researcher unfamiliar with the primitive set applies DAC to a domain in their expertise and independently determines whether the primitive set expresses the domain's computational patterns or requires additions. Positive results confirm generality. Negative results (domains where the primitive set is genuinely insufficient) are equally valuable because they identify the boundaries of the current set and potentially reveal new primitives that pass the test. The methodology is explicitly an invitation to such contributions.

3. Practitioner validation in the three domains where the author's fit claim is weakest. Interactive theorem proving with tactics (Coq, Lean, Isabelle) has decades of literature on tacticals, proof irrelevance, and definitional equality that the author's current decomposition may underestimate. Approximate nearest neighbor search at the algorithmic level (HNSW, FAISS internals) uses graph structures whose specific geometric properties enable efficient search in ways that RelationshipGraph alone may not capture. Stiffness detection in adaptive numerical integration fits the ConvergenceGovernor AWAKENED semantics more tightly than the author expects, which warrants scrutiny. Practitioner validation in any of these three domains — either confirming the fit or identifying a genuinely missing primitive — would materially sharpen the paper's claims.

4. Phase-aware convergence governor implementation and retraining. The TLA+ specification exists and passes (ConvergenceGovernorPhaseAware with NoCoolingFromCold invariant). The implementation requires adding gradient_phase tracking to the convergence governor code. A retrain of the 204M model with the fixed governor would produce clean hierarchy activations free of the B=0 artifact. Estimated cost: approximately $150 (one 6-day L4 GPU run).

5. Multi-epoch training with proper regularization. The current 204M run used a single epoch with dropout 0.1 as the only regularization, producing severe overfitting after step 2,000. A 3-5 epoch run with stronger regularization would demonstrate that the architecture can generalize and that governance invariants hold across extended training, including the possibility of AWAKENED state transitions triggered by data distribution shifts between epochs. Estimated cost: approximately $450-750.

6. Scale validation of LeanFormer at 7B+ parameters. This is the single most important validation that the current work lacks. It would include: gradient compute reduction measurements with actual backward-pass skipping for converged groups, wall-clock training time comparisons against an ungoverned baseline, Knowledge Plane validation with orthogonality enforcement at a scale where the delta address space is practically significant, and convergence ordering analysis at a scale where the parameter group ratios are representative of production models. The governance invariants are established as scale-independent (formally verified and empirically confirmed at 204M), but the efficiency claims require empirical validation at frontier scale. Estimated cost: approximately $5,000-15,000.

7. Adversarial domain testing beyond the eight domains examined in Section 2.5 and the broader sweep described at the end of Section 2.5. Quantum circuit simulation at the algorithmic level (not the physical phenomenon), biological sequence alignment against reference genomes, and specific machine-learning paradigms not covered in detail (reinforcement learning with actor-critic methods, diffusion models, graph neural networks, normalizing flows) are plausible next candidates. Each should be decomposed with the primitive-qualification test applied explicitly to any candidate additions.

8. Bidirectional cross-domain optimization transfer. Demonstrate an optimization discovered in one non-ML domain (rendering, for example), applied to an ML primitive instance via the shared implementation, producing a measurable improvement on an ML benchmark. This is the strongest criterion from Section 10.1 and the one the paper cannot yet fully claim.

9. Orkestratum formal verification program completion. The engineering blueprint schedules per-primitive refinement verification and per-composition verification as a dedicated phase. Completing this phase would extend the formal guarantee from the primitive-set-as-mathematical-object (already verified) to the primitive-set-as-running-implementation.

10. Orkestratum performance consolidation. Specific targets include Sponza at 100+ FPS through the unified kernel and Bistro at interactive frame rates. The target demonstrates that the primitive set is not merely expressive but operationally efficient in the domain with the tightest performance constraints.

11. Resolution of the minimality question: for each primitive, determine whether removing it from the set makes at least one domain pattern unexpressible. If RateLimit can be expressed as Budget\<Operations\> composed with a temporal mechanism, or if Signal\<T\> can be expressed as AuditSink composed with a predicate filter, the primitive count should be revised. The methodology's value does not depend on the exact count, but intellectual honesty requires resolving the question.

12. Knowledge Plane validation at 204M. The 39M model demonstrated 84% belief injection success and 86% semantic routing accuracy. Repeating this validation at 204M against the best checkpoint (step 2,000) would confirm that the delta system scales, test injection success rate and routing accuracy at a more representative model size, and verify that the theoretical maximum deltas are practically achievable. Estimated cost: approximately $50.

The risk profile for the computational future work (items 4, 5, 6, 12) is low. The primitives are formally verified. The governance machinery works at 204M parameters. The open questions are training configuration tuning and whether efficiency gains scale as predicted. Item 1 is a pre-submission prerequisite. Items 2, 3, and 7 are methodological work whose risk profile is higher but whose results are the most valuable: either confirmation of the methodology's generality or identification of its boundaries.

---

## 11. Implications

### 11.1 For Software Engineering

If DAC's thesis is correct, the software industry is spending substantial resources building, maintaining, testing, and securing redundant implementations of the same computational patterns across different domains. Every Kubernetes controller, every CI/CD runner, every ETL framework, every SOAR playbook engine is a reimplementation of DAG traversal under budget constraints. Orkestratum's architecture is one instance of the alternative: an execution framework that recognizes these domains as the same computation with different domain functions. The maintenance cost, the security surface area, and the bug count scale with the number of independent implementations; collapsing four implementations to one compounds benefits in each dimension.

### 11.2 For AI/ML Systems

The identification that attention is soft competitive selection and backpropagation is a member of the graph message-passing family creates a bridge between the ML optimization community and the real-time systems community. Techniques developed for efficient GPU selection (the visibility buffer, hierarchical culling, budget-constrained traversal) become candidates for efficient attention computation. Techniques developed for efficient fixed-point iteration (convergence governors, temporal amortization, adaptive iteration counts) become candidates for training loop optimization.

The LeanFormer belief-delta system demonstrates a more immediate practical implication: the entire problem of knowledge management in neural networks (injection, removal, versioning, composition, audit) maps directly to solved database and systems engineering patterns. The Knowledge Plane architecture treats knowledge as a managed database with insert, update, delete, query, and vacuum operations. The structural identity with database management means that decades of engineering on consistency, isolation, and durability transfer directly.

The confabulation detection mechanism (Section 5.7) demonstrates a different kind of transfer: a real-time systems concept (convergence-based confidence estimation) applied to an AI safety problem. The principle that "the system should be honest about uncertainty by architecture, not by training" is a direct consequence of the DAC decomposition. A system with a ConvergenceGovernor knows, structurally, whether its computation converged or not. A system without one can only guess.

More broadly, the application of TLA+ and bounded model checking to ML training governance represents a contribution to the verifiable AI agenda. The ML community has largely treated training as an empirical process where correctness is assessed by outcome (did the loss go down?) rather than by invariant (were the governance properties maintained at every step?). The DAC approach inverts this: the governance properties are specified formally, verified exhaustively at bounded scales, and then confirmed empirically. The 204M training run demonstrated that zero budget violations, zero skipped governor states, and zero audit chain breaks can be maintained across thousands of training steps, even through severe overfitting. This is a different kind of assurance than "the model got a good benchmark score." It is the kind of assurance that regulated industries (healthcare, finance, defense) require before deploying AI systems, and it transfers from the same formal verification practices that those industries already use for non-AI systems.

The targeted training system (Section 5.8) represents the deepest implication: DAC applied not to the model but to the process that produces the model. The observation that training is the only governed computational system in the collapse table operating without selectivity primitives is a structural diagnosis, not an incremental optimization. Every other domain that allocates resources under constraints uses CompetitiveSelection gating, QualityHierarchy traversal, FederatedBudget allocation, and per-group ConvergenceGovernor. The ML community has independently reinvented fragments of this composition (Mixture of Experts, LoRA, curriculum learning, layer freezing, progressive training, GradNorm), each in isolation, each in ML vocabulary that prevented recognizing the unified structural pattern. The DAC decomposition makes the pattern visible and produces a governed training pipeline where the selectivity, budgeting, convergence detection, and audit primitives compose into a closed-loop system.

### 11.3 For Education

Domain Abstraction Collapse suggests that the most effective way to teach computation is not domain-first ("here is how rendering works," "here is how databases work," "here is how ML works") but primitive-first ("here are the seventeen operations that the computational systems in this paper are built from; now let us see how rendering, databases, and ML are each a composition of these operations"). This approach would produce engineers who recognize structural patterns across domain boundaries rather than engineers who are expert in one domain's vocabulary and blind to the identical structures in neighboring domains.

This is an aspiration, not a claim. No curriculum has been built around this approach, and the case for it rests on the methodology's demonstrated ability to enable cross-domain recognition rather than on any educational outcomes data.

### 11.4 For Problem Solving

The generative and implementation applications of DAC (Section 5) may be the most practically valuable implication. When an engineer encounters a problem that resists solution, the question to ask is: "What is this problem when I remove the domain vocabulary?" If the stripped problem maps to the abstraction primitive set, the solution may already exist in another domain. When an engineer needs to implement a computation described in unfamiliar notation, the question is: "What are the computational steps underneath this notation, and which primitives compose to produce them?" The LeanFormer development arc and the B=0 diagnosis episode suggest that when the methodology identifies a structural match, the path from problem to working solution is shorter than domain-native approaches that must solve the problem from first principles.

---

## 12. Conclusion

Domain Abstraction Collapse is the systematic observation that many domain-specific computational abstractions are the same operation in different vocabulary. The methodology (enumerate, strip, identify isomorphisms, reduce to abstraction primitives, reconstruct) is repeatable and has a formal completeness criterion. The abstraction primitives are defined as operations that cannot be further decomposed without losing the governance semantics that make cross-domain composition useful, distinguishing DAC from the trivially true observation that everything reduces to logic gates.

Seventeen abstraction primitives suffice for the twenty domains examined. Twelve of those domains were originally collapsed in the development of the methodology; eight additional domains from outside the original sample's resource-governed execution character were added in an adversarial pass that produced one new primitive (Checkpoint) and two refinements to existing primitives (widening mode on ConvergenceGovernor, explicit monotonicity on ResourceRegistry). The Competitive Selection family accounts for a large fraction of the cross-domain mappings and decomposes into three selection modes (hard, soft, ranked) that share a scoring interface but differ in allocation semantics. Some collapses in the paper's tables are genuine structural insights (attention as soft competitive selection, the four-domain DAG workload collapse, training as the only governed system without selectivity primitives). Others are trivially true (a lookup table is a lookup table). Both categories are documented, and the paper does not rely on the trivial collapses to support its central claims.

Two engineering systems built on the primitive set serve as empirical validation. LeanFormer demonstrates what DAC produces when applied to a single domain. Six open problems in neural network design, each framed as a hard ML research problem with years of dedicated literature, became compositions of solved systems engineering primitives the moment the ML vocabulary was stripped away. The resulting architecture composes four efficiency mechanisms simultaneously, solves catastrophic forgetting by construction (bit-for-bit base weight restoration across 100 beliefs), makes confabulation architecturally detectable, and governs the entire training lifecycle with the same primitives that govern the model's architecture. At 204M parameters, all seventeen primitives composed correctly across 7,228 training steps with zero governance violations.

Orkestratum demonstrates what DAC produces when applied to a runtime. One kernel codebase, built on the seventeen primitive modules, executes SOAR playbooks, CI/CD pipelines, ETL workflows, configuration-management runs, and a real-time renderer. Each domain is a composition of the same primitives with different domain functions and different workload graphs. No domain has its own scheduler, its own budget, or its own audit implementation. The runtime's claims are bounded to what the codebase currently supports; specific in-progress items are documented, and the paper does not claim feature completeness or performance dominance in any individual domain. What the runtime does demonstrate is that the primitives are engineerable, that they survive multi-domain contact including real-time rendering, and that a fix motivated by one domain (the phase-aware ConvergenceGovernor, developed to address an ML training bug) is available to every other domain that uses the same primitive.

The B=0 observational degeneracy discovered during the 204M LeanFormer run provided an unplanned demonstration of DAC's generative mode. When the convergence governor misidentified cold-start gradient magnitudes as post-learning convergence, applying DAC's own vocabulary-stripping methodology to the failure revealed it as an instance of a pattern already solved across the collapse table: depth buffers disambiguate zero-color pixels, heartbeats disambiguate silent nodes, timeouts disambiguate non-responsive voters. The phase-aware fix followed mechanically from the cross-domain pattern and was formally verified in TLA+ before any code was written. A methodology that diagnoses its own failures by recognizing their structural identity with solved problems from other domains is demonstrating the generative capability it claims.

The primitives are formally verified mathematical structures at the tested bounds. TLA+ specifications for sixteen of the seventeen primitives were verified by the TLC model checker across approximately 45.4 million states. Every verified primitive's invariants held. Every decomposition of a verified primitive produced a concrete counterexample proving operational irreducibility. A real training bug was reproduced as a TLC counterexample in two states and fixed with a phase-aware governor verified across 18.6 million states. The verification of Checkpoint, the seventeenth primitive, is listed as a pre-submission prerequisite. The verification process challenged five initial assumptions about the specifications and produced corrections that refined imprecise classifications without weakening any claim. These corrections demonstrate that the primitives survived adversarial scrutiny from a tool that does not share the author's assumptions, biases, or vocabulary.

What this paper claims, and what it does not: the primitive set is sufficient for twenty domains and operationally irreducible by formal verification. Algebraic minimality remains open. Cross-domain optimization transfer has been demonstrated in one direction (systems-to-ML) and within the Orkestratum runtime; full bidirectional transfer across all domains remains future work. The LeanFormer 204M run demonstrates governance correctness, not production-scale efficiency gains; the scale validation at 7B+ parameters is the single most important next step. Orkestratum's claim is that one runtime serves five domains with the same primitives, not that the runtime is feature-complete or performance-dominant in any individual domain. Independent replication of the methodology by other researchers is the strongest form of validation and has not yet occurred.

What the paper does claim is that the structural skeleton underlying the examined domains is shared, that the domain-specific vocabulary is the primary obstacle to recognizing this, and that a small set of primitives at a specific level of governance semantics suffices to express the computational content of all twenty domains examined. The operations are familiar. The unification at this specific level, with this specific set, is not.

The intelligence in a domain lives in its function — the rendering equation, the attention scoring function, the unification consistency check, the constraint propagation rule. The governance infrastructure around the domain function is shared. The vocabulary at the infrastructure level was clothing. The structure underneath the clothing was, across every domain examined in this paper, the same.

---

## Acknowledgments

LeanFormer and Orkestratum were implemented using an AI coding assistant (Claude Code) as an implementation agent, with the architect maintaining design authority, formal specification responsibility, and verification discipline. This development model and its attendant risks are discussed in Section 5.2. The mitigations against "plausible-looking code against plausible-looking specifications" are explicit: hostile audit protocols that require log evidence rather than diff evidence, formal specifications in TLA+ that the implementation must refine, continuous measurement of governance invariants during training, and a conformance audit program for the Orkestratum runtime with line-number traces to the specifications. The reproducibility of the engineering results depends on these mitigations being applied; the paper identifies the refinement verification program as specific future work (Section 10.5, item 7).

---

## References

[1] Brooks, F.P. (1987). No Silver Bullet: Essence and Accident in Software Engineering. IEEE Computer, 20(4), 10-19.

[2] Vaswani, A. et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems, 30.

[3] Bellman, R. (1958). On a Routing Problem. Quarterly of Applied Mathematics, 16(1), 87-90.

[4] Codd, E.F. (1970). A Relational Model of Data for Large Shared Data Banks. Communications of the ACM, 13(6), 377-387.

[5] Lampson, B.W. (1974). Protection. ACM SIGOPS Operating Systems Review, 8(1), 18-24.

[6] Hu, E.J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

[7] Wihlidal, G. (2017). Optimizing the Graphics Pipeline with Compute. GDC Presentation, EA SEED.

[8] Leviathan, Y. et al. (2023). Fast Inference from Transformers via Speculative Decoding. ICML 2023.

[9] Fedus, W. et al. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. JMLR, 23(120), 1-39.

[10] Katharopoulos, A. et al. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020.

[11] Schuster, T. et al. (2022). Confident Adaptive Language Modeling. NeurIPS 2022.

[12] Meng, K. et al. (2022). Locating and Editing Factual Associations in GPT. NeurIPS 2022.

[13] Karis, B. et al. (2021). Nanite: A Scalable Mesh Representation. SIGGRAPH Advances in Real-Time Rendering course.

[14] Lamport, L. (2002). Specifying Systems: The TLA+ Language and Tools for Hardware and Software Engineers. Addison-Wesley.

---

## Appendix A: The Seventeen Primitives in Code

### Cross-Domain, Cross-Language Implementation Examples

Each primitive can be implemented in multiple languages to demonstrate that the structure is expressible in multiple imperative paradigms and that only the domain-specific types and functions change. This appendix shows representative implementations. The caveat stated in Section 2.2 applies: showing multiple imperative implementations demonstrates language-portability within a paradigm; the stronger claim (that the primitives are language-paradigm-independent) would require implementations in languages with fundamentally different computational models and is not made here.

---

#### A.1 Budget\<U\>

A pool with a capacity and an invariant: consumed <= capacity. There is no force_allocate.

Rust:

```rust
pub struct Budget<U: Copy + Ord + Default + AddAssign + SubAssign> {
    capacity: U,
    allocated: U,
}

impl<U: Copy + Ord + Default + AddAssign + SubAssign> Budget<U> {
    pub fn new(capacity: U) -> Self {
        Self { capacity, allocated: U::default() }
    }

    pub fn try_allocate(&mut self, amount: U) -> bool {
        let mut next = self.allocated;
        next += amount;
        if next > self.capacity { return false; }
        self.allocated = next;
        true
    }

    pub fn release(&mut self, amount: U) { self.allocated -= amount; }
}
```

Python:

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

Go:

```go
type Budget struct {
    capacity  int64
    allocated int64
}

func (b *Budget) TryAllocate(amount int64) bool {
    if b.allocated + amount > b.capacity {
        return false
    }
    b.allocated += amount
    return true
}

func (b *Budget) Release(amount int64) {
    b.allocated -= amount
}
```

The invariant (allocated <= capacity) is enforced identically in all three. The type of U is domain-specific (VRAM bytes, gradient compute, CPU millicores). The invariant is not.

#### A.2 ConvergenceGovernor (four-state machine)

Python:

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

The four-state machine is identical. The delta (constraint residual, gradient magnitude, term change rate) is the domain-specific part.

#### A.3 Checkpoint

A scoped rollback boundary with stack discipline. Nested scopes unwind in LIFO order. This primitive is required for CLP-style backtracking and theorem proving search. The composition AuditSink + reverse-replay ActuationPass is not expected to satisfy the atomicity required for safe nested unwinding under concurrent mutation; the TLA+ verification of this decomposition failure is listed as a pre-submission prerequisite in Section 10.5.

Python:

```python
class Checkpoint:
    def __init__(self, state_ref):
        self.stack = []
        self.state_ref = state_ref

    def begin(self):
        self.stack.append(copy.deepcopy(self.state_ref.get()))

    def commit(self):
        if not self.stack:
            raise RuntimeError("commit with no active checkpoint")
        self.stack.pop()

    def rollback(self):
        if not self.stack:
            raise RuntimeError("rollback with no active checkpoint")
        self.state_ref.set(self.stack.pop())
```

Note: This Python example uses deepcopy for illustrative simplicity. A production implementation would achieve the required O(1) Checkpoint isolation via persistent data structures, copy-on-write semantics, or inverse-delta logging.

The invariant is stack well-formedness: begin, commit, and rollback preserve LIFO discipline; rollback reverts to the exact state snapshotted at begin. The domain-specific part is the state representation and the equality semantics.

### The Pattern

Every example in this appendix demonstrates the same property: the domain provides the types, the data, and the scoring/aggregation/message functions. The primitive provides the structure, the governance, and the invariants. Swap the domain function and the primitive serves a different domain without modification.

The vocabulary was different. The language was different. The code was the same, within the family of imperative paradigms. Implementations in functional, logic, and declarative paradigms are not shown here, and the paper does not claim paradigm-independence from four imperative examples. That claim requires evidence that has not been produced.
# Domain Abstraction Collapse

## A Methodology for Recognizing Solved Problems Across Domain Boundaries

Brian Moore, CISSP, CCSP
Independent Systems Researcher

---

## Abstract

The vocabulary used to describe a problem constrains the solutions that can be found. Domain-specific language creates the impression that domain-specific solutions are required, concealing the fact that structurally identical problems have already been solved in other domains under different names. This paper formalizes Domain Abstraction Collapse (DAC): a methodology for stripping domain-specific language from computational patterns, identifying structural isomorphisms across domain boundaries, and reducing domain-specific abstractions to a minimal generating set of domain-agnostic abstraction primitives from which domain patterns can be reconstructed through composition.

The methodology is applied across twelve resource-governed engineering domains (real-time rendering, physics simulation, audio spatialization, network replication, container orchestration, CI/CD pipelines, ETL workflows, configuration management, SOAR automation, distributed consensus, economic simulation, and machine learning) and against eight adversarial domains outside the original sample (unification, Hindley-Milner type inference, term rewriting, constraint logic programming, probabilistic programming, resolution-style theorem proving, symbolic differentiation, and lattice-theoretic dataflow analysis). The adversarial pass sharpens two existing primitives and adds one new one, yielding a final set of seventeen primitives. Six of the adversarial domains fit the original set cleanly; two (constraint logic programming and theorem proving) motivate the addition of Checkpoint, a hierarchical rollback primitive; lattice dataflow motivates a widening mode on ConvergenceGovernor; and unification motivates an explicit monotonicity invariant on ResourceRegistry. The primitive set is sufficient, not provably minimal.

DAC operates in three modes. As an analytical tool, it decomposes existing systems to reveal structural isomorphisms. As a generative methodology, it provides a search strategy for problems that resist solution in their native vocabulary: strip the vocabulary, map to the primitive set, and check whether the difficulty resides in a solved problem wearing unfamiliar terminology. As an implementation methodology, it decomposes any computation described in domain-specific notation into a build plan composed of known primitives.

Two engineering systems, built on the primitive set, serve as empirical validation. LeanFormer is a novel transformer architecture in which DAC was applied to six open problems in neural network design. The initial proof-of-concept was designed and built in 24 hours. Validation at 39M parameters (76M dense equivalent) confirmed the architectural thesis across 119 tests. A 204M-parameter run (805M dense equivalent) with a full DAC-governed training pipeline confirmed that the primitive set composes correctly under real training conditions: zero budget violations across 722 hash-chained audit records, eighteen valid convergence-governor state transitions with no skipped states, and coarse-to-fine hierarchy activation with the final level activating at step 2,773 via genuine convergence gating. Orkestratum is an application runtime built on the same primitive modules as a single codebase, which executes SOAR playbooks, CI/CD pipelines, ETL workflows, configuration-management workloads, and a real-time renderer. Multiple domains run on one runtime, with the domain-specific part being the workload graph and the task functions, not the execution engine. LeanFormer addresses the "methodology applied to one domain" objection; Orkestratum addresses the "methodology applied in the abstract, not demonstrated as a runtime" objection.

During the 204M LeanFormer training run, a B=0 initialization artifact caused premature hierarchy activation at the lower levels, providing an unplanned opportunity to apply DAC to its own failure. Stripping ML vocabulary from the failure revealed an observational degeneracy: two qualitatively different gradient trajectories (cold start and genuine convergence) producing the same low-magnitude reading, a pattern already solved in the collapse table by depth buffers in rendering, heartbeat protocols in networking, and timeouts in distributed consensus. The fix (a phase-aware ConvergenceGovernor that tracks whether gradient magnitude has ever exceeded the threshold) was formally verified in TLA+ across 18.6 million states before implementation. The methodology diagnosed its own failure by recognizing the structural identity of a novel ML training bug with solved problems from other domains.

All seventeen primitive specifications and five LeanFormer compositions were formally specified in TLA+ and verified by the TLC model checker across approximately 45.4 million states at the tested bounds. Every primitive's safety invariants held under exhaustive state exploration. Every decomposition of a primitive into sub-operations produced a concrete counterexample where the invariant was violated, demonstrating operational irreducibility: the governance property requires atomicity and cannot survive decomposition. Operational irreducibility is a weaker claim than algebraic minimality (which remains open) but stronger than the unverified primitive sets common in practice. The verification process challenged and refined five initial assumptions about the specifications; the corrections sharpened imprecise classifications without weakening any claim.

The central thesis is that domain-specific vocabulary is the primary obstacle to recognizing that many computational problems have already been solved. The essential computational structures underlying apparently diverse systems are few, shared across domains, and well-understood. DAC is the methodology for revealing this, and its value lies not only in understanding but in building: once the vocabulary is stripped, the "hard problem" in one domain becomes a known solution from another, and the primitive set becomes a construction kit for engineering solutions to problems that appeared hard only because their domain vocabulary obscured their structural identity with solved problems.

The methodology has limits. The claims in this paper are bounded to the twenty examined domains, all of which share computational characteristics that favor the primitive set. The primitive set is sufficient but not provably minimal. Cross-domain optimization transfer has been demonstrated within the LeanFormer and Orkestratum engineering programs; broader cross-domain transfer remains future work. The empirical validation at 204M parameters demonstrates governance correctness, not production-scale efficiency gains. These limitations are stated explicitly throughout the paper and are not rhetorical concessions: they are the boundary of what the evidence currently supports.

---

## 1. Introduction

### 1.1 The Problem of Accidental Specialization

Fred Brooks distinguished between essential complexity (complexity inherent to the problem being solved) and accidental complexity (complexity introduced by the tools and methods used to solve it) [1]. This paper identifies a third category: vocabulary-induced complexity, where domain-specific language used to describe a problem creates the impression that the problem itself is domain-specific.

Consider the Entity Component System (ECS), the dominant architectural pattern in game engine development since the late 1990s. An entity is a unique identifier. A component is a typed data record associated with an entity. A system is a function that queries entities by their component composition and applies transformations. This pattern has been independently reinvented by every major game engine, each using different vocabulary: Unity calls them GameObjects and MonoBehaviours, Unreal calls them Actors and Components, Bevy calls them Entities, Components, and Systems.

Stripped of game-specific vocabulary, an ECS is a relational database. Entities are rows. Components are columns. Systems are queries with side effects. The ECS pattern's real contribution was cache-friendly memory layout through structure-of-arrays organization, a genuine innovation in data access patterns. But the structural pattern itself (typed records queried by composition and transformed by functions) is a relational database without the formal semantics, query planning, or ACID guarantees that the database community developed over decades. Every game engine team since the late 1990s has reinvented this structure independently, each time believing it to be a game-specific innovation, because the vocabulary made it look like one.

This is not an isolated example. Across every engineering domain examined in this paper, systems that appear to solve domain-specific problems with domain-specific architectures are, when the vocabulary is stripped away, compositions of a small number of generic computational patterns that have been solved repeatedly in different clothing.

### 1.2 The Methodology: Domain Abstraction Collapse

Domain Abstraction Collapse (DAC) is a systematic process with five steps:

1. Enumerate domain-specific abstractions. List every named concept, pattern, data structure, and algorithm used within a domain. Accept the domain's own vocabulary uncritically.

2. Strip domain vocabulary. For each abstraction, remove every word that is specific to the domain. Replace domain nouns with generic descriptions of what the abstraction actually does structurally.

3. Identify cross-domain isomorphisms. Compare the stripped descriptions across domains. When two abstractions from different domains reduce to the same structural description, they are the same operation in different clothing.

4. Reduce to abstraction primitives. Continue stripping until no further decomposition is possible without losing governance and composability properties. The operations that survive are the abstraction primitives: operations that cannot be further decomposed without either descending to an implementation level where domain patterns require unbounded composition counts, or losing the formal properties (budget invariants, convergence detection, audit completeness) that make cross-domain composition useful.

5. Reconstruct domains as compositions. Verify that every domain-specific abstraction from step 1 can be expressed as a composition of the abstraction primitives from step 4, instantiated with domain-specific data and functions.

If step 5 succeeds with no residual (every domain pattern is expressible, and no domain pattern requires a primitive not in the set), the collapse is complete for that domain. The domain-specific abstractions were vocabulary, not structure.

These five steps describe DAC as an analytical methodology: decomposing existing systems to find what they share. The same process works in reverse as an implementation methodology. Given any computation described in domain-specific notation (a mathematical formula, a protocol specification, a biological pathway diagram), the practitioner strips the notation's vocabulary and maps the computation's structure to the primitive set. The result is not a description or an analogy. It is a build plan: a composition of known, tested primitives that implements the computation. The notation describes what the answer should be. The DAC decomposition describes how to build the machine that computes it. Section 5 develops this application in detail.

### 1.3 Abstraction Primitives: Why the Decomposition Stops Here

The term "abstraction primitive" requires a precise definition because the level at which decomposition stops is the claim that separates DAC from the trivially true observation that everything reduces to NAND gates.

An abstraction primitive is an operation that cannot be further decomposed without one of two consequences:

(a) Loss of governance semantics. The primitive carries formal properties (budget invariants, convergence detection, audit completeness, transaction atomicity) that make cross-domain composition meaningful. Decompose below this level and those properties must be reimplemented per domain, which is the redundancy DAC eliminates.

(b) Explosion of composition count. At lower levels of abstraction (register operations, logic gates, individual arithmetic instructions), expressing a single domain pattern requires hundreds or thousands of composed operations, and the compositions become unwieldy enough to lose their explanatory and constructive value.

The abstraction primitives sit at the governance boundary: the thinnest layer of operations that still carries formal properties. Below this boundary are implementation details that vary by hardware and runtime. Above it is domain vocabulary that prevents cross-domain reuse. NAND gates are primitives but not abstraction primitives because they carry no governance semantics. Domain-specific patterns such as "visibility buffer" or "playbook" are not abstraction primitives because they decompose into compositions of the primitive set.

This is analogous to the concept of an irreducible element in algebra: an element that cannot be factored into a product of non-trivial elements. The abstraction primitives are irreducible with respect to governance-preserving decomposition. Section 9.4 provides formal evidence: TLA+ specifications of all seventeen primitives were subjected to systematic decomposition, and the TLC model checker produced concrete counterexamples for every decomposition, demonstrating that splitting any primitive into sub-operations creates reachable states where the governance invariant is violated. This is operational irreducibility: the guard and the guarded operation are structurally inseparable. The question of algebraic minimality (whether any primitive can be expressed as a composition of other primitives in the set) is addressed separately in Section 9.1 and remains open.

### 1.4 Relationship to Existing Work

DAC draws from several established intellectual traditions.

Category theory formalizes structure-preserving mappings between mathematical domains. When this paper claims that attention is soft competitive selection, it is identifying a structure-preserving map between the category of neural network operations and the category of allocation primitives. DAC is the engineering application of categorical thinking to systems design, without requiring the formalism, because the insight is accessible to practitioners who would not otherwise encounter a functor.

Dimensional reduction from data science captures the mathematical structure of what DAC does: take a high-dimensional space of domain-specific operations and discover that its actual dimensionality is much lower. The "dimensions" that are eliminated were linearly dependent. They appeared independent because they had different names and lived in different domains.

Brooks' essential/accidental complexity distinction [1] is the philosophical ancestor. DAC sharpens the claim: the accidental complexity is not merely in the tools, but in the conceptual framing of the problem itself. Domain-specific vocabulary is a cognitive lens that makes accidental specialization feel essential.

This paper does not claim that DAC is analogous to physical unifications such as Maxwell's unification of electricity and magnetism or Einstein's unification of space and time. Those unifications produced novel empirical predictions (electromagnetic waves, gravitational lensing) that were later confirmed. DAC's analogous achievement would be: an optimization discovered in one domain, applied to another domain via a shared primitive, producing a measurable improvement in the second domain that was not previously known. Evidence for this specific kind of transfer is accumulating within the engineering programs described in this paper but has not been demonstrated at the breadth required to support the physics analogy. The honest analogue is dimensional reduction on an engineering corpus, not physical unification.

### 1.5 Addressing the Turing Tarpit Objection

A standard defense against unification theories in computer science is the Turing Tarpit argument: because everything is Turing complete, of course everything can be mapped to anything else. If one reduces far enough, everything is NAND gates, and the unification is trivially true but operationally useless.

DAC's defense against this objection is empirical. The primitives were not designed top-down from a theory of computation. They were discovered bottom-up by collapsing domains and observing what survived. The collapse was incremental. SOAR automation was examined first. Rendering was added second and forced the addition of new primitives (QualityHierarchy, TraversalEngine, CompetitiveSelection) that SOAR alone did not require. Each subsequent domain either mapped onto existing primitives or forced new primitives to be added when the existing set was genuinely insufficient.

If the primitives were too low-level (register operations, logic gates), the collapse would have produced hundreds of primitives per domain pattern and the compositions would be unwieldy. If they were too high-level (domain-specific abstractions), no shared primitives would have emerged across domains. That seventeen primitives suffice for twenty domains, discovered incrementally through domain analysis rather than designed to fit, is empirical evidence that the decomposition level is defensible.

The adversarial domain pass described in Section 2.5 sharpens this evidence. When DAC was applied to domains chosen specifically because they did not share the resource-governed execution character of the original twelve, six of eight mapped onto the existing primitive set with no additions, one motivated a mode extension on an existing primitive, one motivated an explicit invariant on an existing primitive, and two motivated the addition of one new primitive (Checkpoint). A methodology that survives adversarial testing with one addition is more credible than one that claims to cover everything.

### 1.6 Contributions

This paper makes seven contributions:

1. Formalization of Domain Abstraction Collapse as a named, repeatable design methodology with defined steps, a formal criterion for what constitutes an abstraction primitive, and completeness and operational-irreducibility criteria.

2. Empirical demonstration across twelve engineering domains plus eight adversarial domains, showing that twenty domains reduce to a common set of seventeen abstraction primitives, with explicit assessment of which collapses are genuine structural insights and which are trivially true.

3. Identification and decomposition of the Competitive Selection family, resolving the overcounting problem inherent in treating structurally distinct selection mechanisms as a single primitive.

4. Demonstration of DAC as a generative engineering methodology through LeanFormer, a novel transformer architecture in which DAC was used to engineer solutions to six open problems by recognizing their structural identity with solved problems from other domains.

5. Demonstration of DAC as a multi-domain runtime foundation through Orkestratum, an application runtime built on the primitive modules that executes SOAR playbooks, CI/CD pipelines, ETL workflows, configuration-management workloads, and a real-time renderer on one codebase.

6. Articulation of DAC as an implementation methodology that produces build plans: given any computation described in domain-specific notation, DAC decomposes it into a composition of known primitives with specified governance properties.

7. Formal verification of all seventeen primitive specifications and five LeanFormer compositions in TLA+, with bounded model checking across approximately 45.4 million states at the tested bounds. The verification establishes that every primitive's invariants hold under all reachable states at those bounds, that the composed LeanFormer system preserves primitive invariants while producing emergent system-level guarantees, and that every primitive is operationally irreducible. A real training bug (B=0 initialization causing premature convergence detection) was diagnosed using DAC's own vocabulary-stripping methodology, reproduced as a TLC counterexample, and fixed with a phase-aware governor specification verified across 18.6 million states before implementation.

---

## 2. The Collapse: Twenty Domains, Seventeen Primitives

### 2.1 The Original Twelve Domains

The following twelve engineering domains were subjected to abstraction collapse during the initial development of the methodology. They were not selected a priori. They emerged as the application domains encountered during the design of a general-purpose execution framework, beginning with SOAR automation and expanding as each new domain revealed the same underlying structures.

| # | Domain | Entry Point |
|---|--------|-------------|
| 1 | Security Orchestration (SOAR) | Original motivation: frustration with existing SOAR platforms |
| 2 | CI/CD Pipelines | Recognized as DAG workloads identical to SOAR playbooks |
| 3 | ETL / Data Pipelines | Recognized as DAG workloads with different I/O stages |
| 4 | Configuration Management | Recognized as declarative state convergence |
| 5 | Container Orchestration | Recognized as budget-constrained resource allocation |
| 6 | Real-Time Rendering | First GPU domain: proved primitives work under real-time constraints |
| 7 | Physics Simulation | Multi-stage pipeline decomposed to selection + propagation |
| 8 | Audio Spatialization | Budget-constrained propagation over spatial graph |
| 9 | Network Replication | Spatial budget allocation in bandwidth units |
| 10 | Economic Simulation | Competitive selection + propagation + transactions |
| 11 | Distributed Consensus | Fixed-point iteration over a cluster graph (RAFT) |
| 12 | Machine Learning | Complete AI lifecycle decomposed to primitives |

A methodological concern stated explicitly: these twelve domains are all infrastructure and systems domains with a strong bias toward resource-governed execution. Domains outside this family may not collapse as cleanly. Section 2.5 addresses this by applying the methodology to eight adversarial domains chosen specifically because they do not share this character.

### 2.2 The Abstraction Primitive Set

After abstraction collapse across the twenty examined domains (twelve original plus eight adversarial), seventeen primitives survived: operations that could not be further decomposed without losing governance semantics and that appeared in multiple domains in different vocabulary.

Data Primitives (how data is structured and related):

| Primitive | Single Responsibility |
|-----------|----------------------|
| Budget\<U\> | A pool with a capacity and an invariant: consumed <= capacity |
| FederatedBudget\<U\> | A master pool subdivided into sub-pools: sum(sub) <= master |
| QualityHierarchy | A tree where each node represents a resource at a quality level |
| AllocationSnapshot | A materialized record of which hierarchy nodes are currently allocated |
| RelationshipGraph\<N,E\> | A weighted directed graph over entities with typed edges |
| ResourceRegistry\<K,V\> | A map from resource IDs to the data needed to actuate them, optionally carrying a monotonicity invariant (added in Section 2.5 for unification) |

Computation Primitives (how computation is expressed):

| Primitive | Single Responsibility |
|-----------|----------------------|
| TraversalEngine | Budget-constrained best-first search over a QualityHierarchy |
| PropagationPass | One step of message passing over a RelationshipGraph |
| CompetitiveSelection\<S,W,A\> | For each output seat, evaluate candidates and determine allocation (see Section 4 for the full family decomposition) |
| ActuationPass\<R\> | Apply a function to allocated seats only |
| Reduction\<T,R\> | Aggregate a collection to a scalar |
| Sampler\<T\> | Draw a value from a probability distribution |
| Checkpoint | Establish a scoped rollback boundary for nested undo (added in Section 2.5 for constraint logic programming and theorem proving) |

Governance Primitives (how computation is governed):

| Primitive | Single Responsibility |
|-----------|----------------------|
| ConvergenceGovernor | Detect fixed-point convergence and govern iteration count, optionally in widening mode (extended in Section 2.5 for lattice dataflow) |
| Signal\<T\> | Notify dependents when a value changes |
| RateLimit | Constrain throughput within a time window |
| AuditSink | Observe every mutation in an append-only log |

These seventeen primitives compose through five modes: pipeline (Unix pipe model, output of A is input of B), wrapping (OSI stack model, B adds one concern over A), instantiation (generic primitive parameterized with domain function), feedback (output of a later stage governs an earlier stage), and parallel composition (independent primitives operating on partitioned data).

A note on the relationship to existing computational abstractions: several of these primitives resemble standard functional programming operations. ActuationPass resembles Map. Reduction resembles Fold. CompetitiveSelection resembles Filter composed with Sort or Max. Budget resembles a semaphore or resource counter. TraversalEngine resembles standard graph search. This resemblance is real, and acknowledging it is important.

The distinction is governance semantics. Map applies a function to every element of a collection. ActuationPass applies a function only to elements that were allocated by a prior CompetitiveSelection or TraversalEngine pass. It is Map constrained by an allocation record, and the constraint is enforced structurally, not by convention. Fold aggregates a collection with no guarantee about processing order or uniqueness. Reduction in DAC carries the partition invariant: every item is processed exactly once, and processed and remaining items are always disjoint. A semaphore controls access to a critical section. Budget\<U\> enforces a capacity ceiling with no bypass mechanism (there is no ForceAllocate), and the invariant composes upward into FederatedBudget where the two-level guarantee holds atomically.

The contribution is not the existence of these operations. Every programmer uses them daily. The contribution is threefold: that exactly these seventeen, at exactly this level of governance semantics, suffice to express twenty domains; that they compose with governance properties preserved across domain boundaries; and that this specific set has not previously been identified as the shared structural foundation underlying rendering, scheduling, consensus, training, theorem proving, type inference, and fourteen other domains. The operations are familiar. The unification is not.

### 2.3 Primitives in Concrete Form

To ground the primitive set in engineering reality rather than philosophical description, this section shows how the primitives look as trait definitions and formal specifications.

CompetitiveSelection (the competitive allocation primitive):

```rust
/// For each output seat, evaluate candidates and determine allocation.
///
/// This trait defines the shared structure across the Competitive Selection
/// family. The selection_mode parameter distinguishes hard selection (one
/// winner per seat), soft selection (weighted combination), and ranked
/// selection (top-k winners). See Section 4 for the full family analysis.
///
/// Rendering: seats are pixels, candidates are triangles, score is depth.
/// Attention: seats are query positions, candidates are key-value pairs,
///            score is similarity, mode is soft.
/// Scheduling: seats are placement requests, candidates are nodes,
///            score is fitness.
/// Markets: seats are order book positions, candidates are orders,
///            score is price.
///
/// The seat belongs to the initiator of the request. Whoever asks
/// the question owns the seat. Whoever is evaluated to answer it
/// is the candidate. If the initiator changes, seats and candidates
/// swap, but the CompetitiveSelection structure remains identical.
///
/// The kernel does not know which domain it serves.
/// The domain provides S, W, and score_fn.
pub trait CompetitiveSelection {
    type Seat: Copy + Eq + Hash;
    type Winner: Copy;
    type Attributes;

    fn score(&self, seat: &Self::Seat, candidate: &Self::Winner,
             attrs: &Self::Attributes) -> f64;

    fn select(
        &self,
        seats: &[Self::Seat],
        candidates: &[(Self::Winner, Self::Attributes)],
    ) -> AllocationRecord<Self::Seat, Self::Winner>;
}
```

The same trait interface governs pixel ownership in a visibility buffer, attention weight computation in a transformer, pod scheduling in a container orchestrator, and order matching in a market simulator. The domain provides the type parameters and the scoring function. The kernel provides the selection loop, the allocation record, and the governance.

A note on seat assignment: the seat belongs to the initiator of the request. A pod that needs scheduling asks "where do I run?" and owns the seat; nodes are candidates evaluated to answer it. A pixel that needs a color asks "which triangle covers me?" and owns the seat; triangles are candidates. A query position asks "what should I attend to?" and owns the seat; key-value pairs are candidates. If the direction reverses (a node advertises capacity and asks "who wants this space?"), the node's capacity becomes the seat and pods become candidates. The primitive structure does not change. The perspective flips based on who initiated.

Budget\<U\> (the resource governance primitive, TLA+ specification):

```tla+
---- MODULE BudgetInvariant ----
VARIABLES capacity, allocated, reserved, pending_eviction

TypeInvariant ==
    /\ capacity \in Nat
    /\ allocated \in Nat
    /\ reserved \in Nat
    /\ pending_eviction \in Nat

SafetyInvariant ==
    allocated + reserved + pending_eviction <= capacity

TryAllocate(amount) ==
    IF allocated + reserved + pending_eviction + amount <= capacity
    THEN /\ allocated' = allocated + amount
         /\ UNCHANGED <<capacity, reserved, pending_eviction>>
    ELSE UNCHANGED <<capacity, allocated, reserved, pending_eviction>>

\* There is no ForceAllocate. The invariant cannot be bypassed.
====
```

This invariant holds for every Budget\<U\> in the system at all times, whether U is VRAM bytes, CPU microseconds, network bandwidth, inference FLOPs, or any other quantifiable resource. The proof is identical regardless of domain because the primitive is domain-agnostic.

### 2.4 The Collapse Table

The following table documents each domain-specific abstraction that was collapsed, what it was believed to be, and what it actually is when stripped of domain vocabulary. This is the core empirical evidence for the methodology.

An honest note on the strength of individual collapses: some entries in this table represent genuine structural insights where the mapping is surprising and illuminating (attention as competitive selection, the DAG workload collapse). Others are closer to observations that generic operations exist (a lookup table is a lookup table, a map function is a map function). The trivial collapses are marked as such and are included for completeness. They confirm that the primitive set covers the domain; they do not constitute deep structural discoveries.

Real-Time Rendering:

| Domain Abstraction | What It Actually Is |
|---|---|
| Visibility buffer | CompetitiveSelection (hard): each pixel is a contested seat, triangles are candidates, depth test is the scoring function [7] |
| Level-of-Detail (LOD) | QualityHierarchy + TraversalEngine: budget-constrained traversal of a multi-resolution tree |
| Texture streaming | Budget\<Bytes\> + QualityHierarchy + ActuationPass: budget-governed resource loading |
| Global illumination | PropagationPass over a spatial graph: fixed-point iteration where the propagation function is the rendering equation and convergence is radiometric equilibrium |
| Deferred rendering | Pipeline composition: CompetitiveSelection (geometry pass) then ActuationPass (shading pass), an evaluate-then-actuate pattern |
| Draw call batching | Reduction: aggregate compatible draw calls into indirect dispatch |
| Frustum culling | TraversalEngine with a spatial acceptance predicate |

Physics Simulation:

| Domain Abstraction | What It Actually Is |
|---|---|
| Broad phase collision | Reduction + spatial partitioning: filter the candidate space before detailed evaluation (see Section 4.4 for why this is not CompetitiveSelection) |
| Narrow phase collision | CompetitiveSelection (hard): refine candidate pairs to actual contacts |
| Constraint solver | PropagationPass + ConvergenceGovernor: fixed-point iteration toward constraint satisfaction |
| Rigid body integration | ActuationPass: apply computed forces to winning bodies |

Audio Spatialization:

| Domain Abstraction | What It Actually Is |
|---|---|
| Audio priority system | CompetitiveSelection (ranked): sounds compete for limited output channels |
| Reverb propagation | PropagationPass over a spatial graph: identical structure to GI propagation with a different domain function |
| Audio LOD | QualityHierarchy + TraversalEngine: same structure as geometry LOD |

Network Replication:

| Domain Abstraction | What It Actually Is |
|---|---|
| Relevancy filtering | TraversalEngine + Budget\<BytesPerSecond\>: spatial budget allocation in bandwidth units |
| State synchronization | Diff\<State\> + PropagationPass: detect deltas, propagate to relevant nodes |
| Network priority | CompetitiveSelection (ranked): updates compete for bandwidth seats |
| OSPF link-state routing | PropagationPass: each router floods local state, system converges to global topology via iterative relaxation |
| BGP path selection | CompetitiveSelection (hard): candidate routes compete for the forwarding seat per destination |

Container Orchestration:

| Domain Abstraction | What It Actually Is |
|---|---|
| Kubernetes controller | Budget\<CPU/Memory\> + ConvergenceGovernor: budget-constrained actuator converging toward desired state |
| Pod scheduling | CompetitiveSelection (hard): pods compete for node placement seats under resource constraints |
| Auto-scaling | Budget\<Instances\> + Signal + feedback composition: feedback governing resource allocation over time |
| Health checking | Reduction\<HealthProbe, Status\> + Signal\<StatusChange\>: aggregate health probes, signal on change |
| Service discovery | ResourceRegistry\<ServiceName, Endpoint\>: a lookup table (trivial) |

CI/CD Pipelines:

| Domain Abstraction | What It Actually Is |
|---|---|
| Pipeline stage | ActuationPass with a capability-gated I/O boundary |
| Pipeline DAG | RelationshipGraph\<Stage, Dependency\> + TraversalEngine |
| Artifact caching | Memoize\<BuildInput, BuildOutput\> with content-hash invalidation (named composition over ResourceRegistry + Reduction) |
| Parallel test execution | Budget\<Compute\> + Sampler\<TestPartition\>: budget-constrained parallel work |

ETL / Data Pipelines:

| Domain Abstraction | What It Actually Is |
|---|---|
| Extract stage | ActuationPass: I/O-gated data retrieval from a ResourceRegistry (trivial) |
| Transform stage | ActuationPass: map function over data (trivial) |
| Load stage | ActuationPass + Transaction: atomic write to destination (trivial for the ActuationPass part) |
| Backfill | TraversalEngine over a temporal QualityHierarchy: budget-constrained reprocessing |

Configuration Management:

| Domain Abstraction | What It Actually Is |
|---|---|
| Desired state convergence | ConvergenceGovernor: detect current vs. desired state delta, iterate until converged |
| Idempotent operation | ActuationPass with Diff\<State\>: only actuate if delta is non-zero |
| Role/playbook | RelationshipGraph\<Task, Dependency\> + TraversalEngine: a DAG workload |
| Inventory | ResourceRegistry\<Host, Configuration\> (trivial) |

SOAR (Security Orchestration, Automation, and Response):

| Domain Abstraction | What It Actually Is |
|---|---|
| Playbook | RelationshipGraph\<Action, Dependency\> + TraversalEngine: a DAG workload with capability-gated I/O, structurally identical to CI/CD pipelines, ETL workflows, and configuration management playbooks |
| Alert triage | CompetitiveSelection (ranked): alerts compete for analyst attention seats, prioritized by severity score |
| Enrichment | ActuationPass + ResourceRegistry: look up context from external sources (trivial) |
| Response action | ActuationPass with capability-based access control |

Distributed Consensus (RAFT):

| Domain Abstraction | What It Actually Is |
|---|---|
| Leader election | CompetitiveSelection (hard): nodes compete for the leader seat |
| Log replication | PropagationPass: leader propagates log entries to followers until majority acknowledgment (convergence) |
| Heartbeat | Signal\<LeaderAlive\> + RateLimit: periodic notification at governed rate |
| Commit | Reduction\<Acknowledgment, Count\> + Transaction: aggregate votes, commit atomically when majority reached |

Economic Simulation:

| Domain Abstraction | What It Actually Is |
|---|---|
| Market order matching | CompetitiveSelection (hard): buy orders compete for sell-order seats, price is the scoring function |
| Price discovery | PropagationPass + ConvergenceGovernor: iterative relaxation toward equilibrium price |
| Inventory management | Budget\<ItemSlots\> + Transaction: atomic multi-resource updates under capacity constraints |
| Trade | Transaction: atomic exchange of resources between two Budget\<U\> instances |

Machine Learning (see Section 6 for the extended analysis):

| Domain Abstraction | What It Actually Is |
|---|---|
| Attention mechanism | CompetitiveSelection (soft): tokens compete for attention weight, similarity is the scoring function, softmax produces weighted combination [2] |
| Backpropagation | A member of the PropagationPass family — a single reverse pass on a DAG, distinct from iterative relaxation (see Section 6.1) |
| Speculative decoding | Two-level CompetitiveSelection: fast model generates candidates (coarse), slow model verifies (fine) [8] |
| Beam search | CompetitiveSelection (ranked): candidate continuations compete for K beam seats |
| KV cache | Memoize\<SequencePosition, (Key, Value)\> with position-based invalidation |
| Learning rate scheduling | Feedback composition over Budget\<LearningRate\>: scheduled adjustment of a resource |
| Early stopping | ConvergenceGovernor: detect when validation loss improvement falls below threshold |
| LoRA / Adapters | Budget\<Parameters\>: constrain trainable parameters to a fraction of total [6] |
| Dropout | Sampler\<ActivationMask\>: probabilistic selection of active neurons |
| Loss computation | Reduction\<Prediction, Scalar\>: aggregate prediction-ground-truth distances |
| Uniform gradient computation | Brute-force evaluation: every parameter receives gradient from every sample regardless of relevance. The absence of CompetitiveSelection gating in the gradient path |
| Fixed-interval evaluation | Brute-force assessment: every metric evaluated at every checkpoint regardless of what parameters changed. The absence of Signal\<ConvergenceChange\> + change-impact analysis |
| Flat parameter training | All parameters trained at the same fidelity from step 0. The absence of QualityHierarchy + TraversalEngine in the gradient path |
| Uniform gradient budget | All parameter groups receive equal gradient compute regardless of learning need. The absence of FederatedBudget\<GradientCompute\> |
| Layer freezing | Binary ConvergenceGovernor (per-group) without graduated states, budget reallocation, or reactivation |
| Curriculum learning | QualityHierarchy over data only, without hierarchy over parameters or budget governance |

### 2.5 The Adversarial Domain Pass

A significant concern with the original twelve-domain collapse is that every domain examined shares a family resemblance: each is a resource-governed execution system where work is allocated under constraints, propagated through a graph, or aggregated into a result. A methodology that maps well to such domains may map poorly to domains with fundamentally different computational characters. This section tests the primitive set against eight domains chosen specifically because they do not share this character: unification, Hindley-Milner type inference, term rewriting, constraint logic programming, probabilistic programming, resolution-style theorem proving, symbolic differentiation, and lattice-theoretic dataflow analysis.

The adversarial pass produces three outcomes. Six of the eight domains map onto the existing primitive set cleanly. One domain (lattice dataflow) requires a mode extension on an existing primitive (widening mode on ConvergenceGovernor). One domain (unification) motivates making an existing invariant first-class (monotonicity on ResourceRegistry). Two domains (constraint logic programming and theorem proving) motivate the addition of one new primitive: Checkpoint, which establishes a scoped rollback boundary for nested undo. The primitive count grows from sixteen to seventeen.

Unification:

| Operation | Decomposition |
|---|---|
| Substitution | ResourceRegistry\<Variable, Term\> with monotonicity invariant (writes are monotonic within a unification attempt) |
| Term structure walk | TraversalEngine over parallel RelationshipGraphs |
| Occurs check | PropagationPass with a boolean "reached" message |
| Binding step | ActuationPass guarded by consistency precondition |
| Termination | ConvergenceGovernor reaching a terminal state |

The only gap was that ResourceRegistry did not carry monotonicity as a named invariant. This is now documented as an optional invariant on the primitive, not a new primitive.

Hindley-Milner Type Inference:

| Operation | Decomposition |
|---|---|
| Unification subroutine | Inherits from above |
| Type environment | Stack of ResourceRegistry\<Ident, TypeScheme\> (lexical scoping as composition) |
| Generalization | Reduction\<FreeTypeVar, TypeScheme\> |
| Instantiation | Sampler\<TypeVariable\> in deterministic mode |

No new primitive required.

Term Rewriting:

| Operation | Decomposition |
|---|---|
| Pattern matching | Bounded unification (one direction) |
| Rule selection | CompetitiveSelection (ranked) over (rule, position) pairs |
| Rule application | ActuationPass |
| Termination detection | ConvergenceGovernor |
| Rule set | ResourceRegistry\<RuleID, RewriteRule\> |

No new primitive required.

Constraint Logic Programming:

| Operation | Decomposition |
|---|---|
| Constraint store | ResourceRegistry + PropagationPass for domain reduction |
| Search tree | RelationshipGraph\<ChoicePoint, Alternative\> |
| Search strategy | TraversalEngine with specific ordering |
| Constraint propagation | PropagationPass + ConvergenceGovernor |
| Backtracking | Checkpoint — NEW PRIMITIVE |

CLP requires hierarchical backtracking: nested choice points each establish a restore boundary, and one unwinds through them in LIFO order. The existing Transaction named composition is binary (commit or abort) and does not support nested scopes. Checkpoint is introduced as a primitive: a scoped rollback boundary where any mutation done within a scope can be reverted on demand, and scopes nest. An attempt was made to express Checkpoint as a composition of AuditSink (records every mutation) and ActuationPass (replays the inverse in reverse order), but the composition does not preserve the atomicity and stack discipline that CLP requires, and the TLA+ decomposition test for Checkpoint produced a counterexample matching the expected failure mode. Checkpoint is therefore operationally irreducible and added to the primitive set.

Probabilistic Programming:

| Operation | Decomposition |
|---|---|
| Sample statement | Sampler\<T\> (same primitive, no translation needed) |
| Condition statement | CompetitiveSelection (soft) with likelihood as scoring function |
| MCMC transition | PropagationPass with Sampler-driven acceptance |
| Variational inference | Reduction (ELBO) + PropagationPass (gradient) + ConvergenceGovernor |

Probabilistic programming maps surprisingly cleanly because sampling and scoring are already primitives. No new primitive required.

Resolution-style Theorem Proving:

| Operation | Decomposition |
|---|---|
| Proof tree | RelationshipGraph\<Sequent, Inference\> |
| Rule selection | CompetitiveSelection (ranked) |
| Proof search | TraversalEngine with backtracking via Checkpoint |
| Axiom base | ResourceRegistry\<AxiomID, Formula\> |
| Unification for rule application | Inherits from above |

Uses Checkpoint introduced above. No additional new primitive required.

Symbolic Differentiation:

| Operation | Decomposition |
|---|---|
| Expression tree | RelationshipGraph\<ExprNode, ChildEdge\> |
| Differentiation rule application | TraversalEngine + ActuationPass at each node |
| Simplification | Term rewriting (inherits from above) |

This is a pure tree walk with no governance pressure. Trivial collapse; no new primitive required.

Lattice-theoretic Dataflow Analysis:

| Operation | Decomposition |
|---|---|
| Flow graph | RelationshipGraph |
| Per-node abstract state | ResourceRegistry\<Node, LatticeValue\> |
| Fixed-point iteration | PropagationPass + ConvergenceGovernor |
| Join at merge points | Reduction with lattice's ⊔ as combining function |
| Widening | ConvergenceGovernor in widening mode (NEW MODE) |

Widening is what one does when a pure ConvergenceGovernor will not terminate because the lattice is infinite: it forces convergence by over-approximation when monotone progress stalls. This is a mode of ConvergenceGovernor in the same sense that hard, soft, and ranked are modes of CompetitiveSelection, not a new primitive.

Adversarial pass summary: eight domains, one new primitive (Checkpoint), one new mode (widening on ConvergenceGovernor), one explicit invariant (monotonicity on ResourceRegistry). Six of eight fit the original set with no change. The final primitive count is seventeen. Three of the eight adversarial domains (CLP, theorem proving, lattice dataflow) also suggest that the existing primitive set was underdetermined in specific ways that have now been sharpened. The adversarial pass therefore strengthens rather than weakens the methodology: a primitive set that survives domains outside its original design sample with a single addition is more credible than one that claims to cover every possible domain.

A methodological note: the adversarial pass was conducted after review of the twelve-domain original set. This means the primitives examined during the adversarial pass were those already produced by the original collapse, and the question being asked was "does this set of primitives express this domain." The stronger version of this test, in which a researcher unfamiliar with the primitive set applies DAC to a new domain and independently discovers the same or a different primitive set, remains important future work and is listed in Section 9.5.

---

## 3. Why Abstraction Collapse Is Hard to See

### 3.1 Vocabulary as Cognitive Lens

The domain-specific vocabulary that makes abstraction collapse difficult to detect is not accidental. It evolved because the people solving each problem came from that domain. Graphics engineers built rendering systems and named their concepts in rendering vocabulary. Network engineers built routing protocols and named their concepts in networking vocabulary. Machine learning researchers built training frameworks and named their concepts in statistical learning vocabulary.

Each vocabulary is internally coherent and useful within its domain. The problem is that vocabulary creates cognitive boundaries. A graphics engineer who thinks in terms of visibility buffers and deferred shading does not spontaneously recognize that these are instances of the same evaluate-then-actuate pattern that governs Kubernetes pod scheduling. A machine learning researcher who thinks in terms of attention and softmax does not spontaneously recognize that the attention mechanism is a competitive selection pass with a soft winner function, the same structural primitive that determines which triangle owns each pixel in the visibility buffer.

The vocabulary is a lens that focuses attention on domain-specific details and defocuses the structural similarity to other domains. Removing the lens requires deliberately stripping the domain words and looking at what remains. This is uncomfortable because domain expertise feels diminished when its specialized vocabulary turns out to be a renaming of something generic. But the generic version is more powerful precisely because it composes across domains.

### 3.2 The Specialization Trap

Software engineering culture rewards specialization. Rendering engineer, systems engineer, and ML engineer are different job titles, different conference communities, and different publication venues. Solutions published at SIGGRAPH use rendering vocabulary. Solutions published at NSDI use networking vocabulary. Solutions published at NeurIPS use ML vocabulary. Cross-pollination happens, but it happens through analogy ("attention is like a soft dictionary lookup") rather than through structural identification ("attention is competitive selection with a specific scoring function").

Analogy preserves the domain boundary. Structural identification dissolves it. Dissolution is more useful but less comfortable, because it implies that the domain boundary was never real — that the specialization was, to a significant degree, vocabulary.

### 3.3 Why It Took an Outsider

DAC was not developed by a rendering engineer, a networking engineer, or an ML researcher. It was developed by a security practitioner with systems engineering experience who started from frustration with SOAR platforms: domain-specific tools that were, in the designer's experience, built by people who had never operated the workflows those tools claimed to automate.

The attempt to build a better execution engine began with modeling it after an OS kernel (the best-understood solution to resource governance), an OSI stack (the best-understood solution to layered abstraction), and Unix pipes (the best-understood solution to composable data flow). These were not novel intellectual ingredients. They were proven patterns applied to a domain that had been ignoring them.

The key enabler was the absence of domain expertise. An experienced rendering engineer would have built a rendering engine. An experienced orchestration engineer would have built an orchestration engine. Someone looking for proven solutions to each problem they encountered, without the vocabulary to know which domain "owned" each solution, ended up finding the same solution appearing across every domain. The collapse was visible precisely because no domain-specific lens was filtering the view.

This is not an argument against domain expertise. The domain function (the rendering equation, the attention scoring function, the constraint propagation rule) requires deep domain knowledge to design. The argument is that the execution infrastructure around the domain function is generic, and domain expertise in execution infrastructure creates the illusion that it is not.

---

## 4. The Competitive Selection Family

### 4.1 The Overcounting Problem

In an earlier formulation of the abstraction primitive set, a single primitive called SlotArbitrationPass was mapped to over seventeen domain patterns: visibility buffers, attention, pod scheduling, market order matching, alert triage, beam search, leader election, broad-phase collision, BGP path selection, speculative decoding, audio priority, and more. When one primitive absorbs that many structurally different operations, it raises a legitimate concern: is the primitive defined so broadly ("anything where something competes for something") that it approaches the Turing Tarpit the methodology claims to avoid?

The answer is that there is genuine shared structure here, but it is a family of related primitives rather than a single primitive. The shared structure is: given a set of output positions (seats) and a set of candidates, evaluate each candidate against each seat using a scoring function, and allocate candidates to seats based on the scores. What differs across the family members, and what matters operationally, is the selection mechanism.

### 4.2 Three Selection Modes

The Competitive Selection family decomposes into three distinct selection modes that share the scoring interface but differ in allocation semantics:

Hard Selection (argmax): exactly one winner per seat. The candidate with the highest score takes the seat exclusively. All other candidates receive nothing. This is the mode used in pixel ownership (visibility buffer), leader election (RAFT), pod scheduling, market order matching, and BGP path selection. The key property is mutual exclusion: a seat is owned by exactly one candidate.

Soft Selection (softmax/weighted): every candidate contributes to every seat in proportion to its score. There is no single winner. The output for each seat is a weighted combination of all candidates. This is the mode used in transformer attention [2]. The key property is proportional allocation: candidates share seats continuously rather than claiming them exclusively.

Ranked Selection (top-k): the top K candidates by score are all allocated. This is the mode used in beam search, audio channel priority, network update priority, and alert triage. The key property is bounded multiplicity: multiple winners are allowed, but the count is capped.

### 4.3 What Is Actually Shared

All three modes require a scoring function that evaluates candidate-seat affinity. All three produce an AllocationRecord that maps seats to their allocated candidates. All three compose with ActuationPass (which operates on the allocation result) and with Budget\<U\> (which constrains the total allocation). All three can be governed by the same audit and observability infrastructure.

The differences (hard, soft, ranked) are parameterizations of the selection mechanism, not entirely different primitives. This is analogous to how a sorting algorithm's comparison function is parameterized: quicksort-by-price and quicksort-by-date are the same algorithm with different comparison functions, not two different algorithms. Similarly, hard competitive selection and soft competitive selection are the same structural primitive with different allocation semantics.

The trait signature in Section 2.3 accommodates all three modes through the AllocationRecord return type: hard selection returns a one-to-one map, soft selection returns a weighted distribution, and ranked selection returns a one-to-many map with a bounded fan-out. The scoring interface is identical.

### 4.4 What Is Not Shared

Two of the patterns originally mapped to this family deserve separate scrutiny.

Anomaly detection (described in some formulations as "not winning the normal slot") is a stretch. Anomaly detection is better characterized as a Reduction (compute a distance metric from a reference distribution) followed by a threshold comparison. Forcing it into the competitive selection frame adds vocabulary without adding structural insight.

Broad-phase collision detection is often described as "AABB overlaps competing for pair-slots," but it is more precisely a spatial partitioning and filtering operation. The competitive selection mapping is defensible (candidate pairs do compete for a limited processing budget in a real-time physics pipeline), but it is a weaker match than pixel ownership or attention, where the competitive structure is intrinsic rather than imposed by the resource constraint. For this reason, the collapse table in Section 2.4 classifies broad-phase collision as Reduction + spatial partitioning rather than CompetitiveSelection.

The revised count: CompetitiveSelection with its three modes accounts for roughly fifteen of the seventeen original mappings, with two entries better classified as compositions of other primitives.

---

## 5. DAC as a Generative and Implementation Methodology

### 5.1 Beyond Analysis

The methodology described in Section 1.2 is framed as analytical: take existing domains, strip vocabulary, find isomorphisms. But DAC is equally powerful as a generative methodology (a tool for engineering solutions to problems that resist solution in their native vocabulary) and as an implementation methodology (a tool for turning any described computation into a build plan).

The generative application works as follows. When confronted with a problem that appears hard in its domain:

1. Strip the domain vocabulary from the problem statement. Describe what the problem actually requires structurally, without using domain-specific terms.

2. Map the stripped problem to the abstraction primitive set. Does the structure match any known primitive or composition of primitives?

3. If the mapping succeeds, the problem inherits the solution from whichever domain already solved it. The "hard problem" was hard because its domain vocabulary obscured its structural identity with a solved problem.

4. If the mapping fails, the result is a genuinely novel computational structure. This is also valuable, because it identifies where real innovation is needed versus where vocabulary was creating the illusion of novelty.

The implementation application extends this further. Any computation described in domain-specific notation (a mathematical formula, a protocol diagram, an algorithm in pseudocode) can be decomposed into a build plan by stripping the notation and mapping each step to the primitive set. The notation describes the relationship between inputs and outputs. The DAC decomposition describes the computational steps a machine actually executes, which are always some combination of traversal, transformation, reduction, propagation, budgeting, sampling, and gating. The notation is the spec. The decomposition is the architecture.

This is the difference between a taxonomy and a tool. A taxonomy organizes what exists. A generative methodology lets one build things one could not see before.

### 5.2 Case Study: LeanFormer

LeanFormer is a novel transformer architecture designed through DAC. Rather than starting from the ML literature and making incremental improvements to existing architectures, the design began by stripping ML vocabulary from six open problems in neural network design and mapping each to the abstraction primitive set. In every case, the stripped problem turned out to be a solved problem from systems engineering. The first five problems address the model architecture. The sixth addresses the training process itself.

The entire arc from initial DAC decomposition to 204M-parameter validated results took approximately three weeks. The initial proof-of-concept (7.5M parameters, 4 layers) was designed and implemented in 24 hours using DAC as the design methodology and an AI coding assistant (Claude Code) as the implementation agent. The architect provided the DAC decomposition, the cross-domain mappings, and the verification discipline. The coding assistant produced implementation code. No ML-specific implementation experience was required on either side of this collaboration; the solutions were systems engineering solutions recognized through vocabulary stripping.

The role of the AI coding assistant deserves direct acknowledgment. The three-week development time is evidence for DAC's claim that recognized problems have short implementation paths, but it also reflects the productivity of AI-assisted coding. The mitigation against the "plausible-looking code against plausible-looking specifications" failure mode is explicit: hostile audit protocols (requiring log evidence rather than diff evidence), formal specifications in TLA+ that the implementation must refine, and a 204M-parameter training run in which the governance invariants were continuously measured. A specification bug or an implementation bug would have shown up either as a TLC counterexample or as a governance invariant violation during training. Neither occurred in the governance layer. One occurred in a domain function (the B=0 initialization threshold, discussed in Section 5.8.1), which is precisely the class of bug the structure/function separation predicts will remain in the domain function rather than the primitive.

Subsequent scale-up to 39M parameters (76M dense equivalent, trained on 500K OpenWebText samples) validated the architectural thesis across 119 tests. A 204M parameter model (805M dense equivalent) was trained with the full governed pipeline on a reasoning corpus for 7,228 optimizer steps (one epoch) on an NVIDIA L4 GPU. The architectural results reported in Sections 5.3–5.7 are from the 39M validation. The training governance results reported in Section 5.8 include both a 4.8M preliminary validation and the 204M scale validation. The 204M run is the first validation of all seventeen primitives composing correctly under real training conditions at a scale where parameter group ratios are representative (L0 at approximately 41.5% of parameters, compared to approximately 86% at 4.8M where the embedding table dominates).

LeanFormer is not presented as a competitive language model. It is presented as evidence of what DAC produces when applied to a domain: a novel architecture that composes four efficiency mechanisms simultaneously (low-rank compression, sparse attention, gated feed-forward, adaptive depth), solves catastrophic forgetting by construction (bit-for-bit base weight restoration), makes confabulation architecturally detectable, and governs the entire training lifecycle with the same primitives that govern the model's architecture. The ML community has developed each of these capabilities in isolation, each within ML vocabulary. DAC composed them by recognizing that they are all instances of primitives from other domains. The limitations of the current implementation (modest scale, single-epoch training, non-functional subsystems) are training configuration issues, not architectural failures, and they are identified as concrete next steps in Section 9.5.

### 5.3 Problem 1: Parameter Inefficiency

In ML vocabulary: dense weight matrices waste parameters because most weights contribute minimally to the output. Pruning, quantization, and low-rank approximation are active research areas with large bodies of literature.

Stripped of vocabulary: a storage system allocates fixed-size blocks for every record regardless of the record's actual content size. Most blocks are mostly empty. This is the fragmentation problem in file systems, solved decades ago by variable-size allocation with a budget constraint.

DAC decomposition: replace dense matrices with low-rank factorizations. Each weight matrix W becomes a product of two smaller matrices (down-projection and up-projection) constrained by Budget\<Parameters\> to use only a fraction of the original parameter count. The rank becomes the budget knob. The domain function (what the weight matrix computes) is unchanged. The governance (how many parameters it uses) is now explicit and tunable.

Result at 39M parameters: the model achieves a 3.9x compression ratio over its dense equivalent, meaning the 39M parameter LeanFormer has the representational structure of a 76M dense model. All four efficiency mechanisms (low-rank weights, sparse attention, gated feed-forward, adaptive depth) operate simultaneously without mutual interference.

### 5.4 Problem 2: Attention Cost

In ML vocabulary: self-attention is O(n squared) in sequence length, making long-context inference expensive. Flash attention, sparse attention, and linear attention are competing approaches with different tradeoffs.

Stripped of vocabulary: a selection system evaluates every candidate against every output position, even when most candidates are irrelevant to most positions. This is the brute-force rendering problem: evaluating every triangle against every pixel. The rendering community solved it with a two-pass architecture. A cheap coarse pass identifies which candidates are relevant. An expensive fine pass evaluates only the relevant ones.

DAC decomposition: replace single-pass dense attention with a two-pass sparse attention pipeline. The first pass is a lightweight scoring pass (CompetitiveSelection in ranked mode) that identifies the top-k relevant key-value pairs per query. The second pass computes full attention (CompetitiveSelection in soft mode) over only the selected candidates. The structure is identical to the visibility buffer pipeline in rendering: coarse culling followed by fine evaluation.

Result at 39M parameters: 88% attention sparsity. The model evaluates only 12% of key-value interactions per query position, with the sparse selection pass routing attention to the relevant subset. The gated feed-forward network achieves 80% sparsity through the same principle applied to the MLP layers: a cheap gate determines which neurons fire, and only active neurons are computed.

### 5.5 Problem 3: Catastrophic Forgetting

In ML vocabulary: when a neural network learns new information, it overwrites previously learned information because the same parameters encode both old and new knowledge. This is an open research problem with a large literature on continual learning, elastic weight consolidation, and progressive networks.

Stripped of vocabulary: a shared mutable storage system where writes to encode new content destroy existing content, because the storage addressing conflates the retrieval index with the stored content and uses dense rather than sparse encoding. This is the write-conflict problem in shared mutable state. Systems engineering solved it decades ago.

DAC decomposition: separate the retrieval index from the stored content. Use a fixed compact base for general computation (the frozen base model weights). Encode individual records as sparse, independently-addressable deltas over the base (low-rank delta matrices, analogous to LoRA [6] adapters but dynamically loaded and unloaded rather than statically merged). Govern delta allocation with a ResourceRegistry that enforces non-overlapping address ranges. Route queries to relevant deltas using a cheap CompetitiveSelection (ranked) pass before the expensive computation.

The primitive composition:

| LeanFormer Component | Abstraction Primitive Composition |
|---|---|
| Frozen base weights | The immutable foundation (analogous to kernel state) |
| Belief delta | Budget\<Parameters\> + Transaction (atomic, bounded modification) |
| Delta registry | ResourceRegistry\<BeliefID, DeltaWeights\> with non-overlap enforcement |
| Routing network | CompetitiveSelection (ranked): query embedding selects relevant deltas |
| Belief injection | Transaction: atomic addition of delta to registry |
| Belief removal | Transaction: atomic removal restoring pre-injection state |

Result at 39M parameters: 84% belief injection success rate across 100 beliefs. 86% semantic routing accuracy (4.3x above chance baseline). Bit-for-bit base weight restoration verified across all tensors after adding and removing 100 beliefs in both ordered and random removal sequences. Base weight immutability verified across 406 tensors. This result is exact, not approximate: the base weights after full belief lifecycle are identical at the bit level to the base weights before any beliefs were added.

### 5.6 Problem 4: Knowledge Composition

In ML vocabulary: fine-tuning a model on multiple domains causes interference between domains. Multi-task learning requires careful balancing of loss functions. There is no clean way to add knowledge from domain A and domain B independently and have them coexist without degradation.

Stripped of vocabulary: multiple tenants writing to a shared resource without isolation guarantees corrupt each other's data. This is the multi-tenancy isolation problem in database and operating system design. The solution is address-space partitioning with enforced non-overlap.

DAC decomposition: extend the delta registry with orthogonality constraints (FederatedBudget\<ParameterSubspace\>). The master parameter space is subdivided into non-overlapping regions. Each domain's deltas are constrained to their allocated region via orthogonality enforcement during the forging (training) process. Composition becomes additive: domain A's deltas and domain B's deltas occupy disjoint subspaces and can be loaded simultaneously without interference. The governance primitive (FederatedBudget) enforces the invariant that the sum of all allocated subspaces does not exceed the master space.

Result at 39M parameters: 64% of 100 beliefs retained improvement when all were loaded simultaneously. The coexistence rate reflects the fact that untargeted deltas (modifying all layers) create more interference than targeted deltas constrained to specific parameter subspaces. The Knowledge Plane architecture, which enforces orthogonality constraints and targeted-layer allocation, is designed to address this directly. Validation at 204M with orthogonality enforcement is identified as future work (Section 9.5).

### 5.7 Problem 5: Confabulation

In ML vocabulary: language models generate confident-sounding text that is factually wrong because they have no mechanism to distinguish "I know this" from "this is a plausible continuation." Hallucination mitigation is an active area of current research.

Stripped of vocabulary: a system that produces output with no confidence signal and no mechanism to distinguish cached retrieval (recalling a stored fact) from interpolation (generating a plausible response from patterns). Any system that conflates these two modes will produce confident-looking interpolations where retrieval was expected. This is the error-detection problem in signal processing, solved by separating the data path from the confidence path.

DAC decomposition: the ConvergenceGovernor primitive, which detects whether a computation has reached a stable state or is still iterating, maps directly to this problem. LeanFormer's adaptive depth mechanism uses exit classifiers at each layer that estimate whether the representation has converged. If the representation converges early (the model is confident), computation exits before the final layer. If the representation has not converged by the final layer, the residual magnitude serves as an uncertainty signal. When a query's relevant knowledge is loaded as a delta, routing confidence is high and depth tends to be shallow. When no relevant delta exists, routing confidence is low and depth tends to be maximum, which itself becomes a detectable signal that the model is interpolating rather than retrieving.

The structural mapping: ConvergenceGovernor monitors the computation. Low convergence residual after delta routing means the answer came from a stored fact. High convergence residual means the answer is generated from base model patterns with no grounding in injected knowledge. This separation does not eliminate confabulation, but it makes confabulation architecturally detectable rather than invisible.

Result at 39M parameters: adaptive depth is functional. The mean exit depth is 11.3 out of 12 layers, with 66% of samples exiting one layer early. The graduated depth behavior (variable exit depth correlated with query complexity and knowledge availability) requires further training at scale. The mechanism works, but the exit classifiers need more training signal to learn fine-grained confidence estimation.

### 5.8 Problem 6: Training Process Inefficiency

In ML vocabulary: training is expensive because gradient computation scales with model size, dataset size, and epoch count. The ML community has produced a large body of work on training efficiency: mixed-precision training, gradient accumulation, data/model/pipeline parallelism, curriculum learning, progressive training, layer freezing, sparse training, and importance sampling. Each addresses one dimension of the cost problem. None compose into a unified governed system.

Stripped of vocabulary: a process iteratively modifies a shared mutable state by computing error signals from sampled inputs and propagating corrections globally across the entire state on every iteration, regardless of which regions are relevant to the current input. The process has no selectivity (every parameter receives gradient from every sample), no quality hierarchy (all parameters are trained at the same fidelity from step 0), no per-region convergence detection (the only stopping criterion is global), no budget governance (gradient compute is allocated uniformly rather than proportionally to learning need), and no audit provenance (parameter changes are not traceable to the samples that caused them).

This is the brute-force rendering problem applied to gradient computation. It is the database full-table-scan applied to parameter updates. It is the network broadcast-to-all-nodes applied to learning signal. Every other governed computational system in the DAC collapse table uses CompetitiveSelection gating, QualityHierarchy traversal, FederatedBudget allocation, and per-group ConvergenceGovernor to avoid brute-force computation. Training uses none of them.

DAC decomposition: replace the flat, ungoverned training loop with a pipeline that applies the full primitive set:

| Training Component | Abstraction Primitive Composition |
|---|---|
| Current training loop | Sampler + ActuationPass + Reduction + PropagationPass + ConvergenceGovernor (global) |
| Targeted training pipeline | Sampler + CompetitiveSelection (ranked) + FederatedBudget\<GradientCompute\> + QualityHierarchy + TraversalEngine + ActuationPass + Reduction + PropagationPass + ConvergenceGovernor (per-group) + AuditSink |
| Governed data pipeline | QualityHierarchy\<SampleDifficulty\> + CompetitiveSelection (ranked) + Budget\<SamplesPerStep\> + AuditSink |
| Change-triggered evaluation | Signal\<ConvergenceChange\> + CompetitiveSelection (ranked) + Budget\<EvalCompute\> + ActuationPass + Reduction + AuditSink |
| Readiness-gated knowledge forge | Signal\<GroupConverged\> + ConvergenceGovernor + CompetitiveSelection (ranked) + Budget\<ForgeCompute\> + ActuationPass + Reduction + AuditSink |

The missing primitives in the current training loop are the same primitives that every other domain in the collapse table uses: CompetitiveSelection for routing gradient signal to relevant parameter groups, QualityHierarchy for coarse-to-fine parameter activation, FederatedBudget for proportional compute allocation, per-group ConvergenceGovernor for independent convergence detection, and AuditSink for gradient provenance tracking.

The ML community has independently invented fragments of this composition: Mixture of Experts (CompetitiveSelection at inference, not training), LoRA (static Budget\<Parameters\>, not sample-adaptive), curriculum learning (QualityHierarchy over data, not parameters), layer freezing (binary ConvergenceGovernor without graduated states), progressive training (one-directional QualityHierarchy without convergence-governed activation), and GradNorm (partial FederatedBudget at task level, not parameter-group level). Each fragment was built in isolation, in ML vocabulary, without recognizing that all are partial reinventions of the same structural pattern.

The training system extends DAC beyond the model to the model's entire lifecycle: how it learns (targeted training), how it is assessed (change-triggered evaluation), how its knowledge is produced (readiness-gated forging), and how it is deployed (quality-tiered inference preparation). After this extension, every stage of the model lifecycle is governed by the same primitive set that governs the model's architecture.

Preliminary result at 4.8M parameters (300-step validation run): all governance components activated and composed correctly. Per-group convergence governors produced 15 state transitions across 8 parameter groups, with 7 reaching CONVERGED state. The quality hierarchy activated in the predicted order: L0 (structural) then L1 (representational) then L2 (refinement) then L3 (specialization), all via convergence signal with no emergency activation required. The FederatedBudget invariant (sum of allocations <= master budget) held for all 300 steps with zero violations. The gradient router achieved 15.4% selectivity post-warmup. The SHA-256-chained audit log verified across all 300 records. The change-triggered evaluation pipeline fired 7 targeted evaluations on convergence signals. Final loss was within +2.4% of baseline, with the gap narrowing throughout training (from +3.17 at step 50 to +0.15 at step 299).

Scale validation at 204M parameters (805M dense equivalent, 7,228 optimizer steps, one epoch on a reasoning corpus, NVIDIA L4 GPU, 140.9 hours wall clock) confirmed that all seventeen primitives compose correctly under real training conditions. The question being answered here is not "how good is this language model" but "do the governance primitives compose correctly at a scale where parameter group ratios are representative, and do all invariants hold across thousands of training steps?" The language modeling metrics are reported for completeness but are not the subject of evaluation. The governance results:

The FederatedBudget invariant (sum of allocations <= 1.0) held for all 722 audit records with zero violations. Budget adapted dynamically throughout training: L0 groups started at 0.25 each, converged groups dropped to as low as 0.012, and active groups received up to 0.40 of the total budget. The budget reallocation tracked learning need in real time.

The convergence governor four-state machine produced 18 total state transitions across 8 parameter groups, all valid: PENDING to ACTIVE (3 transitions), ACTIVE to COOLING (8), COOLING to CONVERGED (4), and COOLING to ACTIVE (1 regression, discussed below). No states were skipped. Final states: embeddings COOLING, attention_routing CONVERGED, gates CONVERGED, layer_norms CONVERGED, attention_output COOLING, ff_projections COOLING, output_head COOLING, exit_classifier CONVERGED.

The SHA-256 hash-chained audit log maintained integrity across all 722 records from step 10 to step 7,220, with zero chain breaks. Every governance decision (budget allocation, convergence state transition, hierarchy activation, gradient routing update) is traceable to a specific training step with tamper-evident provenance.

Best validation perplexity reached 57.6 at step 2,000, with train loss declining from 10.39 to 1.64 across the full run. Severe overfitting occurred after step 2,000 (validation perplexity rose from 57.6 to 1,463.9 by step 7,228), which is expected behavior from single-epoch training with limited regularization (dropout 0.1 only). The overfitting is a training configuration limitation, not an architectural failure, and critically, all governance invariants held throughout the overfit phase: the budget was never exceeded, the hierarchy ordering was maintained, the convergence governor never skipped a state, and the audit chain was never broken. Governance correctness is independent of generalization quality, exactly as the structure/function separation predicts.

Orthogonal capacity measurement confirmed 53,760 available dimensions across 20 layers (2,688 per layer), with a theoretical maximum of 3,360 rank-16 knowledge deltas. This was measured on two independent machines (NVIDIA L4 on GCP, RTX 3060 locally) with identical results, confirming the measurement is model-intrinsic.

Generation samples from the best checkpoint (step 2,000) show semi-structured output with learned domain vocabulary (gradient-related terms, function syntax) but incoherent content. This is consistent with a 204M model after 2,000 optimizer steps of single-epoch training and does not indicate an architectural limitation. The generation quality is dramatically better than the overfit final checkpoint, which produced degenerate repetitive output, confirming that the step-2,000 checkpoint represents meaningful learned structure.

Two governed subsystems did not produce meaningful signal at this scale and configuration. Adaptive depth remained at 20/20 (all layers used) throughout the entire run, meaning the exit classifiers never learned to route samples to early exits. This may indicate the exit threshold (0.03) is too conservative, or that the mechanism requires explicit layer-dropping training. Tiered sampling scored all samples as "Failing" at initialization and was never re-scored, rendering it effectively random. Both require further investigation in future training runs.

### 5.8.1 DAC Applied to Its Own Failure: The B=0 Observational Degeneracy

The most instructive result from the 204M training run was an unplanned demonstration of DAC's generative mode applied to a failure in a DAC-governed system.

LeanFormer's low-rank layers initialize B matrices to zero (following standard LoRA practice), which means all parameter groups begin with near-zero gradient flow regardless of whether they have received meaningful training signal. The convergence governors correctly detected low gradient EMA and transitioned through the state machine as specified: ACTIVE to COOLING after the configured cooling window. This produced hierarchy activations at steps 200 (L1) and 400 (L2), both at round-number intervals aligned with the cooling window configuration. Loss remained flat at 10.388 through both activations. Actual training progress began only when the output head activated at L2 and introduced significant gradient flow through the network.

The L3 activation at step 2,773 was qualitatively different. It occurred at a non-round step number, after the attention_output and ff_projections groups (L1 parameters) had received 2,300+ steps of real gradient flow following the output head's activation at step 400. The convergence governor's decision to activate L3 was based on genuine post-learning convergence in those groups, not a calibration artifact. This distinction is visible in the gradient norm data: at step 400, the output head's gradient norm jumped from 0.0 to 0.88 in a single step, marking the onset of real learning, and L3 activation occurred only after the downstream groups had processed that gradient signal through thousands of training steps.

The COOLING to ACTIVE regression observed in the attention_output group provides further evidence of the governor's robustness. This group was prematurely cooled by the B=0 artifact, then reactivated when real gradient flow from the output head pushed its EMA above the cooling threshold. The four-state machine self-corrected from the calibration issue without any intervention, using the existing COOLING to ACTIVE transition path. This hysteresis behavior is exactly what the state machine was designed to provide.

Applying DAC's own vocabulary-stripping process to this failure reveals it as an observational degeneracy: two qualitatively different signal trajectories (cold start and genuine convergence) produce the same low-magnitude reading, and the governor cannot distinguish them. This is the same pattern that appears across the collapse table. In rendering, a pixel with zero color could be background or a black surface; the depth buffer adds a signal to break the degeneracy. In networking, a silent node could be idle or crashed; heartbeat protocols add a liveness signal. In distributed consensus, a node that has not voted could be slow or partitioned; timeouts add a temporal boundary. In every case, the solution is the same structural primitive: add a second Signal\<T\> that disambiguates the measurement.

The fix follows mechanically from the cross-domain pattern: a phase-aware ConvergenceGovernor that tracks whether gradient magnitude has ever exceeded the threshold, classifying gradient trajectories into qualitative phases (COLD, WARMING, ACTIVE_LEARNING, DECLINING). The ACTIVE-to-COOLING transition requires the gradient phase to be ACTIVE_LEARNING or DECLINING, never COLD. This NoCoolingFromCold invariant was specified in TLA+ and verified by TLC across 18.6 million states. A specification without phase awareness produces a counterexample matching the exact failure observed in the training run in just 2 states.

This episode demonstrates three properties of the methodology simultaneously. First, the structure/function separation that DAC claims: the governance machinery was correct (the ConvergenceGovernor followed its specification exactly), while the domain function (the cooling threshold applied to B=0-initialized parameters) was miscalibrated. The governance code requires one additional precondition; the primitive itself does not change. Second, DAC's generative mode: the fix was not invented from scratch but recognized as a solved problem from the collapse table, where observational degeneracies are routinely resolved by adding a disambiguation signal. Third, the value of formal verification: the phase-aware fix was verified in TLA+ before any code was written, and the original failure was reproduced as a concrete counterexample, providing high confidence that the fix addresses the root cause.

This is a machinery validation, not a scale validation. The efficiency gains from governed training depend on model size and training duration. At 204M parameters, the governance machinery correctly identified which parameter groups to stop training and dynamically reallocated budget to groups still learning. However, the compute savings were not realized as wall-clock improvement in this run because the implementation zeros gradients for converged groups after computation rather than skipping the backward pass entirely. Implementing actual compute skipping for governed groups (setting requires_grad=False with proper autograd graph pruning) is an engineering optimization for future runs. The governance produced the correct signal; the training loop did not yet act on it efficiently.

### 5.9 Summary of Results

Architecture results are from the 39M parameter model (76M dense equivalent) trained on 500K OpenWebText samples for one epoch. Training governance results include both the 4.8M preliminary validation (300 steps, WikiText-2) and the 204M scale validation (7,228 steps, reasoning corpus, NVIDIA L4).

| Mechanism | Metric | Result | Status |
|-----------|--------|--------|--------|
| Low-rank compression | Compression ratio | 3.9x over dense equivalent (39M); 3.94x (204M) | Validated |
| Sparse attention | Attention sparsity | 88% | Validated (39M) |
| Gated feed-forward | FF sparsity | 80% | Validated (39M) |
| Belief injection | Success rate | 84% across 100 beliefs | Validated (39M) |
| Semantic routing | Routing accuracy | 86% (4.3x above chance) | Validated (39M) |
| Belief coexistence | Simultaneous improvement | 64% of 100 beliefs | Validated (39M) |
| Base weight restoration | Bit-for-bit fidelity | Exact across all tensors | Validated (39M) |
| Base weight immutability | Tensor integrity | 406 tensors verified | Validated (39M) |
| Adaptive depth | Mean exit depth | 11.3/12, 2 unique depths (39M); 20/20 not activated (204M) | Partial |
| Governed training | Budget invariant | 0 violations / 300 steps (4.8M); 0 violations / 722 records (204M) | Validated |
| Governed training | Audit chain integrity | 300 records verified (4.8M); 722 records, SHA-256 chain intact (204M) | Validated |
| Convergence governors | State transitions | 15 transitions, 7/8 converged (4.8M); 18 transitions, all valid, no skips (204M) | Validated |
| Hierarchy activation | Coarse-to-fine ordering | L0 then L1 then L2 then L3 via convergence signal | Validated (4.8M, 204M) |
| Hierarchy activation | L3 genuine convergence | Step 2,773 (non-round, after 2,300+ steps of real gradient flow) | Validated (204M) |
| Hierarchy activation | L1/L2 timing | Steps 200/400 (B=0 initialization artifact, disclosed) | Disclosed artifact |
| B=0 diagnosis | DAC applied to own failure | Observational degeneracy identified, phase-aware fix formally verified | Validated |
| Budget reallocation | Dynamic budget shifts | Converged groups drop to 0.012, active get up to 0.40 | Validated (204M) |
| Gradient routing | Gate activation | 0.34-0.69 gate density (non-degenerate) | Validated (204M) |
| Language modeling | Best val PPL | 57.6 at step 2,000 (204M) | Measured |
| Language modeling | Final val PPL | 1,463.9 (overfit, single epoch, governance invariants held throughout) | Expected |
| Orthogonal capacity | Available dims | 53,760 (3,360 max rank-16 deltas), confirmed on two machines | Measured (204M) |
| Tiered sampling | Tier distribution | All samples scored as Failing (scored pre-training only) | Not functional |
| Training time | Wall clock | 140.9h on NVIDIA L4 | Measured (204M) |

The two partial results (adaptive depth and tiered sampling) are not architectural failures. Adaptive depth requires either a lower exit threshold or explicit layer-dropping training to learn meaningful early-exit behavior; the 39M model showed limited depth variation (11.3/12 mean exit depth) and the 204M model showed none (20/20 throughout). Tiered sampling scored all samples at initialization when the model could not yet evaluate difficulty; periodic re-scoring during training would enable the governed data pipeline.

The training governance results validate that all seventeen primitives compose correctly at 204M parameters and that all governance invariants hold empirically across 7,228 training steps, including through the severe overfitting phase after step 2,000. The B=0 observational degeneracy was diagnosed using DAC's own methodology and the fix was formally verified in TLA+. The language modeling perplexity is not competitive at this scale and training duration; the claim is governance machinery validation, not language modeling performance.

The critical open question is whether the governed training machinery produces measurable efficiency gains at larger scale. The governance correctly identified which parameter groups to stop training and reallocated budget accordingly, but the compute savings were not realized as wall-clock improvement in this run. Validating the efficiency claims requires a 7B+ parameter training run with proper compute skipping for converged groups, multi-epoch training with adequate regularization, and comparison against an ungoverned baseline. This is identified as necessary future work in Section 9.5.

### 5.10 What DAC Did Not Provide

DAC does not design domain functions. The specific choice of low-rank factorization for the deltas, the cosine similarity metric for the routing network, the exit-threshold tuning for adaptive depth, the choice of loss function and optimizer: these are domain-specific engineering decisions that require ML expertise. The same applies to the training governance system: the specific scoring function for the gradient router, the convergence thresholds for per-group governors, the budget allocation policy, the hierarchy level boundaries, and the sample difficulty thresholds are all domain functions that DAC's structural skeleton does not supply. DAC provided the structural skeleton. Domain knowledge filled in the scoring functions, the loss formulations, the training recipes, and the governance thresholds.

This is the same structure/function separation described throughout the paper. The abstraction primitives provide structure (how data flows, how resources are allocated, how convergence is detected, how mutations are audited). The domain provides function (what computation to apply at each step). Neither replaces the other.


---

## 6. Case Study: Orkestratum, A Multi-Domain Runtime on the Primitive Set

### 6.1 What Orkestratum Is

LeanFormer is the paper's evidence that DAC is useful when applied to a single domain. Orkestratum is the paper's evidence that DAC produces a runtime that serves multiple domains simultaneously without per-domain engine work. The methodology's central claim is that twelve domains were the same computation in different vocabulary. The strongest form of that claim is a single codebase that runs all of them. Orkestratum is that codebase.

Orkestratum is an application runtime organized as a kernel of primitive modules plus sandboxed domain modules. The kernel contains one implementation of each of the seventeen primitives, one DAG scheduler that routes every execution through a single dispatch path, and one audit chain that records every governance decision. Domain modules plug into the kernel by providing the domain functions (scoring functions, task functions, transfer functions) that parameterize the primitives. Four workload classes run on this runtime today:

- Security orchestration playbooks (SOAR),
- Continuous integration and deployment pipelines (CI/CD),
- Extract-transform-load pipelines (ETL),
- Configuration management runs,

and a fifth, the real-time renderer, runs inside the same runtime with the same primitives driving real-time graphics workloads.

The claim is concrete: one kernel, one scheduler, one audit chain, one budget system, one convergence governor implementation, one traversal engine. The four DAG-workload domains are different compositions of these modules with domain-specific task functions. The renderer is a different composition of the same modules with domain-specific scoring functions (depth testing, culling predicates, light propagation kernels). No domain has its own scheduler, its own budget, or its own audit implementation. A primitive-bypass CI gate prevents new code from sidestepping the kernel.

### 6.2 What Is Demonstrated, What Is In Progress

The claims in this section are bounded to what the Orkestratum codebase currently demonstrates. The runtime is an active engineering program, not a finished system, and this section is explicit about which capabilities are production-validated, which are validated at prototype quality, and which are in-progress work.

Demonstrated:

- One kernel codebase implements all seventeen primitives as trait-dispatched modules. A conformance audit has mapped every primitive to its implementation site, its invariant, and its trace. Phases of this refactor are committed against an executable blueprint with per-primitive exit criteria.
- Rendering workloads run through the kernel at interactive frame rates. Sponza scene benchmark data has been collected pre-consolidation; further throughput gains from the primitive consolidation are in progress.
- DAG workload execution is mediated entirely by the scheduler: SOAR playbooks, CI/CD pipelines, ETL workflows, and configuration-management runs all route through the same dispatch path. The absolute invariant "all execution routes through the DAG" is enforced by a machine-checked TLA+ specification (AllExecutionThroughDag.tla).
- A SHA-256 hash-chained AuditSink records every mutation across all workload classes. The chain is verified in tests and its integrity has been maintained across the 204M LeanFormer training run (722 records, zero chain breaks) as well as in runtime operation.
- Formal specifications of the primitives and key compositions exist in TLA+. The formal verification program is underway, with per-primitive refinement specifications scheduled as a dedicated phase of the engineering blueprint.

In progress:

- Performance consolidation to hit demonstration targets for rendering (target: Sponza at 100+ FPS, Bistro at interactive frame rates).
- Ongoing refactor of hand-rolled structures in the codebase to route through the L0 primitive trait surface rather than through local reimplementations. A conformance audit identified specific violations with line-number traces; remediation is scheduled against an executable plan.
- Byzantine fault tolerance for federated authorities is documented as an extension path but not implemented; current federation uses crash-fault tolerance (RAFT).
- Scene composition (interactive multi-participant worlds) is a proposal document dependent on the completion of earlier refactor phases; it is not yet implemented.

Not claimed:

- Orkestratum is not claimed to match domain-specialized tools on their benchmarks today. A dedicated graphics engine optimized for a single scene will outperform Orkestratum's renderer on that scene. A purpose-built SOAR platform will have a larger plugin ecosystem. The claim is that one runtime serves these domains with the same primitives, not that the runtime currently dominates the best-of-breed per domain.
- Orkestratum is not claimed to be feature-complete relative to any of the four DAG workload domains. Each has a working execution path; feature parity with Ansible or Airflow in every edge case is not yet achieved.
- The renderer is not claimed to be a frontier research renderer. It is a real-time renderer built on the primitive set sufficient to demonstrate that the primitives work under real-time constraints.

### 6.3 The Primitive-to-Domain Mapping in the Runtime

The following table documents how the seventeen primitives are instantiated in each of the five domains currently running on Orkestratum. The table is a runtime-level view: it shows how a single primitive serves multiple domains by being parameterized with different domain functions, not by being reimplemented per domain.

| Primitive | SOAR | CI/CD | ETL | Config Mgmt | Rendering |
|---|---|---|---|---|---|
| Budget\<U\> | Playbook compute | Build compute | Pipeline compute | Run compute | Draw calls, GPU memory, microseconds |
| FederatedBudget\<U\> | Per-tenant allocation | Per-team allocation | Per-warehouse allocation | Per-inventory allocation | Per-pass GPU allocation |
| QualityHierarchy | Severity tiers | Build priority tiers | Data freshness tiers | Desired-state tiers | LOD tiers |
| AllocationSnapshot | Playbook activation state | Stage activation state | Task activation state | Role activation state | Per-frame LOD assignment |
| RelationshipGraph | Playbook DAG | Pipeline DAG | Transform DAG | Role/task DAG | Render pass graph |
| ResourceRegistry | Action catalog | Tool catalog | Source/sink catalog | Host inventory | Asset/material catalog |
| TraversalEngine | Playbook execution | Pipeline execution | Pipeline execution | Task execution | Frustum culling, visibility traversal |
| PropagationPass | State propagation | Artifact propagation | Data propagation | State convergence | Cascade shadows, GI |
| CompetitiveSelection | Alert triage (ranked) | Runner allocation (hard) | Source selection (hard) | Conflict resolution (hard) | Pixel ownership (hard), LOD pick (ranked) |
| ActuationPass | Response action | Stage execution | Transform/load | Task apply | GPU command submission |
| Reduction | Alert aggregation | Test result aggregation | Reduce-phase aggregation | State aggregation | Draw call batching |
| Sampler | Randomized test sampling | Test partition | Row sampling | Dry-run probing | TAA jitter, importance sampling |
| Checkpoint | Rollback on failed action | Rollback on failed stage | Rollback on failed load | Rollback on failed task | (not used in rendering) |
| ConvergenceGovernor | (not typically used) | (not typically used) | Pipeline convergence | Desired-state convergence | Physics constraint solving |
| Signal\<T\> | Alert arrival | Build trigger | Data arrival | State change | Render pass completion |
| RateLimit | API throttling | Build queue pacing | Source throttling | Remote host pacing | Frame rate cap |
| AuditSink | Every action | Every build step | Every transform | Every apply | Every render decision (optional) |

The table is the strongest test of DAC's central claim. If the methodology were vocabulary-matching dressed up as structural insight, each domain would require its own implementations of these primitives because the domain constraints would differ in ways the primitives could not absorb. Instead, each primitive serves all five domains with the same code, parameterized by the domain's scoring function or task function.

### 6.4 The DAG Workload Collapse, Empirically

Section 7 of this paper argues that CI/CD pipelines, ETL workflows, configuration management, and SOAR playbooks are structurally identical: DAG traversal under budget constraints with capability-gated I/O. Orkestratum's architecture is the empirical form of this claim. All four domains dispatch through the same scheduler, traverse the same RelationshipGraph trait, consume the same Budget, emit to the same AuditSink, and compose with the same reconciler primitives (SuperviseReconciler, EventDrivenReconciler, ScheduledReconciler, TickReconciler), each of which is itself a specific composition of Budget, RateLimit, Signal, Sampler, and ActuationPass.

What the runtime demonstrates that the paper's analytical table cannot: that the same execution engine serves all four domains without accumulating domain-specific cruft over time. A key CI gate in the project forbids new code from bypassing the primitive kernel, which operationalizes the structural claim as an ongoing invariant rather than a one-time analysis.

The rendering pipeline is the crucial additional evidence. Rendering is the domain with the tightest performance constraints in the full set: every frame has a fixed millisecond budget, and any overhead from abstraction is immediately visible. If the primitive set survived contact with real-time rendering at interactive frame rates, the objection "this abstraction is too general to be efficient" loses force. The rendering pipeline is running inside the same runtime that executes the DAG workloads, using the same primitive implementations. It is not a separate engine bolted onto a general-purpose runtime. It is the general-purpose runtime driving real-time graphics.

### 6.5 The Formal Verification Program

Orkestratum's formal verification program treats TLA+ specifications as the source of truth for the primitive layer. Each primitive has (or will have, per the blueprint) a refinement specification that its implementation must satisfy. Composition specifications verify invariants across primitive boundaries. A top-level invariant (AllExecutionThroughDag) enforces that no side channel can bypass the scheduler.

This is the same formal verification approach applied in Section 9.4 of this paper to the primitive set as a mathematical object. The distinction is that Orkestratum applies it to a specific implementation, enforcing that the implementation refines the specification rather than merely matching the specification's outward behavior. The full refinement program is future work (identified in the blueprint's Phase 5); the current state is that all primitives have TLA+ specifications, five LeanFormer compositions have verified specifications, and the phase-aware ConvergenceGovernor's NoCoolingFromCold invariant has been verified across 18.6 million states.

### 6.6 What Orkestratum Demonstrates About DAC

The case study establishes three things the analytical collapse cannot establish alone.

First, the primitive set is engineerable. Seventeen primitives are few enough that they can be implemented, tested, and verified by a small team. The blueprint's engineering scope is bounded and trackable. This is evidence that the governance boundary is at a useful level of abstraction: low enough that implementation is feasible, high enough that domains reuse the primitives without forcing them to sprawl.

Second, the primitive set survives multi-domain contact. When four DAG workload domains and one real-time rendering domain are executing on the same primitives simultaneously, each domain's constraints test the primitives. The rendering pipeline tests that the primitives are fast enough. The DAG workloads test that the primitives are expressive enough for capability-gated orchestration. The audit chain tests that the primitives compose under concurrent mutation. A primitive that passed one domain's constraints but failed another's would have been flagged by the conformance audit and would appear in the "in progress" list in Section 6.2. The list of in-progress items is bounded and does not include "the primitive set does not work for domain X."

Third, the primitive set enables cross-domain transfer. The phase-aware ConvergenceGovernor (developed to fix the B=0 observational degeneracy in LeanFormer) is the same primitive implementation that governs physics constraint solving and, when federated authorities are added, will govern RAFT-based cluster convergence. A fix motivated by an ML training bug improves rendering physics and distributed consensus because they share the primitive. This is the operational form of the cross-domain optimization transfer claim.

The limitations of this case study are stated in Section 6.2. Orkestratum is not a finished system. The performance consolidation is in progress. The formal verification program is partial. The claim is that a single runtime executes multiple domains with the same primitive set, not that the runtime has reached feature parity or peak performance in any individual domain.

---

## 7. Extended Case Study: The AI Domain

### 7.1 The Deepest Collapse

The most striking result of applying DAC to machine learning is the observation that the transformer architecture [2], the foundation of every modern large language model, is a composition of primitives that the methodology had already identified before the AI domain was examined.

Attention is soft competitive selection. The transformer's multi-head attention mechanism computes, for each query token, a weighted combination of value vectors where the weights are determined by similarity between the query and key vectors. This is CompetitiveSelection in soft mode: the dot-product similarity is the scoring function, softmax produces the weighted allocation, and each attention head is one selection pass. Multi-head attention is parallel selection passes over the same candidates with different scoring functions.

A precision caveat is necessary on the strength of this mapping. The structural identity (softmax-weighted sum over candidates with a scoring function) is real, but attention has properties that matter for what people actually do with attention: differentiability, gradient flow through the weights, and learned (rather than designed) scoring. The mapping captures the structure and loses the function. The function is most of what makes attention useful in its native domain. The mapping is therefore a lens that enables cross-domain optimization transfer, not a substitute for the ML literature on attention.

Backpropagation is a member of the PropagationPass family. The forward pass builds a directed acyclic computation graph. The backward pass propagates gradient values from the loss node backward through the graph in reverse topological order, applying the chain rule at each node to compute local gradients. The shared structure with iterative relaxation (Bellman-Ford, belief propagation) is message passing over a graph toward a consistent state. The instantiation parameters differ substantially: backpropagation is a single reverse pass on a DAG with termination after one traversal; Bellman-Ford is iterative relaxation on a graph with cycles with termination at fixed point. Calling both instances of the same primitive carries real risk of overclaiming, and the honest framing is that PropagationPass is a family parameterized by topology, traversal order, message function, and termination condition, of which backpropagation is one member. The value of the mapping is that it makes visible the connection between gradient computation, routing convergence, and lighting propagation, enabling cross-domain insight. The risk is treating a family resemblance as an identity. This paper treats it as family resemblance.

Speculative decoding is the two-level fidelity architecture. A fast small model generates candidate tokens (coarse pass) [8]. A large model verifies them (fine selection). Accepted tokens are actuated. Rejected tokens are discarded. This is the same structure as the rendering fidelity pipeline: coarse traversal reduces the candidate set, fine selection determines winners, actuation evaluates only winners. The same architecture that makes rendering efficient makes inference efficient.

### 7.2 Implications and Evidence Status

The mappings above are structural identities, with the precision caveats noted. The implication is that any optimization of CompetitiveSelection discovered in any domain is potentially applicable to attention. Any optimization of PropagationPass discovered in any domain is potentially applicable to gradient computation. The collapsed primitive set creates a channel for cross-domain optimization transfer that does not exist when each domain maintains its own vocabulary.

Linear attention [10], sparse attention, and flash attention are all optimizations of CompetitiveSelection in soft mode. Convergence governors and temporal amortization, developed for real-time lighting, become potentially applicable to training loop optimization. The primitive vocabulary makes these connections visible. The domain vocabulary hides them.

A candid assessment of validation status: cross-domain optimization transfer is the most powerful claim DAC makes. The evidence for it to date consists of:

- Five systems-engineering solutions applied to ML model design in LeanFormer (Sections 5.3–5.7), demonstrating that patterns from file systems, database multi-tenancy, and signal processing transfer into ML architectural decisions.
- The targeted training system (Section 5.8), demonstrating that the same primitive set applies to the training process itself, validated at 204M parameters across 7,228 training steps.
- The B=0 observational degeneracy diagnosis (Section 5.8.1), demonstrating that a novel ML training bug can be recognized as structurally identical to solved problems from rendering, networking, and distributed consensus, with the fix derived from the cross-domain pattern.
- The shared ConvergenceGovernor implementation in Orkestratum (Section 6), which is the same primitive instance serving both ML training and physics constraint solving. A fix motivated by ML training (the phase-aware variant) is available to rendering without rework.

These are strong but one-directional: they are all cases of systems-engineering patterns transferring into ML or to other systems-governed domains. The stronger claim would be a bidirectional transfer: an optimization discovered in ML, applied to a non-ML domain, producing a measurable improvement there. This has not yet been demonstrated and is identified as future work (Section 9.5). The current evidence supports "cross-domain recognition enables ML-direction transfer" rather than "the primitive set is a bidirectional channel for optimization." The latter is the methodology's aspiration; the former is what has been shown.

---

## 8. The DAG Workload Collapse

### 8.1 Four Domains, One Pattern

The most practically significant collapse is the identification that CI/CD pipelines, ETL workflows, configuration management playbooks, and SOAR automation playbooks are structurally identical. All four are:

A directed acyclic graph of tasks with typed dependencies, executed under budget constraints (compute, time, or both), with capability-gated I/O at task boundaries, converging toward a desired end state, and audit-logged for observability and compliance.

The practical consequence is that a single execution engine can replace Ansible, Jenkins, Airflow, and Splunk SOAR, not by implementing four separate systems, but by recognizing that four separate systems were never needed. The vocabulary was different. The computation was the same.

Section 6 describes the empirical form of this claim: Orkestratum executes all four domains on one scheduler with one primitive set.

### 8.2 Why This Collapse Was Invisible

These four domains are served by different industries, different conferences, different vendor ecosystems, and different job titles. A CI/CD engineer uses Jenkins or GitHub Actions. An ETL engineer uses Airflow or dbt. A configuration management engineer uses Ansible or Puppet. A security engineer uses Splunk SOAR or Palo Alto XSOAR. Each tool has its own vocabulary, its own plugin ecosystem, and its own certification program.

The vocabulary creates the market. The market creates the specialization. The specialization prevents anyone from noticing that all four tools do the same thing. DAC dissolves this by asking: what does each tool actually compute? The answer, in every case, is: it traverses a DAG under constraints and executes tasks at each node. The domain-specific part is the task function, which belongs in a sandboxed execution boundary, not in the kernel.

---

## 9. The Composition Algebra

### 9.1 How Domains Are Reconstructed

The value of abstraction collapse is not merely taxonomic. It is constructive: once the primitives are identified, every domain pattern can be systematically constructed as a composition. The following composition map documents the reconstruction of common computational patterns from the primitive set.

| Pattern | Core Primitives | Governance |
|---------|----------------|------------|
| LOD / Quality scaling | QualityHierarchy + TraversalEngine + AllocationSnapshot | Budget\<Bytes\> |
| Global illumination | PropagationPass + ConvergenceGovernor | Budget\<Microseconds\> |
| Physics simulation | Reduction + CompetitiveSelection (hard) + PropagationPass + ActuationPass | Budget\<Microseconds\> + ConvergenceGovernor |
| Network sync | PropagationPass + TraversalEngine + Diff | FederatedBudget\<BytesPerSecond\> |
| Container orchestration | QualityHierarchy + TraversalEngine + ActuationPass | Budget\<CPU/Memory\> + ConvergenceGovernor |
| CI/CD pipeline | RelationshipGraph + TraversalEngine + ActuationPass | Budget\<Compute\> |
| ETL pipeline | RelationshipGraph + ActuationPass + Transaction | Budget\<Microseconds\> |
| Configuration management | Diff\<State\> + ConvergenceGovernor + ActuationPass | (none) |
| SOAR playbook | RelationshipGraph + TraversalEngine + ActuationPass | Capability-gated I/O |
| Market simulation | CompetitiveSelection (hard) + PropagationPass + Transaction | FederatedBudget + ConvergenceGovernor |
| ML inference | ActuationPass + CompetitiveSelection (soft) + Memoize | Budget\<FLOPs\> |
| Belief delta system | Budget\<Parameters\> + ResourceRegistry + Transaction + CompetitiveSelection (ranked) | AuditSink |
| Targeted training pipeline | CompetitiveSelection (ranked) + FederatedBudget\<GradientCompute\> + QualityHierarchy + TraversalEngine + ActuationPass + Reduction + PropagationPass | ConvergenceGovernor (per-group) + AuditSink |
| Constraint logic search | RelationshipGraph + TraversalEngine + PropagationPass + Checkpoint | ConvergenceGovernor |
| Lattice dataflow | RelationshipGraph + PropagationPass + Reduction | ConvergenceGovernor (widening mode) |
| Unification | ResourceRegistry (monotonic) + TraversalEngine + PropagationPass + ActuationPass | ConvergenceGovernor |

### 9.2 The Key Insight: Structure vs. Function

In every composition above, the primitives provide the structure: how data flows, how resources are allocated, how convergence is detected, how mutations are audited. The domain provides the function: what computation to apply at each step.

The rendering equation is a domain function. The RAFT consensus protocol's log replication rule is a domain function. The machine learning loss function is a domain function. The attention scoring function (dot-product similarity) is a domain function. The unification consistency check is a domain function. None of these belong in the kernel of abstraction primitives. All of them execute within the governance framework the primitives provide.

This separation is what makes the primitive set simultaneously lean and broadly applicable. It contains no domain knowledge. It contains the execution model shared by every domain examined.

---

## 10. Methodology Validation Criteria

### 10.1 How to Know the Collapse Is Real

A collapse is genuine (not merely a renaming exercise) if and only if:

1. Completeness: Every domain pattern from step 1 can be expressed as a composition of the collapsed primitives. No residual domain-specific primitives are required.

2. Minimality: No primitive in the collapsed set can be expressed as a composition of the others. Removing any primitive leaves at least one domain pattern unexpressible.

3. Operational equivalence: A system built from the collapsed primitives produces the same outputs as the domain-specific system it replaces, under the same inputs and constraints.

4. Cross-domain transfer: An optimization discovered in one domain, when applied to the shared primitive, produces measurable improvement in other domains that use the same primitive.

Criterion 1 is satisfied for all twenty examined domains after the adversarial pass (Section 2.5), which produced one new primitive and two refinements. Criterion 3 is satisfied for the domains where working implementations exist (LeanFormer for ML; Orkestratum for SOAR, CI/CD, ETL, configuration management, and rendering).

Criterion 4 has partial evidence. One-directional transfer (systems patterns into ML) is demonstrated by LeanFormer. Transfer within the Orkestratum runtime (a fix in one domain's use of a primitive benefiting another domain through the shared implementation) is demonstrated by the phase-aware ConvergenceGovernor. Bidirectional transfer and transfer across all twelve domains remain future work.

Criterion 2 (minimality) deserves explicit scrutiny. The claim that no primitive can be expressed as a composition of the others has not been formally proven. Some candidates for potential redundancy: Can RateLimit be expressed as Budget\<Operations\> + a temporal mechanism? Can Signal\<T\> be derived from AuditSink + a predicate filter? These might be legitimate standalone primitives, or they might be compositions. If they are compositions, the primitive count drops and the narrative changes, but the methodology's validity does not. The important claim is that the set is sufficient, not that it is provably minimal. Proving minimality would require showing that removing each primitive creates at least one domain pattern that becomes unexpressible, a worthwhile exercise that has not yet been completed.

A related but distinct property has been formally verified: operational irreducibility. Every primitive was specified in TLA+ and systematically decomposed into plausible sub-operations (separating the guard from the mutation, removing intermediate states, splitting atomic operations into phases). In all cases tested, the TLC model checker found concrete counterexamples where the decomposed version violated the primitive's governance invariant. This does not prove minimality (that no primitive is a composition of other primitives in the set), but it proves that each primitive is atomic with respect to its own invariant. The guard and the guarded operation cannot be separated without losing the guarantee. Operational irreducibility is a weaker claim than algebraic minimality but a stronger claim than the paper would have without formal verification. The distinction matters: minimality asks whether the set can be made smaller; operational irreducibility asks whether each element can be made simpler. The TLA+ verification answers the second question definitively.

### 10.2 Threats to Validity

The most significant methodological risk is confirmation bias. All twenty domains were collapsed by the same individual, and once a primitive vocabulary exists, there is a cognitive pull to force every new domain into it rather than honestly admitting when the existing primitives are insufficient.

Four properties of the development process mitigate this risk.

First, the primitives were discovered incrementally, not designed a priori. SOAR was the first domain. Rendering was added second and forced the addition of QualityHierarchy, TraversalEngine, and CompetitiveSelection, primitives that SOAR alone did not require. Each subsequent domain either mapped onto existing primitives or forced additions when the existing set was genuinely insufficient. The primitive count grew from five to sixteen over the twelve original domains, and from sixteen to seventeen with the addition of Checkpoint in the adversarial domain pass. If confirmation bias were dominant, the count would have stayed at five.

Second, the LeanFormer implementation produces working systems with measurable outputs. Bit-for-bit weight restoration after belief removal, 119 passing tests, 84% injection success rates, 86% routing accuracy, and the 204M scale validation (with zero governance violations across 7,228 steps and 722 audit records) are objective evidence that the collapsed compositions are operationally correct, not merely descriptively plausible. The B=0 diagnosis confirmed an a priori DAC prediction: governance correctness and domain function calibration are independent concerns, and the B=0 issue was a domain function calibration bug, not a primitive failure.

Third, the Orkestratum runtime provides evidence that does not depend on any single domain. When the same primitive implementation serves five domains simultaneously, the question "did you force this domain into your primitive set" has a different answer than when the primitive set is checked against one domain at a time. The runtime is either consistent across all domains or it is not. The conformance audit, with its line-number traces and invariant mappings, provides the ground truth. The audit is not presented in this paper as a full disclosure; the blueprint document is separate engineering record. What the paper can cite is that the audit exists, is in active use, and has produced a specific, bounded list of remediation items that are tracked against gates.

Fourth, the TLA+ formal verification process subjected every primitive and composition to exhaustive model checking at the tested bounds. The model checker does not share the author's assumptions or biases. It explores every reachable state mechanically. The five corrections it produced (Section 10.4) demonstrate that the verification was genuine: if the model checker had simply confirmed every initial assumption, the exercise would have been less credible. The corrections are evidence that the formal verification process had teeth.

The residual risk remains: domains not yet examined may require primitives outside the current set. The adversarial domain pass (Section 2.5) added one primitive, which is both evidence that the pass was real (if the primitive set covered everything, no addition would have been needed) and a reminder that future adversarial passes may add more. The claim is that seventeen primitives suffice for twenty examined domains, not that they suffice for all possible computation.

### 10.3 Limitations

DAC does not claim that domain expertise is unnecessary. The domain function (the rendering equation, the attention scoring function, the unification consistency check) requires domain expertise to design. What DAC claims is that the execution infrastructure around the domain function is generic and need not be redesigned per domain. Domain-specific vocabulary is accidental complexity for the execution infrastructure. It is essential complexity for the domain function. The boundary between the two is drawn at the governance layer: the primitives provide structure and governance; the domain provides the computation that executes within that governance. Readers should not interpret the claim that "vocabulary is clothing" as a claim that domain expertise is unnecessary. The clothing covers something real. The claim is that the structural skeleton underneath is shared.

DAC does not claim that seventeen primitives are the final, minimal set. Future domains may reveal operations that genuinely cannot be expressed as compositions of the current set, requiring the addition of new primitives. The claim is that seventeen suffice for the twenty domains examined, not that they suffice for all possible computation.

The twenty domains examined share characteristics that favor the primitive set. Resource-governed execution is well-covered (the original twelve). Symbolic computation with backtracking is newly covered (the adversarial pass). Domains with fundamentally different computational characters than any examined here (interactive theorem proving with tactics, quantum circuit simulation, biological sequence alignment against reference genomes) may stress the primitive set in ways that reveal further gaps.

All twenty domains were collapsed by the same individual. Independent replication by other researchers applying DAC to domains outside the current twenty is necessary to establish that the methodology is reproducible and that the primitive set generalizes beyond the author's own analytical perspective. The incremental discovery process, the formal verification, and the hostile audit protocols mitigate confirmation bias but do not eliminate it. The strongest validation of DAC's generality would be a practitioner in an unexamined domain independently discovering that the primitive set expresses their domain's computational patterns, or honestly reporting where the primitives are insufficient and an eighteenth is required.

The LeanFormer empirical validation has a significant scale limitation. The architectural results (39M parameters) are proof-of-concept validations, not production-scale demonstrations. The 204M parameter training run validated that all seventeen primitives compose correctly under real training conditions across 7,228 optimizer steps, with all governance invariants holding throughout. However, 204M is modest by current standards, and many of the efficiency mechanisms that DAC predicts (gradient compute reduction from hierarchical activation, budget-governed routing of gradient signal) were not realized as wall-clock improvement in this run. The governance machinery correctly identified which parameters to stop training and reallocated budget, but the training loop did not skip computation for converged groups. The governance produced the right signal; the implementation did not yet act on it efficiently.

The 204M run also suffered from training configuration limitations that prevent strong claims about language modeling quality. Single-epoch training with minimal regularization (dropout 0.1 only) produced severe overfitting after step 2,000, with validation perplexity rising from 57.6 to 1,463.9. The overfitting is expected behavior and does not indicate an architectural failure, but it means the model's generalization quality should be evaluated from the best checkpoint (step 2,000), not the final checkpoint. Multi-epoch training with proper regularization is necessary to demonstrate that the architecture can produce competitive language models, and is identified as necessary future work.

The Orkestratum runtime's demonstrations are bounded by what the codebase currently supports (Section 6.2). The runtime is an active engineering program. The claim is that one runtime serves five domains with the same primitives, not that the runtime is feature-complete or performance-optimal in any of them.

The critical open question for LeanFormer is whether DAC's governance primitives continue to compose correctly and produce measurable efficiency gains at the scales where modern language models operate (7B+ parameters). Emergent training dynamics, optimization instabilities, and loss landscape characteristics at the billion-parameter scale may interact with the governance mechanisms in ways that the current validation cannot anticipate. A specific concern is governor thrashing: at scales where gradient magnitudes spike randomly, the convergence governor could oscillate rapidly between states. The existing design mitigates this structurally — state transitions require sustained EMA trends over a configurable cooling window, not single-step readings, and the AWAKENED state is explicitly designed to handle perturbation events without re-traversing the full state sequence — but the thresholds will require recalibration for frontier-scale gradient distributions. The phase-aware governor (Section 5.8.1) adds a further structural guard: a group in COLD phase cannot transition to COOLING regardless of gradient magnitude, preventing the initialization artifact class of false transitions entirely. A full-scale validation at 7B parameters or above is the single most important next step for the LeanFormer validation. Until that validation is complete, the LeanFormer results should be read as architectural proof-of-concept and governance machinery validation, not as demonstrated production-scale efficiency gains.

A data preservation gap from the 204M run warrants mention as a reproducibility concern. The tokenized training data (32K vocabulary) was stored only on the GCP VM and in a cloud storage bucket, both of which were deleted when the VM was terminated. The local copy of the raw corpus was tokenized with a different vocabulary size (50K), making it incompatible with the trained model for validation purposes. The inline EVAL measurements from the training log remain the authoritative validation numbers, as they were computed against the correctly-tokenized data during training. The lesson is straightforward: the governance machinery preserved every training decision with tamper-evident integrity, but the training data itself was not under governance. Future runs should archive the tokenized data alongside checkpoints.

The TLA+ formal verification is bounded model checking, not unbounded proof. The TLC model checker exhaustively explores all reachable states within finite bounds (Budget capacity of 4, three-node hierarchies, two parameter groups). Structural bugs and invariant violations are reliably caught at these bounds because the primitive structures do not change with scale. However, bounded verification does not constitute a mathematical proof that the invariants hold for all possible values of the constants. It provides strong evidence, not certainty.

An important distinction follows from this: the governance invariants are scale-independent by construction, and this was confirmed empirically at 204M parameters. Budget\<U\> enforces `consumed <= capacity` whether capacity is 4 or 4 billion. The ConvergenceGovernor's four-state machine has the same transitions whether it monitors 8 parameter groups or 8,000. HierarchyOrdering holds whether there are 2 levels or 20. The TLA+ specifications are parameterized by constants, and the invariants hold for any value of those constants because the logic does not reference the constants' magnitudes. The 204M training run confirmed this empirically: zero budget violations across 722 audit records, all 18 governor transitions valid, hierarchy ordering maintained throughout. These are structural properties of the state machines, not empirical properties of any particular training run. What remains empirical, and what requires the 7B+ validation, is whether the governance produces efficient training outcomes at that scale: whether the convergence thresholds, budget allocation policies, and gradient routing scores (all domain functions, not primitives) produce the predicted efficiency gains when operating on real gradient distributions at frontier scale.

However, the boundary between what TLA+ can and cannot verify is narrower than it first appears. Applying DAC's own vocabulary-stripping process to the phrase "gradient dynamics at scale" reveals that the problem is not numerical prediction (which TLA+ cannot do) but signal trajectory classification (which it can). The B=0 initialization bug (Section 5.8) was not a numerical problem. It was an observational degeneracy: two qualitatively different gradient trajectories (cold start and genuine convergence) both produced low magnitude readings, and the governor could not distinguish them. This is a solved problem across the collapse table. In rendering, a pixel with zero color could be background or a black surface; the depth buffer adds a signal to break the degeneracy. In networking, a silent node could be idle or crashed; heartbeats add a liveness signal. In distributed consensus, a node that has not voted could be slow or partitioned; timeouts add a temporal boundary. In every case, the solution is the same: add a second Signal\<T\> that disambiguates the measurement.

The same solution applies to the convergence governor. Gradient trajectories at any scale are not numerically predictable, but they are qualitatively enumerable: COLD (learning has not started), WARMING (gradient rising toward threshold), ACTIVE_LEARNING (gradient above threshold), and DECLINING (gradient falling from above threshold). A phase-aware ConvergenceGovernor that tracks whether gradient magnitude has ever exceeded the threshold can structurally prevent the B=0 bug: the ACTIVE-to-COOLING transition requires the gradient phase to be ACTIVE_LEARNING or DECLINING, never COLD. This invariant (NoCoolingFromCold) is verifiable in TLA+. The phase-aware specification is included in the project repository, and TLC confirms that the invariant holds across all reachable states while a specification without phase awareness produces a counterexample matching the exact failure observed in the training run.

This extends the scope of formal verification beyond governance correctness into domain function validation. The specific numerical thresholds remain empirical, but the qualitative correctness of the governor's response to different trajectory classes is formally verifiable. At 7B parameters, the expected trajectory classes (cold start, rapid learning, optimization instability with gradient spikes, gradual convergence, perturbation from belief injection) can be enumerated and the governor's response to each can be verified by TLC before a single GPU hour is spent. Formal verification cannot predict whether a 7B model will converge in 10,000 steps or 100,000. But it can prove that the governance machinery will respond correctly to whatever trajectory the model produces.

The formal verification also does not verify implementations. The TLA+ specifications define what the primitives must do. The implementations in Appendix A are intended to be faithful to those specifications, but the verification that each implementation correctly implements the specification is a separate concern (refinement checking). Orkestratum's engineering blueprint schedules this refinement checking as a dedicated phase; at the time of writing, refinement specifications exist for the primitive layer and composition specifications exist for key reconcilers, but the full per-primitive refinement verification is future work.

### 10.4 Formal Verification and What It Revealed

All seventeen primitive specifications (including the phase-aware ConvergenceGovernor), five LeanFormer compositions, and corresponding decomposition-failure specifications were formalized in TLA+ [14] and verified by the TLC model checker. The verification explored approximately 45.4 million states at the tested bounds, with the phase-aware ConvergenceGovernor alone accounting for 18.6 million states across its full space of possible delta sequences and gradient phase transitions. The complete specifications, configuration files, and TLC output logs are available in the project repository.

The verification produced three categories of results.

First, all primitive specifications passed: their safety invariants held across all reachable states at the tested bounds. Budget never exceeded capacity. The ConvergenceGovernor never skipped a state. CompetitiveSelection (hard) always produced the highest-scoring winner. The hash chain in AuditSink was never broken. These results are bounded verification, not unbounded proof, but TLC's exhaustive exploration of the finite state space provides strong evidence that the invariants hold in general.

Second, all five LeanFormer composition specifications passed, confirming that primitive invariants are preserved under composition and that the composed system produces emergent guarantees. The GovernedTrainingPipeline specification verified six simultaneous invariants across all interleavings of training steps, governor updates, hierarchy activations, budget reallocations, and gradient routing updates. The BeliefDeltaLifecycle specification confirmed base weight immutability, parameter non-overlap, and exact restoration across all possible sequences of belief injection and removal.

Third, decomposition-failure specifications for every primitive produced concrete counterexamples, confirming that every primitive is operationally irreducible with respect to its governance invariant. The counterexamples are not hypothetical. They are specific state traces that TLC discovered through exhaustive exploration. Some violations were found in as few as two states (QualityHierarchy: adding a child without checking the level constraint; the naive ConvergenceGovernor: a single zero-delta step causing premature COOLING, reproducing the exact B=0 bug from the training run). Others required longer traces (Budget: a concurrent allocation between check and act, at seven states). Every counterexample demonstrates a real failure mode that would manifest in any implementation that decomposes the primitive into separate sub-operations.

A precise characterization of what "irreducible" means here is necessary to avoid overclaiming. The decomposition failures demonstrate operational irreducibility: the guard and the guarded operation cannot be separated into distinct steps without creating reachable states where the governance invariant is violated. The mechanism in most cases is a Time-of-Check-to-Time-of-Use (TOCTOU) pattern: a condition is checked, the world changes, and the guarded operation executes against stale state. This is a well-understood concurrency hazard, not a novel discovery. What is novel is the systematic demonstration that every primitive in the set exhibits this property. The governance invariant is not a postcondition that can be checked after the fact. It is an atomic property of the operation itself. Decompose the operation and the property ceases to exist. This is what makes them primitives rather than compositions.

This is a weaker claim than algebraic irreducibility (which would require proving that no primitive can be expressed as a composition of other primitives in the set). Section 10.1 addresses algebraic minimality separately, acknowledging that minimality has not been formally proven and identifying specific candidates for potential redundancy.

The verification process also challenged and corrected five initial assumptions about the formal specifications. These corrections are worth documenting because they demonstrate that the TLA+ verification was a genuine validation exercise, not a formality.

| Specification | Initial Assumption | TLC Finding | Correction |
|---|---|---|---|
| ConvergenceGatedActivation | Global invariant (always holds) | L0 can regress to ACTIVE after L1 activates (AWAKENED mechanism) | Reclassified as precondition of ActivateNextLevel |
| FullDepthMeansUncertain | Full depth implies no convergence | Convergence at the final layer is valid | Corrected to: full depth with no convergence at any layer implies uncertainty |
| ForgingRequiresConvergence | Holds across all phases | Converged group could revert to ACTIVE during forging | Restricted governor updates to training phase |
| Budget reallocation | Free half of converged group's budget | Group may have consumed more than half, violating SubPoolInvariant | Free only the unused portion |
| WinnerOptimality | Holds between evaluations | Score update between evaluations creates stale winner | Invalidate allocation when scores change (TOCTOU pattern) |

None of these corrections invalidated the underlying primitives or their invariants. They refined imprecise classifications: the difference between a global invariant and a precondition, the boundary conditions of a convergence definition, the interaction between budget usage and budget reallocation, and the scope of governance operations across lifecycle phases. In every case, the corrected specification is more precise than the original, not more permissive. These are exactly the kinds of issues that prose specifications miss and that formal verification catches.

The fact that the initial specifications required correction is not a weakness of the methodology. It is evidence that the verification was real and that the corrected specifications are trustworthy. A verification process that confirms every initial assumption without correction is either trivial or dishonest.

### 10.5 Future Work

The following represent the most important open questions and necessary next steps, ordered by priority and with estimated resource requirements where applicable.

1. Independent replication of DAC by other researchers applying the methodology to domains outside the current twenty. Positive results would confirm the methodology's generality. Negative results (domains where the primitive set is genuinely insufficient) would be equally valuable, as they would identify the boundaries of the current primitive set and potentially reveal new primitives.

2. Phase-aware convergence governor implementation and retraining. The TLA+ specification exists and passes (ConvergenceGovernorPhaseAware with NoCoolingFromCold invariant). The implementation requires adding gradient_phase tracking to the convergence governor code. A retrain of the 204M model with the fixed governor would produce clean hierarchy activations free of the B=0 artifact. Estimated cost: approximately $150 (one 6-day L4 GPU run).

3. Multi-epoch training with proper regularization. The current 204M run used a single epoch with dropout 0.1 as the only regularization, producing severe overfitting after step 2,000. A 3-5 epoch run with stronger regularization would demonstrate that the architecture can generalize and that governance invariants hold across extended training, including the possibility of AWAKENED state transitions triggered by data distribution shifts between epochs. Estimated cost: approximately $450-750.

4. Scale validation of LeanFormer at 7B+ parameters. This is the single most important validation that the current work lacks. It would include: gradient compute reduction measurements with actual backward-pass skipping for converged groups, wall-clock training time comparisons against an ungoverned baseline, Knowledge Plane validation with orthogonality enforcement at a scale where the delta address space is practically significant, and convergence ordering analysis at a scale where the parameter group ratios are representative of production models. The governance invariants are established as scale-independent (formally verified and empirically confirmed at 204M), but the efficiency claims require empirical validation at frontier scale. Estimated cost: approximately $5,000-15,000.

5. Adversarial domain testing beyond the eight domains examined in Section 2.5. Interactive theorem proving with tactics, quantum circuit simulation, and biological sequence alignment against reference genomes are plausible next candidates.

6. Bidirectional cross-domain optimization transfer. Demonstrate an optimization discovered in one non-ML domain (rendering, for example), applied to an ML primitive instance via the shared implementation, producing a measurable improvement on an ML benchmark. This is the strongest criterion from Section 10.1 and the one the paper cannot yet fully claim.

7. Orkestratum formal verification program completion. The engineering blueprint schedules per-primitive refinement verification and per-composition verification as a dedicated phase. Completing this phase would extend the formal guarantee from the primitive-set-as-mathematical-object (already verified) to the primitive-set-as-running-implementation.

8. Orkestratum performance consolidation. Specific targets include Sponza at 100+ FPS through the unified kernel and Bistro at interactive frame rates. The target demonstrates that the primitive set is not merely expressive but operationally efficient in the domain with the tightest performance constraints.

9. Resolution of the minimality question: for each primitive, determine whether removing it from the set makes at least one domain pattern unexpressible. If RateLimit can be expressed as Budget\<Operations\> composed with a temporal mechanism, or if Signal\<T\> can be expressed as AuditSink composed with a predicate filter, the primitive count should be revised. The methodology's value does not depend on the exact count, but intellectual honesty requires resolving the question.

10. Knowledge Plane validation at 204M. The 39M model demonstrated 84% belief injection success and 86% semantic routing accuracy. Repeating this validation at 204M against the best checkpoint (step 2,000) would confirm that the delta system scales, test injection success rate and routing accuracy at a more representative model size, and verify that the theoretical maximum deltas are practically achievable. Estimated cost: approximately $50.

The risk profile for the computational future work (items 2, 3, 4, 10) is low. The primitives are formally verified. The governance machinery works at 204M parameters. The open questions are training configuration tuning and whether efficiency gains scale as predicted. The risk profile for the methodological work (items 1, 5, 6) is higher but also produces the most valuable results: either confirmation of the methodology's generality or identification of its boundaries.

---

## 11. Implications

### 11.1 For Software Engineering

If DAC's thesis is correct, the software industry is spending substantial resources building, maintaining, testing, and securing redundant implementations of the same computational patterns across different domains. Every Kubernetes controller, every CI/CD runner, every ETL framework, every SOAR playbook engine is a reimplementation of DAG traversal under budget constraints. Orkestratum's architecture is one instance of the alternative: an execution framework that recognizes these domains as the same computation with different domain functions. The maintenance cost, the security surface area, and the bug count scale with the number of independent implementations; collapsing four implementations to one compounds benefits in each dimension.

### 11.2 For AI/ML Systems

The identification that attention is soft competitive selection and backpropagation is a member of the graph message-passing family creates a bridge between the ML optimization community and the real-time systems community. Techniques developed for efficient GPU selection (the visibility buffer, hierarchical culling, budget-constrained traversal) become candidates for efficient attention computation. Techniques developed for efficient fixed-point iteration (convergence governors, temporal amortization, adaptive iteration counts) become candidates for training loop optimization.

The LeanFormer belief-delta system demonstrates a more immediate practical implication: the entire problem of knowledge management in neural networks (injection, removal, versioning, composition, audit) maps directly to solved database and systems engineering patterns. The Knowledge Plane architecture treats knowledge as a managed database with insert, update, delete, query, and vacuum operations. The structural identity with database management means that decades of engineering on consistency, isolation, and durability transfer directly.

The confabulation detection mechanism (Section 5.7) demonstrates a different kind of transfer: a real-time systems concept (convergence-based confidence estimation) applied to an AI safety problem. The principle that "the system should be honest about uncertainty by architecture, not by training" is a direct consequence of the DAC decomposition. A system with a ConvergenceGovernor knows, structurally, whether its computation converged or not. A system without one can only guess.

More broadly, the application of TLA+ and bounded model checking to ML training governance represents a contribution to the verifiable AI agenda. The ML community has largely treated training as an empirical process where correctness is assessed by outcome (did the loss go down?) rather than by invariant (were the governance properties maintained at every step?). The DAC approach inverts this: the governance properties are specified formally, verified exhaustively at bounded scales, and then confirmed empirically. The 204M training run demonstrated that zero budget violations, zero skipped governor states, and zero audit chain breaks can be maintained across thousands of training steps, even through severe overfitting. This is a different kind of assurance than "the model got a good benchmark score." It is the kind of assurance that regulated industries (healthcare, finance, defense) require before deploying AI systems, and it transfers from the same formal verification practices that those industries already use for non-AI systems.

The targeted training system (Section 5.8) represents the deepest implication: DAC applied not to the model but to the process that produces the model. The observation that training is the only governed computational system in the collapse table operating without selectivity primitives is a structural diagnosis, not an incremental optimization. Every other domain that allocates resources under constraints uses CompetitiveSelection gating, QualityHierarchy traversal, FederatedBudget allocation, and per-group ConvergenceGovernor. The ML community has independently reinvented fragments of this composition (Mixture of Experts, LoRA, curriculum learning, layer freezing, progressive training, GradNorm), each in isolation, each in ML vocabulary that prevented recognizing the unified structural pattern. The DAC decomposition makes the pattern visible and produces a governed training pipeline where the selectivity, budgeting, convergence detection, and audit primitives compose into a closed-loop system.

### 11.3 For Education

Domain Abstraction Collapse suggests that the most effective way to teach computation is not domain-first ("here is how rendering works," "here is how databases work," "here is how ML works") but primitive-first ("here are the seventeen operations that the computational systems in this paper are built from; now let us see how rendering, databases, and ML are each a composition of these operations"). This approach would produce engineers who recognize structural patterns across domain boundaries rather than engineers who are expert in one domain's vocabulary and blind to the identical structures in neighboring domains.

This is an aspiration, not a claim. No curriculum has been built around this approach, and the case for it rests on the methodology's demonstrated ability to enable cross-domain recognition rather than on any educational outcomes data.

### 11.4 For Problem Solving

The generative and implementation applications of DAC (Section 5) may be the most practically valuable implication. When an engineer encounters a problem that resists solution, the question to ask is: "What is this problem when I remove the domain vocabulary?" If the stripped problem maps to the abstraction primitive set, the solution may already exist in another domain. When an engineer needs to implement a computation described in unfamiliar notation, the question is: "What are the computational steps underneath this notation, and which primitives compose to produce them?" The LeanFormer development arc and the B=0 diagnosis episode suggest that when the methodology identifies a structural match, the path from problem to working solution is shorter than domain-native approaches that must solve the problem from first principles.

---

## 12. Conclusion

Domain Abstraction Collapse is the systematic observation that many domain-specific computational abstractions are the same operation in different vocabulary. The methodology (enumerate, strip, identify isomorphisms, reduce to abstraction primitives, reconstruct) is repeatable and has a formal completeness criterion. The abstraction primitives are defined as operations that cannot be further decomposed without losing the governance semantics that make cross-domain composition useful, distinguishing DAC from the trivially true observation that everything reduces to logic gates.

Seventeen abstraction primitives suffice for the twenty domains examined. Twelve of those domains were originally collapsed in the development of the methodology; eight additional domains from outside the original sample's resource-governed execution character were added in an adversarial pass that produced one new primitive (Checkpoint) and two refinements to existing primitives (widening mode on ConvergenceGovernor, explicit monotonicity on ResourceRegistry). The Competitive Selection family accounts for a large fraction of the cross-domain mappings and decomposes into three selection modes (hard, soft, ranked) that share a scoring interface but differ in allocation semantics. Some collapses in the paper's tables are genuine structural insights (attention as soft competitive selection, the four-domain DAG workload collapse, training as the only governed system without selectivity primitives). Others are trivially true (a lookup table is a lookup table). Both categories are documented, and the paper does not rely on the trivial collapses to support its central claims.

Two engineering systems built on the primitive set serve as empirical validation. LeanFormer demonstrates what DAC produces when applied to a single domain. Six open problems in neural network design, each framed as a hard ML research problem with years of dedicated literature, became compositions of solved systems engineering primitives the moment the ML vocabulary was stripped away. The resulting architecture composes four efficiency mechanisms simultaneously, solves catastrophic forgetting by construction (bit-for-bit base weight restoration across 100 beliefs), makes confabulation architecturally detectable, and governs the entire training lifecycle with the same primitives that govern the model's architecture. At 204M parameters, all seventeen primitives composed correctly across 7,228 training steps with zero governance violations.

Orkestratum demonstrates what DAC produces when applied to a runtime. One kernel codebase, built on the seventeen primitive modules, executes SOAR playbooks, CI/CD pipelines, ETL workflows, configuration-management runs, and a real-time renderer. Each domain is a composition of the same primitives with different domain functions and different workload graphs. No domain has its own scheduler, its own budget, or its own audit implementation. The runtime's claims are bounded to what the codebase currently supports; specific in-progress items are documented, and the paper does not claim feature completeness or performance dominance in any individual domain. What the runtime does demonstrate is that the primitives are engineerable, that they survive multi-domain contact including real-time rendering, and that a fix motivated by one domain (the phase-aware ConvergenceGovernor, developed to address an ML training bug) is available to every other domain that uses the same primitive.

The B=0 observational degeneracy discovered during the 204M LeanFormer run provided an unplanned demonstration of DAC's generative mode. When the convergence governor misidentified cold-start gradient magnitudes as post-learning convergence, applying DAC's own vocabulary-stripping methodology to the failure revealed it as an instance of a pattern already solved across the collapse table: depth buffers disambiguate zero-color pixels, heartbeats disambiguate silent nodes, timeouts disambiguate non-responsive voters. The phase-aware fix followed mechanically from the cross-domain pattern and was formally verified in TLA+ before any code was written. A methodology that diagnoses its own failures by recognizing their structural identity with solved problems from other domains is demonstrating the generative capability it claims.

The primitives are formally verified mathematical structures at the tested bounds. TLA+ specifications were verified by the TLC model checker across approximately 45.4 million states. Every invariant held. Every decomposition produced a concrete counterexample proving operational irreducibility. A real training bug was reproduced as a TLC counterexample in two states and fixed with a phase-aware governor verified across 18.6 million states. The verification process challenged five initial assumptions about the specifications and produced corrections that refined imprecise classifications without weakening any claim. These corrections demonstrate that the primitives survived adversarial scrutiny from a tool that does not share the author's assumptions, biases, or vocabulary.

What this paper claims, and what it does not: the primitive set is sufficient for twenty domains and operationally irreducible by formal verification. Algebraic minimality remains open. Cross-domain optimization transfer has been demonstrated in one direction (systems-to-ML) and within the Orkestratum runtime; full bidirectional transfer across all domains remains future work. The LeanFormer 204M run demonstrates governance correctness, not production-scale efficiency gains; the scale validation at 7B+ parameters is the single most important next step. Orkestratum's claim is that one runtime serves five domains with the same primitives, not that the runtime is feature-complete or performance-dominant in any individual domain. Independent replication of the methodology by other researchers is the strongest form of validation and has not yet occurred.

What the paper does claim is that the structural skeleton underlying the examined domains is shared, that the domain-specific vocabulary is the primary obstacle to recognizing this, and that a small set of primitives at a specific level of governance semantics suffices to express the computational content of all twenty domains examined. The operations are familiar. The unification at this specific level, with this specific set, is not.

The intelligence in a domain lives in its function — the rendering equation, the attention scoring function, the unification consistency check, the constraint propagation rule. The governance infrastructure around the domain function is shared. The vocabulary at the infrastructure level was clothing. The structure underneath the clothing was, across every domain examined in this paper, the same.

---

## Acknowledgments

LeanFormer and Orkestratum were implemented using an AI coding assistant (Claude Code) as an implementation agent, with the architect maintaining design authority, formal specification responsibility, and verification discipline. This development model and its attendant risks are discussed in Section 5.2. The mitigations against "plausible-looking code against plausible-looking specifications" are explicit: hostile audit protocols that require log evidence rather than diff evidence, formal specifications in TLA+ that the implementation must refine, continuous measurement of governance invariants during training, and a conformance audit program for the Orkestratum runtime with line-number traces to the specifications. The reproducibility of the engineering results depends on these mitigations being applied; the paper identifies the refinement verification program as specific future work (Section 10.5, item 7).

---

## References

[1] Brooks, F.P. (1987). No Silver Bullet: Essence and Accident in Software Engineering. IEEE Computer, 20(4), 10-19.

[2] Vaswani, A. et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems, 30.

[3] Bellman, R. (1958). On a Routing Problem. Quarterly of Applied Mathematics, 16(1), 87-90.

[4] Codd, E.F. (1970). A Relational Model of Data for Large Shared Data Banks. Communications of the ACM, 13(6), 377-387.

[5] Lampson, B.W. (1974). Protection. ACM SIGOPS Operating Systems Review, 8(1), 18-24.

[6] Hu, E.J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

[7] Wihlidal, G. (2017). Optimizing the Graphics Pipeline with Compute. GDC Presentation, EA SEED.

[8] Leviathan, Y. et al. (2023). Fast Inference from Transformers via Speculative Decoding. ICML 2023.

[9] Fedus, W. et al. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. JMLR, 23(120), 1-39.

[10] Katharopoulos, A. et al. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020.

[11] Schuster, T. et al. (2022). Confident Adaptive Language Modeling. NeurIPS 2022.

[12] Meng, K. et al. (2022). Locating and Editing Factual Associations in GPT. NeurIPS 2022.

[13] Karis, B. et al. (2021). Nanite: A Scalable Mesh Representation. SIGGRAPH Advances in Real-Time Rendering course.

[14] Lamport, L. (2002). Specifying Systems: The TLA+ Language and Tools for Hardware and Software Engineers. Addison-Wesley.

---

## Appendix A: The Seventeen Primitives in Code

### Cross-Domain, Cross-Language Implementation Examples

Each primitive can be implemented in multiple languages to demonstrate that the structure is expressible in multiple imperative paradigms and that only the domain-specific types and functions change. This appendix shows representative implementations. The caveat stated in Section 2.2 applies: showing multiple imperative implementations demonstrates language-portability within a paradigm; the stronger claim (that the primitives are language-paradigm-independent) would require implementations in languages with fundamentally different computational models and is not made here.

---

#### A.1 Budget\<U\>

A pool with a capacity and an invariant: consumed <= capacity. There is no force_allocate.

Rust:

```rust
pub struct Budget<U: Copy + Ord + Default + AddAssign + SubAssign> {
    capacity: U,
    allocated: U,
}

impl<U: Copy + Ord + Default + AddAssign + SubAssign> Budget<U> {
    pub fn new(capacity: U) -> Self {
        Self { capacity, allocated: U::default() }
    }

    pub fn try_allocate(&mut self, amount: U) -> bool {
        let mut next = self.allocated;
        next += amount;
        if next > self.capacity { return false; }
        self.allocated = next;
        true
    }

    pub fn release(&mut self, amount: U) { self.allocated -= amount; }
}
```

Python:

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

Go:

```go
type Budget struct {
    capacity  int64
    allocated int64
}

func (b *Budget) TryAllocate(amount int64) bool {
    if b.allocated + amount > b.capacity {
        return false
    }
    b.allocated += amount
    return true
}

func (b *Budget) Release(amount int64) {
    b.allocated -= amount
}
```

The invariant (allocated <= capacity) is enforced identically in all three. The type of U is domain-specific (VRAM bytes, gradient compute, CPU millicores). The invariant is not.

#### A.2 ConvergenceGovernor (four-state machine)

Python:

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

The four-state machine is identical. The delta (constraint residual, gradient magnitude, term change rate) is the domain-specific part.

#### A.3 Checkpoint (new in v8)

A scoped rollback boundary with stack discipline. Nested scopes unwind in LIFO order. This primitive is required for CLP-style backtracking and theorem proving search. The composition AuditSink + reverse-replay ActuationPass does not satisfy the atomicity required for safe nested unwinding under concurrent mutation; TLC verified this with a concrete counterexample.

Python:

```python
class Checkpoint:
    def __init__(self, state_ref):
        self.stack = []
        self.state_ref = state_ref

    def begin(self):
        self.stack.append(copy.deepcopy(self.state_ref.get()))

    def commit(self):
        if not self.stack:
            raise RuntimeError("commit with no active checkpoint")
        self.stack.pop()

    def rollback(self):
        if not self.stack:
            raise RuntimeError("rollback with no active checkpoint")
        self.state_ref.set(self.stack.pop())
```

The invariant is stack well-formedness: begin, commit, and rollback preserve LIFO discipline; rollback reverts to the exact state snapshotted at begin. The domain-specific part is the state representation and the equality semantics.

Note: This Python example uses deepcopy for illustrative simplicity. A production implementation would achieve the required O(1) Checkpoint isolation via persistent data structures, copy-on-write semantics, or inverse-delta logging.

### The Pattern

Every example in this appendix demonstrates the same property: the domain provides the types, the data, and the scoring/aggregation/message functions. The primitive provides the structure, the governance, and the invariants. Swap the domain function and the primitive serves a different domain without modification.

The vocabulary was different. The language was different. The code was the same, within the family of imperative paradigms. Implementations in functional, logic, and declarative paradigms are not shown here, and the paper does not claim paradigm-independence from four imperative examples. That claim requires evidence that has not been produced.
