# DRAFT - Domain Abstraction Collapse

## Recognizing Solved Problems Across Domain Boundaries

Brian Moore, CISSP, CCSP
Independent Systems Researcher

---

## Abstract

Vocabulary across engineering domains diverges while the governance structure underneath recurs. Budget invariants, convergence detection, audit completeness, and atomic mutation appear under names that rarely cross between graphics, machine learning, distributed consensus, or networking. This paper names that pattern Domain Abstraction Collapse (DAC) and proposes that the recurring structure is sharp enough to specify formally and to compose deliberately.

The starting point was practical. I was building an execution kernel intended to serve three automation domains (ETL, SOAR, configuration management) that the relevant communities already treated as workloads over a streaming DAG. The goal was an orthogonal kernel with a composable module system. Achieving it required identifying which operations the three domains genuinely shared at the structural level, separated from the domain-specific framing each tool applied. After working through nine more domains (real-time rendering, physics simulation, audio spatialization, network replication, container orchestration, CI/CD pipelines, distributed consensus, economic simulation, machine learning) the same shapes kept recurring. Sixteen primitives covered the patterns I found across the twelve. The set is not proven minimal; sufficiency for the patterns I examined is the load-bearing claim.

DAC operates analytically (decomposing existing systems to surface shared structure) and constructively (converting any computation described in domain notation into a build plan composed of primitives). The constructive direction covers both generative search, when domain vocabulary is concealing a solution another domain has already produced, and implementation, when the build plan is itself the deliverable.

LeanFormer is the worked example. It is an artifact of applying DAC to a domain where I do not have research training. My background is security and Linux systems engineering and automation. I am not an ML engineer. The work tested whether a non-expert applying DAC carefully could produce a workable architecture for problems the ML literature treats as open. The output is a 204M-parameter transformer whose model architecture, training pipeline, and knowledge management system are all compositions of the sixteen primitives. After I built each component from the DAC decomposition, I checked the result against the ML literature. Several components closely resemble established techniques: the belief delta system resembles LoRA, the gated attention resembles sparse attention variants, the parameter group hierarchy resembles curriculum learning. I take that convergence as a rediscovery signal. Starting from a structural skeleton derived independently of ML, the architecture landed near solutions ML researchers reached through domain-internal reasoning. All sixteen primitive invariants held empirically across 7,228 training steps with zero governance violations across 722 hash-chained audit records. Best validation perplexity reached 57.6 at step 2,000. The model is not competitive as a language model and I do not claim it is. The claim is rediscovery and machinery validation. Comparative baselines (vanilla LoRA, ungoverned training, dense parameter equivalents) are open work identified in Section 9.5.

During the 204M run, a B=0 initialization artifact caused two parameter groups to be misclassified as converged before they had received meaningful gradient signal. Stripping ML vocabulary described the failure as an observational degeneracy: two qualitatively different gradient trajectories produced the same low-magnitude reading. The same shape appears in rendering (depth buffers separate background from black surfaces), in networking (heartbeats separate idle from crashed nodes), in distributed consensus (timeouts separate slow from partitioned voters), and in database concurrency control (MVCC version numbers separate stale from current reads). The fix has the same shape across all of them: a second signal that disambiguates the measurement. A phase-aware ConvergenceGovernor that records whether gradient magnitude has ever exceeded the threshold was specified in TLA+ and verified by TLC across roughly 18.6 million states. The original failure reproduces as a 2-state TLC counterexample without the phase guard.

Forty-four TLA+ specifications (twenty-one primitive specifications, eighteen decomposition-failure specifications, five LeanFormer compositions) were verified by TLC across roughly 44.6 million states for the universal verification and 178,000 additional states for the LeanFormer compositions. Every primitive safety invariant held. Every decomposition into separable check-then-act steps produced a concrete TOCTOU counterexample, establishing that the primitives carry atomicity-dependent invariants at their boundaries. The verification process surfaced and corrected five imprecise specification choices.

The point of the paper is bounded. Across the twelve engineering domains I examined, structurally similar operations appear under different vocabularies, and sixteen primitives covered the patterns. The methodology works, in my experience, by asking the same second question of every new domain: how does this really work, underneath the names?

---

## 1. Introduction

### 1.1 Patterns Hidden by Vocabulary

Fred Brooks distinguished between essential complexity (complexity inherent to the problem) and accidental complexity (complexity introduced by the tools and methods used to solve it) [1]. There is a third category worth naming: vocabulary-induced complexity, where the domain-specific language used to describe a problem creates the impression that the problem itself is domain-specific.

Consider the Entity Component System (ECS), an architectural pattern in game engine development since the late 1990s. An entity is a unique identifier. A component is a typed data record associated with an entity. A system is a function that queries entities by their component composition and applies transformations. The pattern shows up in major game engines under different names: GameObjects and MonoBehaviours in Unity, Actors and Components in Unreal, Entities and Components and Systems in Bevy, GameObjects and Components in Godot. Each engine names the pieces differently while preserving the same shape underneath.

Stripped of game-specific vocabulary, an ECS is a relational table with column-oriented storage. Entities map to rows; components map to columns; systems are queries with side effects. The cache-friendly memory layout is a real engineering contribution. The structural pattern (typed records queried by composition and transformed by functions) is well-trodden ground in the database community, which has spent decades developing query planning, transaction semantics, schema evolution, and access control for that shape. The observation that ECS resembles a database is not original to me; data-oriented design literature and Tim Sweeney in public talks have made versions of it. What is worth noting is the scale of the parallel reinvention. Many game engine teams have built ECS infrastructure independently within game-engine vocabulary, without much cross-pollination from the database side.

This is one example of a recurring pattern. Across the engineering domains I examined, systems that appear to solve domain-specific problems with domain-specific architectures often turn out, when the vocabulary is stripped, to be compositions of a small number of computational patterns that have been solved repeatedly under different names.

### 1.2 The Methodology: Domain Abstraction Collapse

In plain terms, DAC is a habit. When you learn a new technology and ask "how does this work?" you get an answer that is technically correct but full of domain jargon you have to look up. So you ask the second question: how does this *really* work? That second question, asked deliberately and across enough domains to start noticing patterns, is DAC. It is pattern matching against solutions you already know, with the domain vocabulary stripped off so the matching is possible. The five steps below are the systematic version of that habit.

Domain Abstraction Collapse (DAC) is a five-step process:

1. Enumerate domain-specific abstractions. List every named concept, pattern, data structure, and algorithm used within a domain. Accept the domain's own vocabulary.
2. Strip domain vocabulary. For each abstraction, remove every word that is specific to the domain. Replace domain nouns with generic descriptions of what the abstraction does structurally.
3. Identify cross-domain similarities. Compare the stripped descriptions across domains. When two abstractions from different domains reduce to the same structural description, they are candidates for the same operation under different names.
4. Reduce to abstraction primitives. Continue stripping until further decomposition would either lose governance properties (budget invariants, convergence detection, audit completeness) or descend to an implementation level where domain patterns require unbounded composition counts. The operations that survive this process are the abstraction primitives.
5. Reconstruct domains as compositions. Verify that every domain-specific abstraction from step 1 can be expressed as a composition of the abstraction primitives from step 4, instantiated with domain-specific data and functions.

If step 5 succeeds with no residual (every domain pattern is expressible, no domain pattern requires a primitive not in the set), the collapse is complete for that domain. The domain-specific abstractions were vocabulary, not structure.

These five steps describe DAC as an analytical methodology. The same process works in reverse. Given a computation described in domain-specific notation (a mathematical formula, a protocol specification, a biological pathway diagram), strip the notation's vocabulary and map the computation's structure to the primitive set. The result is a build plan composed of primitives that implements the computation. Section 5 develops this constructive application.

### 1.3 Abstraction Primitives: Why the Decomposition Stops Here

The level at which decomposition stops is the part of the claim that needs care. Set it too low and the result reduces trivially to NAND gates with no operational value. Set it too high and domain vocabulary creeps back in.

An abstraction primitive is an operation that cannot be further decomposed without one of two consequences:

(a) Loss of governance semantics. The primitive carries formal properties (budget invariants, convergence detection, audit completeness, transaction atomicity) that make cross-domain composition meaningful. Decompose below this level and those properties have to be reimplemented per domain.

(b) Explosion of composition count. At lower levels of abstraction (register operations, logic gates, individual arithmetic instructions), expressing a single domain pattern requires hundreds or thousands of composed operations and the compositions become unwieldy enough to lose their explanatory and constructive value.

The abstraction primitives sit at what I call the governance boundary: the thinnest layer of operations that still carries formal properties. Below this boundary are implementation details that vary by hardware and runtime. Above it is domain vocabulary that prevents cross-domain reuse. NAND gates are primitives but not abstraction primitives because they carry no governance semantics. Domain-specific patterns like "visibility buffer" or "playbook" are not primitives at all because they decompose into compositions of the abstraction primitive set.

A note on counting. The paper refers to sixteen primitives in conceptual terms. The TLA+ specifications expand this to eighteen spec modules because CompetitiveSelection is split into hard, soft, and ranked variants for verification clarity. Two cross-primitive composition theorems (TraversalBudgetComposition, SelectThenActuate) and a phase-aware variant of ConvergenceGovernor (introduced in Section 5.8.1) bring the verified spec count to twenty-one. Throughout the paper, "sixteen primitives" refers to the conceptual set; spec counts are noted explicitly when they matter.

A second note on what is and is not claimed. Each primitive's invariant has been formally checked under the assumption that primitive operations are atomic. Section 9.4 and Appendix D describe systematic decompositions of each primitive into separable check-then-act steps and the TOCTOU patterns that result. This shows that the primitives, as specified, depend on atomicity at the primitive boundary. It is consistent with how atomic operations behave in concurrent systems generally. It does not show that the primitive set is algebraically minimal in the sense that no primitive can be expressed as a composition of the others. Algebraic minimality is open and is discussed in Section 9.1.

### 1.4 Related Work and Intellectual Lineage

DAC sits in a tradition of work that strips domain vocabulary to find shared structure, without introducing new mathematical formalism beyond what the engineering audience already uses.

Brooks' essential/accidental complexity distinction [1] is the closest philosophical ancestor. The contribution here is to point at vocabulary specifically as a source of accidental complexity that operates at the level of conceptual framing, not just at the level of tools.

Pattern languages, including Alexander's original work in architecture [a-1] and Gamma et al.'s software design patterns [a-2], identify recurring structural motifs across many designs and give them names. DAC has the same instinct but pushes it further in two ways: the patterns are stripped of all domain-specific framing rather than kept at the design-pattern level, and the catalog is closed (the claim is that sixteen primitives suffice for the twelve domains examined, not that they are entries in an open list).

Information hiding (Parnas) [a-3] and abstract data types (Liskov) [a-4] articulated the principle that interfaces should hide implementation details that vary across cases. The primitives here can be read as abstract data types whose interfaces are governance invariants and whose implementations vary per domain.

Coordination languages (Linda [a-5], Reo) and process algebras (CSP [a-6], the actor model [a-7], pi-calculus) formalized concurrent and distributed computation. Several of the sixteen primitives have shape similar to operations in these formalisms: RateLimit resembles a token bucket from queueing theory; PropagationPass resembles message-passing semantics. DAC is engineering-flavored rather than algebraic; the relationship is family resemblance, not derivation.

Wing's work on computational thinking [a-8] argues that recognizing the computational structure underneath domain problems is a general intellectual skill. DAC is a specific procedure for doing that for a class of resource-governed execution systems.

Workflow management formalisms (van der Aalst's workflow patterns [a-9]) cataloged control-flow patterns across many process-management systems. The DAG workload collapse in Section 7 overlaps with that work.

Category theory provides a formal language for structure-preserving mappings between domains. Some of the cross-domain identifications in this paper could be formalized as functors or as adjunctions. The paper does not develop that formalization because the engineering audience for whom the methodology is most useful does not typically reach for it. Categorical formalization is a worthwhile direction for future work.

The contribution of this paper has four parts: a description of DAC as a named, repeatable process for recognizing structural similarity across domain boundaries; a specific catalog of sixteen primitives that came out of applying DAC to twelve engineering domains; formal specifications and verification of each primitive plus the decomposition-failure verifications and the LeanFormer composition checks; and a worked example (LeanFormer) showing that the catalog is sufficient to design a non-trivial system end-to-end.

### 1.5 The Turing Tarpit Question

A reasonable objection to any claim of cross-domain structural similarity is the Turing Tarpit argument: because all general computation is Turing-equivalent, of course any computational pattern can be mapped to any other, and the mapping is operationally useless if pursued far enough. Reduce far enough and everything is NAND gates.

The defense here is empirical rather than theoretical. The sixteen primitives were not designed top-down from a theory of computation. The starting point was three automation domains (ETL, SOAR, configuration management) known in their respective communities to be expressible as workloads over a streaming DAG. The original goal was practical: build an orthogonal kernel with a composable module system that could serve all three from one execution engine. As I worked through additional domains, asking how each worked underneath its vocabulary, the same shapes kept appearing.

Rendering came in via orchestration. Once I noticed that orchestration is substrate-agnostic (a game engine orchestrating rendering, physics, and audio subsystems at per-tick granularity is doing the same thing as a Kubernetes scheduler placing pods or a hypervisor placing VMs), rendering was pulled into the analysis as an orchestration domain operating under tighter constraints. The level-of-detail (LOD) system in rendering shared shape with OSPF priority propagation in routing. Both are budget-bounded selections of what to compute or synchronize given a partial view of the system. That parallel forced the addition of QualityHierarchy and TraversalEngine, plus CompetitiveSelection: primitives the original three automation domains did not require. Each subsequent domain either mapped onto existing primitives or forced new primitives to be added when the existing set was genuinely insufficient.

If the primitives were too low-level (register operations, logic gates), the collapse would have produced hundreds of primitives per domain pattern. If they were too high-level (domain-specific abstractions), no shared primitives would have emerged across domains. The fact that sixteen primitives sufficed for twelve domains, discovered through incremental domain analysis rather than designed to fit a target count, is the evidence that the decomposition level is in roughly the right place. It is not proof of optimality.

### 1.6 Contributions

The paper makes the following contributions:

1. A description of Domain Abstraction Collapse as a named, repeatable design methodology with defined steps, a working definition of what counts as an abstraction primitive, and a sufficiency criterion.
2. A catalog of sixteen primitives produced by applying DAC to twelve engineering domains, with the genuine collapses and renamings called out honestly.
3. Identification of the Competitive Selection family as parameterized rather than monolithic. The original primitive set had collapsed seventeen domain patterns into a single primitive; treating the family as three parameterized modes (hard, soft, ranked) resolves that overcounting.
4. LeanFormer, a transformer architecture designed end-to-end through DAC by a non-ML practitioner, with measurements at 39M parameters across 119 tests and a 204M-parameter governance-machinery validation across 7,228 training steps.
5. An articulation of DAC as a constructive methodology that converts domain-notation computations into build plans composed of primitives.
6. Demonstration that the same primitive set can govern a system's architecture, its training, evaluation, knowledge management, and deployment, validated at 204M parameters with all governance invariants holding throughout the run.
7. TLA+ specifications for all twenty-one primitive specs, eighteen decomposition-failure specs, and five LeanFormer compositions, verified by TLC across roughly 44.6M states for the universal primitive verification and 178K additional states for the LeanFormer case study. The verification establishes that primitive invariants hold under all reachable states at the tested bounds, that compositions preserve those invariants, and that decomposition into separable check-then-act steps systematically introduces TOCTOU windows that violate the invariants. A B=0 initialization bug observed during the 204M run was reproduced as a 2-state TLC counterexample, and a phase-aware fix was verified across roughly 18.6 million states.

---

## 2. The Collapse: Twelve Domains, Sixteen Primitives

### 2.1 The Domains Examined

The following twelve engineering domains were subjected to abstraction collapse. They were not selected from an a priori list. They emerged as the application domains encountered during the design of a general-purpose execution framework. SOAR automation came first; each subsequent domain pulled in adjacent material.

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
| 12 | Machine Learning | Final test: generative application of the methodology to an unfamiliar domain |

### 2.2 The Abstraction Primitive Set

After abstraction collapse across all twelve domains, sixteen primitives covered the patterns: operations that could not be further decomposed without losing governance semantics, and that appeared in multiple domains under different vocabulary.

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

A note on atomicity. Several compositions in Sections 2.4 and 8 require that a sequence of primitive operations execute atomically (atomic write to destination, atomic exchange between two Budget instances). Atomicity is treated as a runtime requirement that the primitive set assumes, not as an additional primitive. The TLA+ specifications model atomicity by treating each primitive operation as a single transition. In implementation, ensuring atomicity is the responsibility of the runtime (transaction manager, software transactional memory, or equivalent). Adding Transaction as a primitive would either be redundant with that or would expand into the territory of distributed transaction protocols that are out of scope.

### 2.3 The Primitive Trait Signature

The structural skeleton of a primitive, in Rust trait syntax for concreteness:

```rust
trait AbstractionPrimitive {
    type Input;
    type Output;
    type Invariant;

    fn apply(&self, input: Self::Input) -> Result<Self::Output, InvariantViolation>;
    fn invariant(&self) -> Self::Invariant;
}
```

The shape is uniform: a primitive takes a typed input, produces a typed output (or an invariant violation), and exposes its governance invariant. The trait is realized differently per primitive (Budget's invariant is a numeric ceiling; ConvergenceGovernor's is a state-machine property; AuditSink's is a hash-chain property). Each primitive is a self-contained unit that can be composed with others without leaking its governance to its caller.

### 2.4 Domain Patterns Reduced to Primitive Compositions

The following tables document the reduction of common domain patterns to primitive compositions. They are not exhaustive. They are representative of the kind of mapping the methodology produces. Mappings that are renamings rather than informative collapses are flagged.

Real-Time Rendering:

| Domain Abstraction | What It Is |
|---|---|
| Visibility buffer | CompetitiveSelection (hard): triangles compete for pixel ownership, depth is the scoring function |
| Deferred shading | ActuationPass over the AllocationSnapshot from the visibility CompetitiveSelection |
| Level of detail (LOD) | QualityHierarchy + TraversalEngine + AllocationSnapshot under Budget\<Bytes\> |
| Frustum culling | Reduction over RelationshipGraph (spatial partition) (largely a renaming) |
| Global illumination | PropagationPass + ConvergenceGovernor under Budget\<Microseconds\> |
| Temporal amortization | Budget\<TimeSlot\> + RateLimit + ConvergenceGovernor (feedback) |

Physics Simulation:

| Domain Abstraction | What It Is |
|---|---|
| Broad-phase collision | Reduction over RelationshipGraph (spatial partition); see Section 4.3 for why this is not CompetitiveSelection |
| Narrow-phase collision | CompetitiveSelection (hard): contact pairs compete for resolution priority |
| Constraint solver | PropagationPass + ConvergenceGovernor: iterative relaxation of constraint forces |
| Integration step | ActuationPass: apply force-derived velocity changes to position |

Audio Spatialization:

| Domain Abstraction | What It Is |
|---|---|
| Spatial audio mixer | PropagationPass + Budget\<Voices\>: sound propagates over scene graph under voice budget |
| Audio LOD / culling | QualityHierarchy + TraversalEngine: distant sources at lower fidelity |
| Reverb buses | RelationshipGraph + PropagationPass: signal flows over a routing graph |
| Voice priority | CompetitiveSelection (ranked): voices compete for the K available channels |

Network Replication:

| Domain Abstraction | What It Is |
|---|---|
| Interest management | CompetitiveSelection (ranked): entities compete for relevance to each viewer |
| Bandwidth budget | FederatedBudget\<BytesPerSecond\>: master bandwidth subdivided per viewer |
| Delta compression | Reduction\<(Old, New) → ChangeRecord\>: aggregate state diffs |
| Reliable channels | RateLimit + ConvergenceGovernor: bound retransmits, converge on acknowledgment |

Container Orchestration:

| Domain Abstraction | What It Is |
|---|---|
| Pod scheduling | CompetitiveSelection (hard): nodes compete for pod placement, fit is the scoring function |
| Resource quotas | FederatedBudget\<CPU/Memory\>: cluster master subdivided per namespace |
| Liveness probes | Signal\<Healthy\> + RateLimit: periodic check at governed rate |
| Reconciliation loop | ConvergenceGovernor + ActuationPass: detect drift, drive toward desired state |

CI/CD Pipelines:

| Domain Abstraction | What It Is |
|---|---|
| Pipeline DAG | RelationshipGraph\<Stage, Dependency\> + TraversalEngine |
| Artifact caching | ResourceRegistry\<BuildInput, BuildOutput\> + Budget\<CacheCapacity\> with content-hash invalidation |
| Parallel test execution | Budget\<Compute\> + Sampler\<TestPartition\>: budget-constrained parallel work |

ETL / Data Pipelines:

| Domain Abstraction | What It Is |
|---|---|
| Extract stage | ActuationPass: I/O-gated data retrieval from a ResourceRegistry (largely a renaming) |
| Transform stage | ActuationPass: map function over data (largely a renaming) |
| Load stage | ActuationPass + Budget\<DestinationCapacity\> + ResourceRegistry\<Key, Record\> with atomic write boundary |
| Backfill | TraversalEngine over a temporal QualityHierarchy: budget-constrained reprocessing |

Configuration Management:

| Domain Abstraction | What It Is |
|---|---|
| Desired state convergence | ConvergenceGovernor: detect current vs. desired state delta, iterate until converged |
| Idempotent operation | ActuationPass with Reduction\<(Old, New) → ChangeRecord\>: only actuate if delta is non-zero |
| Role/playbook | RelationshipGraph\<Task, Dependency\> + TraversalEngine: a DAG workload |
| Inventory | ResourceRegistry\<Host, Configuration\> (largely a renaming) |

SOAR (Security Orchestration, Automation, and Response):

| Domain Abstraction | What It Is |
|---|---|
| Playbook | RelationshipGraph\<Action, Dependency\> + TraversalEngine: a DAG workload with capability-gated I/O, structurally the same shape as CI/CD pipelines, ETL workflows, and configuration management playbooks |
| Alert triage | CompetitiveSelection (ranked): alerts compete for analyst attention seats, prioritized by severity score |
| Enrichment | ActuationPass + ResourceRegistry: look up context from external sources (largely a renaming) |
| Response action | ActuationPass with capability-based access control |

Distributed Consensus (RAFT):

| Domain Abstraction | What It Is |
|---|---|
| Leader election | CompetitiveSelection (hard): nodes compete for the leader seat |
| Log replication | PropagationPass: leader propagates log entries to followers until majority acknowledgment |
| Heartbeat | Signal\<LeaderAlive\> + RateLimit: periodic notification at governed rate |
| Commit | Reduction\<Acknowledgment, Count\> + Budget\<CommitSlot\> + ResourceRegistry\<TxnID, Record\> with atomic commit boundary |

Economic Simulation:

| Domain Abstraction | What It Is |
|---|---|
| Market order matching | CompetitiveSelection (hard): buy orders compete for sell-order seats, price is the scoring function |
| Price discovery | PropagationPass + ConvergenceGovernor: iterative relaxation toward equilibrium price |
| Inventory management | Budget\<ItemSlots\> + ResourceRegistry\<ItemID, Slot\>: multi-resource updates under capacity constraints with atomic boundary |
| Trade | Budget\<U_source\> + Budget\<U_target\>: atomic exchange between two Budget instances |

Machine Learning (extended discussion in Section 6):

| Domain Abstraction | What It Is |
|---|---|
| Attention mechanism | CompetitiveSelection (soft): tokens compete for attention weight, similarity is the scoring function, softmax produces weighted combination [2] |
| Backpropagation | PropagationPass (single-pass variant): reverse message passing on the computation graph (precision caveat in Section 6.1) |
| Speculative decoding | Two-level CompetitiveSelection: fast model generates candidates (coarse), slow model verifies (fine) [8] |
| Beam search | CompetitiveSelection (ranked): candidate continuations compete for K beam seats |
| KV cache | ResourceRegistry\<SequencePosition, (Key, Value)\> + Budget\<CacheCapacity\> with position-based invalidation |
| Learning rate scheduling | Budget\<TimeSlot\> + RateLimit + ConvergenceGovernor (feedback): PID-like control governing a resource (learning rate) over time |
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

---

## 3. Why the Pattern Is Hard to See

### 3.1 Vocabulary as Cognitive Lens

The domain-specific vocabulary that makes abstraction collapse difficult to detect is not arbitrary. It evolved because the people solving each problem came from that domain. Each domain's vocabulary grew up among the people working in it: graphics engineers in rendering, network engineers in routing, ML researchers in statistical learning, audio engineers in spatialization. The names are internally coherent within each domain and rarely cross over.

Each vocabulary is internally coherent and useful within its domain. The side effect is that vocabulary creates cognitive boundaries. A graphics engineer who thinks in terms of visibility buffers and deferred shading does not necessarily recognize that these are instances of the same evaluate-then-actuate pattern that governs Kubernetes pod scheduling. An ML researcher who thinks in terms of attention and softmax does not necessarily recognize that the attention mechanism shares structure with the competitive selection pass that determines which triangle owns each pixel.

The vocabulary focuses attention on domain-specific details and tends to hide the structural similarity to other domains. Recognizing the similarity requires deliberately stripping the domain words and looking at what remains.

### 3.2 The Specialization Trap

Software engineering culture rewards specialization. Rendering engineer, systems engineer, ML engineer, security engineer: each is a different job title, conference community, and publication venue. Solutions published at SIGGRAPH use rendering vocabulary; the same kind of structural pattern published at NSDI uses networking vocabulary, and at NeurIPS it shows up in ML vocabulary. Cross-domain references do happen, but they tend to happen through analogy ("attention is like a soft dictionary lookup") rather than through explicit identification ("attention shares structure with competitive selection under a specific scoring function").

Analogy preserves the domain boundary. Explicit identification reduces it. The latter is operationally more useful but is less common in practice because the incentives of specialized publication venues do not reward it.

### 3.3 The Path I Took

DAC was not developed by a rendering engineer, a networking engineer, or an ML researcher. My background is security and Linux systems engineering and automation. The methodology grew out of a practical problem. I was working across three automation domains (ETL, SOAR, configuration management) that the relevant communities already knew were expressible as workloads over a streaming DAG, and I wanted to build a single execution engine for all three rather than maintain three vocabularies and three plugin ecosystems. That meant deliberately looking for the structural operations all three shared, separated from the domain-specific framing each tool puts on those operations. The intent from the start was to identify automation primitives that could compose into an orthogonal kernel.

The attempt to build a better execution engine began with modeling it after well-understood systems-engineering patterns. An OS kernel handles resource governance. An OSI stack handles layered abstraction. Unix pipes handle composable data flow. Capability-based access control (Lampson) handles authority delegation. These were not novel intellectual ingredients. They were proven patterns from systems engineering applied to a problem I was already trying to solve.

The methodology grew from there. The first major expansion came from orchestration. I noticed that orchestration is substrate-agnostic. A container scheduler placing pods on nodes, a hypervisor placing VMs on hosts, an OS scheduler placing processes on cores, and a game engine running its rendering, physics, audio, and AI subsystems at per-tick granularity are all doing the same thing structurally: allocating bounded resources to consumers under priority and capacity constraints, on a clock. The substrate is not the operation. That recognition pulled rendering into the analysis, because game engines run orchestration internally at very high frequency and very tight constraints, which made them a useful stress test of the same primitives.

Once rendering was on the table, the cross-domain parallels came quickly. The level-of-detail (LOD) system in rendering chooses, for each piece of geometry visible to the camera, what level of fidelity to render at. This looked structurally similar to OSPF's link-state propagation, where each node's priority calculation determines which neighbors should have synchronized state based on topology and route value. The shape underneath both is the same: given a bounded budget and a partial view of the system, what should be computed or synchronized first? That parallel forced QualityHierarchy into the primitive set (the LOD tree, the OSPF area hierarchy), along with TraversalEngine (the budget-bounded walk that visits the prioritized subset) and CompetitiveSelection (which triangle owns each pixel; which path wins each route).

That progression through orchestration into rendering and then routing is where the methodology stopped feeling like ad hoc cross-domain analogy and started feeling like a process worth naming. I kept asking the same question of each new domain ("how does this *really* work?") and kept getting answers that mapped onto operations I had already named. At some point the abstractions had grown enough teeth that they were worth writing down. DAC is the name I gave, retroactively, to a process I had been doing for some time without naming it.

The path that led to DAC was not "an outsider noticed something insiders missed." Many of the individual cross-domain observations in this paper have been made before by people in those domains. ECS-as-database has been observed in game-development circles. Attention-as-kernel-lookup is in the original Transformer paper's framing. Backpropagation-as-message-passing is the standard view in graph-based autodiff. Configuration management as fixed-point convergence has been formalized in academic work on Puppet and Chef. What I did was apply the same vocabulary-stripping process across enough domains that the pattern of patterns became visible, and then specify and verify the resulting primitive set formally. The contribution is the synthesis and the verification, not the individual observations.

The reason I was the one to do this synthesis, rather than someone with deeper domain expertise in any single area, is contingent. I started with a goal (a single kernel for three automation domains) that required finding shared structure, and I kept asking how each new domain I encountered worked rather than accepting its vocabulary at face value. That made the cross-domain similarities easier to notice. A domain expert with the same goal would likely have arrived at a similar set.

---

## 4. The Competitive Selection Family

### 4.1 The Overcounting Problem

In the original formulation, a single primitive called SlotArbitrationPass was mapped to over seventeen domain patterns: visibility buffers, attention, pod scheduling, market order matching, alert triage, beam search, leader election, broad-phase collision, BGP path selection, speculative decoding, audio priority, and more. When one primitive absorbs that many structurally different operations, it raises a legitimate concern: is the primitive defined so broadly ("anything where something competes for something") that it approaches the Turing tarpit the methodology claims to avoid?

The answer is that there is genuine shared structure here, but it is a family of related primitives rather than a single primitive. The shared structure is: given a set of output positions (seats) and a set of candidates, evaluate each candidate against each seat using a scoring function, and allocate candidates to seats based on the scores. What differs across the family members, and what matters operationally, is the selection mechanism.

### 4.2 Three Selection Modes

The Competitive Selection family decomposes into modes that share the scoring interface but differ in their allocation semantics.

Hard Selection (argmax) gives exactly one winner per seat. The candidate with the highest score takes the seat exclusively. All other candidates receive nothing. This is the mode used in pixel ownership (visibility buffer), leader election (RAFT), pod scheduling, market order matching, and BGP path selection. The key property is mutual exclusion: a seat is owned by exactly one candidate.

Soft Selection (softmax/weighted) gives every candidate a contribution to every seat in proportion to its score. There is no single winner. The output for each seat is a weighted combination of all candidates. This is the mode used in transformer attention [2]. The key property is proportional allocation: candidates share seats continuously rather than claiming them exclusively.

Ranked Selection (top-k) gives the top K candidates by score all allocated. This is the mode used in beam search, audio channel priority, network update priority, and alert triage. The key property is bounded multiplicity: multiple winners are allowed but the count is capped.

Every mode requires a scoring function that evaluates candidate-seat affinity, produces an AllocationRecord that maps seats to allocated candidates, and composes with ActuationPass and Budget\<U\> the same way. The differences are parameterizations of the selection mechanism. Differentiability differs across the modes (softmax is differentiable, argmax is not), but differentiability is a property of the scoring/allocation function the domain supplies, not a property of the primitive's structure. The same applies to Reduction parameterized by an associative operator: sum, max, and mode have different differentiability properties yet share a primitive. Treating hard, soft, and ranked as modes of one family is consistent with how the rest of the primitive set handles function variation.

### 4.3 What Is Not Shared

Two patterns originally mapped to this family deserve separate scrutiny.

Anomaly detection (described in some formulations as "not winning the normal slot") is a stretch. Anomaly detection is better characterized as a Reduction (compute a distance metric from a reference distribution) followed by a threshold comparison. Forcing it into the competitive selection frame adds vocabulary without adding structural insight.

Broad-phase collision detection is often described as "AABB overlaps competing for pair-slots," but it is more precisely a spatial partitioning and filtering operation. The competitive selection mapping is defensible (candidate pairs compete for a limited processing budget in a real-time physics pipeline), but it is a weaker match than pixel ownership or attention, where the competitive structure is intrinsic rather than imposed by the resource constraint. The collapse table in Section 2.4 classifies broad-phase collision as Reduction + spatial partitioning rather than CompetitiveSelection.

The revised count: CompetitiveSelection with its three modes accounts for roughly fifteen of the seventeen original mappings, with two entries better classified as compositions of other primitives.

---

## 5. DAC as a Constructive Methodology: The LeanFormer Artifact

### 5.1 What This Section Is and Is Not

The methodology described in Section 1.2 is framed analytically: take existing domains, strip vocabulary, find similarities. The same process works constructively. When confronted with a problem that resists solution in its native vocabulary, strip the vocabulary and check whether the structure matches a primitive or composition of primitives. If it does, the problem may inherit a solution from whichever domain has already worked through it.

LeanFormer is the worked example. I want to be direct about what it is and is not, because the framing matters.

LeanFormer is an artifact of applying DAC to a domain where I do not have research training. My background is security and Linux systems engineering and automation. I am not an ML engineer. The work tested whether a non-expert applying DAC carefully could produce a workable architecture for problems the ML literature treats as open. The result is a 204M-parameter transformer composed entirely of the sixteen DAC primitives, where every primitive invariant held empirically through 7,228 training steps with zero governance violations.

What this section establishes:

- The architecture composes from the sixteen primitives.
- The training run completed without governance invariant violations.
- The components I derived from DAC decomposition correspond closely to recognizable ML techniques (LoRA-style low-rank deltas, sparse attention, hierarchical training).

What this section does not establish:

- That LeanFormer matches or beats any baseline on language modeling capability.
- That the belief delta system protects knowledge better than vanilla LoRA.
- That governed training is more efficient than ungoverned training in wall-clock terms at this scale.
- That the confabulation signal correlates with output accuracy.

Comparative baselines on each of these questions are open work for which I have concrete experimental designs in Section 9.5. They require funding I do not currently have.

This is a paper about DAC. LeanFormer is presented as the artifact of a non-expert applying DAC to ML, and the convergence between independent DAC-derived architecture and established ML practice is the rediscovery signal I want to draw attention to. I am not trying to empirically prove LeanFormer in this paper.

### 5.2 The Six Problems and the Decomposition Process

I started by listing six problems in neural network design that came up while thinking about how transformers actually work: parameter inefficiency, attention cost, catastrophic forgetting, knowledge composition, confabulation, and training process inefficiency. I did not start from the ML literature. I did not know LoRA existed when I started. I decomposed each problem in DAC vocabulary first and only checked the literature after the architecture was in place.

For each problem the process was the same: describe the problem without ML vocabulary, map the structural description to the primitive set, check whether the resulting composition has a counterpart in another domain.

Total time on the LeanFormer work, including initial DAC decomposition through 204M-parameter measured results, was roughly three weeks. The initial proof-of-concept (7.5M parameters, 4 layers) was designed and implemented in 24 hours using DAC as the design methodology and Claude Code as the implementation agent. The architect provided the DAC decomposition, the cross-domain mappings, and the verification discipline. Claude Code produced all implementation code. No deep ML implementation experience was required on either side of this collaboration; the structural solutions were systems engineering solutions identified through vocabulary stripping. The domain functions (loss formulation, optimizer choice, scoring functions) still required ML knowledge, and those parts of the work involved more iteration and external reference than the structural decomposition did.

Subsequent scale-up to 39M parameters (76M dense equivalent, trained on 500K OpenWebText samples) covered 119 tests. A 204M parameter model (805M dense equivalent) was trained with the full governed pipeline on a reasoning corpus for 7,228 optimizer steps (one epoch) on an NVIDIA L4 GPU. The architectural results in Sections 5.3-5.7 are from the 39M run. The training governance results in Section 5.8 include both the 4.8M preliminary run and the 204M scale run.

### 5.3 Problem 1: Parameter Inefficiency

In ML vocabulary: dense transformer parameters are mostly redundant. Many heads attend to similar things; many feed-forward neurons activate together; weight matrices have effective rank far below their nominal dimension.

Stripped of vocabulary: a system that allocates a fixed compute budget uniformly to consumers without regard for whether each consumer is producing distinct work.

DAC decomposition: Budget\<Parameters\> with low-rank factorization. Apply Budget to enforce a parameter ceiling and use rank-decomposed weight matrices to keep the parameter count below the ceiling while preserving expressive capacity. This is rediscovery of the LoRA pattern [6] applied not as a fine-tuning adapter but as a primary architectural choice.

Result at 39M parameters: 3.9x parameter ratio against dense equivalent (76M dense capacity in 39M trainable parameters). Same ratio held at 204M (3.94x).

### 5.4 Problem 2: Attention Cost

In ML vocabulary: full attention is O(N²) in sequence length. Standard mitigations (linear attention [10], sparse attention) trade quality for compute.

Stripped of vocabulary: a competitive selection pass over N candidates against N seats produces N² candidate-seat affinity scores, but most of those scores fall below any meaningful threshold. The full evaluation is wasted on positions that will not contribute meaningfully to the output.

DAC decomposition: CompetitiveSelection (soft) with a learned gating function that excludes low-affinity candidate-seat pairs from the attention computation entirely. This is Sampler\<AttentionMask\> + CompetitiveSelection (soft), where the sampler reduces the candidate set before scoring. Rediscovery of sparse attention.

Result at 39M parameters: 88% attention sparsity. The capability impact of the sparsity has not been measured against dense baseline (Section 9.5).

### 5.5 Problem 3: Catastrophic Forgetting

In ML vocabulary: training a model on task B after task A degrades task A performance because the same parameters are updated for both. The original capability is overwritten.

Stripped of vocabulary: a system updates a shared resource for two distinct purposes without any architectural separation between the resources used for each purpose.

DAC decomposition: ResourceRegistry\<BeliefID, DeltaWeights\> + Budget\<Parameters\> + AuditSink. Frozen base weights serve as the structural backbone; new knowledge is stored as additive low-rank deltas indexed by belief ID. Belief injection is an atomic registry insertion. Belief removal is an atomic registry deletion that restores the base weights bit-for-bit. This is structurally close to LoRA adapter stacking, with one architectural difference: the registry enforces non-overlapping parameter address ranges across deltas, so two beliefs cannot interfere through shared parameter slices.

| Pattern | Composition |
|---|---|
| Frozen base | ResourceRegistry\<TensorID, FrozenWeights\>, immutable after init |
| Belief delta | Budget\<Parameters\> + ResourceRegistry\<BeliefID, DeltaWeights\> with atomic write boundary |
| Routing layer | CompetitiveSelection (ranked) over belief deltas |
| Combined inference | Base computation + active deltas applied additively |
| Belief removal | Delete from ResourceRegistry, restore base weights bit-for-bit |

Result at 39M parameters: 84% belief injection success across 100 beliefs. Bit-for-bit base weight restoration on belief removal across all 100 beliefs and 406 base tensors. The architectural separation removes the shared-parameter interference channel by construction. Whether it produces less belief-to-belief interference than vanilla LoRA stacking in practice is the open empirical question (Section 9.5).

### 5.6 Problem 4: Knowledge Composition

In ML vocabulary: combining multiple pieces of knowledge into a coherent answer is an open problem. Mechanistic interpretability work [12] suggests that knowledge is distributed across attention heads in ways that are difficult to compose deliberately.

Stripped of vocabulary: a routing system needs to combine information from multiple address ranges into a single output, where the relevance of each range to the query is determined by a similarity comparison.

DAC decomposition: a semantic routing layer that uses CompetitiveSelection (ranked) to identify which belief deltas are relevant to a given input, scored by cosine similarity between the input embedding and each delta's address vector. Selected deltas are activated for the forward pass. Unselected deltas remain dormant.

Result at 39M parameters: 86% routing accuracy on a held-out test set against a 5-way clustering of 100 beliefs (chance ~20%). 64% of beliefs improve simultaneously without interfering with each other.

### 5.7 Problem 5: Confabulation

In ML vocabulary: language models produce outputs that look fluent and confident but are factually wrong. The model output carries no signal that distinguishes high-confidence from low-confidence cases. There is no architectural correlate of uncertainty.

Stripped of vocabulary: a process that produces an output without converging requires a way to signal that it has not converged, separately from the output itself.

DAC decomposition: an adaptive depth mechanism with a per-layer ConvergenceGovernor. At each layer, the governor evaluates whether further computation is likely to change the output. If converged, exit early. If full depth is reached without convergence, output a confidence flag indicating non-convergence. The flag is the structural correlate of "I don't know."

Result at 39M parameters: mean exit depth of 11.3/12, with two distinct exit depths used. This is a partial result. The adaptive depth mechanism is not yet being used aggressively, which suggests the exit threshold needs tuning or that explicit layer-dropping training is needed before the model develops early-exit behavior. At 204M, the mechanism did not activate at all (20/20 throughout). The architectural primitive composition is in place; whether the resulting confidence signal correlates with output accuracy is an empirical question this paper does not answer (Section 9.5).

### 5.8 Problem 6: Training Process Inefficiency

In ML vocabulary: standard training applies the same rules everywhere. Every parameter receives gradient signal from every sample regardless of relevance. Every metric is evaluated at fixed intervals regardless of what changed. The result is computationally expensive and treats the system as uniform when it is not.

Stripped of vocabulary: a resource-allocation system that distributes a finite budget (compute) uniformly across consumers (parameter groups) regardless of which consumers are still actively producing useful work. This is what Kubernetes pod scheduling, Linux process scheduling, and rendering LOD systems address: resource governance with variable allocation based on observed need.

DAC decomposition: apply the same primitive set that governs the model's architecture to govern the training process itself. The full composition uses a FederatedBudget\<GradientCompute\>, a master gradient compute budget subdivided per parameter group with per-group allocation that adapts to observed convergence state. Parameter groups are organized by QualityHierarchy into levels (L0 structural, L1 representational, L2 refinement, L3 specialization), so coarse levels train first and finer levels activate via convergence signal. Each parameter group has its own four-state ConvergenceGovernor (PENDING, ACTIVE, COOLING, CONVERGED, with AWAKENED for reactivation after perturbation). CompetitiveSelection (ranked) over gradients gives the gradient router a way to score each (sample, parameter group) pair and gate the gradient signal to the most relevant pairs. When a group converges, downstream subsystems receive the event through Signal\<GroupConverged\>. AuditSink logs every governance decision (budget allocation, state transition, hierarchy activation, gradient routing update) to a hash-chained log with SHA-256 integrity.

Preliminary result at 4.8M parameters (300-step validation run): all governance components activated and composed correctly. Per-group convergence governors produced 15 state transitions across 8 parameter groups, with 7 reaching CONVERGED state. The quality hierarchy activated in the predicted order (L0, then L1, then L2, then L3), all via convergence signal with no emergency activation required. The FederatedBudget invariant (sum of allocations <= master budget) held for all 300 steps with zero violations. The gradient router achieved 15.4% selectivity post-warmup. The SHA-256-chained audit log verified across all 300 records. The change-triggered evaluation pipeline fired 7 targeted evaluations on convergence signals. Final loss was within +2.4% of baseline, with the gap narrowing throughout training. At 4.8M parameters, the embedding table dominates (~86% of parameters), so this run primarily tested whether the governance machinery could be wired up correctly, not whether it would produce different behavior at scale.

Scale validation at 204M parameters (805M dense equivalent, 7,228 optimizer steps, one epoch on a reasoning corpus, NVIDIA L4 GPU, 140.9 hours wall clock) tested whether the governance primitives compose correctly at a scale where parameter group ratios are representative.

The FederatedBudget invariant (sum of allocations <= 1.0) held for all 722 audit records with zero violations. Budget adapted dynamically throughout training: L0 groups started at 0.25 each, converged groups dropped to as low as 0.012, and active groups received up to 0.40 of the total budget.

The convergence governor four-state machine produced 18 total state transitions across 8 parameter groups, all valid: PENDING to ACTIVE (3 transitions), ACTIVE to COOLING (8), COOLING to CONVERGED (4), and COOLING to ACTIVE (1 regression). No states were skipped.

The SHA-256 hash-chained audit log maintained integrity across all 722 records from step 10 to step 7,220, with zero chain breaks.

Best validation perplexity reached 57.6 at step 2,000, with train loss declining from 10.39 to 1.64 across the full run. Severe overfitting occurred after step 2,000. This is expected behavior from single-epoch training with limited regularization (dropout 0.1 only). All governance invariants held throughout the overfit phase: the budget was never exceeded, the hierarchy ordering was maintained, the convergence governor never skipped a state, and the audit chain was never broken. Governance correctness held independently of generalization quality.

Orthogonal capacity measurement confirmed 53,760 available dimensions across 20 layers (2,688 per layer), with a theoretical maximum of 3,360 rank-16 knowledge deltas. This was measured on two independent machines (NVIDIA L4 on GCP, RTX 3060 locally) with identical results.

Two governed subsystems did not produce meaningful signal at this scale. Adaptive depth remained at 20/20 throughout the entire run. Tiered sampling scored all samples as "Failing" at initialization and was never re-scored. Both require further work in future training runs.

The "primitives compose correctly" claim requires care. What was directly measured is that each primitive's local invariant held throughout the run. This is consistent with the primitives composing correctly, but it does not by itself establish emergent system-level guarantees that go beyond the per-primitive invariants. The composition-level invariants verified in TLA+ (Appendix C) do address composition properties. The empirical run confirms those invariants are not violated in practice at 204M parameters and 7,228 steps. The accurate framing is that every primitive's invariants held empirically and the TLA+ verification covered the composition-level invariants formally.

### 5.8.1 The B=0 Initialization Episode

The most informative result from the 204M training run was a failure mode that DAC's vocabulary-stripping process turned out to apply to.

LeanFormer's low-rank layers initialize B matrices to zero (following standard LoRA practice), which means all parameter groups begin with near-zero gradient flow regardless of whether they have received meaningful training signal. The convergence governors correctly detected low gradient EMA and transitioned through the state machine as specified: ACTIVE to COOLING after the configured cooling window. This produced hierarchy activations at steps 200 (L1) and 400 (L2), both at round-number intervals aligned with the cooling window configuration. Loss remained flat at 10.388 through both activations. Real training progress began only when the output head activated at L2 and introduced significant gradient flow through the network.

The L3 activation at step 2,773 was qualitatively different. It occurred at a non-round step number, after the attention_output and ff_projections groups (L1 parameters) had received 2,300+ steps of real gradient flow following the output head's activation. The L3 activation was driven by post-learning convergence in those groups, not by a calibration artifact from initialization.

Applying the vocabulary-stripping process to this failure described it as an observational degeneracy: two qualitatively different gradient trajectories (cold start and post-learning convergence) produced the same low-magnitude reading, and the governor could not distinguish them from the magnitude alone. The same shape shows up in several other domains. A rendering pipeline cannot tell a background pixel from a black surface using color alone, so the depth buffer carries an additional signal that resolves the ambiguity. A networking stack cannot tell an idle node from a crashed one without a liveness signal, which is what heartbeat protocols provide. Distributed consensus cannot tell a slow voter from a partitioned one without a temporal boundary, so timeouts impose one. Database concurrency control cannot tell a stale read from a current one without a version number, which is what MVCC supplies. The fix has the same shape in every case: a second signal that breaks the observational tie.

A phase-aware ConvergenceGovernor tracks whether gradient magnitude has ever exceeded the threshold and classifies gradient trajectories into qualitative phases (COLD, WARMING, ACTIVE_LEARNING, DECLINING). The ACTIVE-to-COOLING transition requires the gradient phase to be ACTIVE_LEARNING or DECLINING, never COLD. This NoCoolingFromCold invariant was specified in TLA+ and verified by TLC across roughly 18.6 million states. A specification without phase awareness produces a counterexample matching the exact failure observed in the training run in just 2 states.

Two honest observations. The bug class (zero-as-ambiguous-signal) is well-understood, and the fix (a latching phase tracker) is a standard hysteresis pattern. The contribution of DAC here is not the discovery of a new engineering technique. It is that the vocabulary-stripping process surfaced the connection to the cross-domain pattern, which gave a direct route from "this is the kind of bug it is" to "this is the kind of fix that works." A skilled control engineer or ML researcher could have arrived at the same fix without the methodology; the methodology made the recognition faster for someone whose training was not in either area.

The narrative structure of "the methodology applied to its own failure" is satisfying but not by itself evidence of generative power. The vocabulary stripping happened after the bug was understood; there is no controlled comparison showing DAC produced the diagnosis faster than direct ML reasoning would have. What can be claimed is that the diagnosis route was available, was used, produced a fix, and the fix has been formally verified.

### 5.9 Summary of LeanFormer Measurements

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

### 5.10 What DAC Did Not Provide

DAC does not design domain functions. The specific choice of low-rank factorization for the deltas, the cosine similarity metric for the routing network, the exit-threshold tuning for adaptive depth, the choice of loss function and optimizer: these are domain-specific engineering decisions that require ML expertise. The same applies to the training governance system. The specific scoring function for the gradient router, the convergence thresholds for per-group governors, the budget allocation policy, the hierarchy level boundaries, and the sample difficulty thresholds are all domain functions that DAC's structural skeleton does not supply.

DAC provided the structural skeleton. Domain knowledge filled in the scoring functions, the loss formulations, the training recipes, and the governance thresholds. This is the structure/function separation described throughout the paper. The abstraction primitives provide structure (how data flows, how resources are allocated, how convergence is detected, how mutations are audited). The domain provides function (what computation to apply at each step). Neither replaces the other.

---

## 6. The AI Domain Mappings

### 6.1 Three Mappings, with Precision Caveats

The most far-reaching application of DAC in this paper is to machine learning. The transformer architecture [2] can be expressed as a composition of primitives that the methodology had identified before the AI domain was examined.

Attention is soft competitive selection. The transformer's multi-head attention mechanism computes, for each query token, a weighted combination of value vectors where the weights are determined by similarity between the query and key vectors. This is CompetitiveSelection in soft mode: the dot-product similarity is the scoring function, softmax produces the weighted allocation, and each attention head is one selection pass. Multi-head attention is parallel selection passes over the same candidates with different scoring functions.

Backpropagation is a single-pass variant of PropagationPass. The forward pass builds a directed acyclic computation graph. The backward pass propagates gradient values from the loss node backward through the graph in reverse topological order. At each node it applies the chain rule to compute local gradients.

A precision caveat is necessary here. Backpropagation is a single reverse pass on a DAG. It does not require iterative relaxation to a fixed point the way Bellman-Ford does on graphs with cycles [3]. The mapping is not "backpropagation equals Bellman-Ford." The mapping is that both backpropagation and Bellman-Ford are instances of PropagationPass: the shared structural primitive of message passing over a graph toward a consistent state. The instantiation parameters differ (graph topology, traversal order, message function, termination condition). The primitive is the message-passing structure.

This mapping is the weakest of the three. The shared structure (messages flowing through a graph) is real but broad enough that calling both instances of the same primitive carries some risk of overclaiming. A more accurate framing is that PropagationPass is a family of operations parameterized by topology and termination, and backpropagation is one member of that family.

Speculative decoding is the two-level fidelity architecture. A fast small model generates candidate tokens (coarse pass) [8]. A large model verifies them (fine selection). Accepted tokens are actuated. Rejected tokens are discarded. This is the same shape as the rendering fidelity pipeline: coarse traversal reduces the candidate set, fine selection determines winners, actuation evaluates only winners.

### 6.2 What This Implies About Cross-Domain Optimization Transfer

These are structural mappings, with the precision caveats above. The implication is that any optimization of CompetitiveSelection, discovered in any domain, is a candidate for transfer to attention. Any optimization of PropagationPass, discovered in any domain, is a candidate for transfer to gradient computation. The shared primitive vocabulary creates a channel for cross-domain optimization transfer that is harder to see when each domain maintains its own vocabulary.

Linear attention [10], sparse attention, and flash attention can be read as optimizations of CompetitiveSelection in soft mode. Convergence governors and temporal amortization, developed for real-time lighting, are candidates for application to training loop optimization.

A candid assessment of validation status: cross-domain optimization transfer is the most useful claim DAC makes, and the evidence so far runs in one direction (systems engineering optima → ML). The LeanFormer architecture (Sections 5.3-5.7) shows five systems engineering solutions applied to model design problems. The targeted training system (Section 5.8) shows the same primitive set applied to the training process itself. The B=0 observational degeneracy diagnosis (Section 5.8.1) shows DAC's vocabulary-stripping process applied to a failure in a DAC-governed system. All three demonstrations move structural insights from systems engineering into ML. A stronger demonstration would be a transfer in the opposite direction: an optimization that originated in ML, applied to (say) rendering or consensus, with measurable improvement. That has not yet been done and is identified as future work in Section 9.5.

---

## 7. The DAG Workload Pattern

### 7.1 Four Domains, One Pattern

The most practically actionable cross-domain mapping in this work is the identification that CI/CD pipelines, ETL workflows, configuration management playbooks, and SOAR automation playbooks share a common shape.

All four are the same kind of workload: a directed acyclic graph of tasks with typed dependencies, executed under budget constraints (compute, time, or both), with capability-gated I/O at task boundaries. Execution converges toward a desired end state and is audit-logged for observability and compliance.

If this mapping is correct in practice, the consequence is that a single execution engine could replace Ansible, Jenkins, Airflow, and Splunk SOAR. Not by reimplementing four separate systems, but by recognizing that the underlying execution pattern is shared. The "domain-specific" part is the task function, which belongs in a sandboxed execution boundary, not in the kernel. Workflow management research [a-9] has cataloged similar patterns under different names; this section's contribution is identifying that the DAC primitive set covers the four named domains specifically.

### 7.2 Why This Pattern Was Hard to Notice

These four domains are served by different industries, conferences, vendor ecosystems, and job titles. CI/CD engineers use Jenkins or GitHub Actions, ETL engineers use Airflow or dbt, configuration management engineers use Ansible or Puppet, security engineers use Splunk SOAR or Palo Alto XSOAR. Each tool has its own vocabulary, plugin ecosystem, and certification program.

That layered specialization is what made the shared shape hard to see. A single practitioner rarely uses all four. The DAC process here asks the same second question of each tool: what does the engine compute, with the vocabulary stripped? The answer in every case is the same: a DAG traversal under constraints, with task execution at each node. The task function is the part that varies.

---

## 8. Composition Patterns

### 8.1 How Domains Are Reconstructed

The point of identifying primitives is constructive: once they are named, every domain pattern can be expressed as a composition. The following table documents the reconstruction of common computational patterns from the primitive set. This is a catalog of compositions, not a formal algebra.

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

In every composition above, the primitives provide the structure (how data flows, how resources are allocated, how convergence is detected, how mutations are audited). The domain provides the function (what computation to apply at each step). Several compositions require atomicity at a boundary. Atomicity is treated as a runtime requirement, not as an additional primitive (Section 2.2).

### 8.2 Structure vs. Function

The boundary between structure and function is the part of this work that requires the most care. The rendering equation is a domain function. Newton's laws are a domain function. The RAFT consensus protocol's log replication rule is a domain function. The machine learning loss function is a domain function. The attention scoring function (dot-product similarity) is a domain function.

Where the boundary gets fuzzy: the softmax inside CompetitiveSelection (soft) is a specific mathematical operation that is not part of CompetitiveSelection (hard). Treating softmax as part of the primitive's "structure" rather than as a domain-supplied function is a judgment that the differentiability and distribution-output properties are operationally important features of the soft-selection mode. This is the same kind of judgment as treating sum, max, and mode as different parameterizations of Reduction.

This separation is what makes the primitive set simultaneously lean and broadly applicable. It contains no domain knowledge. It contains the execution model that recurred across the domains examined.

---

## 9. Methodology Validation

### 9.1 How to Know the Mapping Is Real

A cross-domain mapping is meaningful (not a renaming exercise) if and only if:

1. Sufficiency: every domain pattern from step 1 can be expressed as a composition of the collapsed primitives. No residual domain-specific primitives are required.
2. Minimality: no primitive in the collapsed set can be expressed as a composition of the others. Removing any primitive leaves at least one domain pattern unexpressible.
3. Operational equivalence: a system built from the collapsed primitives produces the same outputs as the domain-specific system it replaces, under the same inputs and constraints.
4. Cross-domain transfer: an optimization discovered in one domain, when applied to the shared primitive, produces measurable improvement in other domains that use the same primitive.

Sufficiency (criterion 1) is satisfied for all twelve domains examined, with the caveat that "satisfied" means "I was able to express every domain pattern I considered as a composition," which is dependent on which patterns were considered.

Operational equivalence (criterion 3) is satisfied for the domains where working implementations exist (LeanFormer demonstrates this for the ML composition).

Cross-domain transfer (criterion 4) is partially demonstrated: the LeanFormer architecture, the targeted training system, and the B=0 diagnosis all transfer structural insights from systems engineering into ML. Transfer in the opposite direction has not been demonstrated.

Minimality (criterion 2) is open. Specific candidates worth investigating: can RateLimit be expressed as Budget\<Operations\> with periodic reset? Can Signal\<T\> be derived from AuditSink with a predicate filter? If either decomposition succeeds, the primitive count drops below sixteen. The methodology's validity does not depend on the exact count; sufficiency is the load-bearing criterion.

### 9.2 What the Twelve Domains Cover

The twelve domains share a structural ancestor: they are all resource-governed execution systems where work flows through a graph under capacity constraints. Within that family, the coverage is broad. Stream processing, event-driven systems, observability pipelines, build systems, deployment orchestration, secrets rotation, certificate management, materialized views, distributed gossip, reactive systems, package management, all have the same shape at the kernel level.

The methodology has not been tested against domains with fundamentally different computational character: backtracking search (SAT/SMT solving), interactive theorem proving, term rewriting, or pure symbolic computation. The DAC primitive set may or may not extend to those domains. That is open work.

### 9.3 What TLA+ Does and Does Not Show

TLA+ formal verification is bounded model checking, not unbounded proof. The TLC model checker exhaustively explores all reachable states within finite bounds (Budget capacity of 4, three-node hierarchies, two parameter groups). Structural bugs and invariant violations are reliably caught at these bounds because the primitive structures do not change with scale. Bounded verification does not constitute a mathematical proof that the invariants hold for all possible values of the constants. It provides strong evidence, not certainty.

The governance invariants are scale-independent by construction, and this was confirmed empirically at 204M parameters. Budget\<U\> enforces `consumed <= capacity` whether capacity is 4 or 4 billion. The ConvergenceGovernor's four-state machine has the same transitions whether it monitors 8 parameter groups or 8,000. HierarchyOrdering holds whether there are 2 levels or 20. The TLA+ specifications are parameterized by constants, and the invariants hold for any value of those constants because the logic does not reference the constants' magnitudes. The 204M training run confirmed this empirically: zero budget violations across 722 audit records, all 18 governor transitions valid, hierarchy ordering maintained throughout. These are structural properties of the state machines, not empirical properties of any particular training run. What remains empirical, and what requires the 7B+ work, is whether the governance produces efficient training outcomes at that scale.

The formal verification does not verify implementations. The TLA+ specifications define what the primitives must do. The Python, Rust, Go, and TypeScript implementations in Appendix A are intended to be faithful to those specifications, but the verification that each implementation correctly implements the specification is a separate concern (refinement checking) that has not been performed.

### 9.4 Formal Verification: Results

All twenty-one primitive specifications, eighteen decomposition-failure specifications, and five LeanFormer compositions were formalized in TLA+ [14] and verified by the TLC model checker. The twenty-one primitive specifications comprise eighteen primitive modules (the sixteen primitives, with CompetitiveSelection split into three modules), two cross-primitive composition theorems (TraversalBudgetComposition, SelectThenActuate), and the phase-aware ConvergenceGovernor. The universal primitive verification explored roughly 44.6 million states across 39 specifications. The five LeanFormer case-study compositions were verified separately across roughly 178,000 additional states. The complete specifications, configuration files, and TLC output logs are available in the project repository.

The verification produced three categories of results.

First, all twenty-one primitive specifications passed: their safety invariants held across all reachable states at the tested bounds. Budget never exceeded capacity. The ConvergenceGovernor never skipped a state. CompetitiveSelection (hard) always produced the highest-scoring winner. The hash chain in AuditSink was never broken.

Second, all five LeanFormer composition specifications passed. Primitive invariants are preserved under composition, and the composed system produces composition-level invariants drawn from the constituent primitives. The GovernedTrainingPipeline specification (Appendix C.3) verified thirteen simultaneous invariants across all interleavings of training steps, phase-aware governor updates, hierarchy activations, signal-gated transitions, ranked routing updates, and audit log appends. The new NoCoolingFromCold invariant holds under composition, demonstrating the B=0 fix at the case-study level.

Third, all eighteen decomposition specifications produced concrete counterexamples (Appendix D). Decomposing a primitive into separable check-then-act steps introduces a TOCTOU window where the primitive's invariant is violated. The counterexamples are not hypothetical. They are state traces that TLC discovered through exhaustive exploration. Some violations were found in as few as two states (QualityHierarchy: adding a child without checking the level constraint; the basic ConvergenceGovernor: a single zero-delta step causing premature COOLING, which reproduces the exact B=0 bug from the training run). Others required longer traces.

These decomposition results establish that each primitive's invariant requires atomicity at the primitive boundary. They do not establish algebraic minimality (that no primitive can be expressed as a composition of others in the set). The two properties are different. The first is what was verified; the second remains open. The decomposition technique, in general, is a known consequence of how atomic operations behave in concurrent systems. What the verification adds is the systematic application of that technique to every primitive in the set, with concrete counterexamples produced by an unbiased model checker rather than imagined ones.

The verification process challenged and corrected five initial assumptions about the formal specifications.

| Specification | Initial Assumption | TLC Finding | Correction |
|---|---|---|---|
| ConvergenceGatedActivation | Global invariant (always holds) | L0 can regress to ACTIVE after L1 activates (AWAKENED mechanism) | Reclassified as precondition of ActivateNextLevel |
| FullDepthMeansUncertain | Full depth implies no convergence | Convergence at the final layer is valid | Corrected to: full depth with no convergence at any layer implies uncertainty |
| ForgingRequiresConvergence | Holds across all phases | Converged group could revert to ACTIVE during forging | Restricted governor updates to training phase |
| Budget reallocation | Free half of converged group's budget | Group may have consumed more than half; this can violate SubPoolInvariant | Free only the unused portion |
| WinnerOptimality | Holds between evaluations | Score update between evaluations creates stale winner | Invalidate allocation when scores change (TOCTOU pattern) |

In every case, the corrected specification is more precise than the original, not more permissive. These are the kinds of issues that prose specifications miss and that formal verification catches.

### 9.5 Open Work and Funded Next Steps

The following are the most important open questions and necessary next steps. Items requiring funding are noted with estimated costs.

**1. Phase-aware convergence governor implementation and clean retrain.** The TLA+ specification exists and passes. The implementation requires adding gradient_phase tracking to the convergence governor code, adding a peak_observed flag, and enforcing the NoCoolingFromCold precondition on the ACTIVE-to-COOLING transition. A retrain of the 204M model with the fixed governor would produce clean hierarchy activations free of the B=0 artifact. Estimated cost: ~$150 (one 6-day L4 GPU run).

**2. Architectural ablations on the 39M setup.** The architectural results in Sections 5.3-5.7 currently lack baselines. A set of three ablations would resolve this:

(a) LeanFormer with dense attention vs. sparse attention. Same model, same training, attention sparsity flag flipped. Compares val PPL, wall-clock, and capability on a held-out task. Establishes whether the DAC-derived sparse attention costs capability against a dense baseline.

(b) Vanilla LoRA at rank 16 vs. LeanFormer's belief deltas at rank 16, single belief. Same dataset, same training. Tests whether LeanFormer's belief delta matches vanilla LoRA at single-task fine-tuning, which is the rediscovery confirmation.

(c) Dense baseline at matched effective parameter count. A standard transformer at 39M dense params, same training. Establishes that the LeanFormer compression is not pathologically destructive.

Estimated cost: ~$200-400.

**3. Catastrophic forgetting comparison.** The architectural separation between base weights and belief deltas is structurally close to LoRA adapter stacking. The empirical question is whether LeanFormer's enforced non-overlapping address spaces produce less belief-to-belief interference than vanilla LoRA stacking, where multiple adapters target the same parameter slices. Three protocols on the same base model:

(a) Sequential full fine-tuning baseline. Train task A, measure capability. Continue training on task B, re-measure capability on A. Standard catastrophic forgetting demonstration.

(b) Vanilla LoRA stack baseline. Train LoRA adapter A, save. Train separate LoRA adapter B from base, save. Load A only, evaluate. Load B only, evaluate. Tests whether LoRA-as-it-actually-works exhibits interference.

(c) LeanFormer belief injection. Inject belief A, evaluate. Inject belief B (orthogonal address space), evaluate both A and B. Remove belief A, verify base restored bit-for-bit.

Either outcome is publishable. A measurable interference reduction relative to vanilla LoRA stacking would support the architectural separation claim. Equivalent interference would contract the claim to "LeanFormer does what LoRA does, derived independently," which is itself the rediscovery signal the paper centers on. Estimated cost: ~$200-400.

**4. Confabulation calibration.** On a held-out QA task, partition predictions by exit depth. Test whether accuracy is higher when exit depth is shallow (early convergence) and lower when full depth is reached without convergence. Establishes whether the architectural confidence signal correlates with output correctness or is currently inert. This requires only inference on a fixed checkpoint. Estimated cost: ~$50.

**5. Governance machinery comparison at 204M.** Same architecture, governance machinery disabled (no FederatedBudget enforcement, no ConvergenceGovernor state machine, no hierarchy gating, no audit chain). The honest framing is what governance buys versus what it costs. Governance probably costs PPL because it gates gradient flow and adds overhead; it buys SHA-chained auditability, dynamic budget reallocation, and convergence-driven hierarchy activation. The comparison measures both sides. Estimated cost: ~$1,000-1,500.

**6. Multi-epoch training with proper regularization.** The current 204M run used a single epoch with dropout 0.1 as the only regularization, which produced severe overfitting after step 2,000. A 3-5 epoch run with stronger regularization (dropout 0.15-0.2, weight decay sweep) would address the overfitting and provide a cleaner training trajectory for evaluating the governance machinery's behavior on a model that is actually learning rather than overfitting. Estimated cost: ~$900-1,500.

**7. Scale validation of LeanFormer at 7B+ parameters.** The most important validation that the current work lacks. Includes gradient compute reduction measurements with actual backward-pass skipping for converged groups, wall-clock training time comparisons against an ungoverned baseline, Knowledge Plane validation with orthogonality enforcement at a scale where the delta address space is practically significant. The governance invariants are established as scale-independent. The efficiency claims require empirical validation at frontier scale. Estimated cost: ~$5,000-15,000 depending on GPU tier and training duration.

**8. Cross-domain optimization transfer in the opposite direction.** The methodology's strongest theoretical claim is that optimizations transfer across domains via shared primitives. So far, all demonstrations move structural insights from systems engineering into ML. A demonstration in the opposite direction (an ML-derived attention optimization applied to a non-ML CompetitiveSelection use case in rendering or networking, with measurable improvement) would substantially strengthen the central claim.

**9. Independent replication of DAC by other researchers** applying the methodology to domains outside the current twelve. Positive results would confirm generality. Negative results would identify the boundaries of the current primitive set and may reveal new primitives.

**10. Adversarial domain testing** against domains with fundamentally different computational character (backtracking search, term rewriting, interactive theorem proving). The current twelve domains are all resource-governed DAG execution systems. The methodology may or may not extend to domains where work is not graph-shaped at the kernel level.

**11. Independent implementation of two or more non-ML domains** from the collapse table (real-time rendering pipeline, network routing system) to provide stronger evidence for cross-domain primitive transfer.

**12. Refinement checking** between TLA+ specifications and implementations, verifying that the Python, Rust, Go, and TypeScript code in Appendix A correctly implements the formal specifications in Appendix B.

**13. Investigation of specific minimality candidates.** Whether RateLimit can be expressed as Budget\<Operations\> with periodic reset, and whether Signal\<T\> can be derived from AuditSink with a predicate filter. If either decomposition succeeds, the primitive count drops below sixteen.

**14. Categorical formalization** of the cross-domain mappings as functors or adjunctions, for the audience that prefers that level of formality.

Items 1 through 7 are the priority sequence for converting the LeanFormer artifact into a body of work with comparative evidence behind each architectural claim. Items 8 through 14 strengthen the methodology itself. Items 1-5 together require roughly $1,500-2,300 of GPU compute. Item 7 alone is the largest single expense and produces the strongest single result.

---

## 10. Conclusion

Sixteen abstraction primitives were sufficient for the twelve engineering domains I examined. The set is not proven minimal. Other domains may require additions. The Competitive Selection family accounts for a large fraction of the cross-domain mappings and decomposes into modes (hard, soft, ranked) that share a scoring interface but differ in allocation semantics. Some mappings in the collapse table are genuinely informative: attention as soft competitive selection, the four-domain DAG workload pattern, training as a governed system that was missing selectivity primitives. Others are largely renamings: ETL extract/transform stages as ActuationPass, inventory as ResourceRegistry. Both categories are documented in the body of the paper without dressing up the renamings as discoveries.

The methodology behind the catalog (asking how each domain works underneath its vocabulary, and noticing when the answer matches an operation that another domain already named) is the part of the work I expect to generalize. The specific primitive set is the artifact of applying that methodology to the twelve domains I happened to encounter.

LeanFormer was a deliberate test of whether the methodology can produce a workable architecture for a domain the practitioner does not have research-level expertise in. I am not an ML engineer. Six problems in neural network design (parameter inefficiency, attention cost, catastrophic forgetting, knowledge composition, confabulation, training process inefficiency) mapped to compositions of systems engineering primitives once the ML vocabulary was stripped. The resulting architecture composes four efficiency mechanisms simultaneously, addresses base-weight forgetting through architectural separation (bit-for-bit base weight restoration across 100 beliefs), makes confabulation architecturally detectable, and governs the entire training lifecycle with the same primitives that govern the model's architecture. The components I derived through DAC turn out to closely resemble established ML techniques (LoRA, sparse attention, hierarchical training). I take that convergence as a rediscovery signal: starting from a structural skeleton derived independently of ML, the architecture landed near solutions ML researchers reached through domain-internal reasoning.

At 204M parameters, all sixteen primitives' invariants held empirically across 7,228 training steps with zero governance violations. The model is not competitive as a language model. That gap reflects parts of the problem DAC does not address (training recipe, regularization, data curation, optimizer choice, loss formulation) and parts that need more compute than I had available. Comparative baselines on each architectural claim (vanilla LoRA, ungoverned training, dense parameter equivalents) are the next concrete experimental work. The structural skeleton DAC produced runs and behaves correctly under its specifications. Whether each component beats its established counterpart on capability is the funded question.

The B=0 episode during the 204M run was a useful side experiment. The bug class (zero-as-ambiguous-signal) is well-understood, and the fix (a latching phase tracker) is a standard hysteresis pattern. What DAC contributed was the recognition route: stripping the ML vocabulary from the failure produced a description that matched a pattern already solved in rendering, networking, distributed consensus, and database concurrency control. The fix was specified in TLA+ and verified by TLC before any code was written.

The TLA+ verification provides bounded evidence rather than unbounded proof. Forty-four specifications (twenty-one primitive specifications, eighteen decomposition-failure specifications, five LeanFormer compositions) were verified by the TLC model checker across roughly 44.6 million states for the universal primitive verification plus roughly 178,000 additional states for the LeanFormer case study. Every primitive's safety invariant held. Every decomposition into separable check-then-act steps produced a concrete TOCTOU counterexample, establishing that each primitive's invariant requires atomicity at the primitive boundary. The verification process challenged five initial assumptions about the specifications and produced corrections that refined imprecise classifications without weakening any falsified claim.

The other useful next step beyond the funded baseline work is independent application of the methodology by other practitioners to domains I have not examined. The methodology is a habit, not a result. If other people apply it and find that the primitive set is insufficient for their domains, the catalog gets new primitives or the existing ones get revised. Either outcome strengthens the work.

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

Python (ML training convergence):

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

The four-state machine has the same shape across all three implementations. The delta input changes meaning depending on the domain (constraint residual in physics, gradient magnitude in training, term change rate in consensus, transaction conflict rate in databases), but the state machine logic is unchanged.

### The Pattern

Every example in this appendix shows the same property: the domain provides the types, the data, and the scoring or aggregation or message functions, while the primitive provides the structure, the governance, and the invariants. Swap the domain function and the primitive serves a different domain without modification. The implementations vary in language and idiom, but the operation each one performs is the same operation.
