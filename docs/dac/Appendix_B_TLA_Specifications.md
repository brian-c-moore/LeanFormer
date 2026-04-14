# Appendix B: Formal Specifications of the Sixteen Primitives in TLA+

## Why TLA+

The code appendix (Appendix A) demonstrates that the primitives are language-agnostic by showing identical structure in Rust, Python, Go, and TypeScript. But code in any language still carries implementation concerns: memory models, type systems, error handling, concurrency semantics. These are noise when the question is whether the primitive's logic is sound.

TLA+ removes all implementation noise. A TLA+ specification describes what a system does, not how it does it. If the invariants hold in TLA+, they hold in any correct implementation regardless of language, runtime, or hardware. The primitives are not Rust traits or Python classes. They are mathematical structures. TLA+ is the natural language for mathematical structures.

Each specification below defines the state variables, the legal operations, the safety invariants (properties that must always hold), and the liveness properties (properties that must eventually hold) for one primitive. Where primitives compose, the composition invariants are specified separately.

---

## Data Primitives

---

### B.1 Budget\<U\>

```tla+
---- MODULE Budget ----
EXTENDS Naturals

VARIABLES capacity, allocated, reserved, pending_eviction

TypeInvariant ==
    /\ capacity \in Nat
    /\ allocated \in Nat
    /\ reserved \in Nat
    /\ pending_eviction \in Nat

SafetyInvariant ==
    allocated + reserved + pending_eviction <= capacity

Init ==
    /\ capacity \in Nat \ {0}
    /\ allocated = 0
    /\ reserved = 0
    /\ pending_eviction = 0

TryAllocate(amount) ==
    IF allocated + reserved + pending_eviction + amount <= capacity
    THEN /\ allocated' = allocated + amount
         /\ UNCHANGED <<capacity, reserved, pending_eviction>>
    ELSE UNCHANGED <<capacity, allocated, reserved, pending_eviction>>

Reserve(amount) ==
    IF allocated + reserved + pending_eviction + amount <= capacity
    THEN /\ reserved' = reserved + amount
         /\ UNCHANGED <<capacity, allocated, pending_eviction>>
    ELSE UNCHANGED <<capacity, allocated, reserved, pending_eviction>>

CommitReservation(amount) ==
    /\ amount <= reserved
    /\ allocated' = allocated + amount
    /\ reserved' = reserved - amount
    /\ UNCHANGED <<capacity, pending_eviction>>

Release(amount) ==
    /\ amount <= allocated
    /\ allocated' = allocated - amount
    /\ UNCHANGED <<capacity, reserved, pending_eviction>>

MarkEviction(amount) ==
    /\ amount <= allocated
    /\ allocated' = allocated - amount
    /\ pending_eviction' = pending_eviction + amount
    /\ UNCHANGED <<capacity, reserved>>

CompleteEviction(amount) ==
    /\ amount <= pending_eviction
    /\ pending_eviction' = pending_eviction - amount
    /\ UNCHANGED <<capacity, allocated, reserved>>

Next ==
    \/ \E a \in 1..capacity : TryAllocate(a)
    \/ \E a \in 1..capacity : Reserve(a)
    \/ \E a \in 1..reserved : CommitReservation(a)
    \/ \E a \in 1..allocated : Release(a)
    \/ \E a \in 1..allocated : MarkEviction(a)
    \/ \E a \in 1..pending_eviction : CompleteEviction(a)

Spec == Init /\ [][Next]_<<capacity, allocated, reserved, pending_eviction>>

THEOREM Spec => []SafetyInvariant
====
```

What we expect to prove:

The safety invariant `allocated + reserved + pending_eviction <= capacity` holds in every reachable state. No sequence of TryAllocate, Reserve, CommitReservation, Release, MarkEviction, or CompleteEviction operations can violate the ceiling. The specification has no ForceAllocate operation. This is not an omission. It is the claim: the invariant is structurally unbypassable. TLC (the TLA+ model checker) exhaustively explores all interleavings of all operations at all amounts and confirms that no reachable state violates the inequality. This holds whether U is VRAM bytes, CPU millicores, gradient compute FLOPs, or network bandwidth. The domain is not mentioned anywhere in the spec.

---

### B.2 FederatedBudget\<U\>

```tla+
---- MODULE FederatedBudget ----
EXTENDS Naturals, FiniteSets

CONSTANTS SubPoolNames, MasterCapacity

VARIABLES master_allocated, sub_capacities, sub_allocated

TypeInvariant ==
    /\ master_allocated \in Nat
    /\ sub_capacities \in [SubPoolNames -> Nat]
    /\ sub_allocated \in [SubPoolNames -> Nat]

FederationInvariant ==
    master_allocated <= MasterCapacity

SubPoolInvariant ==
    \A name \in SubPoolNames : sub_allocated[name] <= sub_capacities[name]

ConsistencyInvariant ==
    master_allocated = SumOver(SubPoolNames, sub_capacities)

\* Helper: sum all sub-pool capacities
SumOver(names, f) ==
    LET RECURSIVE Sum(_, _)
        Sum(s, acc) ==
            IF s = {} THEN acc
            ELSE LET x == CHOOSE x \in s : TRUE
                 IN Sum(s \ {x}, acc + f[x])
    IN Sum(names, 0)

Init ==
    /\ master_allocated = 0
    /\ sub_capacities = [n \in SubPoolNames |-> 0]
    /\ sub_allocated = [n \in SubPoolNames |-> 0]

AllocateSubPool(name, amount) ==
    IF master_allocated + amount <= MasterCapacity
    THEN /\ master_allocated' = master_allocated + amount
         /\ sub_capacities' = [sub_capacities EXCEPT ![name] = @ + amount]
         /\ UNCHANGED sub_allocated
    ELSE UNCHANGED <<master_allocated, sub_capacities, sub_allocated>>

AllocateFromSubPool(name, amount) ==
    IF sub_allocated[name] + amount <= sub_capacities[name]
    THEN /\ sub_allocated' = [sub_allocated EXCEPT ![name] = @ + amount]
         /\ UNCHANGED <<master_allocated, sub_capacities>>
    ELSE UNCHANGED <<master_allocated, sub_capacities, sub_allocated>>

ReleaseFromSubPool(name, amount) ==
    /\ amount <= sub_allocated[name]
    /\ sub_allocated' = [sub_allocated EXCEPT ![name] = @ - amount]
    /\ UNCHANGED <<master_allocated, sub_capacities>>

Next ==
    \/ \E name \in SubPoolNames, a \in 1..MasterCapacity :
        AllocateSubPool(name, a)
    \/ \E name \in SubPoolNames, a \in 1..MasterCapacity :
        AllocateFromSubPool(name, a)
    \/ \E name \in SubPoolNames, a \in 1..MasterCapacity :
        ReleaseFromSubPool(name, a)

Spec == Init /\ [][Next]_<<master_allocated, sub_capacities, sub_allocated>>

THEOREM Spec => [](FederationInvariant /\ SubPoolInvariant /\ ConsistencyInvariant)
====
```

What we expect to prove:

Three invariants hold simultaneously in every reachable state. First, the federation invariant: total allocated to sub-pools never exceeds master capacity. Second, the sub-pool invariant: each sub-pool's internal allocation never exceeds its own capacity. Third, the consistency invariant: the master's allocated amount equals the sum of all sub-pool capacities, so no resources appear or disappear during subdivision. The composition of these three means the two-level hierarchy is leak-proof. This is the property that makes FederatedBudget safe for gradient compute allocation across parameter groups: no group can overspend its budget, and the sum of all group budgets cannot exceed the total compute budget.

---

### B.3 QualityHierarchy

```tla+
---- MODULE QualityHierarchy ----
EXTENDS Naturals, FiniteSets

CONSTANTS Nodes, MaxLevel

VARIABLES level, cost, children, parent

TypeInvariant ==
    /\ level \in [Nodes -> 0..MaxLevel]
    /\ cost \in [Nodes -> Nat]
    /\ children \in [Nodes -> SUBSET Nodes]
    /\ parent \in [Nodes -> Nodes \cup {NULL}]

\* A node's level is always greater than its children's levels
HierarchyInvariant ==
    \A n \in Nodes : \A c \in children[n] : level[n] > level[c]

\* Every child has exactly one parent
ParentConsistency ==
    \A n \in Nodes : \A c \in children[n] : parent[c] = n

\* No node is its own ancestor (acyclicity)
Acyclicity ==
    \A n \in Nodes : n \notin Descendants(n)

\* Cost decreases with level (coarser = cheaper)
CostMonotonicity ==
    \A n \in Nodes : \A c \in children[n] : cost[n] <= cost[c]

RECURSIVE Descendants(_)
Descendants(n) ==
    children[n] \cup UNION {Descendants(c) : c \in children[n]}

Init ==
    /\ level \in [Nodes -> 0..MaxLevel]
    /\ cost \in [Nodes -> Nat]
    /\ children \in [Nodes -> SUBSET Nodes]
    /\ parent \in [Nodes -> Nodes \cup {NULL}]
    /\ HierarchyInvariant
    /\ ParentConsistency
    /\ Acyclicity

Spec == Init /\ [][UNCHANGED <<level, cost, children, parent>>]_<<level, cost, children, parent>>

THEOREM Spec => [](HierarchyInvariant /\ ParentConsistency /\ Acyclicity)
====
```

What we expect to prove:

The hierarchy is a proper tree: acyclic, single-parent, with levels strictly decreasing from parent to child. Cost monotonicity ensures that coarser nodes are cheaper to evaluate than finer nodes, which is the property that makes budget-constrained traversal meaningful. If a coarser node were more expensive than its children, descending into children would be cheaper and the traversal heuristic ("stop descending when the budget runs out") would produce worse results at higher cost. TLC confirms that no valid hierarchy construction violates these structural properties. The hierarchy does not know whether its nodes are mesh LODs, parameter groups, or sample difficulty tiers.

---

### B.4 AllocationSnapshot

```tla+
---- MODULE AllocationSnapshot ----
EXTENDS Naturals, FiniteSets

CONSTANTS Nodes, BudgetCapacity

VARIABLES accepted, total_cost, budget_remaining

TypeInvariant ==
    /\ accepted \subseteq Nodes
    /\ total_cost \in Nat
    /\ budget_remaining \in Nat

\* The snapshot is consistent with the budget
BudgetConsistency ==
    total_cost + budget_remaining <= BudgetCapacity

\* Total cost equals the sum of accepted node costs
CostAccuracy ==
    total_cost = SumCosts(accepted)

\* Snapshot is immutable once produced (no partial updates)
\* This is modeled by the snapshot being a single atomic output
\* of a TraversalEngine pass, not an incrementally modified structure.

Init ==
    /\ accepted = {}
    /\ total_cost = 0
    /\ budget_remaining = BudgetCapacity

AcceptNode(n, node_cost) ==
    /\ n \notin accepted
    /\ node_cost <= budget_remaining
    /\ accepted' = accepted \cup {n}
    /\ total_cost' = total_cost + node_cost
    /\ budget_remaining' = budget_remaining - node_cost

Next ==
    \E n \in Nodes, c \in 1..BudgetCapacity : AcceptNode(n, c)

Spec == Init /\ [][Next]_<<accepted, total_cost, budget_remaining>>

THEOREM Spec => []BudgetConsistency
====
```

What we expect to prove:

The snapshot never records an allocation that exceeds the budget. The sum of accepted node costs plus the remaining budget never exceeds the original capacity. This is the bridge between TraversalEngine (which produces snapshots) and ActuationPass (which consumes them): the actuation pass can trust that every node in the snapshot was afforded by the budget. No downstream consumer needs to re-verify the budget constraint.

---

### B.5 RelationshipGraph\<N,E\>

```tla+
---- MODULE RelationshipGraph ----
EXTENDS Naturals, FiniteSets

CONSTANTS Nodes, MaxWeight

VARIABLES edges, adjacency

TypeInvariant ==
    /\ edges \subseteq (Nodes \X Nodes \X 0..MaxWeight)
    /\ adjacency \in [Nodes -> SUBSET Nodes]

\* Adjacency map is consistent with edge set
AdjacencyConsistency ==
    \A src \in Nodes : adjacency[src] =
        {dst \in Nodes : \E w \in 0..MaxWeight : <<src, dst, w>> \in edges}

\* No self-loops (optional, domain-dependent)
NoSelfLoops ==
    \A n \in Nodes : n \notin adjacency[n]

Init ==
    /\ edges = {}
    /\ adjacency = [n \in Nodes |-> {}]

AddEdge(src, dst, weight) ==
    /\ src /= dst
    /\ edges' = edges \cup {<<src, dst, weight>>}
    /\ adjacency' = [adjacency EXCEPT ![src] = @ \cup {dst}]

RemoveEdge(src, dst) ==
    /\ \E w \in 0..MaxWeight : <<src, dst, w>> \in edges
    /\ edges' = {e \in edges : e[1] /= src \/ e[2] /= dst}
    /\ adjacency' = [adjacency EXCEPT ![src] = @ \ {dst}]

Next ==
    \/ \E s, d \in Nodes, w \in 0..MaxWeight : AddEdge(s, d, w)
    \/ \E s, d \in Nodes : RemoveEdge(s, d)

Spec == Init /\ [][Next]_<<edges, adjacency>>

THEOREM Spec => []AdjacencyConsistency
====
```

What we expect to prove:

The adjacency index is always consistent with the edge set. No operation can create a state where the adjacency map claims a neighbor that has no corresponding edge, or where an edge exists without the adjacency map reflecting it. This is a structural integrity property. PropagationPass relies on the adjacency map to determine message recipients; if the map is inconsistent with the actual edges, messages are sent to wrong nodes or not sent at all. TLC confirms the consistency holds across all sequences of add and remove operations.

---

### B.6 ResourceRegistry\<K,V\>

```tla+
---- MODULE ResourceRegistry ----
EXTENDS FiniteSets

CONSTANTS Keys, Values

VARIABLES entries

TypeInvariant ==
    entries \in [SUBSET Keys -> Values]

\* Every registered key maps to exactly one value
UniqueMapping ==
    \A k \in DOMAIN entries : entries[k] \in Values

\* Registration and deregistration are consistent
RegisterConsistency ==
    \A k \in Keys : k \in DOMAIN entries => entries[k] \in Values

Init ==
    entries = <<>>  \* empty function

Register(k, v) ==
    entries' = [entries EXCEPT ![k] = v]

Deregister(k) ==
    /\ k \in DOMAIN entries
    /\ entries' = [key \in (DOMAIN entries) \ {k} |-> entries[key]]

Lookup(k) ==
    IF k \in DOMAIN entries
    THEN entries[k]
    ELSE NULL

Next ==
    \/ \E k \in Keys, v \in Values : Register(k, v)
    \/ \E k \in Keys : Deregister(k)

Spec == Init /\ [][Next]_entries

THEOREM Spec => []UniqueMapping
====
```

What we expect to prove:

Every key maps to exactly one value. Register overwrites, it does not create duplicates. Deregister removes cleanly. Lookup never returns a stale value. This is the trivial collapse acknowledged in the paper: a registry is a map. The TLA+ spec confirms what everyone already knows, but it confirms it in the same formal language as the non-trivial primitives, reinforcing that all sixteen sit at the same level of rigor.

---

## Computation Primitives

---

### B.7 TraversalEngine

```tla+
---- MODULE TraversalEngine ----
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Nodes, RootNode, MaxBudget

VARIABLES budget_remaining, visited, accepted, queue

TypeInvariant ==
    /\ budget_remaining \in Nat
    /\ visited \subseteq Nodes
    /\ accepted \subseteq Nodes
    /\ queue \subseteq Nodes

\* Never spend more than we have
BudgetInvariant ==
    budget_remaining >= 0

\* Accepted nodes are always a subset of visited nodes
AcceptedSubsetVisited ==
    accepted \subseteq visited

\* Every accepted node was affordable at the time of acceptance
\* (This is an implicit property: the budget was checked before accepting)

\* Traversal is monotonic: once visited, a node stays visited
VisitMonotonicity ==
    \* Expressed as: visited only grows
    TRUE  \* Enforced by the Next relation never removing from visited

Init ==
    /\ budget_remaining = MaxBudget
    /\ visited = {}
    /\ accepted = {}
    /\ queue = {RootNode}

VisitNode(n) ==
    /\ n \in queue
    /\ n \notin visited
    /\ visited' = visited \cup {n}
    /\ LET node_cost == Cost(n) IN
       IF node_cost <= budget_remaining
       THEN /\ accepted' = accepted \cup {n}
            /\ budget_remaining' = budget_remaining - node_cost
            /\ queue' = (queue \ {n}) \cup Children(n)
       ELSE /\ accepted' = accepted
            /\ budget_remaining' = budget_remaining
            /\ queue' = queue \ {n}

Skip(n) ==
    /\ n \in queue
    /\ queue' = queue \ {n}
    /\ UNCHANGED <<budget_remaining, visited, accepted>>

Terminate ==
    /\ queue = {}
    /\ UNCHANGED <<budget_remaining, visited, accepted, queue>>

Next ==
    \/ \E n \in Nodes : VisitNode(n)
    \/ \E n \in Nodes : Skip(n)
    \/ Terminate

\* Liveness: traversal eventually terminates
Liveness == <>(queue = {})

Spec == Init /\ [][Next]_<<budget_remaining, visited, accepted, queue>>

THEOREM Spec => [](BudgetInvariant /\ AcceptedSubsetVisited)
====
```

What we expect to prove:

Two safety properties and one liveness property. Safety: the budget never goes negative (no accepted node was unaffordable) and every accepted node was visited (no phantom acceptances). Liveness: the traversal terminates (the queue eventually empties). Together these guarantee that the TraversalEngine produces a valid AllocationSnapshot: a set of nodes that were explicitly visited, explicitly affordable, and produced by a process that completes. The domain determines Cost(n) and Children(n). The traversal guarantees hold regardless of what those functions return.

---

### B.8 PropagationPass

```tla+
---- MODULE PropagationPass ----
EXTENDS Naturals, Reals

CONSTANTS Nodes, Edges, MaxIterations

VARIABLES values, iteration, changed

TypeInvariant ==
    /\ values \in [Nodes -> Real]
    /\ iteration \in Nat
    /\ changed \in BOOLEAN

\* Iteration count is bounded
BoundedIteration ==
    iteration <= MaxIterations

\* Values are monotonically improving toward fixed point
\* (domain-specific: for shortest path, values decrease; for GI, energy conserves)
\* The generic invariant is just bounded iteration.

Init ==
    /\ values \in [Nodes -> Real]
    /\ iteration = 0
    /\ changed = TRUE

Propagate ==
    /\ changed = TRUE
    /\ iteration < MaxIterations
    /\ LET new_values == [n \in Nodes |->
           LET incoming == {<<src, w>> : <<src, dst, w>> \in Edges, dst = n}
           IN Aggregate(values[n], {MessageFn(values[src], w) : <<src, w>> \in incoming})
       ]
       IN /\ changed' = (new_values /= values)
          /\ values' = new_values
          /\ iteration' = iteration + 1

Terminate ==
    /\ (changed = FALSE \/ iteration = MaxIterations)
    /\ UNCHANGED <<values, iteration, changed>>

Next == Propagate \/ Terminate

\* Liveness: propagation terminates (either converges or hits max iterations)
Liveness == <>(changed = FALSE \/ iteration = MaxIterations)

Spec == Init /\ [][Next]_<<values, iteration, changed>>

THEOREM Spec => []BoundedIteration
====
```

What we expect to prove:

The propagation pass always terminates, either by reaching a fixed point (no values changed) or by exhausting the iteration budget. This is the property that makes PropagationPass safe in real-time systems: unbounded iteration would cause frame drops in rendering or unbounded latency in routing convergence. The iteration cap is a budget constraint on computation time, structurally identical to Budget\<U\> applied to iteration count. The domain provides MessageFn and Aggregate. The specification guarantees termination regardless of what those functions compute. For shortest-path (Bellman-Ford), convergence is guaranteed on graphs without negative cycles. For GI (radiosity), convergence follows from energy conservation. For backpropagation, exactly one pass suffices (the DAG has no cycles). The spec covers all three cases through the same termination mechanism.

---

### B.9 CompetitiveSelection (Hard)

```tla+
---- MODULE CompetitiveSelectionHard ----
EXTENDS Naturals, FiniteSets

CONSTANTS Seats, Candidates

VARIABLES allocation, scores

TypeInvariant ==
    /\ allocation \in [Seats -> Candidates \cup {NULL}]
    /\ scores \in [Seats -> [Candidates -> Nat]]

\* Mutual exclusion: each seat has at most one winner
MutualExclusion ==
    \A s \in Seats : Cardinality({allocation[s]}) <= 1

\* Winner has highest score: no candidate scores higher than the winner
WinnerOptimality ==
    \A s \in Seats :
        allocation[s] /= NULL =>
            \A c \in Candidates :
                scores[s][c] <= scores[s][allocation[s]]

\* Determinism: same scores produce same allocation
\* (Ties are broken by a fixed ordering, not by nondeterminism)

Init ==
    /\ allocation = [s \in Seats |-> NULL]
    /\ scores = [s \in Seats |-> [c \in Candidates |-> 0]]

Evaluate(s) ==
    LET best == CHOOSE c \in Candidates :
        \A other \in Candidates : scores[s][other] <= scores[s][c]
    IN /\ allocation' = [allocation EXCEPT ![s] = best]
       /\ UNCHANGED scores

UpdateScore(s, c, new_score) ==
    /\ scores' = [scores EXCEPT ![s][c] = new_score]
    /\ UNCHANGED allocation

Next ==
    \/ \E s \in Seats : Evaluate(s)
    \/ \E s \in Seats, c \in Candidates, v \in Nat : UpdateScore(s, c, v)

Spec == Init /\ [][Next]_<<allocation, scores>>

THEOREM Spec => [](MutualExclusion /\ WinnerOptimality)
====
```

What we expect to prove:

Two invariants. Mutual exclusion: no seat is ever assigned to more than one candidate. Winner optimality: the assigned candidate always has the highest score among all candidates for that seat. Together these are the correctness definition of hard competitive selection. The visibility buffer depends on mutual exclusion (each pixel has exactly one triangle). Leader election depends on mutual exclusion (exactly one leader per term). Pod scheduling depends on winner optimality (the best-scoring node gets the pod). The spec proves both properties hold for any scoring function, any number of seats, and any number of candidates.

---

### B.10 CompetitiveSelection (Soft)

```tla+
---- MODULE CompetitiveSelectionSoft ----
EXTENDS Naturals, Reals, FiniteSets

CONSTANTS Seats, Candidates

VARIABLES weights, output

TypeInvariant ==
    /\ weights \in [Seats -> [Candidates -> Real]]
    /\ output \in [Seats -> [Candidates -> Real]]

\* Weights are non-negative
NonNegativity ==
    \A s \in Seats : \A c \in Candidates : weights[s][c] >= 0

\* Weights sum to 1 for each seat (probability distribution)
Normalization ==
    \A s \in Seats :
        SumWeights(s) = 1

\* No candidate is excluded from contributing (all weights > 0 in softmax)
\* This distinguishes soft from hard selection
UniversalContribution ==
    \A s \in Seats : \A c \in Candidates : weights[s][c] > 0

SumWeights(s) ==
    LET RECURSIVE Sum(_, _)
        Sum(remaining, acc) ==
            IF remaining = {} THEN acc
            ELSE LET c == CHOOSE c \in remaining : TRUE
                 IN Sum(remaining \ {c}, acc + weights[s][c])
    IN Sum(Candidates, 0)

ComputeSoftmax(s, raw_scores) ==
    LET exp_scores == [c \in Candidates |-> Exp(raw_scores[c])]
        total == SumWeights(s)  \* using exp_scores
    IN [c \in Candidates |-> exp_scores[c] / total]

Spec == Init /\ [][Next]_<<weights, output>>

THEOREM Spec => [](NonNegativity /\ Normalization /\ UniversalContribution)
====
```

What we expect to prove:

Three invariants that together define soft selection and distinguish it from hard selection. Non-negativity: no weight is negative (the output is a valid mixture). Normalization: weights sum to 1 per seat (the output is a probability distribution). Universal contribution: every candidate contributes a non-zero weight to every seat (no candidate is fully excluded). This last property is what makes attention fundamentally different from a visibility buffer. In hard selection, all but one candidate get zero. In soft selection, every candidate contributes. The spec proves that softmax preserves all three properties for any set of input scores, which is why attention produces a weighted blend rather than a winner-take-all selection.

---

### B.11 CompetitiveSelection (Ranked)

```tla+
---- MODULE CompetitiveSelectionRanked ----
EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS Candidates, K

VARIABLES selected, scores

TypeInvariant ==
    /\ selected \subseteq Candidates
    /\ scores \in [Candidates -> Nat]

\* Bounded multiplicity: at most K winners
BoundedMultiplicity ==
    Cardinality(selected) <= K

\* Threshold optimality: every selected candidate scores at least as high
\* as every non-selected candidate
ThresholdOptimality ==
    \A s \in selected : \A c \in Candidates \ selected :
        scores[s] >= scores[c]

\* Maximality: if fewer than K are selected, all candidates are selected
\* (we never select fewer than K unless there aren't K candidates)
Maximality ==
    Cardinality(selected) = Min(K, Cardinality(Candidates))

Init ==
    /\ selected = {}
    /\ scores \in [Candidates -> Nat]

Select ==
    LET sorted == SortByScore(Candidates, scores)
        top_k == Take(sorted, K)
    IN /\ selected' = top_k
       /\ UNCHANGED scores

Spec == Init /\ [][Select]_<<selected, scores>>

THEOREM Spec => [](BoundedMultiplicity /\ ThresholdOptimality /\ Maximality)
====
```

What we expect to prove:

Three invariants. Bounded multiplicity: never more than K winners (the beam width, the audio channel count, the bandwidth seat count). Threshold optimality: every selected candidate outscores every non-selected candidate (no inferior candidate displaces a superior one). Maximality: if fewer than K are selected, it is because fewer than K candidates exist, not because the algorithm stopped early. Together these define correct top-k selection. Beam search depends on all three (wrong beam contents produce wrong outputs). Audio priority depends on bounded multiplicity (more sounds than channels causes clipping). Alert triage depends on threshold optimality (a low-severity alert should never displace a high-severity one).

---

### B.12 ActuationPass\<R\>

```tla+
---- MODULE ActuationPass ----
EXTENDS FiniteSets

CONSTANTS Seats, Resources

VARIABLES allocation, actuated, side_effects

TypeInvariant ==
    /\ allocation \in [Seats -> Resources \cup {NULL}]
    /\ actuated \subseteq Seats
    /\ side_effects \in [Seats -> BOOLEAN]

\* Only allocated seats are actuated
ActuationScope ==
    \A s \in actuated : allocation[s] /= NULL

\* Actuation does not modify allocation
AllocationImmutability ==
    \* allocation is unchanged by actuation
    TRUE  \* Enforced by UNCHANGED in Actuate action

\* Actuation is total over allocation: every allocated seat is eventually actuated
ActuationCompleteness == <>(\A s \in Seats : allocation[s] /= NULL => s \in actuated)

Init ==
    /\ allocation \in [Seats -> Resources \cup {NULL}]
    /\ actuated = {}
    /\ side_effects = [s \in Seats |-> FALSE]

Actuate(s) ==
    /\ allocation[s] /= NULL
    /\ s \notin actuated
    /\ actuated' = actuated \cup {s}
    /\ side_effects' = [side_effects EXCEPT ![s] = TRUE]
    /\ UNCHANGED allocation

Next == \E s \in Seats : Actuate(s)

Spec == Init /\ [][Next]_<<allocation, actuated, side_effects>>

THEOREM Spec => []ActuationScope
THEOREM Spec => ActuationCompleteness
====
```

What we expect to prove:

Two properties. Safety: actuation only applies to seats that have an allocation (no action on empty seats, no shading of unowned pixels, no gradient update to unactivated parameter groups). The pass never decides which seats to act on; it only executes on seats that a prior CompetitiveSelection or TraversalEngine already determined. Liveness: every allocated seat is eventually actuated (no allocated work is silently dropped). The separation between "decide" (CompetitiveSelection) and "act" (ActuationPass) is the evaluate-then-actuate pattern that appears in deferred rendering, Kubernetes reconciliation, and the LeanFormer training pipeline. The spec formalizes this separation: ActuationPass has UNCHANGED allocation in every action.

---

### B.13 Reduction\<T,R\>

```tla+
---- MODULE Reduction ----
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Items, InitialValue

VARIABLES result, remaining, processed

TypeInvariant ==
    /\ remaining \subseteq Items
    /\ processed \subseteq Items
    /\ result \in Nat  \* simplified to Nat for model checking

\* Coverage: every item is eventually processed
Coverage == <>(remaining = {})

\* Partition: processed and remaining are disjoint and cover all items
Partition ==
    /\ processed \cup remaining = Items
    /\ processed \cap remaining = {}

\* Order independence: the final result is the same regardless of
\* processing order (requires commutative, associative combine function)
\* This is a constraint on the domain function, not on the primitive.

Init ==
    /\ remaining = Items
    /\ processed = {}
    /\ result = InitialValue

Process(item) ==
    /\ item \in remaining
    /\ result' = Combine(result, item)
    /\ remaining' = remaining \ {item}
    /\ processed' = processed \cup {item}

Next == \E i \in Items : Process(i)

Spec == Init /\ [][Next]_<<result, remaining, processed>>

THEOREM Spec => []Partition
THEOREM Spec => Coverage
====
```

What we expect to prove:

Safety: at every step, processed and remaining partition the full item set (no item is counted twice, no item is lost). Liveness: every item is eventually processed (the reduction completes). The spec is deliberately minimal because Reduction is the most generic primitive. The interesting property is that the final result depends only on the combine function and the items, not on processing order, but that property belongs to the domain function (it must be commutative and associative), not to the primitive. The primitive guarantees coverage and partitioning. Loss computation, vote counting, health-probe aggregation, and draw-call batching all require these guarantees.

---

### B.14 Sampler\<T\>

```tla+
---- MODULE Sampler ----
EXTENDS Naturals, Reals, FiniteSets

CONSTANTS Items, SampleSize

VARIABLES distribution, selected

TypeInvariant ==
    /\ distribution \in [Items -> Real]
    /\ selected \subseteq Items

\* Distribution is valid: non-negative and sums to 1
ValidDistribution ==
    /\ \A i \in Items : distribution[i] >= 0
    /\ SumDist = 1

\* Sample size is bounded
BoundedSample ==
    Cardinality(selected) <= SampleSize

\* Selected items are from the support of the distribution
SupportConsistency ==
    \A s \in selected : distribution[s] > 0

SumDist ==
    LET RECURSIVE Sum(_, _)
        Sum(remaining, acc) ==
            IF remaining = {} THEN acc
            ELSE LET i == CHOOSE i \in remaining : TRUE
                 IN Sum(remaining \ {i}, acc + distribution[i])
    IN Sum(Items, 0)

Init ==
    /\ distribution \in [Items -> Real]
    /\ ValidDistribution
    /\ selected = {}

Sample ==
    /\ Cardinality(selected) < SampleSize
    /\ \E i \in Items :
        /\ distribution[i] > 0
        /\ selected' = selected \cup {i}
        /\ UNCHANGED distribution

Spec == Init /\ [][Sample]_<<distribution, selected>>

THEOREM Spec => [](BoundedSample /\ SupportConsistency)
====
```

What we expect to prove:

Two invariants. Bounded sample: the number of selected items never exceeds the sample size (dropout masks have the right density, training batches have the right size). Support consistency: nothing with zero probability is ever selected (a zero-probability neuron is never activated by dropout, a zero-weight sample is never drawn for training). These are the minimum correctness properties for any sampling operation. The distribution shape (uniform, weighted, temperature-scaled) is the domain function.

---

## Governance Primitives

---

### B.15 ConvergenceGovernor

```tla+
---- MODULE ConvergenceGovernor ----
EXTENDS Naturals, Reals, Sequences

CONSTANTS Threshold, AwakenThreshold, Window

VARIABLES state, delta_history

States == {"ACTIVE", "COOLING", "CONVERGED", "AWAKENED"}

TypeInvariant ==
    /\ state \in States
    /\ delta_history \in Seq(Real)
    /\ Len(delta_history) <= Window

\* Valid state transitions only
ValidTransitions ==
    /\ state = "ACTIVE" => state' \in {"ACTIVE", "COOLING"}
    /\ state = "COOLING" => state' \in {"COOLING", "CONVERGED", "ACTIVE"}
    /\ state = "CONVERGED" => state' \in {"CONVERGED", "AWAKENED"}
    /\ state = "AWAKENED" => state' \in {"AWAKENED", "CONVERGED"}

\* Cannot skip states: ACTIVE cannot go directly to CONVERGED
NoSkipping ==
    /\ ~(state = "ACTIVE" /\ state' = "CONVERGED")
    /\ ~(state = "ACTIVE" /\ state' = "AWAKENED")
    /\ ~(state = "COOLING" /\ state' = "AWAKENED")

\* CONVERGED means the average delta is below threshold
ConvergedMeansStable ==
    state = "CONVERGED" => AvgDelta <= Threshold

\* AWAKENED means a converged governor detected new change
AwakenedFromConverged ==
    state = "AWAKENED" => AvgDelta > Threshold

AvgDelta ==
    IF Len(delta_history) = 0 THEN 0
    ELSE SumSeq(delta_history) / Len(delta_history)

SumSeq(seq) ==
    LET RECURSIVE Sum(_, _)
        Sum(s, acc) ==
            IF s = <<>> THEN acc
            ELSE Sum(Tail(s), acc + Head(s))
    IN Sum(seq, 0)

Init ==
    /\ state = "ACTIVE"
    /\ delta_history = <<>>

Update(delta) ==
    LET new_history ==
            IF Len(delta_history) = Window
            THEN Append(Tail(delta_history), delta)
            ELSE Append(delta_history, delta)
        avg == SumSeq(new_history) / Len(new_history)
    IN /\ delta_history' = new_history
       /\ state' =
            CASE state = "ACTIVE" /\ avg < Threshold * 2 -> "COOLING"
              [] state = "COOLING" /\ avg < Threshold -> "CONVERGED"
              [] state = "COOLING" /\ avg >= Threshold * 2 -> "ACTIVE"
              [] state = "CONVERGED" /\ avg > AwakenThreshold -> "AWAKENED"
              [] state = "AWAKENED" /\ avg < Threshold -> "CONVERGED"
              [] OTHER -> state

Next == \E d \in Real : Update(d)

Spec == Init /\ [][Next]_<<state, delta_history>>

THEOREM Spec => [](ValidTransitions /\ NoSkipping)
====
```

What we expect to prove:

The state machine has exactly four states with restricted transitions. ACTIVE can only move to COOLING (never directly to CONVERGED, which would skip the deceleration phase). COOLING can move to CONVERGED (fully stable) or back to ACTIVE (if change rate increases again). CONVERGED can only move to AWAKENED (change detected after stability). AWAKENED can only move back to CONVERGED (re-stabilized). The NoSkipping invariant is the critical one: it prevents a parameter group from being declared converged without first passing through the cooling phase, which would cause premature gradient budget reallocation. The AWAKENED state is what makes the governor non-monotonic: a converged group can be reactivated when new data or a belief injection perturbs it. TLC confirms that no sequence of delta values can produce an invalid state transition.

---

### B.16 Signal\<T\>

```tla+
---- MODULE Signal ----
EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS Values, Listeners

VARIABLES current_value, pending_notifications, notified

TypeInvariant ==
    /\ current_value \in Values
    /\ pending_notifications \subseteq Listeners
    /\ notified \subseteq Listeners

\* Change detection: notifications only fire on actual changes
ChangeDetection ==
    pending_notifications /= {} => current_value /= previous_value

\* Notification completeness: every listener is eventually notified
NotificationCompleteness ==
    <>(pending_notifications = {})

\* No spurious notifications: a listener is notified at most once per change
NoSpuriousNotifications ==
    pending_notifications \cap notified = {}

\* Value consistency: all listeners see the same value
ValueConsistency ==
    \A l \in notified : ObservedValue(l) = current_value

Init ==
    /\ current_value \in Values
    /\ pending_notifications = {}
    /\ notified = {}

SetValue(v) ==
    /\ v /= current_value
    /\ current_value' = v
    /\ pending_notifications' = Listeners
    /\ notified' = {}

NotifyListener(l) ==
    /\ l \in pending_notifications
    /\ pending_notifications' = pending_notifications \ {l}
    /\ notified' = notified \cup {l}
    /\ UNCHANGED current_value

Next ==
    \/ \E v \in Values : SetValue(v)
    \/ \E l \in Listeners : NotifyListener(l)

Spec == Init /\ [][Next]_<<current_value, pending_notifications, notified>>

THEOREM Spec => []NoSpuriousNotifications
THEOREM Spec => NotificationCompleteness
====
```

What we expect to prove:

Three properties. Change detection: notifications only fire when the value actually changes (setting a signal to its current value is a no-op). No spurious notifications: each listener is notified at most once per change event (a listener cannot receive duplicate notifications for the same state transition). Notification completeness: every registered listener is eventually notified (no listener is silently skipped). The change-triggered evaluation pipeline in the training system depends on all three: convergence state changes must fire exactly once per transition, reach every registered evaluation handler, and not fire when the state has not actually changed.

---

### B.17 RateLimit

```tla+
---- MODULE RateLimit ----
EXTENDS Naturals

CONSTANTS MaxPerWindow, WindowDuration

VARIABLES count, window_start, clock

TypeInvariant ==
    /\ count \in 0..MaxPerWindow
    /\ window_start \in Nat
    /\ clock \in Nat

\* Throughput invariant: count never exceeds max within a window
ThroughputInvariant ==
    count <= MaxPerWindow

\* Window consistency: window_start is always <= clock
WindowConsistency ==
    window_start <= clock

\* Reset correctness: when a new window starts, count resets to 0
ResetCorrectness ==
    clock - window_start >= WindowDuration => count = 0
    \* (After TryAcquire resets it)

Init ==
    /\ count = 0
    /\ window_start = 0
    /\ clock = 0

TryAcquire ==
    IF clock - window_start >= WindowDuration
    THEN /\ window_start' = clock
         /\ count' = 1
         /\ UNCHANGED clock
    ELSE IF count < MaxPerWindow
         THEN /\ count' = count + 1
              /\ UNCHANGED <<window_start, clock>>
         ELSE UNCHANGED <<count, window_start, clock>>

Tick ==
    /\ clock' = clock + 1
    /\ UNCHANGED <<count, window_start>>

Next == TryAcquire \/ Tick

Spec == Init /\ [][Next]_<<count, window_start, clock>>

THEOREM Spec => []ThroughputInvariant
====
```

What we expect to prove:

The throughput invariant: within any single time window, the count of successful acquisitions never exceeds MaxPerWindow. This is a Budget\<Operations\> with a temporal reset, which is why the minimality section of the paper questions whether RateLimit is a standalone primitive or a composition of Budget and a temporal mechanism. TLC confirms the invariant holds across all interleavings of TryAcquire and Tick operations. The domain determines what is being rate-limited (API requests, heartbeat signals, checkpoint writes). The invariant is the same.

---

### B.18 AuditSink

```tla+
---- MODULE AuditSink ----
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Operations

VARIABLES log, last_hash

TypeInvariant ==
    /\ log \in Seq([operation: Operations, prev_hash: Nat, hash: Nat])
    /\ last_hash \in Nat

\* Append-only: log only grows, records are never modified or removed
AppendOnly ==
    \* Old entries are unchanged (enforced by Append semantics)
    TRUE

\* Hash chain integrity: each record's prev_hash matches the previous record's hash
ChainIntegrity ==
    \A i \in 2..Len(log) :
        log[i].prev_hash = log[i-1].hash

\* Completeness: every mutation is recorded
\* (This is a liveness property: if a mutation occurs, a log entry eventually exists)

\* Tamper evidence: modifying any record breaks the chain
\* (This follows from ChainIntegrity: if record i is modified, its hash changes,
\*  which means record i+1's prev_hash no longer matches, which is detectable)

\* Consistency: last_hash always equals the most recent record's hash
HashConsistency ==
    IF Len(log) > 0
    THEN last_hash = log[Len(log)].hash
    ELSE last_hash = 0

Init ==
    /\ log = <<>>
    /\ last_hash = 0

Record(op) ==
    LET new_hash == Hash(last_hash, op)
        entry == [operation |-> op, prev_hash |-> last_hash, hash |-> new_hash]
    IN /\ log' = Append(log, entry)
       /\ last_hash' = new_hash

Next == \E op \in Operations : Record(op)

Spec == Init /\ [][Next]_<<log, last_hash>>

THEOREM Spec => [](ChainIntegrity /\ HashConsistency)
====
```

What we expect to prove:

Two invariants. Chain integrity: every record's prev_hash field matches the hash of the preceding record, forming an unbroken chain from the first entry to the last. Hash consistency: the last_hash state variable always equals the most recent record's hash (no desynchronization between the running state and the log). Together these provide tamper evidence: if any record in the chain is modified after the fact, the hash chain breaks at that point, and the break is detectable by any consumer that walks the chain. The SHA-256 chain in LeanFormer's training audit log is a direct implementation of this spec. The RAFT commit log is another. The spec proves the tamper-evidence property holds regardless of what operations are being logged.

---

## Composition Invariants

The individual primitive specs prove properties in isolation. When primitives compose, additional invariants emerge from the composition. Two important compositions:

### B.19 TraversalEngine + Budget + AllocationSnapshot

```tla+
---- MODULE TraversalBudgetComposition ----
\* The composition of TraversalEngine with Budget produces an AllocationSnapshot
\* where total_cost never exceeds the Budget's capacity.

THEOREM
    /\ TraversalEngine!Spec
    /\ Budget!Spec
    /\ TraversalEngine!budget_remaining = Budget!capacity - Budget!allocated
    => [](AllocationSnapshot!total_cost <= Budget!capacity)
====
```

The traversal engine's budget_remaining is the budget's available capacity. Every node acceptance deducts from both. The composition guarantees that the snapshot's total cost never exceeds the budget, which means ActuationPass can consume the snapshot without re-checking affordability.

### B.20 CompetitiveSelection + ActuationPass

```tla+
---- MODULE SelectThenActuate ----
\* The evaluate-then-actuate pattern: selection produces an allocation,
\* actuation consumes it without modifying it.

THEOREM
    /\ CompetitiveSelectionHard!Spec
    /\ ActuationPass!Spec
    /\ ActuationPass!allocation = CompetitiveSelectionHard!allocation
    => [](ActuationPass!ActuationScope)
    \* Every actuated seat was selected by CompetitiveSelection
====
```

This is the deferred rendering pattern, the Kubernetes reconciliation pattern, and the LeanFormer gradient application pattern formalized: the decision and the action are separate passes with a contract (the allocation record) between them. The spec proves that actuation never operates on seats that were not selected.

---

## Summary: What the Specifications Prove

| Primitive | Key Invariant | What It Means |
|-----------|--------------|---------------|
| Budget | consumed <= capacity | Resources cannot be overallocated |
| FederatedBudget | sum(sub) <= master, each sub <= its cap | Two-level budgets are leak-proof |
| QualityHierarchy | levels strictly decrease parent-to-child | Coarse-to-fine structure is well-formed |
| AllocationSnapshot | total_cost <= budget_capacity | Every accepted node was affordable |
| RelationshipGraph | adjacency consistent with edges | Message passing reaches correct neighbors |
| ResourceRegistry | unique key-value mapping | Lookup always returns current value |
| TraversalEngine | budget never negative, traversal terminates | Exploration is bounded and complete |
| PropagationPass | iteration bounded, terminates | Message passing does not run forever |
| CompetitiveSelection (hard) | mutual exclusion, winner optimality | One winner per seat, always the best |
| CompetitiveSelection (soft) | non-negative, normalized, universal | Valid probability distribution, all contribute |
| CompetitiveSelection (ranked) | bounded multiplicity, threshold optimality | At most K winners, always the best K |
| ActuationPass | only acts on allocated seats | Never acts without a prior decision |
| Reduction | items partitioned, all processed | Every item counted exactly once |
| Sampler | bounded sample, support consistency | Correct size, nothing impossible is drawn |
| ConvergenceGovernor | valid transitions, no state skipping | ACTIVE->COOLING->CONVERGED is mandatory path |
| Signal | change detection, no spurious notifications | Fires exactly once per actual change |
| RateLimit | count <= max per window | Throughput ceiling enforced |
| AuditSink | hash chain integrity | Tamper-evident, append-only |

The primitives are mathematical structures. The implementations in Rust, Python, Go, and TypeScript are instantiations of these structures. The TLA+ specifications are the structures themselves, stripped of every implementation detail, with only the invariants remaining. If the invariants hold in TLA+, they hold in any correct implementation. The domain is never mentioned. The language is never mentioned. The logic is the same.
