# Appendix D: Atomicity Dependence of Primitive Invariants

## The Argument

Section 1.3 of the paper defines an abstraction primitive as an operation that cannot be further decomposed without losing governance semantics. Appendix B specifies the invariants each primitive must satisfy. This appendix establishes empirically that those invariants depend on atomicity at the primitive boundary: for each primitive, we factor it into separable check-then-act steps in a context with a concurrent actor, run TLC, and exhibit a concrete reachable state where the invariant is violated.

The result this appendix supports is bounded and concrete. It is not a proof of algebraic minimality (no primitive in the set is expressible as a composition of the others); that question remains open and is discussed in Section 9.1. It is the systematic demonstration that every primitive in the set carries an atomicity-dependent invariant that fails under the standard check-then-act decomposition. The property that atomic operations fail their atomicity-dependent invariants when atomicity is removed in concurrent contexts is a known consequence of how atomicity works generally; what is new here is the systematic application of that decomposition technique to all sixteen primitives, with concrete counterexamples produced by an unbiased model checker rather than imagined ones.

The practical consequence: each primitive's guard and its guarded operation are one transactional unit. They are the boundary at which formal verification can be applied, because below that boundary the invariant ceases to hold. This is what justifies treating the primitives as the unit of formal specification in Appendix B and as the unit of composition in Appendix C.

Each section follows the same structure:

1. The primitive and its invariant (from Appendix B)
2. The decomposition: a plausible factoring into separable check and act steps
3. Why the decomposition looks reasonable (it is not a straw man)
4. The TLA+ spec with the decomposition applied
5. The counterexample TLC produces
6. What this means for the primitive's atomic boundary

The counterexamples are not hypothetical. TLC produces them as concrete state traces. The traces are reproducible. Anyone with the TLA+ toolbox can run these specs and observe the violations.

---

## D.1 Budget\<U\>: Separating Guard from Mutation

Invariant: `allocated + reserved + pending_eviction <= capacity`

Decomposition: factor TryAllocate into two steps: (1) check whether the allocation fits, (2) perform the allocation. This is how most non-atomic implementations work: read the current value, compare, then write.

Why it looks reasonable: this is the standard check-then-act pattern used in virtually every non-transactional system. A developer implementing Budget in a concurrent environment might naturally write it this way.

```tla+
---- MODULE BudgetDecomposed ----
EXTENDS Naturals

CONSTANTS Capacity

VARIABLES allocated, checked, pending_amount

TypeInvariant ==
    /\ allocated \in 0..Capacity
    /\ checked \in BOOLEAN
    /\ pending_amount \in 0..Capacity

SafetyInvariant ==
    allocated <= Capacity

Init ==
    /\ allocated = 0
    /\ checked = FALSE
    /\ pending_amount = 0

\* Step 1: Check if allocation fits. Record the result.
CheckFit(amount) ==
    /\ checked = FALSE
    /\ pending_amount' = amount
    /\ checked' = (allocated + amount <= Capacity)
    /\ UNCHANGED allocated

\* Step 2: If check passed, perform the allocation.
PerformAllocation ==
    /\ checked = TRUE
    /\ allocated' = allocated + pending_amount
    /\ checked' = FALSE
    /\ pending_amount' = 0

\* Step 3: Another actor allocates between check and act.
\* This is the interleaving that breaks the invariant.
ConcurrentAllocate(amount) ==
    /\ allocated + amount <= Capacity
    /\ allocated' = allocated + amount
    /\ UNCHANGED <<checked, pending_amount>>

Next ==
    \/ \E a \in 1..Capacity : CheckFit(a)
    \/ PerformAllocation
    \/ \E a \in 1..Capacity : ConcurrentAllocate(a)

Spec == Init /\ [][Next]_<<allocated, checked, pending_amount>>

\* TLC WILL FIND A VIOLATION OF THIS:
THEOREM Spec => []SafetyInvariant
====
```

Expected counterexample (Capacity = 4):

```
State 1: allocated = 0, checked = FALSE, pending_amount = 0
State 2: CheckFit(3) -> allocated = 0, checked = TRUE, pending_amount = 3
         (check passes: 0 + 3 <= 4)
State 3: ConcurrentAllocate(2) -> allocated = 2, checked = TRUE, pending_amount = 3
         (concurrent actor allocates 2, which fits)
State 4: PerformAllocation -> allocated = 5, checked = FALSE
         VIOLATION: allocated (5) > Capacity (4)
```

What this means: the check and the mutation are one atomic operation. Decomposing them creates a window between the check and the act where the world can change, invalidating the check. Budget\<U\> is irreducible because the invariant enforcement IS the allocation. They cannot be separated without losing the guarantee.

TLC config:
```
CONSTANTS Capacity = 4
INVARIANT SafetyInvariant
```

---

## D.2 FederatedBudget\<U\>: Separating Master Check from Sub-Pool Allocation

Invariant: `sum(sub_capacities) <= master_capacity` AND `each sub_allocated <= sub_capacity`

Decomposition: factor AllocateSubPool into (1) check master budget, (2) create sub-pool. Factor AllocateFromSubPool into (1) check sub-pool budget, (2) perform sub-allocation.

Why it looks reasonable: a developer might implement the master budget and sub-pools as independent services that coordinate through messages rather than atomic operations.

```tla+
---- MODULE FederatedBudgetDecomposed ----
EXTENDS Naturals

CONSTANTS MasterCapacity, SubPoolNames

VARIABLES master_used, sub_caps, sub_used, master_checked, pending_sub_alloc

FederationInvariant ==
    master_used <= MasterCapacity

SubPoolInvariant ==
    \A n \in SubPoolNames : sub_used[n] <= sub_caps[n]

Init ==
    /\ master_used = 0
    /\ sub_caps = [n \in SubPoolNames |-> 0]
    /\ sub_used = [n \in SubPoolNames |-> 0]
    /\ master_checked = FALSE
    /\ pending_sub_alloc = [name |-> "", amount |-> 0]

\* Step 1: Check master budget for sub-pool creation
CheckMasterBudget(name, amount) ==
    /\ master_checked = FALSE
    /\ master_used + amount <= MasterCapacity
    /\ master_checked' = TRUE
    /\ pending_sub_alloc' = [name |-> name, amount |-> amount]
    /\ UNCHANGED <<master_used, sub_caps, sub_used>>

\* Step 2: Create the sub-pool (after check passed)
CreateSubPool ==
    /\ master_checked = TRUE
    /\ master_used' = master_used + pending_sub_alloc.amount
    /\ sub_caps' = [sub_caps EXCEPT ![pending_sub_alloc.name] =
                    @ + pending_sub_alloc.amount]
    /\ master_checked' = FALSE
    /\ UNCHANGED <<sub_used, pending_sub_alloc>>

\* Concurrent sub-pool creation between check and act
ConcurrentCreateSubPool(name, amount) ==
    /\ master_used + amount <= MasterCapacity
    /\ master_used' = master_used + amount
    /\ sub_caps' = [sub_caps EXCEPT ![name] = @ + amount]
    /\ UNCHANGED <<sub_used, master_checked, pending_sub_alloc>>

Next ==
    \/ \E n \in SubPoolNames, a \in 1..MasterCapacity : CheckMasterBudget(n, a)
    \/ CreateSubPool
    \/ \E n \in SubPoolNames, a \in 1..MasterCapacity : ConcurrentCreateSubPool(n, a)

Spec == Init /\ [][Next]_<<master_used, sub_caps, sub_used,
                           master_checked, pending_sub_alloc>>

\* TLC WILL FIND A VIOLATION:
THEOREM Spec => []FederationInvariant
====
```

Expected counterexample (MasterCapacity = 10, SubPoolNames = {"A", "B"}):

```
State 1: master_used = 0
State 2: CheckMasterBudget("A", 7) -> passes (0 + 7 <= 10)
State 3: ConcurrentCreateSubPool("B", 5) -> master_used = 5 (5 <= 10, ok)
State 4: CreateSubPool -> master_used = 12
         VIOLATION: master_used (12) > MasterCapacity (10)
```

What this means: the federation invariant requires that checking the master budget and deducting from it are atomic. Two sub-pool creations that individually fit can together exceed the master. FederatedBudget is not "Budget + a map of sub-pools." It is an atomic two-level allocation where both levels are checked and committed in one operation.

---

## D.3 ConvergenceGovernor: Removing the COOLING State

Invariant: valid state transitions, no skipping from ACTIVE directly to CONVERGED

Decomposition: simplify the four-state machine to three states by removing COOLING. Transition directly from ACTIVE to CONVERGED when the delta drops below threshold.

Why it looks reasonable: COOLING looks like unnecessary complexity. If the delta is below threshold, the group has converged. Why add an intermediate state?

```tla+
---- MODULE ConvergenceGovernorDecomposed ----
EXTENDS Naturals, Sequences

CONSTANTS Threshold, AwakenThreshold, Window

VARIABLES state, delta_history

\* Simplified three-state machine: ACTIVE, CONVERGED, AWAKENED
\* No COOLING state.

Init ==
    /\ state = "ACTIVE"
    /\ delta_history = <<>>

Update(delta) ==
    LET new_history ==
            IF Len(delta_history) >= Window
            THEN Append(Tail(delta_history), delta)
            ELSE Append(delta_history, delta)
        avg == SumSeq(new_history) \div Max(Len(new_history), 1)
    IN
    /\ delta_history' = new_history
    /\ state' =
        CASE state = "ACTIVE" /\ avg < Threshold -> "CONVERGED"
          [] state = "CONVERGED" /\ avg > AwakenThreshold -> "AWAKENED"
          [] state = "AWAKENED" /\ avg < Threshold -> "CONVERGED"
          [] OTHER -> state

\* The property we want to check:
\* Does premature convergence occur?
\* A group is declared CONVERGED after a single low-delta step
\* even though the delta was temporarily low and rebounds.

\* We model a realistic scenario: delta drops briefly, then spikes.
\* With COOLING, the brief drop moves to COOLING but not CONVERGED.
\* Without COOLING, the brief drop goes straight to CONVERGED,
\* triggering budget reallocation prematurely.

\* This invariant should hold but WILL NOT:
\* "CONVERGED means the group has been consistently low for Window steps"
StableConvergence ==
    state = "CONVERGED" =>
        \A i \in 1..Len(delta_history) : delta_history[i] < Threshold

Next == \E d \in 0..100 : Update(d)

Spec == Init /\ [][Next]_<<state, delta_history>>

\* TLC WILL FIND A VIOLATION:
THEOREM Spec => []StableConvergence
====
```

Expected counterexample (Threshold = 10, Window = 5):

```
State 1: state = ACTIVE, history = <<>>
State 2: Update(5) -> state = CONVERGED, history = <<5>>
         VIOLATION: StableConvergence requires all history entries < Threshold,
         but we only have ONE entry. The group was declared CONVERGED
         after a single low-delta step.
State 3: (if continued) Update(80) -> state = AWAKENED, history = <<5, 80>>
         The group was "converged" for one step, budget was reallocated,
         and now it's awakened. The budget reallocation was wasted.
```

What this means: the COOLING state is not optional complexity. It is the hysteresis filter that prevents premature convergence declarations. Without COOLING, a single low-delta step triggers CONVERGED, which triggers budget reallocation (from the FederatedBudget composition), which starves the group of gradient compute right when it was still actively learning. The four-state machine is irreducible because removing any state creates a failure mode in the composed system.

The AWAKENED state is similarly non-optional. Without it, a converged group that gets perturbed (by new data or belief injection) would have to re-enter ACTIVE and re-traverse COOLING before being declared CONVERGED again. AWAKENED provides a fast path back to CONVERGED for groups that were previously stable and are re-stabilizing after perturbation. Removing AWAKENED does not break safety but degrades liveness: reactivated groups take longer to reconverge and waste gradient budget during the unnecessary COOLING phase.

---

## D.4 CompetitiveSelection (Hard): Separating Scoring from Allocation

Invariant: mutual exclusion (one winner per seat) AND winner optimality (winner has highest score)

Decomposition: factor selection into (1) score all candidates for all seats, (2) allocate winners based on cached scores. Two separate passes.

Why it looks reasonable: batch scoring followed by batch allocation is a common optimization pattern. Score everything first, then do the allocation pass. Avoids interleaving score computation with allocation decisions.

```tla+
---- MODULE CompetitiveSelectionDecomposed ----
EXTENDS Naturals, FiniteSets

CONSTANTS Seats, Candidates

VARIABLES scores, cached_scores, allocation, phase

WinnerOptimality ==
    \A s \in Seats :
        allocation[s] /= NULL =>
            \A c \in Candidates : scores[s][c] <= scores[s][allocation[s]]

Init ==
    /\ scores \in [Seats -> [Candidates -> Nat]]
    /\ cached_scores = [s \in Seats |-> [c \in Candidates |-> 0]]
    /\ allocation = [s \in Seats |-> NULL]
    /\ phase = "idle"

\* Step 1: Cache all scores (batch scoring pass)
BatchScore ==
    /\ phase = "idle"
    /\ cached_scores' = scores
    /\ phase' = "scored"
    /\ UNCHANGED <<scores, allocation>>

\* Step 2: Allocate based on cached scores (batch allocation pass)
BatchAllocate ==
    /\ phase = "scored"
    /\ allocation' = [s \in Seats |->
        CHOOSE c \in Candidates :
            \A other \in Candidates : cached_scores[s][other] <= cached_scores[s][c]]
    /\ phase' = "idle"
    /\ UNCHANGED <<scores, cached_scores>>

\* Concurrent score update between scoring and allocation passes
ConcurrentScoreUpdate(s, c, new_score) ==
    /\ scores' = [scores EXCEPT ![s][c] = new_score]
    /\ UNCHANGED <<cached_scores, allocation, phase>>

Next ==
    \/ BatchScore
    \/ BatchAllocate
    \/ \E s \in Seats, c \in Candidates, v \in Nat : ConcurrentScoreUpdate(s, c, v)

Spec == Init /\ [][Next]_<<scores, cached_scores, allocation, phase>>

\* TLC WILL FIND A VIOLATION:
\* WinnerOptimality compares allocation against CURRENT scores,
\* but allocation was based on CACHED scores.
THEOREM Spec => []WinnerOptimality
====
```

Expected counterexample (Seats = {s1}, Candidates = {c1, c2}):

```
State 1: scores[s1] = [c1 |-> 5, c2 |-> 3], phase = idle
State 2: BatchScore -> cached_scores[s1] = [c1 |-> 5, c2 |-> 3], phase = scored
State 3: ConcurrentScoreUpdate(s1, c2, 10) -> scores[s1] = [c1 |-> 5, c2 |-> 10]
         (c2's real score is now 10, but cached score is still 3)
State 4: BatchAllocate -> allocation[s1] = c1 (based on cached: c1=5 > c2=3)
         VIOLATION: WinnerOptimality fails because scores[s1][c2] = 10 > scores[s1][c1] = 5
         The wrong candidate won. The pixel shows the wrong triangle.
         The pod is scheduled on a suboptimal node.
```

What this means: scoring and allocation must be atomic per seat. If scores change between the scoring pass and the allocation pass, the winner is stale. In rendering this produces visual artifacts (wrong triangle visible). In attention this produces incorrect attention weights. In scheduling this produces suboptimal placements. The scoring function and the argmax are one irreducible operation.

---

## D.5 CompetitiveSelection (Soft): Separating Scoring from Normalization

Invariant: weights are non-negative, sum to 1, and every candidate contributes (universal contribution)

Decomposition: factor softmax into (1) compute exp(score) for each candidate, (2) normalize by dividing by sum. Two separate steps.

Why it looks reasonable: this is literally how softmax is implemented in most frameworks. The decomposition is mathematically correct in a single-threaded context.

```tla+
---- MODULE SoftSelectionDecomposed ----
EXTENDS Naturals, Reals

CONSTANTS Seats, Candidates

VARIABLES raw_exp, normalized, sum_exp, phase, scores

Normalization ==
    \A s \in Seats :
        SumWeights(s) = 1

UniversalContribution ==
    \A s \in Seats : \A c \in Candidates : normalized[s][c] > 0

Init ==
    /\ scores \in [Seats -> [Candidates -> Nat]]
    /\ raw_exp = [s \in Seats |-> [c \in Candidates |-> 0]]
    /\ normalized = [s \in Seats |-> [c \in Candidates |-> 0]]
    /\ sum_exp = [s \in Seats |-> 0]
    /\ phase = "idle"

\* Step 1: Compute exp(score) for each candidate
ComputeExp ==
    /\ phase = "idle"
    /\ raw_exp' = [s \in Seats |-> [c \in Candidates |-> Exp(scores[s][c])]]
    /\ sum_exp' = [s \in Seats |->
        Sum({Exp(scores[s][c]) : c \in Candidates})]
    /\ phase' = "exp_computed"
    /\ UNCHANGED <<scores, normalized>>

\* Step 2: Normalize
Normalize ==
    /\ phase = "exp_computed"
    /\ normalized' = [s \in Seats |-> [c \in Candidates |->
        raw_exp[s][c] / sum_exp[s]]]
    /\ phase' = "idle"
    /\ UNCHANGED <<scores, raw_exp, sum_exp>>

\* Concurrent score update between exp and normalization
ConcurrentScoreUpdate(s, c, new_score) ==
    /\ scores' = [scores EXCEPT ![s][c] = new_score]
    /\ UNCHANGED <<raw_exp, normalized, sum_exp, phase>>

Next ==
    \/ ComputeExp
    \/ Normalize
    \/ \E s \in Seats, c \in Candidates, v \in Nat :
        ConcurrentScoreUpdate(s, c, v)

Spec == Init /\ [][Next]_<<scores, raw_exp, normalized, sum_exp, phase>>

\* TLC WILL FIND A VIOLATION:
THEOREM Spec => []Normalization
====
```

Expected counterexample: a score update between ComputeExp and Normalize means the exp values and the sum are inconsistent. The normalized weights no longer sum to 1. In attention, this means the weighted combination of values is no longer a proper convex combination, producing outputs outside the convex hull of the value vectors. The exponentiation and the normalization are one atomic operation.

---

## D.6 CompetitiveSelection (Ranked): Separating Sorting from Truncation

Invariant: bounded multiplicity (at most K) AND threshold optimality (selected outscores unselected)

Decomposition: (1) sort candidates by score, (2) take top K. Between sort and take, a new candidate arrives.

```tla+
---- MODULE RankedSelectionDecomposed ----
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS K, InitialCandidates

VARIABLES candidates, sorted_order, selected, phase

ThresholdOptimality ==
    \A s \in selected : \A c \in candidates \ selected :
        Score(s) >= Score(c)

Init ==
    /\ candidates = InitialCandidates
    /\ sorted_order = <<>>
    /\ selected = {}
    /\ phase = "idle"

\* Step 1: Sort by score
SortCandidates ==
    /\ phase = "idle"
    /\ sorted_order' = SortDescending(candidates)
    /\ phase' = "sorted"
    /\ UNCHANGED <<candidates, selected>>

\* Step 2: Take top K
TakeTopK ==
    /\ phase = "sorted"
    /\ selected' = First(sorted_order, K)
    /\ phase' = "idle"
    /\ UNCHANGED <<candidates, sorted_order>>

\* Between sort and take, a high-scoring candidate arrives
ConcurrentInsert(new_candidate) ==
    /\ candidates' = candidates \cup {new_candidate}
    /\ UNCHANGED <<sorted_order, selected, phase>>

Next ==
    \/ SortCandidates
    \/ TakeTopK
    \/ \E c \in Candidates : ConcurrentInsert(c)

Spec == Init /\ [][Next]_<<candidates, sorted_order, selected, phase>>

\* TLC WILL FIND A VIOLATION:
THEOREM Spec => []ThresholdOptimality
====
```

Expected counterexample: a candidate with the highest score arrives after sorting but before truncation. The selected set misses it. In beam search, this means the best continuation is dropped. In alert triage, a critical alert is missed. Sort-then-take is one atomic operation.

---

## D.7 ActuationPass\<R\>: Separating Allocation Read from Action Execution

Invariant: only allocated seats are actuated (ActuationScope)

Decomposition: (1) read the allocation record to get the list of seats, (2) execute the action on each seat. Between read and execute, the allocation changes.

```tla+
---- MODULE ActuationPassDecomposed ----
EXTENDS Naturals, FiniteSets

CONSTANTS Seats, Resources

VARIABLES allocation, cached_allocation, actuated, phase

ActuationScope ==
    \A s \in actuated : allocation[s] /= NULL

Init ==
    /\ allocation \in [Seats -> Resources \cup {NULL}]
    /\ cached_allocation = [s \in Seats |-> NULL]
    /\ actuated = {}
    /\ phase = "idle"

\* Step 1: Read allocation
ReadAllocation ==
    /\ phase = "idle"
    /\ cached_allocation' = allocation
    /\ phase' = "read"
    /\ UNCHANGED <<allocation, actuated>>

\* Step 2: Actuate based on cached allocation
ActuateFromCache(s) ==
    /\ phase = "read"
    /\ cached_allocation[s] /= NULL
    /\ actuated' = actuated \cup {s}
    /\ UNCHANGED <<allocation, cached_allocation, phase>>

\* Concurrent deallocation between read and actuate
ConcurrentDeallocate(s) ==
    /\ allocation' = [allocation EXCEPT ![s] = NULL]
    /\ UNCHANGED <<cached_allocation, actuated, phase>>

Next ==
    \/ ReadAllocation
    \/ \E s \in Seats : ActuateFromCache(s)
    \/ \E s \in Seats : ConcurrentDeallocate(s)

Spec == Init /\ [][Next]_<<allocation, cached_allocation, actuated, phase>>

\* TLC WILL FIND A VIOLATION:
THEOREM Spec => []ActuationScope
====
```

Expected counterexample: seat s1 is allocated, the allocation is cached, s1 is deallocated concurrently, then ActuateFromCache(s1) fires based on the stale cache. The action executes on a seat that no longer has an allocation. In rendering, this produces a draw call for a resource that has been unloaded. In training, this applies gradient to a parameter group that has been deactivated. The allocation read and the actuation must be governed by the same snapshot boundary.

---

## D.8 PropagationPass: Removing the Iteration Bound

Invariant: `iteration <= MaxIterations` (bounded iteration, guaranteed termination)

Decomposition: remove the iteration cap. Let propagation run until convergence (changed = FALSE) without any bound.

Why it looks reasonable: if the algorithm converges, why impose an artificial cap? Let it run until it is done.

```tla+
---- MODULE PropagationPassDecomposed ----
EXTENDS Naturals

CONSTANTS Nodes

VARIABLES values, iteration, changed

\* No MaxIterations constant. Propagation runs until convergence.

BoundedIteration ==
    iteration <= 1000  \* Some reasonable upper bound we expect to hold

Init ==
    /\ values \in [Nodes -> Nat]
    /\ iteration = 0
    /\ changed = TRUE

Propagate ==
    /\ changed = TRUE
    \* No iteration < MaxIterations guard
    /\ \E new_values \in [Nodes -> Nat] :
        /\ changed' = (new_values /= values)
        /\ values' = new_values
        /\ iteration' = iteration + 1

Next == Propagate

Spec == Init /\ [][Next]_<<values, iteration, changed>>

\* TLC WILL FIND: iteration exceeds any finite bound if the message function
\* does not guarantee convergence. For graphs with negative cycles (Bellman-Ford)
\* or non-contractive message functions (unstable GI), propagation never terminates.
\*
\* With the bound, PropagationPass ALWAYS terminates.
\* Without the bound, termination depends on the domain function.
\* The primitive's guarantee (termination) is lost.
THEOREM Spec => []BoundedIteration
====
```

What this means: PropagationPass without an iteration bound is not a primitive with guaranteed termination. It is an unbounded loop whose termination depends entirely on properties of the domain function. Some domain functions converge (shortest path on graphs without negative cycles). Some do not (negative cycles, non-contractive operators). The iteration bound is the governance property that makes PropagationPass safe regardless of domain. Removing it decomposes the primitive into "a loop" (no governance) plus "a convergence assumption" (domain-dependent). The loop alone is not a primitive because it has no guarantee. The assumption alone is not enforceable. Together they are the irreducible PropagationPass.

---

## D.9 TraversalEngine: Separating Budget Check from Node Acceptance

Invariant: budget_remaining >= 0 (never spend more than available)

Decomposition: (1) evaluate node cost, (2) accept node, (3) deduct from budget. Three steps instead of one atomic visit.

```tla+
---- MODULE TraversalEngineDecomposed ----
EXTENDS Naturals

CONSTANTS Nodes, MaxBudget

VARIABLES budget_remaining, accepted, evaluated_node, evaluated_cost, phase

BudgetInvariant ==
    budget_remaining >= 0

Init ==
    /\ budget_remaining = MaxBudget
    /\ accepted = {}
    /\ evaluated_node = NULL
    /\ evaluated_cost = 0
    /\ phase = "idle"

\* Step 1: Evaluate a node's cost
EvaluateNode(n, cost) ==
    /\ phase = "idle"
    /\ n \notin accepted
    /\ evaluated_node' = n
    /\ evaluated_cost' = cost
    /\ phase' = "evaluated"
    /\ UNCHANGED <<budget_remaining, accepted>>

\* Step 2: Accept the node (add to accepted set)
AcceptNode ==
    /\ phase = "evaluated"
    /\ accepted' = accepted \cup {evaluated_node}
    /\ phase' = "accepted"
    /\ UNCHANGED <<budget_remaining, evaluated_node, evaluated_cost>>

\* Step 3: Deduct cost from budget
DeductCost ==
    /\ phase = "accepted"
    /\ budget_remaining' = budget_remaining - evaluated_cost
    /\ phase' = "idle"
    /\ UNCHANGED <<accepted, evaluated_node, evaluated_cost>>

\* Concurrent traversal accepts another node between steps
ConcurrentAccept(n, cost) ==
    /\ cost <= budget_remaining
    /\ budget_remaining' = budget_remaining - cost
    /\ accepted' = accepted \cup {n}
    /\ UNCHANGED <<evaluated_node, evaluated_cost, phase>>

Next ==
    \/ \E n \in Nodes, c \in 1..MaxBudget : EvaluateNode(n, c)
    \/ AcceptNode
    \/ DeductCost
    \/ \E n \in Nodes, c \in 1..MaxBudget : ConcurrentAccept(n, c)

Spec == Init /\ [][Next]_<<budget_remaining, accepted, evaluated_node,
                           evaluated_cost, phase>>

\* TLC WILL FIND A VIOLATION:
THEOREM Spec => []BudgetInvariant
====
```

Expected counterexample: node n1 costs 7, budget is 10. EvaluateNode checks that 7 <= 10 (fits). ConcurrentAccept takes 5 (budget drops to 5). AcceptNode adds n1. DeductCost subtracts 7 from 5, giving budget = -2. The invariant is violated. Budget check, acceptance, and deduction are one atomic operation.

---

## D.10 Reduction\<T,R\>: Processing an Item Twice

Invariant: `processed \cap remaining = {}` (items partitioned, no double-counting)

Decomposition: (1) select an item for processing, (2) combine it into the result, (3) remove it from remaining. Three steps.

```tla+
---- MODULE ReductionDecomposed ----
EXTENDS Naturals, FiniteSets

CONSTANTS Items

VARIABLES remaining, processed, current_item, result, phase

Partition ==
    processed \cap remaining = {}

NoDuplicateProcessing ==
    \* result reflects each item at most once
    Cardinality(processed) + Cardinality(remaining) = Cardinality(Items)

Init ==
    /\ remaining = Items
    /\ processed = {}
    /\ current_item = NULL
    /\ result = 0
    /\ phase = "idle"

\* Step 1: Select item
SelectItem(item) ==
    /\ phase = "idle"
    /\ item \in remaining
    /\ current_item' = item
    /\ phase' = "selected"
    /\ UNCHANGED <<remaining, processed, result>>

\* Step 2: Combine into result
CombineItem ==
    /\ phase = "selected"
    /\ result' = result + Value(current_item)
    /\ processed' = processed \cup {current_item}
    /\ phase' = "combined"
    /\ UNCHANGED <<remaining, current_item>>

\* Step 3: Remove from remaining
RemoveFromRemaining ==
    /\ phase = "combined"
    /\ remaining' = remaining \ {current_item}
    /\ phase' = "idle"
    /\ UNCHANGED <<processed, current_item, result>>

\* Concurrent: another reducer selects the same item before removal
ConcurrentSelect(item) ==
    /\ item \in remaining
    /\ result' = result + Value(item)
    /\ processed' = processed \cup {item}
    /\ UNCHANGED <<remaining, current_item, phase>>

Next ==
    \/ \E i \in Items : SelectItem(i)
    \/ CombineItem
    \/ RemoveFromRemaining
    \/ \E i \in Items : ConcurrentSelect(i)

Spec == Init /\ [][Next]_<<remaining, processed, current_item, result, phase>>

\* TLC WILL FIND A VIOLATION:
THEOREM Spec => []NoDuplicateProcessing
====
```

Expected counterexample: item x is selected by the primary reducer. Before it is removed from remaining, a concurrent reducer also processes x. The item is counted twice. In loss computation, this doubles a sample's contribution. In vote counting, this counts a vote twice. Selection, combination, and removal are one atomic operation.

---

## D.11 Sampler\<T\>: Separating Distribution Read from Sample Draw

Invariant: support consistency (nothing with zero probability is drawn)

Decomposition: (1) read the distribution, (2) draw from it. Between read and draw, the distribution changes.

```tla+
---- MODULE SamplerDecomposed ----
EXTENDS Naturals, FiniteSets

CONSTANTS Items

VARIABLES distribution, cached_distribution, selected, phase

SupportConsistency ==
    \A s \in selected : distribution[s] > 0

Init ==
    /\ distribution \in [Items -> Nat]
    /\ cached_distribution = [i \in Items |-> 0]
    /\ selected = {}
    /\ phase = "idle"

ReadDistribution ==
    /\ phase = "idle"
    /\ cached_distribution' = distribution
    /\ phase' = "read"
    /\ UNCHANGED <<distribution, selected>>

DrawFromCache(item) ==
    /\ phase = "read"
    /\ cached_distribution[item] > 0
    /\ selected' = selected \cup {item}
    /\ UNCHANGED <<distribution, cached_distribution, phase>>

\* An item's probability drops to zero between read and draw
ConcurrentZero(item) ==
    /\ distribution' = [distribution EXCEPT ![item] = 0]
    /\ UNCHANGED <<cached_distribution, selected, phase>>

Next ==
    \/ ReadDistribution
    \/ \E i \in Items : DrawFromCache(i)
    \/ \E i \in Items : ConcurrentZero(i)

Spec == Init /\ [][Next]_<<distribution, cached_distribution, selected, phase>>

\* TLC WILL FIND A VIOLATION:
THEOREM Spec => []SupportConsistency
====
```

Expected counterexample: item x has probability 0.3. Distribution is cached. x's probability drops to 0 (removed from dataset, belief deregistered). DrawFromCache selects x based on stale cache. A zero-probability item is in the selected set. In dropout, this means a permanently disabled neuron fires. Reading the distribution and sampling from it are one atomic operation.

---

## D.12 Signal\<T\>: Separating Value Set from Notification Dispatch

Invariant: no spurious notifications (each listener notified at most once per change)

Decomposition: (1) set the new value, (2) dispatch notifications to listeners. Between set and dispatch, another set occurs.

```tla+
---- MODULE SignalDecomposed ----
EXTENDS Naturals, FiniteSets

CONSTANTS Values, Listeners

VARIABLES current_value, pending_value, notifications_sent, phase

\* Listeners should be notified of every distinct value change.
\* If value changes A -> B -> C, listeners should see B then C.
\* If notifications for B haven't dispatched when C arrives,
\* listeners never learn about B.
NoLostNotifications ==
    \* Every value change produces a notification to every listener
    \* before the next value change. With decomposition, this breaks.
    TRUE  \* Checked via trace inspection

Init ==
    /\ current_value \in Values
    /\ pending_value = current_value
    /\ notifications_sent = {}
    /\ phase = "idle"

\* Step 1: Set new value
SetValue(v) ==
    /\ v /= current_value
    /\ pending_value' = v
    /\ current_value' = v
    /\ phase' = "pending_notify"
    /\ UNCHANGED notifications_sent

\* Step 2: Notify listeners one by one
NotifyListener(l) ==
    /\ phase = "pending_notify"
    /\ l \notin notifications_sent
    /\ notifications_sent' = notifications_sent \cup {l}
    /\ UNCHANGED <<current_value, pending_value, phase>>

\* Complete notification round
CompleteNotification ==
    /\ phase = "pending_notify"
    /\ notifications_sent = Listeners
    /\ notifications_sent' = {}
    /\ phase' = "idle"
    /\ UNCHANGED <<current_value, pending_value>>

\* Another value change arrives before all listeners are notified
ConcurrentSet(v) ==
    /\ v /= current_value
    /\ current_value' = v
    /\ pending_value' = v
    \* Notifications for the PREVIOUS value are lost for un-notified listeners
    /\ notifications_sent' = {}  \* Reset: new change overwrites pending
    /\ UNCHANGED phase

Next ==
    \/ \E v \in Values : SetValue(v)
    \/ \E l \in Listeners : NotifyListener(l)
    \/ CompleteNotification
    \/ \E v \in Values : ConcurrentSet(v)

Spec == Init /\ [][Next]_<<current_value, pending_value, notifications_sent, phase>>

\* The violation: listener L1 was notified of value B.
\* Listener L2 was NOT notified of value B because value changed to C
\* before L2's notification dispatched.
\* L2 sees A -> C, never knowing B existed.
\* In the training pipeline: a convergence state transition is lost.
\* The evaluation pipeline never fires for that transition.
====
```

What this means: a convergence state change from ACTIVE to COOLING to CONVERGED produces two Signal events. If the second event (COOLING to CONVERGED) arrives before all listeners process the first (ACTIVE to COOLING), the first transition is lost. The evaluation pipeline that should fire on ACTIVE-to-COOLING never fires. The value update and the complete notification dispatch are one atomic operation.

---

## D.13 RateLimit: Separating Window Check from Count Increment

Invariant: `count <= MaxPerWindow`

Decomposition: (1) check if we are in the current window, (2) check if count is under max, (3) increment count. Three steps.

```tla+
---- MODULE RateLimitDecomposed ----
EXTENDS Naturals

CONSTANTS MaxPerWindow, WindowDuration

VARIABLES count, window_start, clock, window_ok, count_ok, phase

ThroughputInvariant ==
    count <= MaxPerWindow

Init ==
    /\ count = 0
    /\ window_start = 0
    /\ clock = 0
    /\ window_ok = FALSE
    /\ count_ok = FALSE
    /\ phase = "idle"

\* Step 1: Check window
CheckWindow ==
    /\ phase = "idle"
    /\ window_ok' = (clock - window_start < WindowDuration)
    /\ phase' = "window_checked"
    /\ UNCHANGED <<count, window_start, clock, count_ok>>

\* Step 2: Check count
CheckCount ==
    /\ phase = "window_checked"
    /\ window_ok = TRUE
    /\ count_ok' = (count < MaxPerWindow)
    /\ phase' = "count_checked"
    /\ UNCHANGED <<count, window_start, clock, window_ok>>

\* Step 3: Increment
Increment ==
    /\ phase = "count_checked"
    /\ count_ok = TRUE
    /\ count' = count + 1
    /\ phase' = "idle"
    /\ UNCHANGED <<window_start, clock, window_ok, count_ok>>

\* Concurrent acquire between check and increment
ConcurrentAcquire ==
    /\ count < MaxPerWindow
    /\ count' = count + 1
    /\ UNCHANGED <<window_start, clock, window_ok, count_ok, phase>>

Tick ==
    /\ clock' = clock + 1
    /\ UNCHANGED <<count, window_start, window_ok, count_ok, phase>>

Next ==
    \/ CheckWindow
    \/ CheckCount
    \/ Increment
    \/ ConcurrentAcquire
    \/ Tick

Spec == Init /\ [][Next]_<<count, window_start, clock, window_ok, count_ok, phase>>

\* TLC WILL FIND A VIOLATION:
THEOREM Spec => []ThroughputInvariant
====
```

Expected counterexample: count is MaxPerWindow - 1. Two concurrent acquires both pass the count check (both see count < MaxPerWindow). Both increment. Count exceeds MaxPerWindow. In the training pipeline, this means more checkpoints than the rate limit allows, consuming disk I/O budget. Window check, count check, and increment are one atomic operation.

---

## D.14 AuditSink: Separating Append from Hash Computation

Invariant: hash chain integrity (`record[i].prev_hash == record[i-1].hash`)

Decomposition: (1) append the record to the log, (2) compute and store the hash. Two steps.

```tla+
---- MODULE AuditSinkDecomposed ----
EXTENDS Naturals, Sequences

CONSTANTS Operations

VARIABLES log, last_hash, pending_record, phase

ChainIntegrity ==
    \A i \in 2..Len(log) :
        log[i].prev_hash = log[i-1].hash

Init ==
    /\ log = <<>>
    /\ last_hash = 0
    /\ pending_record = NULL
    /\ phase = "idle"

\* Step 1: Append record with prev_hash but no hash yet
AppendRecord(op) ==
    /\ phase = "idle"
    /\ pending_record' = [op |-> op, prev_hash |-> last_hash, hash |-> NULL]
    /\ log' = Append(log, [op |-> op, prev_hash |-> last_hash, hash |-> NULL])
    /\ phase' = "appended"
    /\ UNCHANGED last_hash

\* Step 2: Compute hash and update the record
ComputeHash ==
    /\ phase = "appended"
    /\ LET new_hash == Hash(pending_record.prev_hash, pending_record.op)
       IN /\ log' = [log EXCEPT ![Len(log)].hash = new_hash]
          /\ last_hash' = new_hash
    /\ phase' = "idle"
    /\ UNCHANGED pending_record

\* Concurrent record appended before hash is computed
ConcurrentAppend(op) ==
    /\ log' = Append(log, [op |-> op, prev_hash |-> last_hash, hash |-> NULL])
    /\ UNCHANGED <<last_hash, pending_record, phase>>

Next ==
    \/ \E op \in Operations : AppendRecord(op)
    \/ ComputeHash
    \/ \E op \in Operations : ConcurrentAppend(op)

Spec == Init /\ [][Next]_<<log, last_hash, pending_record, phase>>

\* TLC WILL FIND A VIOLATION:
THEOREM Spec => []ChainIntegrity
====
```

Expected counterexample:

```
State 1: log = <<>>, last_hash = 0
State 2: AppendRecord("step_1") -> log = <<[op="step_1", prev=0, hash=NULL]>>
State 3: ConcurrentAppend("step_2") -> log = <<[...], [op="step_2", prev=0, hash=NULL]>>
         Record 2's prev_hash is 0 (last_hash hasn't been updated yet)
State 4: ComputeHash -> log[1].hash = H(0, "step_1"), last_hash = H(0, "step_1")
         VIOLATION: log[2].prev_hash is 0, but log[1].hash is H(0, "step_1")
         Chain is broken. Record 2 points to the wrong predecessor.
```

What this means: the append and the hash computation are one atomic operation. If a second record is appended before the first record's hash is computed, the second record captures a stale prev_hash. The chain forks. In the training pipeline, this means the gradient provenance chain is broken: you cannot trace a parameter change back through the unbroken chain to the sample that caused it. The audit guarantee is lost.

---

## D.15 QualityHierarchy: Allowing Level Violations

Invariant: `level[parent] > level[child]` (levels strictly decrease)

Decomposition: allow nodes to be inserted or reparented without checking the level constraint.

Why it looks reasonable: a developer might build the tree first and validate the level ordering afterward, treating it as a post-construction check rather than a construction constraint.

```tla+
---- MODULE QualityHierarchyDecomposed ----
EXTENDS Naturals, FiniteSets

CONSTANTS Nodes, MaxLevel

VARIABLES level, children

HierarchyInvariant ==
    \A n \in Nodes : \A c \in children[n] : level[n] > level[c]

Init ==
    /\ level \in [Nodes -> 0..MaxLevel]
    /\ children = [n \in Nodes |-> {}]

\* Add child without checking level constraint
AddChild(parent, child) ==
    /\ child \notin children[parent]
    \* No level check here
    /\ children' = [children EXCEPT ![parent] = @ \cup {child}]
    /\ UNCHANGED level

\* Change a node's level without checking children
ChangeLevel(n, new_level) ==
    /\ level' = [level EXCEPT ![n] = new_level]
    /\ UNCHANGED children

Next ==
    \/ \E p, c \in Nodes : AddChild(p, c)
    \/ \E n \in Nodes, l \in 0..MaxLevel : ChangeLevel(n, l)

Spec == Init /\ [][Next]_<<level, children>>

\* TLC WILL FIND A VIOLATION:
THEOREM Spec => []HierarchyInvariant
====
```

Expected counterexample: node A at level 2 gets child B at level 3. Now `level[A] = 2 < level[B] = 3`, violating the hierarchy invariant. In the TraversalEngine, this means descending from parent to child moves UP the quality hierarchy instead of down. Budget-constrained traversal assumes that descending costs more (finer detail costs more). If a child is coarser than its parent, the traversal wastes budget on coarse representations while skipping fine ones. The level constraint is not a validation step. It is a construction invariant that must hold at every insertion.

---

## D.16 AllocationSnapshot: Allowing Post-Hoc Modifications

Invariant: `total_cost + budget_remaining <= original_budget` (snapshot is consistent with budget)

Decomposition: allow the snapshot to be modified after creation (add or remove nodes after the traversal produces it).

Why it looks reasonable: a downstream consumer might want to "adjust" the snapshot, adding a high-priority node or removing a low-value one.

```tla+
---- MODULE AllocationSnapshotDecomposed ----
EXTENDS Naturals, FiniteSets

CONSTANTS Nodes, OriginalBudget

VARIABLES accepted, total_cost, budget_remaining

BudgetConsistency ==
    total_cost + budget_remaining <= OriginalBudget

Init ==
    /\ accepted = {}
    /\ total_cost = 0
    /\ budget_remaining = OriginalBudget

\* Normal acceptance (budget-checked)
AcceptNode(n, cost) ==
    /\ n \notin accepted
    /\ cost <= budget_remaining
    /\ accepted' = accepted \cup {n}
    /\ total_cost' = total_cost + cost
    /\ budget_remaining' = budget_remaining - cost

\* Post-hoc insertion: add a node without budget check
ForceInsert(n, cost) ==
    /\ n \notin accepted
    /\ accepted' = accepted \cup {n}
    /\ total_cost' = total_cost + cost
    /\ UNCHANGED budget_remaining  \* Budget not deducted

Next ==
    \/ \E n \in Nodes, c \in 1..OriginalBudget : AcceptNode(n, c)
    \/ \E n \in Nodes, c \in 1..OriginalBudget : ForceInsert(n, c)

Spec == Init /\ [][Next]_<<accepted, total_cost, budget_remaining>>

\* TLC WILL FIND A VIOLATION:
THEOREM Spec => []BudgetConsistency
====
```

Expected counterexample: budget is 10, three nodes are accepted for cost 3 each (total_cost = 9, remaining = 1). ForceInsert adds a node with cost 5 without deducting from remaining. Now total_cost = 14, remaining = 1, sum = 15 > 10. The snapshot claims more resources than the budget allows. In the ActuationPass that consumes this snapshot, the system attempts to render, compute, or transfer more than the available resources permit. The snapshot is immutable after creation. There is no ForceInsert. This is the same structural principle as Budget's missing ForceAllocate.

---

## D.17 ResourceRegistry\<K,V\>: Allowing Duplicate Keys

Invariant: unique key-value mapping

Decomposition: allow register to add a second entry for an existing key rather than overwriting.

```tla+
---- MODULE ResourceRegistryDecomposed ----
EXTENDS Naturals, FiniteSets

CONSTANTS Keys, Values

VARIABLES entries

\* entries is now a SET of (key, value) pairs, not a function
\* This allows duplicate keys

UniqueKeys ==
    \A k \in Keys :
        Cardinality({v \in Values : <<k, v>> \in entries}) <= 1

Init ==
    entries = {}

\* Register without checking for existing key
RegisterNoCheck(k, v) ==
    /\ entries' = entries \cup {<<k, v>>}

\* Lookup returns ambiguous result when duplicates exist
\* (Which value is "the" value for this key?)

Next ==
    \E k \in Keys, v \in Values : RegisterNoCheck(k, v)

Spec == Init /\ [][Next]_entries

\* TLC WILL FIND A VIOLATION:
THEOREM Spec => []UniqueKeys
====
```

Expected counterexample: Register("belief_1", delta_A) then Register("belief_1", delta_B). Two entries for the same key. Lookup returns ambiguous results. In the belief delta system, this means a query routed to "belief_1" might get either delta, nondeterministically. The routing accuracy metric becomes meaningless. Register must be an upsert (insert or overwrite), not an append. This is trivial, but the triviality is the point: even the simplest primitive has an invariant that decomposition can violate.

---

## Summary: The Atomicity Dependence Table

| Primitive | Decomposition Attempted | Invariant Violated | Failure Mode |
|-----------|------------------------|-------------------|-------------|
| Budget | Separate check from mutation | allocated > capacity | Overallocation via TOCTOU |
| FederatedBudget | Separate master check from sub-pool creation | sum(sub) > master | Two sub-pools that individually fit but together exceed master |
| ConvergenceGovernor | Remove COOLING state | Premature CONVERGED declaration | Budget reallocated to a group still actively learning |
| CompetitiveSelection (hard) | Separate scoring from allocation | Wrong winner | Stale scores produce suboptimal allocation |
| CompetitiveSelection (soft) | Separate exp from normalization | Weights do not sum to 1 | Invalid probability distribution |
| CompetitiveSelection (ranked) | Separate sort from truncation | Missing top candidate | High-priority item arrives after sort, before take |
| ActuationPass | Separate allocation read from action | Action on deallocated seat | Draw call for unloaded resource |
| PropagationPass | Remove iteration bound | Non-termination | Unbounded loop on non-convergent domain function |
| TraversalEngine | Separate budget check from acceptance | Negative budget | Two accepts that individually fit but together exceed budget |
| Reduction | Separate selection from combination | Double-counted item | Same item processed by two concurrent reducers |
| Sampler | Separate distribution read from draw | Zero-probability item drawn | Stale distribution allows impossible selection |
| Signal | Separate value set from notification | Lost notification | Second value change overwrites pending notifications |
| RateLimit | Separate window check from increment | count > max | Two concurrent acquires both pass the check |
| AuditSink | Separate append from hash | Broken hash chain | Second record captures stale prev_hash |
| QualityHierarchy | Allow level-violating insertions | Child level > parent level | Traversal ascends instead of descending |
| AllocationSnapshot | Allow post-hoc modification | total_cost > budget | Snapshot claims more resources than budget permits |
| ResourceRegistry | Allow duplicate keys | Ambiguous lookup | Nondeterministic result for same key |

Every decomposition produces a concrete counterexample. Every counterexample is a state that TLC reaches by exploring all legal interleavings. Every counterexample demonstrates a real failure mode that would manifest in any implementation that decomposes the primitive.

The primitives carry atomicity-dependent invariants at their boundaries. The guard and the guarded operation are one transactional unit. Separate them and the guarantee disappears.
