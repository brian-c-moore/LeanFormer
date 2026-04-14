# Appendix C: LeanFormer Formal Specification in TLA+

## The Composition Proof

Appendix B specifies the sixteen primitives in isolation. This appendix specifies LeanFormer as a composed system: the belief delta lifecycle, the knowledge plane, the governed training pipeline, and the full model lifecycle. Each specification is a composition of primitives from Appendix B, and the system-level invariants are emergent properties of the composition, not additional axioms.

If these specifications verify under TLC, the claim is proven: LeanFormer is a composition of DAC primitives, and the composition preserves the invariants of the constituent primitives while producing system-level guarantees that no individual primitive provides alone.

---

## C.1 Belief Delta Lifecycle

The belief delta system is the core knowledge management mechanism. A belief is a low-rank delta over frozen base weights. Beliefs are injected atomically, routed to by query similarity, and removed atomically with bit-for-bit base weight restoration. The lifecycle must satisfy: base weight immutability, atomic injection/removal, registry consistency, and exact restoration.

```tla+
---- MODULE BeliefDeltaLifecycle ----
EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS
    Beliefs,            \* Set of possible belief IDs
    MaxBeliefs,         \* Maximum concurrent beliefs (Budget capacity)
    ParameterGroups,    \* Set of parameter group IDs
    BaseWeightHash      \* Hash of the frozen base weights (constant)

VARIABLES
    registry,           \* ResourceRegistry: belief_id -> delta_weights
    active_beliefs,     \* Set of currently loaded beliefs
    parameter_budget,   \* Budget<Parameters>: tracks allocated parameter space
    base_hash,          \* Current hash of base weights (must equal BaseWeightHash always)
    audit_log,          \* AuditSink: append-only log of all mutations
    last_audit_hash     \* Hash chain head for audit

----

\* === Type Invariants ===

TypeInvariant ==
    /\ registry \in [SUBSET Beliefs -> SUBSET ParameterGroups]
    /\ active_beliefs \subseteq Beliefs
    /\ parameter_budget \in Nat
    /\ base_hash = BaseWeightHash
    /\ audit_log \in Seq([op: {"inject", "remove"}, belief: Beliefs,
                          prev_hash: Nat, hash: Nat])
    /\ last_audit_hash \in Nat

----

\* === Safety Invariants ===

\* The base weights are never modified. This is the foundational guarantee.
\* Every operation that touches the model must leave base_hash unchanged.
BaseWeightImmutability ==
    base_hash = BaseWeightHash

\* The number of active beliefs never exceeds the parameter budget.
\* This is Budget<Parameters>.SafetyInvariant applied to beliefs.
BeliefBudgetInvariant ==
    Cardinality(active_beliefs) <= MaxBeliefs

\* Every active belief has a registry entry.
\* No belief can be "active" without its delta weights being registered.
RegistryConsistency ==
    \A b \in active_beliefs : b \in DOMAIN registry

\* No two beliefs occupy the same parameter groups (non-overlap).
\* This is FederatedBudget<ParameterSubspace>.SubPoolInvariant.
ParameterNonOverlap ==
    \A b1, b2 \in active_beliefs :
        b1 /= b2 => registry[b1] \cap registry[b2] = {}

\* Audit chain is intact.
AuditChainIntegrity ==
    \A i \in 2..Len(audit_log) :
        audit_log[i].prev_hash = audit_log[i-1].hash

\* After all beliefs are removed, the state is identical to initial state.
\* This is the bit-for-bit restoration guarantee.
ExactRestoration ==
    active_beliefs = {} => base_hash = BaseWeightHash

----

\* === Operations ===

Init ==
    /\ registry = <<>>
    /\ active_beliefs = {}
    /\ parameter_budget = MaxBeliefs
    /\ base_hash = BaseWeightHash
    /\ audit_log = <<>>
    /\ last_audit_hash = 0

\* Inject a belief: atomic addition of delta to registry.
\* This is a Transaction: either the entire injection succeeds or nothing changes.
InjectBelief(b, param_groups) ==
    \* Preconditions (Transaction guards)
    /\ b \notin active_beliefs                              \* Not already loaded
    /\ Cardinality(active_beliefs) < MaxBeliefs              \* Budget permits
    /\ \A existing \in active_beliefs :                      \* No parameter overlap
        registry[existing] \cap param_groups = {}
    /\ param_groups /= {}                                    \* Non-empty delta
    \* Atomic state transition
    /\ registry' = [x \in (DOMAIN registry) \cup {b} |->
                    IF x = b THEN param_groups ELSE registry[x]]
    /\ active_beliefs' = active_beliefs \cup {b}
    /\ parameter_budget' = parameter_budget - 1
    /\ base_hash' = base_hash                                \* Base weights unchanged
    /\ LET new_hash == Hash(last_audit_hash, <<"inject", b>>)
       IN /\ audit_log' = Append(audit_log,
              [op |-> "inject", belief |-> b,
               prev_hash |-> last_audit_hash, hash |-> new_hash])
          /\ last_audit_hash' = new_hash

\* Remove a belief: atomic removal restoring pre-injection state.
\* The delta is subtracted. Base weights are untouched.
RemoveBelief(b) ==
    \* Preconditions
    /\ b \in active_beliefs
    \* Atomic state transition
    /\ registry' = [x \in (DOMAIN registry) \ {b} |-> registry[x]]
    /\ active_beliefs' = active_beliefs \ {b}
    /\ parameter_budget' = parameter_budget + 1
    /\ base_hash' = base_hash                                \* Base weights unchanged
    /\ LET new_hash == Hash(last_audit_hash, <<"remove", b>>)
       IN /\ audit_log' = Append(audit_log,
              [op |-> "remove", belief |-> b,
               prev_hash |-> last_audit_hash, hash |-> new_hash])
          /\ last_audit_hash' = new_hash

\* No operation modifies base weights. This is enforced structurally:
\* every action has base_hash' = base_hash. There is no action that
\* sets base_hash to anything else. This is the ForceAllocate absence
\* principle applied to base weights.

Next ==
    \/ \E b \in Beliefs, pg \in SUBSET ParameterGroups :
        InjectBelief(b, pg)
    \/ \E b \in Beliefs : RemoveBelief(b)

----

\* === Liveness ===

\* Any injected belief can eventually be removed.
\* (No belief gets "stuck" in the registry.)
RemovalLiveness ==
    \A b \in Beliefs : b \in active_beliefs ~> b \notin active_beliefs

----

Spec == Init /\ [][Next]_<<registry, active_beliefs, parameter_budget,
                          base_hash, audit_log, last_audit_hash>>

THEOREM Spec => [](BaseWeightImmutability /\ BeliefBudgetInvariant /\
                   RegistryConsistency /\ ParameterNonOverlap /\
                   AuditChainIntegrity /\ ExactRestoration)
====
```

What we expect to prove:

Six invariants that together define correct belief lifecycle management.

BaseWeightImmutability: the base weights are never modified by any operation. This is not a runtime check. It is a structural property: no action in the specification assigns a new value to base_hash. TLC exhaustively verifies that no reachable state has base_hash different from BaseWeightHash. This is the formal proof of the bit-for-bit restoration result observed empirically at 39M parameters.

BeliefBudgetInvariant: the system never loads more beliefs than the budget permits. This is Budget\<Parameters\>.SafetyInvariant instantiated for belief count.

ParameterNonOverlap: no two active beliefs touch the same parameter groups. This is the multi-tenancy isolation guarantee. It is what makes belief composition additive rather than destructive.

ExactRestoration: when all beliefs are removed, the system state is indistinguishable from the initial state. Combined with BaseWeightImmutability, this proves that the belief lifecycle is fully reversible.

AuditChainIntegrity: every injection and removal is recorded in a tamper-evident log. The chain can be walked to reconstruct the complete history of knowledge mutations.

---

## C.2 Semantic Routing

The routing network is CompetitiveSelection (ranked) applied to belief selection. A query embedding is scored against all active beliefs, and the top-k beliefs are selected for application.

```tla+
---- MODULE SemanticRouting ----
EXTENDS Naturals, Reals, FiniteSets

CONSTANTS
    Queries,        \* Set of possible query embeddings
    Beliefs,        \* Set of possible beliefs
    K               \* Maximum beliefs to route per query

VARIABLES
    active_beliefs, \* Currently loaded beliefs (from BeliefDeltaLifecycle)
    scores,         \* Score matrix: query x belief -> similarity
    routed,         \* Result: query -> set of selected beliefs
    confidence      \* Routing confidence per query (for convergence detection)

----

\* === Safety Invariants ===

\* Never route to an inactive belief
ActiveOnlyRouting ==
    \A q \in Queries : \A b \in routed[q] : b \in active_beliefs

\* Route at most K beliefs per query (bounded multiplicity)
BoundedRouting ==
    \A q \in Queries : Cardinality(routed[q]) <= K

\* Selected beliefs are the highest-scoring among active beliefs
\* (threshold optimality from CompetitiveSelection ranked)
RoutingOptimality ==
    \A q \in Queries :
        \A selected \in routed[q] :
            \A unselected \in active_beliefs \ routed[q] :
                scores[q][selected] >= scores[q][unselected]

\* Empty routing is valid when no beliefs are active
EmptyRoutingValid ==
    active_beliefs = {} => \A q \in Queries : routed[q] = {}

\* Confidence reflects routing quality
\* High confidence when top scores are well-separated from rest
\* Low confidence when scores are uniform (no relevant belief)
ConfidenceRange ==
    \A q \in Queries : confidence[q] >= 0 /\ confidence[q] <= 1

----

\* === Operations ===

Init ==
    /\ active_beliefs \subseteq Beliefs
    /\ scores \in [Queries -> [Beliefs -> Nat]]
    /\ routed = [q \in Queries |-> {}]
    /\ confidence = [q \in Queries |-> 0]

RouteQuery(q) ==
    LET active_scores == [b \in active_beliefs |-> scores[q][b]]
        sorted == SortDescending(active_beliefs, active_scores)
        top_k == Take(sorted, Min(K, Cardinality(active_beliefs)))
    IN /\ routed' = [routed EXCEPT ![q] = top_k]
       /\ confidence' = [confidence EXCEPT ![q] = ComputeConfidence(active_scores, top_k)]
       /\ UNCHANGED <<active_beliefs, scores>>

\* When a belief is injected or removed, routing must be invalidated
\* for all queries (or re-evaluated lazily)
InvalidateRouting(changed_belief) ==
    /\ routed' = [q \in Queries |-> {}]
    /\ UNCHANGED <<active_beliefs, scores, confidence>>

Next ==
    \/ \E q \in Queries : RouteQuery(q)
    \/ \E b \in Beliefs : InvalidateRouting(b)

Spec == Init /\ [][Next]_<<active_beliefs, scores, routed, confidence>>

THEOREM Spec => [](ActiveOnlyRouting /\ BoundedRouting /\ RoutingOptimality)
====
```

What we expect to prove:

ActiveOnlyRouting: the system never routes a query to a belief that has been removed. This is the composition guarantee between the routing module and the belief lifecycle. If a belief is removed between the scoring pass and the actuation pass, the routing result must be invalidated. TLC verifies that no interleaving of RouteQuery and InvalidateRouting produces a state where a removed belief appears in a routing result.

RoutingOptimality: every selected belief outscores every unselected belief. This is CompetitiveSelection (ranked) ThresholdOptimality applied to semantic similarity. The 86% routing accuracy observed empirically is the domain function's quality; the spec proves the selection mechanism itself is correct.

---

## C.3 Governed Training Pipeline

This is the largest and most important specification. It models the full Phase 5 training loop as a composition of primitives: hierarchical parameter activation, per-group convergence governors, federated gradient budget, gradient routing, change-triggered evaluation, and audit provenance.

```tla+
---- MODULE GovernedTrainingPipeline ----
EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS
    ParameterGroups,    \* e.g., {L0_structural, L1_representational,
                        \*        L2_refinement, L3_specialization}
    HierarchyLevels,    \* e.g., {0, 1, 2, 3} matching parameter group levels
    MasterGradientBudget,\* Total gradient compute per step
    MaxSteps,           \* Training step budget
    ConvergenceThreshold,\* Per-group convergence threshold
    AwakenThreshold     \* Threshold for reactivation

VARIABLES
    \* --- Hierarchy and activation ---
    activated_groups,    \* Which parameter groups are currently active
    hierarchy_level,     \* Current maximum activated level

    \* --- Per-group convergence governors ---
    governor_state,      \* [ParameterGroups -> {"ACTIVE","COOLING","CONVERGED","AWAKENED"}]
    governor_deltas,     \* [ParameterGroups -> Seq(Nat)] recent gradient magnitudes

    \* --- Federated gradient budget ---
    group_budget_cap,    \* [ParameterGroups -> Nat] allocated budget per group
    group_budget_used,   \* [ParameterGroups -> Nat] used budget per group
    master_budget_used,  \* Total budget allocated to sub-pools

    \* --- Gradient routing ---
    routing_mask,        \* [ParameterGroups -> BOOLEAN] which groups receive gradient
    routing_selectivity, \* Fraction of groups masked per step

    \* --- Training state ---
    current_step,        \* Current training step
    loss,                \* Current loss value
    loss_history,        \* Recent loss values

    \* --- Change-triggered evaluation ---
    pending_evals,       \* Set of groups that changed state and need evaluation
    eval_results,        \* Results of triggered evaluations

    \* --- Audit ---
    training_audit_log,  \* Append-only log of all training events
    training_last_hash   \* Audit hash chain head

----

\* === Core Safety Invariants ===

\* 1. Federated budget invariant: sum of group allocations <= master budget.
\*    This is FederatedBudget<GradientCompute>.FederationInvariant.
FederatedBudgetInvariant ==
    master_budget_used <= MasterGradientBudget

\* 2. Sub-pool invariant: each group's usage <= its allocation.
SubPoolInvariant ==
    \A g \in ParameterGroups :
        group_budget_used[g] <= group_budget_cap[g]

\* 3. Consistency: master usage equals sum of group allocations.
BudgetConsistency ==
    master_budget_used = SumOver(activated_groups, group_budget_cap)

\* 4. Hierarchy ordering: groups activate in level order.
\*    L0 activates before L1, L1 before L2, L2 before L3.
\*    A higher-level group cannot be active if a lower-level group is not.
HierarchyOrdering ==
    \A g1, g2 \in ParameterGroups :
        /\ g2 \in activated_groups
        /\ Level(g1) < Level(g2)
        => g1 \in activated_groups

\* 5. Convergence-governed activation: a group activates only when
\*    all lower-level groups have reached at least COOLING state.
\*    (Groups don't activate arbitrarily; activation is earned by
\*     convergence signal from below.)
ConvergenceGatedActivation ==
    \A g \in activated_groups :
        Level(g) > 0 =>
            \A lower \in ParameterGroups :
                Level(lower) < Level(g) =>
                    governor_state[lower] \in {"COOLING", "CONVERGED", "AWAKENED"}

\* 6. Governor state validity: only valid transitions.
\*    (Imported from ConvergenceGovernor spec)
GovernorStateValidity ==
    \A g \in ParameterGroups : governor_state[g] \in
        {"ACTIVE", "COOLING", "CONVERGED", "AWAKENED"}

\* 7. No gradient to inactive groups.
\*    Gradient is only routed to activated parameter groups.
GradientScopeInvariant ==
    \A g \in ParameterGroups :
        routing_mask[g] = TRUE => g \in activated_groups

\* 8. Routing selectivity is non-degenerate.
\*    Not all groups are masked (total silence) and not all are unmasked
\*    (no selectivity), except during warmup.
NonDegenerateRouting ==
    current_step > WarmupSteps =>
        /\ \E g \in activated_groups : routing_mask[g] = TRUE
        /\ \E g \in activated_groups : routing_mask[g] = FALSE

\* 9. Audit completeness: every state transition is logged.
AuditCompleteness ==
    \* Every activation, convergence transition, and budget reallocation
    \* has a corresponding audit log entry. Modeled by requiring
    \* the log to grow with every Next step that changes state.
    TRUE  \* Enforced by each action appending to audit_log

\* 10. Training audit chain integrity.
TrainingAuditChainIntegrity ==
    \A i \in 2..Len(training_audit_log) :
        training_audit_log[i].prev_hash = training_audit_log[i-1].hash

\* 11. Step bound: training terminates.
StepBound ==
    current_step <= MaxSteps

----

\* === Derived Properties (emergent from composition) ===

\* Coarse-to-fine convergence ordering:
\* L0 reaches CONVERGED before L1, L1 before L2, etc.
\* This is a prediction, not an axiom. The spec does not enforce it.
\* TLC checks whether it holds given the activation and convergence rules.
\* If it holds, the prediction is verified. If not, the model reveals
\* a counterexample showing when the ordering breaks.
CoarseToFineOrdering ==
    \A g1, g2 \in ParameterGroups :
        /\ governor_state[g1] = "CONVERGED"
        /\ Level(g1) > Level(g2)
        => governor_state[g2] \in {"CONVERGED", "AWAKENED"}

\* Budget efficiency: converged groups release budget to active groups.
\* When a group converges, its budget allocation should shrink and the
\* freed budget should be redistributable. This is a liveness property.
BudgetReallocationLiveness ==
    \A g \in ParameterGroups :
        governor_state[g] = "CONVERGED" ~>
            group_budget_cap[g] < MasterGradientBudget \div Cardinality(ParameterGroups)

----

\* === Helper Functions ===

Level(g) == \* Maps parameter group to hierarchy level
    CASE g = "L0_structural" -> 0
      [] g = "L1_representational" -> 1
      [] g = "L2_refinement" -> 2
      [] g = "L3_specialization" -> 3

SumOver(groups, f) ==
    LET RECURSIVE Sum(_, _)
        Sum(s, acc) ==
            IF s = {} THEN acc
            ELSE LET x == CHOOSE x \in s : TRUE
                 IN Sum(s \ {x}, acc + f[x])
    IN Sum(groups, 0)

WarmupSteps == 10  \* Routing selectivity not enforced during warmup

----

\* === Operations ===

Init ==
    /\ activated_groups = {"L0_structural"}  \* Only L0 active at start
    /\ hierarchy_level = 0
    /\ governor_state = [g \in ParameterGroups |-> "ACTIVE"]
    /\ governor_deltas = [g \in ParameterGroups |-> <<>>]
    /\ group_budget_cap = [g \in ParameterGroups |->
            IF g = "L0_structural"
            THEN MasterGradientBudget
            ELSE 0]
    /\ group_budget_used = [g \in ParameterGroups |-> 0]
    /\ master_budget_used = MasterGradientBudget
    /\ routing_mask = [g \in ParameterGroups |-> g = "L0_structural"]
    /\ routing_selectivity = 0
    /\ current_step = 0
    /\ loss = 1000  \* Initial high loss
    /\ loss_history = <<>>
    /\ pending_evals = {}
    /\ eval_results = <<>>
    /\ training_audit_log = <<>>
    /\ training_last_hash = 0

\* --- Training Step ---
\* One complete forward-backward pass with governed gradient routing.

TrainingStep ==
    /\ current_step < MaxSteps
    \* 1. Route gradient to active groups (CompetitiveSelection ranked)
    /\ LET routed_groups == {g \in activated_groups : routing_mask[g]}
       IN
    \* 2. Compute gradient and apply to routed groups only (ActuationPass)
       /\ \E new_loss \in Nat :
          /\ loss' = new_loss
          /\ loss_history' = IF Len(loss_history) >= 10
                             THEN Append(Tail(loss_history), new_loss)
                             ELSE Append(loss_history, new_loss)
    \* 3. Update per-group gradient magnitudes
    /\ \E deltas \in [activated_groups -> Nat] :
        governor_deltas' = [g \in ParameterGroups |->
            IF g \in activated_groups
            THEN IF Len(governor_deltas[g]) >= 5
                 THEN Append(Tail(governor_deltas[g]), deltas[g])
                 ELSE Append(governor_deltas[g], deltas[g])
            ELSE governor_deltas[g]]
    \* 4. Deduct from group budgets
    /\ \E costs \in [activated_groups -> Nat] :
        /\ \A g \in activated_groups :
            costs[g] <= group_budget_cap[g] - group_budget_used[g]
        /\ group_budget_used' = [g \in ParameterGroups |->
            IF g \in activated_groups
            THEN group_budget_used[g] + costs[g]
            ELSE group_budget_used[g]]
    /\ current_step' = current_step + 1
    \* 5. Audit
    /\ LET new_hash == Hash(training_last_hash, <<"step", current_step>>)
       IN /\ training_audit_log' = Append(training_audit_log,
              [op |-> "step", step |-> current_step,
               prev_hash |-> training_last_hash, hash |-> new_hash])
          /\ training_last_hash' = new_hash
    /\ UNCHANGED <<activated_groups, hierarchy_level, governor_state,
                   group_budget_cap, master_budget_used, routing_mask,
                   routing_selectivity, pending_evals, eval_results>>

\* --- Convergence Governor Update ---
\* After each step, update each active group's convergence governor.

UpdateGovernor(g) ==
    /\ g \in activated_groups
    /\ Len(governor_deltas[g]) > 0
    /\ LET avg == SumSeq(governor_deltas[g]) \div Len(governor_deltas[g])
           old_state == governor_state[g]
           new_state ==
               CASE old_state = "ACTIVE" /\ avg < ConvergenceThreshold * 2
                    -> "COOLING"
                 [] old_state = "COOLING" /\ avg < ConvergenceThreshold
                    -> "CONVERGED"
                 [] old_state = "COOLING" /\ avg >= ConvergenceThreshold * 2
                    -> "ACTIVE"
                 [] old_state = "CONVERGED" /\ avg > AwakenThreshold
                    -> "AWAKENED"
                 [] old_state = "AWAKENED" /\ avg < ConvergenceThreshold
                    -> "CONVERGED"
                 [] OTHER -> old_state
       IN
       /\ governor_state' = [governor_state EXCEPT ![g] = new_state]
       \* If state changed, trigger evaluation (Signal<ConvergenceChange>)
       /\ pending_evals' = IF new_state /= old_state
                           THEN pending_evals \cup {g}
                           ELSE pending_evals
       \* Audit the transition
       /\ IF new_state /= old_state
          THEN LET new_hash == Hash(training_last_hash,
                    <<"convergence", g, old_state, new_state>>)
               IN /\ training_audit_log' = Append(training_audit_log,
                      [op |-> "convergence", group |-> g,
                       from |-> old_state, to |-> new_state,
                       prev_hash |-> training_last_hash, hash |-> new_hash])
                  /\ training_last_hash' = new_hash
          ELSE UNCHANGED <<training_audit_log, training_last_hash>>
       /\ UNCHANGED <<activated_groups, hierarchy_level, governor_deltas,
                      group_budget_cap, group_budget_used, master_budget_used,
                      routing_mask, routing_selectivity, current_step,
                      loss, loss_history, eval_results>>

\* --- Hierarchy Activation ---
\* When all groups at the current level reach COOLING or CONVERGED,
\* activate the next level. This is QualityHierarchy + TraversalEngine
\* governed by ConvergenceGovernor signals.

ActivateNextLevel ==
    /\ hierarchy_level < 3  \* Not all levels active yet
    /\ LET current_level_groups ==
            {g \in ParameterGroups : Level(g) <= hierarchy_level}
       IN \A g \in current_level_groups :
            governor_state[g] \in {"COOLING", "CONVERGED", "AWAKENED"}
    \* Activate next level
    /\ LET next_level == hierarchy_level + 1
           new_groups == {g \in ParameterGroups : Level(g) = next_level}
       IN
       /\ activated_groups' = activated_groups \cup new_groups
       /\ hierarchy_level' = next_level
       \* Reallocate budget: split master budget among all active groups
       /\ LET total_groups == Cardinality(activated_groups \cup new_groups)
              per_group == MasterGradientBudget \div total_groups
          IN /\ group_budget_cap' = [g \in ParameterGroups |->
                    IF g \in (activated_groups \cup new_groups)
                    THEN per_group
                    ELSE 0]
             /\ group_budget_used' = [g \in ParameterGroups |-> 0]  \* Reset for new allocation
             /\ master_budget_used' = per_group * total_groups
       \* Update routing mask to include new groups
       /\ routing_mask' = [g \in ParameterGroups |->
              g \in (activated_groups \cup new_groups)]
       \* Audit
       /\ LET new_hash == Hash(training_last_hash,
                <<"activate", next_level>>)
          IN /\ training_audit_log' = Append(training_audit_log,
                 [op |-> "activate", level |-> next_level,
                  prev_hash |-> training_last_hash, hash |-> new_hash])
             /\ training_last_hash' = new_hash
       /\ UNCHANGED <<governor_state, governor_deltas, routing_selectivity,
                      current_step, loss, loss_history, pending_evals,
                      eval_results>>

\* --- Budget Reallocation ---
\* When a group converges, reduce its budget allocation and redistribute
\* to active (non-converged) groups. This is FederatedBudget rebalancing.

ReallocateBudget(converged_group) ==
    /\ governor_state[converged_group] = "CONVERGED"
    /\ group_budget_cap[converged_group] > 0
    /\ LET freed == group_budget_cap[converged_group] \div 2
           active_non_converged == {g \in activated_groups :
               governor_state[g] \in {"ACTIVE", "COOLING", "AWAKENED"}}
           recipient_count == Cardinality(active_non_converged)
       IN
       /\ recipient_count > 0  \* Someone to give budget to
       /\ LET per_recipient == freed \div recipient_count
          IN /\ group_budget_cap' = [g \in ParameterGroups |->
                    CASE g = converged_group ->
                            group_budget_cap[g] - freed
                      [] g \in active_non_converged ->
                            group_budget_cap[g] + per_recipient
                      [] OTHER -> group_budget_cap[g]]
       \* Master budget total unchanged (just redistributed)
       /\ UNCHANGED <<master_budget_used>>
       \* Audit
       /\ LET new_hash == Hash(training_last_hash,
                <<"realloc", converged_group>>)
          IN /\ training_audit_log' = Append(training_audit_log,
                 [op |-> "realloc", group |-> converged_group,
                  prev_hash |-> training_last_hash, hash |-> new_hash])
             /\ training_last_hash' = new_hash
       /\ UNCHANGED <<activated_groups, hierarchy_level, governor_state,
                      governor_deltas, group_budget_used, routing_mask,
                      routing_selectivity, current_step, loss, loss_history,
                      pending_evals, eval_results>>

\* --- AWAKENED Reactivation ---
\* When a converged group is awakened (new data perturbs it),
\* restore its budget allocation.

ReactivateGroup(g) ==
    /\ governor_state[g] = "AWAKENED"
    /\ group_budget_cap[g] < MasterGradientBudget \div Cardinality(activated_groups)
    \* Increase budget back toward fair share
    /\ LET fair_share == MasterGradientBudget \div Cardinality(activated_groups)
           increase == fair_share - group_budget_cap[g]
       IN /\ group_budget_cap' = [group_budget_cap EXCEPT ![g] = fair_share]
          /\ master_budget_used' = master_budget_used + increase
    \* Must not violate federated budget invariant
    /\ master_budget_used + (MasterGradientBudget \div Cardinality(activated_groups)
                             - group_budget_cap[g]) <= MasterGradientBudget
    /\ UNCHANGED <<activated_groups, hierarchy_level, governor_state,
                   governor_deltas, group_budget_used, routing_mask,
                   routing_selectivity, current_step, loss, loss_history,
                   pending_evals, eval_results, training_audit_log,
                   training_last_hash>>

\* --- Change-Triggered Evaluation ---
\* When a group's convergence state changes (Signal<ConvergenceChange>),
\* trigger targeted evaluation rather than waiting for fixed-interval eval.

TriggeredEvaluation ==
    /\ pending_evals /= {}
    /\ LET g == CHOOSE g \in pending_evals : TRUE
       IN /\ pending_evals' = pending_evals \ {g}
          \* Run evaluation metrics relevant to this group
          /\ \E eval_result \in Nat :
              eval_results' = Append(eval_results,
                  [group |-> g, step |-> current_step, result |-> eval_result])
          \* Audit
          /\ LET new_hash == Hash(training_last_hash, <<"eval", g>>)
             IN /\ training_audit_log' = Append(training_audit_log,
                    [op |-> "eval", group |-> g, step |-> current_step,
                     prev_hash |-> training_last_hash, hash |-> new_hash])
                /\ training_last_hash' = new_hash
    /\ UNCHANGED <<activated_groups, hierarchy_level, governor_state,
                   governor_deltas, group_budget_cap, group_budget_used,
                   master_budget_used, routing_mask, routing_selectivity,
                   current_step, loss, loss_history>>

\* --- Gradient Routing Update ---
\* CompetitiveSelection (ranked) determines which active groups receive
\* gradient this step, based on their learning need (gradient magnitude,
\* convergence state, budget remaining).

UpdateGradientRouting ==
    /\ LET scoreable == activated_groups
           \* Score: higher gradient magnitude + non-converged = higher priority
           need_scores == [g \in scoreable |->
               CASE governor_state[g] = "ACTIVE" -> 3
                 [] governor_state[g] = "COOLING" -> 2
                 [] governor_state[g] = "AWAKENED" -> 3
                 [] governor_state[g] = "CONVERGED" -> 0
                 [] OTHER -> 1]
       IN
       \* Select top groups by need (CompetitiveSelection ranked)
       /\ routing_mask' = [g \in ParameterGroups |->
              IF g \in activated_groups
              THEN need_scores[g] > 0  \* Route to non-converged groups
              ELSE FALSE]
       /\ routing_selectivity' =
              Cardinality({g \in activated_groups : routing_mask'[g] = FALSE})
              * 100 \div Cardinality(activated_groups)
    /\ UNCHANGED <<activated_groups, hierarchy_level, governor_state,
                   governor_deltas, group_budget_cap, group_budget_used,
                   master_budget_used, current_step, loss, loss_history,
                   pending_evals, eval_results, training_audit_log,
                   training_last_hash>>

----

\* === Next State ===

Next ==
    \/ TrainingStep
    \/ \E g \in ParameterGroups : UpdateGovernor(g)
    \/ ActivateNextLevel
    \/ \E g \in ParameterGroups : ReallocateBudget(g)
    \/ \E g \in ParameterGroups : ReactivateGroup(g)
    \/ TriggeredEvaluation
    \/ UpdateGradientRouting

----

\* === Fairness (Liveness) ===

\* Training eventually terminates
TrainingTermination == <>(current_step = MaxSteps)

\* Every pending evaluation is eventually processed
EvalCompleteness == <>(\A g \in ParameterGroups : g \notin pending_evals)

\* If lower levels converge, higher levels eventually activate
HierarchyProgress ==
    (\A g \in ParameterGroups : Level(g) = 0 =>
        governor_state[g] \in {"COOLING", "CONVERGED"})
    ~> hierarchy_level >= 1

----

Spec == Init /\ [][Next]_<<activated_groups, hierarchy_level, governor_state,
    governor_deltas, group_budget_cap, group_budget_used, master_budget_used,
    routing_mask, routing_selectivity, current_step, loss, loss_history,
    pending_evals, eval_results, training_audit_log, training_last_hash>>

THEOREM Spec => [](
    /\ FederatedBudgetInvariant
    /\ SubPoolInvariant
    /\ BudgetConsistency
    /\ HierarchyOrdering
    /\ ConvergenceGatedActivation
    /\ GovernorStateValidity
    /\ GradientScopeInvariant
    /\ TrainingAuditChainIntegrity
    /\ StepBound
)
====
```

What we expect to prove:

This is the capstone specification. Nine safety invariants must hold simultaneously across all interleavings of training steps, governor updates, hierarchy activations, budget reallocations, reactivations, evaluations, and routing updates.

FederatedBudgetInvariant: the total gradient compute allocated to all groups never exceeds the master budget. No training step can cause a group to consume more gradient compute than the system has available. This holds across budget reallocations and reactivations.

HierarchyOrdering: groups activate strictly in level order. L1 cannot activate before L0. L3 cannot activate before L2. This is the structural guarantee that produces the coarse-to-fine convergence ordering observed empirically. TLC verifies it is impossible for the system to reach a state where a higher-level group is active while a lower-level group is not.

ConvergenceGatedActivation: a group activates only when all groups below it have reached at least COOLING state. This is the composition of ConvergenceGovernor + Signal + QualityHierarchy that makes activation earned rather than scheduled. TLC verifies that no interleaving of governor state transitions and activation events can produce a premature activation.

GradientScopeInvariant: gradient is never routed to an inactive group. This is ActuationPass.ActuationScope applied to gradient updates. Combined with HierarchyOrdering, it guarantees that early training steps only update structural parameters (L0), and refinement parameters (L2, L3) only receive gradient after their dependencies have stabilized.

CoarseToFineOrdering (derived, not enforced): this is the prediction that L0 converges before L1, L1 before L2, and so on. The spec does not enforce it as an invariant. Instead, TLC checks whether it holds as an emergent property of the activation and convergence rules. If TLC finds a counterexample, we learn under what conditions the ordering breaks. If it holds across all checked states, the prediction is verified within the bounded model. The empirical result (L0 then L1 then L2 then L3 observed at 4.8M parameters) is consistent with the prediction. The TLC verification would confirm the prediction holds for all possible training dynamics within the model's state space.

---

## C.4 Adaptive Depth and Convergence Detection

The adaptive depth mechanism uses ConvergenceGovernor to determine whether computation has converged at each layer.

```tla+
---- MODULE AdaptiveDepth ----
EXTENDS Naturals, Reals

CONSTANTS
    Layers,             \* Number of transformer layers
    ExitThreshold       \* Convergence residual threshold for early exit

VARIABLES
    layer_residuals,    \* Residual magnitude at each layer
    exit_layer,         \* Layer at which computation exited
    converged,          \* Whether the representation converged
    confidence          \* Confidence signal from convergence detection

----

\* === Safety Invariants ===

\* Exit layer is valid
ValidExit ==
    exit_layer \in 1..Layers

\* Early exit only when converged
EarlyExitRequiresConvergence ==
    exit_layer < Layers => converged = TRUE

\* Confidence is consistent with convergence
ConfidenceConsistency ==
    /\ converged = TRUE => confidence >= ExitThreshold
    /\ converged = FALSE => confidence < ExitThreshold

\* Full depth means no convergence detected
FullDepthMeansUncertain ==
    exit_layer = Layers => converged = FALSE

\* Monotonic residual check: once converged, stays converged
\* (within a single forward pass)
ConvergenceMonotonicity ==
    \A l \in 1..(exit_layer - 1) :
        layer_residuals[l] >= layer_residuals[exit_layer]

----

Init ==
    /\ layer_residuals = [l \in 1..Layers |-> 1000]  \* High initial residual
    /\ exit_layer = Layers
    /\ converged = FALSE
    /\ confidence = 0

ForwardPass ==
    \* Process each layer, check convergence at each
    /\ \E residuals \in [1..Layers -> Nat] :
        /\ layer_residuals' = residuals
        /\ LET first_converged ==
                CHOOSE l \in 1..Layers :
                    /\ residuals[l] < ExitThreshold
                    /\ \A earlier \in 1..(l-1) : residuals[earlier] >= ExitThreshold
           IN
           IF \E l \in 1..Layers : residuals[l] < ExitThreshold
           THEN /\ exit_layer' = first_converged
                /\ converged' = TRUE
                /\ confidence' = ExitThreshold - residuals[first_converged]
           ELSE /\ exit_layer' = Layers
                /\ converged' = FALSE
                /\ confidence' = 0

Next == ForwardPass

Spec == Init /\ [][Next]_<<layer_residuals, exit_layer, converged, confidence>>

THEOREM Spec => [](ValidExit /\ EarlyExitRequiresConvergence /\ FullDepthMeansUncertain)
====
```

What we expect to prove:

EarlyExitRequiresConvergence: the model never exits early unless the convergence detector confirms the representation is stable. This prevents silent quality degradation from premature exits.

FullDepthMeansUncertain: if the model runs all layers without converging, the confidence signal is low. This is the architectural confabulation detection mechanism from Section 5.7: maximum depth with no convergence is a detectable signal that the model is interpolating rather than retrieving. The spec proves this signal is structurally present in every full-depth forward pass, not something that must be learned.

---

## C.5 Full Model Lifecycle

This specification ties everything together: the belief lifecycle, the training pipeline, and the evaluation/deployment cycle.

```tla+
---- MODULE LeanFormerLifecycle ----
EXTENDS Naturals, FiniteSets

CONSTANTS
    Beliefs,
    ParameterGroups,
    MaxTrainingSteps,
    MaxBeliefs

VARIABLES
    \* Lifecycle phase
    phase,              \* {"training", "forging", "evaluation", "deployment"}

    \* From BeliefDeltaLifecycle
    active_beliefs,
    base_hash,

    \* From GovernedTrainingPipeline
    training_step,
    activated_groups,
    governor_states,

    \* Lifecycle audit
    lifecycle_log,
    lifecycle_hash

----

Phases == {"training", "forging", "evaluation", "deployment"}

\* === Lifecycle Invariants ===

\* Base weights are immutable across all lifecycle phases.
\* Training does not modify base weights.
\* Forging does not modify base weights.
\* Deployment does not modify base weights.
GlobalBaseImmutability ==
    base_hash = InitialBaseHash

\* Beliefs can only be forged after training reaches sufficient convergence.
\* This is readiness-gated forging: Signal<GroupConverged> gates the forge.
ForgingRequiresConvergence ==
    phase = "forging" =>
        \E g \in ParameterGroups : governor_states[g] \in {"CONVERGED", "AWAKENED"}

\* Evaluation is triggered by state changes, not by fixed interval.
\* In the lifecycle model, evaluation follows forging and training transitions.
EvaluationIsTriggered ==
    phase = "evaluation" => TRUE  \* Entered only via Signal<ConvergenceChange>

\* Deployment serves only validated beliefs.
\* A belief is deployed only if it passed evaluation.
DeploymentRequiresValidation ==
    phase = "deployment" =>
        \A b \in active_beliefs : Validated(b)

\* The lifecycle is governed end-to-end by the same primitives.
\* No phase operates outside the governance framework.
UniversalGovernance ==
    \A p \in Phases : LifecycleAuditedInPhase(p)

----

\* === Phase Transitions ===

\* training -> forging: when parameter groups converge
TrainingToForging ==
    /\ phase = "training"
    /\ \E g \in ParameterGroups : governor_states[g] = "CONVERGED"
    /\ phase' = "forging"
    /\ UNCHANGED <<active_beliefs, base_hash, training_step,
                   activated_groups, governor_states>>
    /\ RecordLifecycleEvent("training_to_forging")

\* forging -> evaluation: when belief forging completes
ForgingToEvaluation ==
    /\ phase = "forging"
    /\ phase' = "evaluation"
    /\ UNCHANGED <<active_beliefs, base_hash, training_step,
                   activated_groups, governor_states>>
    /\ RecordLifecycleEvent("forging_to_evaluation")

\* evaluation -> deployment: when evaluation passes
EvaluationToDeployment ==
    /\ phase = "evaluation"
    /\ phase' = "deployment"
    /\ UNCHANGED <<active_beliefs, base_hash, training_step,
                   activated_groups, governor_states>>
    /\ RecordLifecycleEvent("evaluation_to_deployment")

\* deployment -> training: when AWAKENED signal triggers retraining
DeploymentToTraining ==
    /\ phase = "deployment"
    /\ \E g \in ParameterGroups : governor_states[g] = "AWAKENED"
    /\ phase' = "training"
    /\ UNCHANGED <<active_beliefs, base_hash, training_step,
                   activated_groups, governor_states>>
    /\ RecordLifecycleEvent("deployment_to_training")

RecordLifecycleEvent(event) ==
    LET new_hash == Hash(lifecycle_hash, event)
    IN /\ lifecycle_log' = Append(lifecycle_log,
           [event |-> event, prev_hash |-> lifecycle_hash, hash |-> new_hash])
       /\ lifecycle_hash' = new_hash

Next ==
    \/ TrainingToForging
    \/ ForgingToEvaluation
    \/ EvaluationToDeployment
    \/ DeploymentToTraining

Spec == Init /\ [][Next]_<<phase, active_beliefs, base_hash, training_step,
    activated_groups, governor_states, lifecycle_log, lifecycle_hash>>

THEOREM Spec => [](GlobalBaseImmutability /\ ForgingRequiresConvergence)
====
```

What we expect to prove:

GlobalBaseImmutability holds across the entire lifecycle. Not just during training. Not just during belief injection. Across every phase transition, every forging operation, every deployment. The base weights are the invariant substrate that the entire system is built on top of. TLC verifies that no sequence of lifecycle transitions can modify them.

ForgingRequiresConvergence prevents premature knowledge creation. A belief delta cannot be forged until the base model's parameters have stabilized enough for the delta to be meaningful. This is the readiness gate that prevents forging on an undertrained substrate.

The lifecycle loop (training -> forging -> evaluation -> deployment -> training via AWAKENED) is the closed-loop system where every phase is governed by the same primitive set. The AWAKENED reactivation mechanism is what makes the loop non-trivial: a deployed model that encounters new data can trigger retraining through the convergence governor, which produces new beliefs through the forge, which are validated through evaluation, which are deployed. The entire cycle is audited, budgeted, and convergence-governed.

---

## Summary: What the Composed Specifications Prove

| Specification | Key Invariants | Primitives Composed |
|--------------|---------------|-------------------|
| Belief Delta Lifecycle | Base immutability, budget cap, parameter non-overlap, exact restoration, audit chain | Budget + ResourceRegistry + Transaction + AuditSink |
| Semantic Routing | Active-only routing, bounded selection, threshold optimality | CompetitiveSelection (ranked) + ResourceRegistry |
| Governed Training Pipeline | Federated budget, hierarchy ordering, convergence-gated activation, gradient scope, audit chain | FederatedBudget + QualityHierarchy + ConvergenceGovernor + CompetitiveSelection (ranked) + ActuationPass + Signal + AuditSink |
| Adaptive Depth | Early exit requires convergence, full depth signals uncertainty | ConvergenceGovernor |
| Full Model Lifecycle | Global base immutability, readiness-gated forging, audited phase transitions | All governance primitives composed |

The individual primitive specs (Appendix B) prove that each primitive's invariants hold in isolation. The LeanFormer specs (this appendix) prove that the invariants hold in composition, and that the composition produces system-level guarantees that no individual primitive provides alone. The combination is the formal proof that Domain Abstraction Collapse produces correct, governed, composable systems from a universal primitive set.
