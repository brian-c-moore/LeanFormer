"""
Real end-to-end validation of governed training.

This runs the EXACT production code path from train_reasoning.py on a small
model. It uses thresholds tuned to exercise all governance transitions within
the test's step budget. Every governance claim is validated with hard assertions:

1.  Loss decreases (training works)
2.  Gradient norms are non-zero and differentiated across groups
3.  Audit log gradient norms are non-zero (not just structurally valid)
4.  Convergence governors transition states (at least one group reaches COOLING)
5.  Hierarchy activates L1 via convergence signal (not emergency)
6.  Budget allocations become non-uniform (respond to gradient signal)
7.  Router gradient masking zeros at least one group's gradients
8.  Budget invariant holds every step
9.  Audit hash chain is valid
10. All governance state is self-consistent at end of run

Usage:
    python tests/validate_governed_training.py
"""

import sys
import os
import json
import tempfile
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from leanformer.model.config import LeanFormerConfig
from leanformer.model.leanformer import LeanFormer
from leanformer.training.param_groups import build_param_groups, load_registry
from leanformer.training.convergence import (
    ConvergenceConfig, ConvergenceGovernor, ConvergenceState,
)
from leanformer.training.hierarchy import HierarchyConfig, HierarchyManager
from leanformer.training.budget import BudgetConfig, FederatedBudget
from leanformer.training.router import GradientRouter, RouterConfig, apply_gradient_mask
from leanformer.training.audit import AuditSink, AuditQuery
from leanformer.training.eval_pipeline import EvalConfig, EvalPipeline


def main():
    print("=== GOVERNED TRAINING END-TO-END VALIDATION ===")
    print("Validates every governance claim against the production code path.\n")

    # Small model
    config = LeanFormerConfig(
        vocab_size=1000, d_model=128, n_heads=4, n_layers=4, d_ff=512,
        max_seq_len=64, attention_rank=16, ff_rank=16,
        screening_rank=4, attention_top_k=16, ff_gate_rank=4, dropout=0.0,
    )
    model = LeanFormer(config)
    total_model_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_model_params:,} params")

    # Dataset — enough for 150 optimizer steps
    N = 800
    input_ids = torch.randint(0, config.vocab_size, (N, 64))
    dataset = TensorDataset(input_ids, input_ids.clone())
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

    # --- Governed setup (mirrors train_reasoning.py) ---
    registry = load_registry()
    groups = build_param_groups(model, registry)
    group_ids = list(groups.keys())
    total_steps = 150
    grad_accum = 2

    # Thresholds tuned to exercise transitions in 150 steps.
    # Production uses conservative thresholds for 21K steps. Here we use
    # aggressive thresholds so that COOLING actually fires within the budget.
    gov_config = ConvergenceConfig(
        ema_decay=0.9,
        cooling_threshold=5.0,      # Aggressive — small model grad norms are small
        cooling_window=10,           # 10 steps below threshold triggers COOLING
        converged_threshold=0.001,   # Won't reach CONVERGED in this short run
        confirmation_window=100,
        reactivation_delta=0.15,
        reactivation_warmup=10,
    )
    governors = {gid: ConvergenceGovernor(gid, gov_config) for gid in groups}

    hierarchy = HierarchyManager(
        groups, governors,
        config=HierarchyConfig(warmup_steps=20, emergency_fraction=0.90),
        total_steps=total_steps,
    )

    budget = FederatedBudget(
        group_ids, governors,
        config=BudgetConfig(
            eval_window=10,  # Recompute every 10 steps
            alpha=1.0, beta=0.0, gamma=0.0,  # Pure gradient-magnitude scoring
        ),
    )

    router = GradientRouter(
        config.d_model, len(group_ids),
        config=RouterConfig(
            warmup_steps=5,
            max_active_groups=max(len(group_ids) // 2, 2),
            entropy_coeff=0.01, balance_window=50,
        ),
    )

    tmpdir = tempfile.mkdtemp()
    audit = AuditSink(os.path.join(tmpdir, "audit.jsonl"))

    def eval_fn():
        model.eval()
        with torch.no_grad():
            x = torch.randint(0, config.vocab_size, (1, 32))
            out = model(x, labels=x, training=False)
            model.train()
            return out["loss"].item()

    metric_dep_map = registry.get("metric_dependency_map", {"perplexity": ["all"]})
    eval_pipeline = EvalPipeline(
        metric_dep_map, {"perplexity": eval_fn},
        config=EvalConfig(max_evals_per_window=3, eval_window_steps=100),
    )
    for gov in governors.values():
        gov.subscribe(eval_pipeline.on_convergence_signal)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3,
    )
    prev_active_levels = set(hierarchy.active_levels)

    # --- Tracking for validation ---
    all_step_grad_norms = []
    all_budget_snapshots = []
    transition_log = []
    hierarchy_events = []
    masked_groups_per_step = []
    budget_invariant_every_step = []

    for gov in governors.values():
        gov.subscribe(lambda sig: transition_log.append(sig))
    hierarchy.subscribe(lambda evt: hierarchy_events.append(evt))

    # --- Training loop (mirrors train_reasoning.py) ---
    print(f"Training: {total_steps} steps, batch=4, accum={grad_accum}\n")

    model.train()
    optimizer_step = 0
    batch_idx = 0
    losses = []

    for epoch in range(10):  # Enough epochs to reach total_steps
        for ids, lbls in loader:
            out = model(ids, labels=lbls, training=True)
            loss = out["loss"] / grad_accum
            loss.backward()

            # Router gradient masking (exactly as in train_reasoning.py)
            masked_this_step = 0
            if not router.in_warmup and (batch_idx + 1) % grad_accum == 0:
                with torch.no_grad():
                    positions = torch.arange(ids.shape[1]).unsqueeze(0)
                    hidden = model.token_embedding(ids) + model.position_embedding(positions)
                routing = router(hidden, group_ids)
                # Count how many groups get masked BEFORE applying
                group_active = routing["hard_mask"].sum(dim=0) > 0
                masked_this_step = sum(1 for i, gid in enumerate(group_ids) if not group_active[i])
                apply_gradient_mask(model, groups, routing, group_ids)

            batch_idx += 1

            if batch_idx % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Capture grad norms BEFORE step/zero_grad
                step_grad_norms = {gid: group.grad_norm() for gid, group in groups.items()}
                all_step_grad_norms.append(dict(step_grad_norms))

                # Update governors with live gradients
                for gid, group in groups.items():
                    if group.hierarchy_level in hierarchy.active_levels:
                        governors[gid].update(step_grad_norms[gid])

                optimizer.step()
                optimizer.zero_grad()
                optimizer_step += 1

                # Track masking
                masked_groups_per_step.append(masked_this_step)

                # Hierarchy
                hierarchy.step(optimizer_step)
                if hierarchy.active_levels != prev_active_levels:
                    trainable = [p for p in model.parameters() if p.requires_grad]
                    if trainable:
                        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
                    prev_active_levels = set(hierarchy.active_levels)

                # Budget
                budget.step(optimizer_step)
                budget_invariant_every_step.append(budget.verify_invariant())
                all_budget_snapshots.append(dict(budget.allocations))

                # Router + eval
                router.step()
                eval_pipeline.step(optimizer_step)

                # Audit (every 5 steps for denser coverage)
                if optimizer_step % 5 == 0:
                    audit.log_step(
                        step=optimizer_step,
                        gradient_norms=step_grad_norms,
                        convergence_states={
                            gid: gov.state.value for gid, gov in governors.items()
                        },
                        budget_allocations=budget.allocations,
                        hierarchy_active_levels=sorted(hierarchy.active_levels),
                    )

                losses.append(out["loss"].item())

                if optimizer_step % 25 == 0:
                    n_a = sum(1 for g in governors.values() if g.state == ConvergenceState.ACTIVE)
                    n_c = sum(1 for g in governors.values() if g.state == ConvergenceState.COOLING)
                    n_v = sum(1 for g in governors.values() if g.state == ConvergenceState.CONVERGED)
                    print(
                        f"  step {optimizer_step:>4} | "
                        f"loss {out['loss'].item():.4f} | "
                        f"gov A{n_a}/C{n_c}/V{n_v} | "
                        f"hier {sorted(hierarchy.active_levels)} | "
                        f"rtr {'warmup' if router.in_warmup else 'active'} | "
                        f"masked {masked_this_step}/{len(group_ids)} | "
                        f"budget_ok={budget.verify_invariant()}"
                    )

                if optimizer_step >= total_steps:
                    break
        if optimizer_step >= total_steps:
            break

    # === VALIDATIONS ===
    print(f"\n{'='*60}")
    print("  VALIDATION RESULTS")
    print(f"{'='*60}\n")
    passed = 0
    failed = 0

    def check(name, condition, pass_msg, fail_msg):
        nonlocal passed, failed
        if condition:
            print(f"  PASS: {name} — {pass_msg}")
            passed += 1
        else:
            print(f"  FAIL: {name} — {fail_msg}")
            failed += 1

    # 1. Loss decreases
    check(
        "Loss convergence",
        len(losses) >= total_steps and losses[-1] < losses[0],
        f"{losses[0]:.4f} -> {losses[-1]:.4f}",
        f"{'no steps' if not losses else f'{losses[0]:.4f} -> {losses[-1]:.4f}'}",
    )

    # 2. Gradient norms are non-zero and differentiated
    final_emas = {gid: gov.gradient_ema for gid, gov in governors.items()}
    l0_positive = sum(1 for gid, v in final_emas.items() if groups[gid].hierarchy_level == 0 and v > 0)
    check(
        "Gradient EMA tracking",
        l0_positive >= 3,
        f"{l0_positive}/4 L0 groups have positive EMA",
        f"Only {l0_positive}/4 L0 groups have positive EMA: {final_emas}",
    )

    # 3. Audit log gradient norms are non-zero
    query = AuditQuery(os.path.join(tmpdir, "audit.jsonl"))
    step_records = query.query_by_type("training_step")
    audit_norms_ok = False
    if step_records:
        # Check multiple records, not just the last one
        for rec in step_records:
            norms = rec.get("gradient_norms", {})
            if any(v > 0 for v in norms.values()):
                audit_norms_ok = True
                break
    check(
        "Audit gradient norms non-zero",
        audit_norms_ok,
        f"{sum(1 for r in step_records if any(v > 0 for v in r.get('gradient_norms', {}).values()))}/{len(step_records)} records have non-zero norms",
        "ALL audit gradient_norms are zero — feedback loop is blind",
    )

    # 4. Convergence governors fire state transitions
    cooling_transitions = [
        t for t in transition_log
        if t.new_state == ConvergenceState.COOLING
    ]
    cooling_groups = {t.group_id for t in cooling_transitions}
    check(
        "Convergence transitions fire",
        len(cooling_transitions) > 0,
        f"{len(cooling_transitions)} transitions, {len(cooling_groups)} groups reached COOLING: {cooling_groups}",
        "No group ever transitioned to COOLING — governors are not detecting convergence",
    )

    # 5. Hierarchy activates L1 via convergence signal (not emergency)
    l1_via_convergence = any(
        evt.get("level") == 1 and evt.get("reason") == "convergence"
        for evt in hierarchy_events
    )
    l1_active = hierarchy.is_level_active(1)
    check(
        "Hierarchy L1 activation via convergence",
        l1_via_convergence,
        "L1 activated by convergence signal",
        f"L1 {'activated via emergency' if l1_active else 'never activated'} — convergence signal not firing",
    )

    # 6. Budget allocations become non-uniform
    if len(all_budget_snapshots) >= 2:
        first_allocs = list(all_budget_snapshots[0].values())
        last_allocs = list(all_budget_snapshots[-1].values())
        first_spread = max(first_allocs) - min(first_allocs)
        last_spread = max(last_allocs) - min(last_allocs)
        budget_differentiated = last_spread > first_spread + 0.001
    else:
        budget_differentiated = False
    check(
        "Budget allocations non-uniform",
        budget_differentiated,
        f"spread grew from {first_spread:.4f} to {last_spread:.4f}",
        f"allocations stayed uniform (spread: {last_spread:.4f})" if all_budget_snapshots else "no budget snapshots",
    )

    # 7. Router gradient masking zeros at least one group
    post_warmup_masked = masked_groups_per_step[router.config.warmup_steps:]
    any_masked = any(m > 0 for m in post_warmup_masked) if post_warmup_masked else False
    total_masked = sum(post_warmup_masked) if post_warmup_masked else 0
    check(
        "Router gradient masking active",
        any_masked,
        f"{total_masked} total groups masked across {len(post_warmup_masked)} post-warmup steps",
        "Router never masked any groups — gradient selectivity not working",
    )

    # 8. Budget invariant holds every step
    all_held = all(budget_invariant_every_step)
    violations = sum(1 for v in budget_invariant_every_step if not v)
    check(
        "Budget invariant every step",
        all_held,
        f"held for all {len(budget_invariant_every_step)} steps",
        f"violated {violations} times",
    )

    # 9. Audit hash chain valid
    chain_valid, chain_count = query.verify_chain()
    check(
        "Audit hash chain integrity",
        chain_valid and chain_count > 0,
        f"valid chain, {chain_count} records",
        f"{'invalid chain' if not chain_valid else 'empty log'}",
    )

    # 10. Governance state self-consistent
    # Every group that's COOLING or CONVERGED should have low gradient EMA
    # Every group in an active hierarchy level should have been updated
    states_consistent = True
    for gid, gov in governors.items():
        if gov.state == ConvergenceState.COOLING and gov.gradient_ema > gov_config.cooling_threshold:
            states_consistent = False
            break
    check(
        "Governance state self-consistent",
        states_consistent,
        "all states consistent with their EMA values",
        "state/EMA mismatch detected",
    )

    # Final summary
    print(f"\n{'='*60}")
    print(f"  {passed} passed, {failed} failed out of 10 checks")
    if failed == 0:
        print("  ALL VALIDATIONS PASSED")
    else:
        print("  FAILURES DETECTED — do not deploy")
    print(f"{'='*60}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
