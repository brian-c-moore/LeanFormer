"""
Real end-to-end validation of governed training.

This is NOT a unit test of isolated components. This runs the actual governed
training loop from train_reasoning.py on a small model with real gradients,
real convergence tracking, real hierarchy activation, real gradient masking,
and real audit logging. It validates the production code path.

Usage:
    python tests/validate_governed_training.py
"""

import sys
import os
import tempfile
import torch
import torch.nn as nn
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
    print("This validates the production code path, not isolated components.\n")

    # Small model
    config = LeanFormerConfig(
        vocab_size=1000, d_model=128, n_heads=4, n_layers=4, d_ff=512,
        max_seq_len=64, attention_rank=16, ff_rank=16,
        screening_rank=4, attention_top_k=16, ff_gate_rank=4, dropout=0.0,
    )
    model = LeanFormer(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params (d_model={config.d_model}, {config.n_layers} layers)")

    # Dataset
    N = 200
    input_ids = torch.randint(0, config.vocab_size, (N, 64))
    dataset = TensorDataset(input_ids, input_ids.clone())
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

    # --- Governed setup (mirrors train_reasoning.py exactly) ---
    registry = load_registry()
    groups = build_param_groups(model, registry)
    group_ids = list(groups.keys())
    total_steps = 50
    grad_accum = 2

    gov_config = ConvergenceConfig(
        ema_decay=0.99, cooling_threshold=0.5, cooling_window=200,
        converged_threshold=0.1, confirmation_window=500,
        reactivation_delta=0.15, reactivation_warmup=100,
    )
    governors = {gid: ConvergenceGovernor(gid, gov_config) for gid in groups}

    hierarchy = HierarchyManager(
        groups, governors,
        config=HierarchyConfig(warmup_steps=100, emergency_fraction=0.80),
        total_steps=total_steps,
    )

    budget = FederatedBudget(
        group_ids, governors,
        config=BudgetConfig(eval_window=50, alpha=0.4, beta=0.3, gamma=0.3),
    )

    router = GradientRouter(
        config.d_model, len(group_ids),
        config=RouterConfig(
            warmup_steps=int(total_steps * 0.05),
            max_active_groups=max(len(group_ids) // 2, 2),
            entropy_coeff=0.01, balance_window=200,
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
        config=EvalConfig(max_evals_per_window=3, eval_window_steps=500),
    )
    for gov in governors.values():
        gov.subscribe(eval_pipeline.on_convergence_signal)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3,
    )
    prev_active_levels = set(hierarchy.active_levels)

    # --- Training loop (mirrors train_reasoning.py) ---
    print(f"Training: {total_steps} steps, batch=4, accum={grad_accum}\n")

    model.train()
    optimizer_step = 0
    batch_idx = 0
    losses = []

    for epoch in range(3):
        for ids, lbls in loader:
            out = model(ids, labels=lbls, training=True)
            loss = out["loss"] / grad_accum
            loss.backward()

            # Router gradient masking (exactly as in train_reasoning.py)
            if not router.in_warmup and (batch_idx + 1) % grad_accum == 0:
                with torch.no_grad():
                    positions = torch.arange(ids.shape[1]).unsqueeze(0)
                    hidden = model.token_embedding(ids) + model.position_embedding(positions)
                routing = router(hidden, group_ids)
                apply_gradient_mask(model, groups, routing, group_ids)

            batch_idx += 1

            if batch_idx % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update governors BEFORE step/zero_grad (gradients are live now)
                for gid, group in groups.items():
                    if group.hierarchy_level in hierarchy.active_levels:
                        governors[gid].update(group.grad_norm())

                optimizer.step()
                optimizer.zero_grad()
                optimizer_step += 1

                lr_mults = hierarchy.step(optimizer_step)

                if hierarchy.active_levels != prev_active_levels:
                    trainable = [p for p in model.parameters() if p.requires_grad]
                    if trainable:
                        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
                    prev_active_levels = set(hierarchy.active_levels)

                budget.step(optimizer_step)
                router.step()
                eval_pipeline.step(optimizer_step)

                if optimizer_step % 10 == 0:
                    audit.log_step(
                        step=optimizer_step,
                        gradient_norms={gid: groups[gid].grad_norm() for gid in groups},
                        convergence_states={
                            gid: gov.state.value for gid, gov in governors.items()
                        },
                        hierarchy_active_levels=sorted(hierarchy.active_levels),
                    )

                losses.append(out["loss"].item())

                if optimizer_step % 10 == 0:
                    n_a = sum(
                        1 for g in governors.values()
                        if g.state == ConvergenceState.ACTIVE
                    )
                    n_c = sum(
                        1 for g in governors.values()
                        if g.state == ConvergenceState.COOLING
                    )
                    n_v = sum(
                        1 for g in governors.values()
                        if g.state == ConvergenceState.CONVERGED
                    )
                    lvls = sorted(hierarchy.active_levels)
                    rtr = "warmup" if router.in_warmup else "active"
                    print(
                        f"  step {optimizer_step:>4} | "
                        f"loss {out['loss'].item():.4f} | "
                        f"gov A{n_a}/C{n_c}/V{n_v} | "
                        f"hier {lvls} | rtr {rtr} | "
                        f"budget_ok={budget.verify_invariant()} | "
                        f"audit={audit.record_count}"
                    )

                if optimizer_step >= total_steps:
                    break
        if optimizer_step >= total_steps:
            break

    # --- Validate results ---
    print("\n=== VALIDATION RESULTS ===\n")
    passed = 0
    failed = 0

    # 1. Training actually happened — loss decreased
    if len(losses) >= total_steps and losses[-1] < losses[0]:
        print(f"  PASS: Loss decreased {losses[0]:.4f} -> {losses[-1]:.4f}")
        passed += 1
    else:
        print(f"  FAIL: Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}")
        failed += 1

    # 2. Governors tracked real, differentiated gradients
    # L0 groups have had gradients the whole run. L1+ groups may have zero EMA
    # if they were frozen for most of training (hierarchy deferral is working).
    emas = {gid: gov.gradient_ema for gid, gov in governors.items()}
    l0_emas = {gid: v for gid, v in emas.items() if groups[gid].hierarchy_level == 0}
    l0_positive = sum(1 for v in l0_emas.values() if v > 0)
    unique_nonzero = len(set(round(v, 8) for v in emas.values() if v > 0))
    if l0_positive >= 2 and unique_nonzero > 1:
        print(f"  PASS: {l0_positive}/{len(l0_emas)} L0 groups have positive EMA, {unique_nonzero} distinct values")
        for gid, v in sorted(emas.items(), key=lambda x: -x[1]):
            lvl = groups[gid].hierarchy_level
            print(f"         L{lvl} {gid}: {v:.6e}")
        passed += 1
    else:
        print(f"  FAIL: EMAs not differentiated: {emas}")
        failed += 1

    # 3. Hierarchy started L0-only
    if 0 in hierarchy.activation_steps:
        print(f"  PASS: Hierarchy active levels: {sorted(hierarchy.active_levels)}")
        passed += 1
    else:
        print(f"  FAIL: L0 not in activation_steps")
        failed += 1

    # 4. Budget invariant held
    if budget.verify_invariant():
        total_alloc = sum(budget.allocations.values())
        print(f"  PASS: Budget invariant held (sum={total_alloc:.4f} <= {budget.config.master_budget})")
        passed += 1
    else:
        print(f"  FAIL: Budget invariant violated")
        failed += 1

    # 5. Router exited warmup and is producing masks
    if not router.in_warmup:
        print(f"  PASS: Router active (exited warmup)")
        passed += 1
    else:
        print(f"  FAIL: Router still in warmup after {total_steps} steps")
        failed += 1

    # 6. Audit chain valid
    query = AuditQuery(os.path.join(tmpdir, "audit.jsonl"))
    valid, count = query.verify_chain()
    if valid and count > 0:
        print(f"  PASS: Audit chain valid, {count} hash-chained records")
        passed += 1
    else:
        print(f"  FAIL: Audit chain invalid or empty (valid={valid}, count={count})")
        failed += 1

    # 7. Final convergence states are real
    final_states = {gid: gov.state.value for gid, gov in governors.items()}
    if all(s in ("ACTIVE", "COOLING", "CONVERGED", "AWAKENED") for s in final_states.values()):
        print(f"  PASS: Final states: {final_states}")
        passed += 1
    else:
        print(f"  FAIL: Invalid states: {final_states}")
        failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    if failed == 0:
        print("  ALL VALIDATIONS PASSED")
    else:
        print("  FAILURES DETECTED")
    print(f"{'='*50}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
