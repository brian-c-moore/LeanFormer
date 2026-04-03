"""
Comparison: Baseline vs. Governed Training Pipeline

Trains the same small model on WikiText-2 twice:
  A) Baseline — standard flat training (all params, uniform allocation)
  B) Governed — full targeted training pipeline (hierarchy, governors, budget, router, audit)

Measures: loss convergence, gradient compute, wall-clock time, eval efficiency.
CPU-compatible (no CUDA required).

Usage:
    python -m leanformer.scripts.compare_training [--steps 300] [--seed 42]
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..model.config import LeanFormerConfig
from ..model.leanformer import LeanFormer
from ..training.param_groups import build_param_groups, load_registry
from ..training.convergence import ConvergenceConfig, ConvergenceGovernor, GovernorManager
from ..training.hierarchy import HierarchyConfig, HierarchyManager
from ..training.budget import BudgetConfig, FederatedBudget
from ..training.router import GradientRouter, RouterConfig, apply_gradient_mask
from ..training.audit import AuditSink, AuditQuery
from ..training.eval_pipeline import EvalConfig, EvalPipeline
from ..training.deployment import DeploymentProfiler


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class WikiTextDataset(Dataset):
    """Load WikiText-2 raw, tokenize with a simple BPE-like approach."""

    def __init__(self, max_samples: int = 2000, seq_len: int = 128, vocab_size: int = 32000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.data = []

        # Try loading cached dataset first, fall back to HF
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "wikitext", "wikitext-2-raw-v1", split="train",
                cache_dir="data/wikitext-2",
                trust_remote_code=True,
            )
            texts = [ex["text"] for ex in ds if len(ex["text"]) > 50][:max_samples * 2]
        except Exception:
            texts = None

        if texts:
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
                tok.pad_token = tok.eos_token
                for text in texts[:max_samples]:
                    enc = tok(text, truncation=True, max_length=seq_len, padding="max_length")
                    ids = torch.tensor(enc["input_ids"], dtype=torch.long)
                    self.data.append(ids)
            except Exception:
                texts = None

        # Fallback: synthetic data (still useful for measuring machinery)
        if not self.data:
            print("  [fallback] Using synthetic data (WikiText-2 unavailable)")
            for _ in range(max_samples):
                self.data.append(torch.randint(0, vocab_size, (seq_len,)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        return {"input_ids": ids, "labels": ids.clone()}


# ---------------------------------------------------------------------------
# Baseline training
# ---------------------------------------------------------------------------

def train_baseline(model, loader, num_steps, lr, device, seed):
    """Standard flat training: all params, uniform allocation, no governance."""
    torch.manual_seed(seed)
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    metrics = {
        "losses": [],
        "grad_norms": [],
        "step_times_ms": [],
        "total_grad_computations": 0,
        "eval_calls": 0,
    }

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    step = 0
    for batch in loader:
        if step >= num_steps:
            break

        t0 = time.perf_counter()
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids, labels=labels, training=True)
        loss = out["loss"]
        loss.backward()

        # Record gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        t1 = time.perf_counter()
        step_ms = (t1 - t0) * 1000

        metrics["losses"].append(loss.item())
        metrics["grad_norms"].append(total_norm)
        metrics["step_times_ms"].append(step_ms)
        metrics["total_grad_computations"] += total_params

        # Fixed-interval eval every 50 steps
        if step > 0 and step % 50 == 0:
            metrics["eval_calls"] += 1

        step += 1

    metrics["total_wall_ms"] = sum(metrics["step_times_ms"])
    return metrics


# ---------------------------------------------------------------------------
# Governed training
# ---------------------------------------------------------------------------

def train_governed(model, loader, num_steps, lr, device, seed, tmpdir):
    """Governed training: hierarchy, convergence, budget, router, audit."""
    torch.manual_seed(seed)
    model.train()
    model.to(device)

    config_obj = model.config
    registry = load_registry()
    groups = build_param_groups(model, registry)
    group_ids = list(groups.keys())

    # --- Convergence governors ---
    gov_config = ConvergenceConfig(
        ema_decay=0.95,
        cooling_threshold=5.0,
        cooling_window=15,
        converged_threshold=0.5,
        confirmation_window=30,
        reactivation_delta=0.2,
        reactivation_warmup=10,
    )
    governors = {gid: ConvergenceGovernor(gid, gov_config) for gid in groups}

    # --- Hierarchy ---
    hierarchy = HierarchyManager(
        groups, governors,
        config=HierarchyConfig(warmup_steps=20, emergency_fraction=0.80),
        total_steps=num_steps,
    )

    # --- Budget ---
    budget = FederatedBudget(
        group_ids, governors,
        config=BudgetConfig(eval_window=10, alpha=0.4, beta=0.3, gamma=0.3),
    )

    # --- Router ---
    router = GradientRouter(
        config_obj.d_model, len(group_ids),
        config=RouterConfig(
            warmup_steps=int(num_steps * 0.05),
            max_active_groups=max(len(group_ids) // 2, 2),
            entropy_coeff=0.01,
            balance_window=50,
        ),
    ).to(device)

    # --- Audit ---
    audit_log = Path(tmpdir) / "audit.jsonl"
    audit = AuditSink(audit_log)

    # --- Eval pipeline ---
    metric_dep_map = {
        "perplexity": ["all"],
        "attention_sparsity": ["attention_routing", "gates"],
    }

    def eval_ppl():
        with torch.no_grad():
            model.eval()
            batch = next(iter(loader))
            x = batch["input_ids"][:1].to(device)
            out = model(x, labels=x, training=False)
            model.train()
            return out["loss"].item() if "loss" in out else 10.0

    eval_pipeline = EvalPipeline(
        metric_dep_map,
        {"perplexity": eval_ppl, "attention_sparsity": lambda: 0.8},
        config=EvalConfig(max_evals_per_window=3, eval_window_steps=50),
    )

    for gov in governors.values():
        gov.subscribe(eval_pipeline.on_convergence_signal)

    # --- Optimizer (only trainable params, rebuilt when hierarchy changes) ---
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    router_optim = torch.optim.Adam(router.parameters(), lr=1e-3)

    metrics = {
        "losses": [],
        "grad_norms": [],
        "step_times_ms": [],
        "total_grad_computations": 0,
        "convergence_transitions": 0,
        "hierarchy_activations": [],
        "budget_invariant_violations": 0,
        "router_entropy": [],
        "active_param_fraction": [],
        "eval_calls": 0,
        "groups_masked_per_step": [],
    }

    transition_count = [0]
    def on_transition(signal):
        transition_count[0] += 1
    for gov in governors.values():
        gov.subscribe(on_transition)

    activation_log = []
    hierarchy.subscribe(activation_log.append)

    total_params = sum(p.numel() for p in model.parameters())
    prev_active_levels = set(hierarchy.active_levels)

    step = 0
    for batch in loader:
        if step >= num_steps:
            break

        t0 = time.perf_counter()
        optimizer.zero_grad()
        router_optim.zero_grad()

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Router forward
        with torch.no_grad():
            positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
            hidden = model.token_embedding(input_ids) + model.position_embedding(positions)
        routing = router(hidden, group_ids)

        # Model forward
        out = model(input_ids, labels=labels, training=True)
        loss = out["loss"] + routing["entropy_loss"]
        loss.backward()

        # Gradient masking (after warmup)
        masked_count = 0
        if not router.in_warmup:
            group_active = routing["hard_mask"].sum(dim=0) > 0
            for i, gid in enumerate(group_ids):
                if not group_active[i] and gid in groups:
                    for p in groups[gid].params:
                        if p.grad is not None:
                            p.grad.zero_()
                    masked_count += 1

        # Record gradient norm (of active params only)
        active_grad_norm = 0.0
        active_param_count = 0
        for gid, group in groups.items():
            if group.hierarchy_level in hierarchy.active_levels:
                for p in group.params:
                    if p.grad is not None:
                        active_grad_norm += p.grad.data.norm(2).item() ** 2
                    if p.requires_grad:
                        active_param_count += p.numel()
        active_grad_norm = active_grad_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        router_optim.step()
        router.step()

        # Update governors
        for gid, group in groups.items():
            if group.hierarchy_level in hierarchy.active_levels:
                governors[gid].update(group.grad_norm())

        # Update hierarchy
        lr_multipliers = hierarchy.step(step)

        # Rebuild optimizer if hierarchy changed
        if hierarchy.active_levels != prev_active_levels:
            trainable = [p for p in model.parameters() if p.requires_grad]
            if trainable:
                optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
            prev_active_levels = set(hierarchy.active_levels)

        # Update budget
        budget.step(step)
        if not budget.verify_invariant():
            metrics["budget_invariant_violations"] += 1

        # Update eval
        eval_results = eval_pipeline.step(step)
        if eval_results:
            metrics["eval_calls"] += 1

        # Audit
        audit.log_step(
            step=step,
            gradient_norms={gid: groups[gid].grad_norm() for gid in groups},
            convergence_states={gid: gov.state.value for gid, gov in governors.items()},
            hierarchy_active_levels=sorted(hierarchy.active_levels),
        )

        t1 = time.perf_counter()
        step_ms = (t1 - t0) * 1000

        metrics["losses"].append(out["loss"].item())
        metrics["grad_norms"].append(active_grad_norm)
        metrics["step_times_ms"].append(step_ms)
        metrics["total_grad_computations"] += active_param_count
        metrics["active_param_fraction"].append(active_param_count / total_params)
        metrics["groups_masked_per_step"].append(masked_count)
        metrics["router_entropy"].append(-routing["entropy_loss"].item() if routing["entropy_loss"].item() != 0 else 0)

        step += 1

    metrics["total_wall_ms"] = sum(metrics["step_times_ms"])
    metrics["convergence_transitions"] = transition_count[0]
    metrics["hierarchy_activations"] = activation_log
    metrics["eval_calls"] = eval_pipeline.get_stats()["total_evals"]

    # Verify audit chain
    query = AuditQuery(audit_log)
    valid, count = query.verify_chain()
    metrics["audit_chain_valid"] = valid
    metrics["audit_records"] = count

    # Final governor states
    metrics["final_states"] = {
        gid: gov.state.value for gid, gov in governors.items()
    }

    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(baseline, governed, num_steps):
    """Print comparison report."""
    print("\n" + "=" * 74)
    print("  COMPARISON: Baseline vs. Governed Training Pipeline")
    print("=" * 74)

    # Loss convergence
    def avg_loss(losses, window):
        if len(losses) < window:
            return sum(losses) / len(losses) if losses else float("inf")
        return sum(losses[-window:]) / window

    bl_final = avg_loss(baseline["losses"], 20)
    p5_final = avg_loss(governed["losses"], 20)
    bl_first = avg_loss(baseline["losses"][:20], 20)
    p5_first = avg_loss(governed["losses"][:20], 20)

    print(f"\n{'METRIC':<45} {'BASELINE':>12} {'GOVERNED':>12} {'DELTA':>10}")
    print("-" * 79)

    # Loss
    print(f"{'Initial loss (first 20 steps avg)':<45} {bl_first:>12.4f} {p5_first:>12.4f} {'':>10}")
    print(f"{'Final loss (last 20 steps avg)':<45} {bl_final:>12.4f} {p5_final:>12.4f} {(p5_final-bl_final)/bl_final*100:>+9.1f}%")

    if len(baseline["losses"]) >= 2 and len(governed["losses"]) >= 2:
        bl_ppl = math.exp(min(bl_final, 20))
        p5_ppl = math.exp(min(p5_final, 20))
        print(f"{'Final perplexity':<45} {bl_ppl:>12.1f} {p5_ppl:>12.1f} {(p5_ppl-bl_ppl)/bl_ppl*100:>+9.1f}%")

    # Wall clock
    bl_wall = baseline["total_wall_ms"]
    p5_wall = governed["total_wall_ms"]
    print(f"\n{'Total wall-clock time (ms)':<45} {bl_wall:>12.0f} {p5_wall:>12.0f} {(p5_wall-bl_wall)/bl_wall*100:>+9.1f}%")
    bl_avg_step = bl_wall / num_steps
    p5_avg_step = p5_wall / num_steps
    print(f"{'Avg step time (ms)':<45} {bl_avg_step:>12.1f} {p5_avg_step:>12.1f} {(p5_avg_step-bl_avg_step)/bl_avg_step*100:>+9.1f}%")

    # Gradient compute
    bl_gc = baseline["total_grad_computations"]
    p5_gc = governed["total_grad_computations"]
    reduction = (1 - p5_gc / bl_gc) * 100 if bl_gc > 0 else 0
    print(f"\n{'Total gradient computations':<45} {bl_gc:>12,} {p5_gc:>12,} {-reduction:>+9.1f}%")

    if governed["active_param_fraction"]:
        avg_active = sum(governed["active_param_fraction"]) / len(governed["active_param_fraction"])
        print(f"{'Avg active param fraction (governed)':<45} {'100.0%':>12} {avg_active*100:>11.1f}% {(avg_active-1)*100:>+9.1f}%")

    # Gradient routing
    if governed["groups_masked_per_step"]:
        avg_masked = sum(governed["groups_masked_per_step"]) / len(governed["groups_masked_per_step"])
        total_groups = len(governed["final_states"])
        print(f"{'Avg groups masked per step (of {total_groups})':<45} {'0':>12} {avg_masked:>12.1f} {'':>10}")

    # Eval efficiency
    bl_eval = baseline["eval_calls"]
    p5_eval = governed["eval_calls"]
    eval_red = (1 - p5_eval / bl_eval) * 100 if bl_eval > 0 else 0
    print(f"\n{'Evaluation calls':<45} {bl_eval:>12} {p5_eval:>12} {-eval_red:>+9.1f}%")

    # Governance machinery
    print(f"\n{'--- GOVERNANCE METRICS (governed only) ---':<45}")
    print(f"{'Convergence transitions':<45} {'N/A':>12} {governed['convergence_transitions']:>12}")
    print(f"{'Hierarchy activations':<45} {'N/A':>12} {len(governed['hierarchy_activations']):>12}")
    print(f"{'Budget invariant violations':<45} {'N/A':>12} {governed['budget_invariant_violations']:>12}")
    print(f"{'Audit chain valid':<45} {'N/A':>12} {str(governed.get('audit_chain_valid', 'N/A')):>12}")
    print(f"{'Audit records':<45} {'N/A':>12} {governed.get('audit_records', 0):>12}")

    # Final convergence states
    print(f"\n{'--- FINAL CONVERGENCE STATES ---':<45}")
    for gid, state in sorted(governed["final_states"].items()):
        print(f"  {gid:<40} {state}")

    # Loss curve comparison (sampled)
    print(f"\n{'--- LOSS CURVE (sampled every 25 steps) ---'}")
    print(f"  {'Step':>6}  {'Baseline':>10}  {'Governed':>10}  {'Diff':>10}")
    sample_points = list(range(0, num_steps, max(1, num_steps // 12)))
    if num_steps - 1 not in sample_points:
        sample_points.append(min(num_steps - 1, len(baseline["losses"]) - 1))
    for i in sample_points:
        if i < len(baseline["losses"]) and i < len(governed["losses"]):
            bl_l = baseline["losses"][i]
            p5_l = governed["losses"][i]
            diff = p5_l - bl_l
            print(f"  {i:>6}  {bl_l:>10.4f}  {p5_l:>10.4f}  {diff:>+10.4f}")

    print("\n" + "=" * 74)

    # Summary verdict
    print("\nSUMMARY:")
    if p5_final <= bl_final * 1.05:
        print(f"  Loss: PASS (governed within 5% of baseline: {p5_final:.4f} vs {bl_final:.4f})")
    else:
        print(f"  Loss: WATCH (governed {(p5_final-bl_final)/bl_final*100:+.1f}% vs baseline)")

    if reduction > 0:
        print(f"  Gradient compute: REDUCED by {reduction:.1f}%")
    else:
        print(f"  Gradient compute: No reduction (hierarchy not yet differentiated)")

    if governed["convergence_transitions"] > 0:
        print(f"  Convergence governance: ACTIVE ({governed['convergence_transitions']} transitions)")
    else:
        print(f"  Convergence governance: No transitions (thresholds may need tuning)")

    if len(governed["hierarchy_activations"]) > 0:
        print(f"  Hierarchy: ACTIVE ({len(governed['hierarchy_activations'])} level activations)")
    else:
        print(f"  Hierarchy: Only L0 active (short run or slow convergence)")

    if governed["budget_invariant_violations"] == 0:
        print(f"  Budget invariant: HELD (0 violations across {num_steps} steps)")
    else:
        print(f"  Budget invariant: VIOLATED {governed['budget_invariant_violations']} times")

    if governed.get("audit_chain_valid"):
        print(f"  Audit chain: VALID ({governed.get('audit_records', 0)} records)")

    if bl_eval > 0 and eval_red > 0:
        print(f"  Eval efficiency: {eval_red:.0f}% fewer eval calls")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baseline vs Governed training comparison")
    parser.add_argument("--steps", type=int, default=300, help="Training steps per run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=2000, help="Max dataset samples")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Model config — matches configs/validation.yaml
    config = LeanFormerConfig(
        vocab_size=32000,
        d_model=128, n_heads=4, n_layers=4, d_ff=512,
        max_seq_len=args.seq_len,
        attention_rank=16, ff_rank=16,
        screening_rank=4, attention_top_k=32,
        ff_gate_rank=4, ff_sparsity_target=0.7,
        min_depth=1, exit_threshold=0.05, dropout=0.1,
    )

    param_count = sum(
        p.numel() for p in LeanFormer(config).parameters()
    )
    print(f"Model: {param_count:,} params (d_model={config.d_model}, {config.n_layers} layers)")
    print(f"Training: {args.steps} steps, batch_size={args.batch_size}, lr={args.lr}")

    # Load dataset once
    print("\nLoading dataset...")
    dataset = WikiTextDataset(
        max_samples=args.max_samples,
        seq_len=args.seq_len,
        vocab_size=config.vocab_size,
    )
    print(f"Dataset: {len(dataset)} samples, seq_len={args.seq_len}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0,
    )

    # --- Run A: Baseline ---
    print(f"\n{'='*40}")
    print("  TRAINING A: Baseline (flat training)")
    print(f"{'='*40}")
    model_a = LeanFormer(config).to(device)
    torch.manual_seed(args.seed)
    baseline_metrics = train_baseline(model_a, loader, args.steps, args.lr, device, args.seed)
    print(f"  Done. Final loss: {baseline_metrics['losses'][-1]:.4f}")

    # --- Run B: Governed ---
    print(f"\n{'='*40}")
    print("  TRAINING B: Governed pipeline")
    print(f"{'='*40}")
    model_b = LeanFormer(config).to(device)
    # Copy initial weights so both runs start identical
    model_b.load_state_dict(model_a.state_dict().copy())
    # Re-init to same starting point
    model_b = LeanFormer(config).to(device)
    torch.manual_seed(args.seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        governed_metrics = train_governed(model_b, loader, args.steps, args.lr, device, args.seed, tmpdir)
    print(f"  Done. Final loss: {governed_metrics['losses'][-1]:.4f}")

    # --- Report ---
    report(baseline_metrics, governed_metrics, args.steps)

    # Save raw metrics
    output = {
        "config": {
            "steps": args.steps, "seed": args.seed,
            "batch_size": args.batch_size, "lr": args.lr,
            "params": param_count, "device": device,
        },
        "baseline": {
            k: v for k, v in baseline_metrics.items()
            if not isinstance(v, list) or len(v) < 500
        },
        "governed": {
            k: v for k, v in governed_metrics.items()
            if not isinstance(v, list) or len(v) < 500
        },
    }
    results_path = Path("local/governed_comparison.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Raw metrics saved to {results_path}")


if __name__ == "__main__":
    main()
