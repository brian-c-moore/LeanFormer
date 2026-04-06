"""
Train LeanFormer Reasoning Core with governed training pipeline.

Uses the tokenized dataset from prepare_reasoning_data.py.
CUDA mixed precision (fp16) with GradScaler throughout.

Governed training: per-group convergence governors, coarse-to-fine hierarchy
activation, federated budget allocation, gradient routing, SHA-256 audit chain.

After main training: exit head tuning phase (1000 steps, lr=1e-3).

Usage:
    python -m leanformer.scripts.train_reasoning
"""

import sys
import time
import math
import json
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from datasets import load_from_disk

from rich.console import Console

from .. import DEFAULT_TOKENIZER
from ..model.config import LeanFormerConfig
from ..model.leanformer import LeanFormer
from ..knowledge_plane.dfs import compute_model_hash
from ..training.param_groups import build_param_groups, load_registry
from ..training.convergence import ConvergenceConfig, ConvergenceGovernor, ConvergenceState
from ..training.hierarchy import HierarchyConfig, HierarchyManager
from ..training.budget import BudgetConfig, FederatedBudget
from ..training.audit import AuditSink, AuditQuery
from ..training.eval_pipeline import EvalConfig, EvalPipeline

console = Console()

CONFIG_PATH = "configs/reasoning_core.yaml"
DATA_DIR = Path("data/reasoning-core-tokenized")


def main():
    if not torch.cuda.is_available():
        console.print("[red]ERROR: CUDA not available.[/red]")
        sys.exit(1)

    if not DATA_DIR.exists():
        console.print(f"[red]ERROR: Tokenized dataset not found at {DATA_DIR}[/red]")
        console.print("Run: python -m leanformer.scripts.prepare_reasoning_data")
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Build model
    model_config = LeanFormerConfig(**cfg["model"])
    model = LeanFormer(model_config).cuda()

    param_count = sum(p.numel() for p in model.parameters())
    dense_equiv = model_config.dense_equivalent_estimate()

    console.print(f"[bold cyan]LeanFormer Reasoning Core Training[/bold cyan]")
    console.print(f"  Parameters:  {param_count:,}")
    console.print(f"  Dense equiv: {dense_equiv:,} ({dense_equiv/param_count:.1f}x compression)")
    console.print(f"  Device:      {torch.cuda.get_device_name(0)}")
    console.print()

    # Load tokenized dataset
    console.print("Loading tokenized reasoning corpus...")
    dataset = load_from_disk(str(DATA_DIR))
    dataset = dataset.with_format("torch")

    # 95/5 train/val split
    total = len(dataset)
    val_size = int(total * 0.05)
    train_size = total - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    console.print(f"  Train: {train_size:,} samples")
    console.print(f"  Val:   {val_size:,} samples")

    # Training config
    tcfg = cfg["training"]
    batch_size = tcfg["batch_size"]
    grad_accum = tcfg["gradient_accumulation"]
    lr = tcfg["lr"]
    warmup_steps = tcfg["warmup_steps"]
    epochs = tcfg["epochs"]
    weight_decay = tcfg.get("weight_decay", 0.01)
    max_grad_norm = tcfg.get("max_grad_norm", 1.0)
    logging_steps = tcfg.get("logging_steps", 100)
    save_steps = tcfg.get("save_steps", 5000)
    eval_steps = tcfg.get("eval_steps", 1000)
    output_dir = Path(tcfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_batch = batch_size * grad_accum
    micro_steps_per_epoch = train_size // batch_size
    total_micro_steps = micro_steps_per_epoch * epochs
    total_steps = total_micro_steps // grad_accum  # optimizer steps

    console.print(f"  Batch: {batch_size} x {grad_accum} accum = {effective_batch} effective")
    console.print(f"  LR: {lr}, warmup: {warmup_steps}")
    console.print(f"  Epochs: {epochs} ({total_steps:,} optimizer steps, {total_micro_steps:,} micro-steps)")
    console.print()

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=True,
        num_workers=2, pin_memory=True,
    )

    # === Governed Training Setup ===
    registry = load_registry()
    groups = build_param_groups(model, registry)
    group_ids = list(groups.keys())

    # Per-group convergence governors
    gov_config = ConvergenceConfig(
        ema_decay=0.99,
        cooling_threshold=0.5,
        cooling_window=200,
        converged_threshold=0.1,
        confirmation_window=500,
        reactivation_delta=0.15,
        reactivation_warmup=100,
    )
    governors = {
        gid: ConvergenceGovernor(
            gid, gov_config,
            initially_active=(groups[gid].hierarchy_level == 0),
        )
        for gid in groups
    }

    # Coarse-to-fine hierarchy
    hierarchy = HierarchyManager(
        groups, governors,
        config=HierarchyConfig(warmup_steps=100, emergency_fraction=0.80),
        total_steps=total_steps,
    )

    # Federated budget
    budget = FederatedBudget(
        group_ids, governors,
        config=BudgetConfig(eval_window=50, alpha=0.4, beta=0.3, gamma=0.3),
    )

    # SHA-256 audit log
    audit_log_path = output_dir / "audit.jsonl"
    audit = AuditSink(audit_log_path)

    # Change-triggered eval
    metric_dep_map = registry.get("metric_dependency_map", {"perplexity": ["all"]})

    def eval_ppl():
        return evaluate(model, val_loader, model_config)

    eval_pipeline = EvalPipeline(
        metric_dep_map,
        {"perplexity": eval_ppl},
        config=EvalConfig(max_evals_per_window=3, eval_window_steps=500),
    )
    for gov in governors.values():
        gov.subscribe(eval_pipeline.on_convergence_signal)

    console.print(f"\n[bold cyan]Governed Training Pipeline[/bold cyan]")
    console.print(f"  Parameter groups: {len(groups)}")
    console.print(f"  Hierarchy levels: {sorted(hierarchy.levels.keys())}")
    console.print(f"  Active at start:  L0 ({sum(g.param_count for g in groups.values() if g.hierarchy_level == 0):,} params)")
    console.print(f"  Audit log:        {audit_log_path}")

    # Optimizer — only trainable params (L0 initially)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )

    # Cosine schedule with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda")

    # Training state
    global_step = 0       # micro-batch counter
    optimizer_step = 0    # actual weight update counter
    best_val_loss = float("inf")
    training_log = []
    start_epoch = 0
    prev_active_levels = set(hierarchy.active_levels)

    # Check for existing checkpoint to resume from
    resume_state = load_training_state(output_dir, model, optimizer, scheduler, scaler)
    if resume_state is not None:
        optimizer_step, global_step, start_epoch, best_val_loss = resume_state
        # Load model weights
        model_path = output_dir / "pytorch_model.bin"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location="cuda", weights_only=True))
        # Load training log if available
        log_path = output_dir / "training_log.json"
        if log_path.exists():
            with open(log_path) as f:
                training_log = json.load(f)
        # Load governance state if available
        gov_state_path = output_dir / "governance_state.pt"
        if gov_state_path.exists():
            gov_state = torch.load(gov_state_path, map_location="cpu", weights_only=False)
            for gid, gs in gov_state.get("governors", {}).items():
                if gid in governors:
                    governors[gid].load_state_dict(gs)
            if "hierarchy" in gov_state:
                hierarchy.load_state_dict(gov_state["hierarchy"])
            if "budget" in gov_state:
                budget.load_state_dict(gov_state["budget"])
            console.print(f"  [green]Resumed governance state[/green]")

    console.print("[bold green]Starting training...[/bold green]\n")
    train_start = time.perf_counter()

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_lm_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_steps = 0
        epoch_start = time.perf_counter()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            labels = batch["labels"].cuda(non_blocking=True)

            with torch.amp.autocast("cuda"):
                out = model(input_ids, labels=labels, training=True)
                loss = out["loss"] / grad_accum

            scaler.scale(loss).backward()

            epoch_loss += out["loss"].item()
            epoch_lm_loss += out["lm_loss"].item()
            epoch_aux_loss += out["aux_loss"].item()
            epoch_steps += 1
            global_step += 1

            # Gradient accumulation step
            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # Capture gradient norms and update governors BEFORE step/zero_grad
                # (gradients are live now, they won't be after zero_grad)
                step_grad_norms = {gid: group.grad_norm() for gid, group in groups.items()}
                for gid, group in groups.items():
                    if group.hierarchy_level in hierarchy.active_levels:
                        governors[gid].update(step_grad_norms[gid])

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1

                # --- Governance updates (per optimizer step) ---

                # Update hierarchy
                lr_multipliers = hierarchy.step(optimizer_step)

                # Rebuild optimizer if hierarchy activated new levels
                if hierarchy.active_levels != prev_active_levels:
                    new_levels = hierarchy.active_levels - prev_active_levels
                    # Activate governors for newly activated groups
                    for gid, group in groups.items():
                        if group.hierarchy_level in new_levels:
                            governors[gid].activate()
                    console.print(
                        f"  [yellow]HIERARCHY step {optimizer_step}: "
                        f"activated L{',L'.join(str(l) for l in sorted(new_levels))} "
                        f"(active: {sorted(hierarchy.active_levels)})[/yellow]"
                    )
                    trainable = [p for p in model.parameters() if p.requires_grad]
                    if trainable:
                        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
                        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                        # Step scheduler to current position
                        for _ in range(optimizer_step):
                            scheduler.step()
                    prev_active_levels = set(hierarchy.active_levels)

                # Update budget
                budget.step(optimizer_step)

                # Update eval pipeline
                eval_results = eval_pipeline.step(optimizer_step)

                # Audit log (uses step_grad_norms captured before zero_grad)
                if optimizer_step % 10 == 0:  # Log every 10 steps to limit I/O
                    audit.log_step(
                        step=optimizer_step,
                        gradient_norms=step_grad_norms,
                        convergence_states={gid: gov.state.value for gid, gov in governors.items()},
                        budget_allocations=budget.allocations,
                        hierarchy_active_levels=sorted(hierarchy.active_levels),
                        loss_before=out["loss"].item(),
                    )

                # Logging (per optimizer step)
                if optimizer_step % logging_steps == 0:
                    avg_loss = epoch_loss / epoch_steps
                    avg_lm = epoch_lm_loss / epoch_steps
                    avg_aux = epoch_aux_loss / epoch_steps
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed = time.perf_counter() - epoch_start
                    tokens_per_sec = epoch_steps * batch_size * model_config.max_seq_len / elapsed
                    vram = torch.cuda.max_memory_allocated() / 1e9

                    stats = out.get("layer_stats", [])
                    gate_losses_log = [s["gate_loss"].item() for s in stats if "gate_loss" in s]
                    avg_gate_loss = sum(gate_losses_log) / max(len(gate_losses_log), 1) if gate_losses_log else 0.0
                    exit_layer = out.get("exit_layer", model_config.n_layers)

                    # Governance telemetry
                    n_active = sum(1 for g in governors.values() if g.state == ConvergenceState.ACTIVE)
                    n_cooling = sum(1 for g in governors.values() if g.state == ConvergenceState.COOLING)
                    n_converged = sum(1 for g in governors.values() if g.state == ConvergenceState.CONVERGED)
                    active_params = sum(
                        g.param_count for g in groups.values()
                        if g.hierarchy_level in hierarchy.active_levels
                    )
                    total_params = sum(g.param_count for g in groups.values())
                    active_frac = active_params / total_params

                    log_entry = {
                        "step": optimizer_step,
                        "epoch": epoch + 1,
                        "loss": avg_loss,
                        "lm_loss": avg_lm,
                        "aux_loss": avg_aux,
                        "gate_loss": avg_gate_loss,
                        "lr": current_lr,
                        "tokens_per_sec": tokens_per_sec,
                        "vram_gb": vram,
                        "exit_layer": exit_layer,
                        "gov_active": n_active,
                        "gov_cooling": n_cooling,
                        "gov_converged": n_converged,
                        "hierarchy_levels": sorted(hierarchy.active_levels),
                        "active_param_frac": round(active_frac, 3),
                        "budget_invariant": budget.verify_invariant(),
                        "audit_records": audit.record_count,
                    }
                    training_log.append(log_entry)

                    gov_str = f"A{n_active}/C{n_cooling}/V{n_converged}"
                    lvl_str = ",".join(f"L{l}" for l in sorted(hierarchy.active_levels))

                    console.print(
                        f"  step {optimizer_step:>6}/{total_steps} | "
                        f"epoch {epoch+1} | "
                        f"loss {avg_loss:.4f} (lm {avg_lm:.4f} + aux {avg_aux:.4f}) | "
                        f"lr {current_lr:.2e} | "
                        f"{tokens_per_sec:.0f} tok/s | "
                        f"vram {vram:.1f}GB | "
                        f"gate {avg_gate_loss:.4f} | "
                        f"exit {exit_layer}/{model_config.n_layers} | "
                        f"gov {gov_str} | "
                        f"hier {lvl_str} | "
                        f"params {active_frac:.0%}"
                    )

                    # Check for NaN
                    if not math.isfinite(avg_loss):
                        console.print("[red]ERROR: NaN/Inf loss detected! Stopping.[/red]")
                        with open(output_dir / "training_log.json", "w") as f:
                            json.dump(training_log, f, indent=2)
                        sys.exit(1)

                # Eval
                if optimizer_step % eval_steps == 0:
                    val_loss = evaluate(model, val_loader, model_config)
                    val_ppl = math.exp(min(val_loss, 20))
                    console.print(
                        f"  [cyan]EVAL step {optimizer_step}: val_loss={val_loss:.4f}, val_ppl={val_ppl:.1f}[/cyan]"
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            model, model_config, output_dir, optimizer_step, val_loss,
                            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            epoch=epoch, global_step=global_step,
                            best_val_loss=best_val_loss, training_log=training_log,
                            governance_state={
                                "governors": {gid: gov.state_dict() for gid, gov in governors.items()},
                                "hierarchy": hierarchy.state_dict(),
                                "budget": budget.state_dict(),
                            },
                        )
                        console.print(f"  [green]New best! Saved to {output_dir}[/green]")
                    model.train()

                # Save checkpoint
                if optimizer_step % save_steps == 0:
                    save_checkpoint(
                        model, model_config,
                        output_dir / f"step-{optimizer_step}",
                        optimizer_step, epoch_loss / epoch_steps,
                        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        epoch=epoch, global_step=global_step,
                        best_val_loss=best_val_loss, training_log=training_log,
                        governance_state={
                            "governors": {gid: gov.state_dict() for gid, gov in governors.items()},
                            "hierarchy": hierarchy.state_dict(),
                            "budget": budget.state_dict(),
                                },
                    )

        # End of epoch
        epoch_elapsed = time.perf_counter() - epoch_start
        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        val_loss = evaluate(model, val_loader, model_config)
        val_ppl = math.exp(min(val_loss, 20))

        # Governance summary for epoch
        states = {g.state.value: 0 for g in governors.values()}
        for g in governors.values():
            states[g.state.value] = states.get(g.state.value, 0) + 1

        console.print(
            f"\n[bold]Epoch {epoch+1}/{epochs}[/bold] | "
            f"train_loss: {avg_train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_ppl: {val_ppl:.1f} | "
            f"time: {epoch_elapsed/3600:.1f}h | "
            f"gov: {states} | "
            f"hier: {sorted(hierarchy.active_levels)} | "
            f"audit: {audit.record_count} records"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, model_config, output_dir, optimizer_step, val_loss,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch, global_step=global_step,
                best_val_loss=best_val_loss, training_log=training_log,
                governance_state={
                    "governors": {gid: gov.state_dict() for gid, gov in governors.items()},
                    "hierarchy": hierarchy.state_dict(),
                    "budget": budget.state_dict(),
                },
            )
            console.print(f"  [green]New best! Saved to {output_dir}[/green]")
        console.print()

    # === Exit head tuning phase ===
    console.print("[bold cyan]Starting exit head tuning...[/bold cyan]\n")
    tune_exit_heads(model, model_config, train_loader, output_dir, training_log)

    # Final
    total_time = time.perf_counter() - train_start
    final_ppl = math.exp(min(best_val_loss, 20))

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"  Total time:  {total_time/3600:.1f}h")
    console.print(f"  Best val loss: {best_val_loss:.4f}")
    console.print(f"  Best val ppl:  {final_ppl:.1f}")
    console.print(f"  Checkpoint:    {output_dir}")

    # Save final checkpoint (model weights only — training is done)
    save_checkpoint(model, model_config, output_dir, optimizer_step, best_val_loss,
                    training_log=training_log)

    # Compute and save model hash
    console.print("\nComputing model hash...")
    model_hash = compute_model_hash(model)
    hash_path = output_dir / "model_hash.txt"
    with open(hash_path, "w") as f:
        f.write(model_hash)
    console.print(f"  Model hash: {model_hash[:16]}...")
    console.print(f"  Saved to: {hash_path}")

    # === Post-training validation ===
    console.print("\n[bold cyan]Post-training validation...[/bold cyan]")
    post_training_validation(model, model_config, val_loader, output_dir)


def tune_exit_heads(model, config, train_loader, output_dir, training_log):
    """
    Adaptive depth is conservative without explicit exit head training.
    Freeze all weights except exit classifier heads, tune for 1000 steps.
    """
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze exit heads only
    exit_params = []
    for name, param in model.named_parameters():
        if "exit_head" in name:
            param.requires_grad = True
            exit_params.append(param)

    console.print(f"  Exit head params: {sum(p.numel() for p in exit_params):,}")

    # More aggressive exit threshold
    model.depth_controller.threshold = 0.04

    optimizer = torch.optim.AdamW(exit_params, lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    step = 0
    max_steps = 1000
    depth_log = []

    for batch in train_loader:
        if step >= max_steps:
            break

        input_ids = batch["input_ids"].cuda(non_blocking=True)
        labels = batch["labels"].cuda(non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            out = model(input_ids, labels=labels, training=True)
            # Only use aux_loss (exit head loss)
            loss = out["aux_loss"]

        if loss.requires_grad:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        step += 1

        if step % 100 == 0:
            exit_layer = out.get("exit_layer", config.n_layers)
            console.print(
                f"  exit_tune step {step}/{max_steps} | "
                f"aux_loss {out['aux_loss'].item():.4f} | "
                f"exit_layer {exit_layer}/{config.n_layers}"
            )
            depth_log.append({
                "step": step,
                "aux_loss": out["aux_loss"].item(),
                "exit_layer": exit_layer,
            })

    # Unfreeze all params for future use
    for param in model.parameters():
        param.requires_grad = True

    # Save exit-tuned checkpoint
    exit_path = output_dir / "exit_tuned.pt"
    torch.save(model.state_dict(), exit_path)
    console.print(f"  Exit-tuned checkpoint saved to {exit_path}")

    # Add depth log to training log
    training_log.append({"phase": "exit_head_tuning", "depth_log": depth_log})


def post_training_validation(model, config, val_loader, output_dir):
    """Post-training validation: perplexity, generation samples, orthogonal capacity."""
    from transformers import AutoTokenizer

    # 1. Validation perplexity
    val_loss = evaluate(model, val_loader, config)
    val_ppl = math.exp(min(val_loss, 20))
    console.print(f"  Validation perplexity: {val_ppl:.1f}")

    # 2. Generation samples
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
    tokenizer.pad_token = tokenizer.eos_token

    reasoning_prompts = [
        "def fibonacci(n):",
        "The derivative of x squared is",
        "If all mammals are warm-blooded, and",
        "The experiment demonstrated that",
        "To solve this problem, first",
    ]

    model.eval()
    generations = []
    for prompt in reasoning_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
        with torch.no_grad():
            output_ids, _ = model.generate(input_ids, max_new_tokens=100, temperature=0.8)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generations.append({"prompt": prompt, "generation": text})
        console.print(f"  [dim]{prompt}[/dim]")
        console.print(f"  {text[:200]}\n")

    # 3. Orthogonal capacity measurement
    capacity = measure_orthogonal_capacity(model, delta_rank=16)
    console.print(f"  Total available dims: {capacity['total_available_dims']}")
    console.print(f"  Theoretical max deltas (rank 16): {capacity['total_theoretical_max_deltas']}")

    # Save all validation results
    validation_results = {
        "val_loss": val_loss,
        "val_perplexity": val_ppl,
        "generations": generations,
        "orthogonal_capacity": capacity,
    }
    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(validation_results, f, indent=2, default=str)
    console.print(f"\n  Validation results saved to {output_dir / 'validation_results.json'}")


def measure_orthogonal_capacity(model, delta_rank=16):
    """
    For each layer's main weight matrices:
    1. Compute SVD
    2. Count significant singular values (> 1% of max)
    3. Compute theoretical max deltas = (d_model - n_significant) / delta_rank
    """
    results = {
        "per_layer": [],
        "d_model": model.config.d_model,
        "delta_rank": delta_rank,
    }
    total_available = 0
    total_max_deltas = 0

    for layer_idx in range(model.config.n_layers):
        layer = model.layers[layer_idx]

        # Attention output projection — get materialized weight
        attn_weight = (layer.attn.out_proj.A @ layer.attn.out_proj.B).detach().float()
        U, S_attn, V = torch.linalg.svd(attn_weight, full_matrices=False)
        threshold = S_attn[0] * 0.01
        n_sig_attn = (S_attn > threshold).sum().item()
        avail_attn = model.config.d_model - n_sig_attn
        max_deltas_attn = avail_attn // delta_rank

        # FF up projection
        ff_weight = (layer.ff.up_proj.A @ layer.ff.up_proj.B).detach().float()
        U, S_ff, V = torch.linalg.svd(ff_weight, full_matrices=False)
        threshold = S_ff[0] * 0.01
        n_sig_ff = (S_ff > threshold).sum().item()
        avail_ff = model.config.d_model - n_sig_ff
        max_deltas_ff = avail_ff // delta_rank

        results["per_layer"].append({
            "layer": layer_idx,
            "attention_significant_svs": n_sig_attn,
            "ff_significant_svs": n_sig_ff,
            "available_dims_attention": avail_attn,
            "available_dims_ff": avail_ff,
            "theoretical_max_deltas_attention": max_deltas_attn,
            "theoretical_max_deltas_ff": max_deltas_ff,
        })
        total_available += avail_attn + avail_ff
        total_max_deltas += max_deltas_attn + max_deltas_ff

    results["total_available_dims"] = total_available
    results["total_theoretical_max_deltas"] = total_max_deltas
    return results


def evaluate(model, val_loader, config):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            labels = batch["labels"].cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                out = model(input_ids, labels=labels, training=False)
            total_loss += out["lm_loss"].item()
            total_steps += 1
            if total_steps >= 100:  # Cap eval at 100 batches for speed
                break
    return total_loss / max(total_steps, 1)


def save_checkpoint(model, config, output_dir, step, val_loss,
                    optimizer=None, scheduler=None, scaler=None,
                    epoch=0, global_step=0, best_val_loss=None,
                    training_log=None, governance_state=None):
    """Save a full training checkpoint (model + training state + governance state).

    Writes to a temp file first, then renames — atomic on most filesystems
    so a crash mid-save won't corrupt the checkpoint.
    """
    import random
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model weights (always saved)
    model_path = output_dir / "pytorch_model.bin"
    tmp_path = output_dir / "pytorch_model.bin.tmp"
    torch.save(model.state_dict(), tmp_path)
    tmp_path.replace(model_path)

    config.save(output_dir / "leanformer_config.json")

    meta = {
        "optimizer_step": step,
        "global_step": global_step,
        "epoch": epoch,
        "val_loss": val_loss,
        "val_ppl": math.exp(min(val_loss, 20)),
        "best_val_loss": best_val_loss if best_val_loss is not None else val_loss,
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Full training state for resume (optimizer, scheduler, scaler, RNG)
    if optimizer is not None:
        training_state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "optimizer_step": step,
            "global_step": global_step,
            "epoch": epoch,
            "best_val_loss": best_val_loss if best_val_loss is not None else val_loss,
            "torch_rng": torch.random.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "python_rng": random.getstate(),
        }
        state_path = output_dir / "training_state.pt"
        tmp_state = output_dir / "training_state.pt.tmp"
        torch.save(training_state, tmp_state)
        tmp_state.replace(state_path)

    # Save governance state for resume
    if governance_state is not None:
        gov_path = output_dir / "governance_state.pt"
        tmp_gov = output_dir / "governance_state.pt.tmp"
        torch.save(governance_state, tmp_gov)
        tmp_gov.replace(gov_path)

    # Save training log if provided
    if training_log is not None:
        with open(output_dir / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)


def load_training_state(output_dir, model, optimizer, scheduler, scaler):
    """Load training state from a checkpoint for resuming training.

    Returns (optimizer_step, global_step, epoch, best_val_loss) or None if
    no training state file exists.
    """
    import random
    output_dir = Path(output_dir)
    state_path = output_dir / "training_state.pt"

    if not state_path.exists():
        return None

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])

    # Restore RNG states
    if state.get("torch_rng") is not None:
        torch.random.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng"])
    if state.get("python_rng") is not None:
        random.setstate(state["python_rng"])

    console.print(f"  [green]Resumed from step {state['optimizer_step']}, "
                  f"epoch {state['epoch']}[/green]")

    return (
        state["optimizer_step"],
        state["global_step"],
        state["epoch"],
        state["best_val_loss"],
    )


if __name__ == "__main__":
    main()
