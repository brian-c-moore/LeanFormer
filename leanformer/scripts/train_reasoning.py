"""
Train LeanFormer Reasoning Core (~66M params, d_model=768) on knowledge-naive corpus.

Uses the tokenized dataset from prepare_reasoning_data.py.
CUDA mixed precision (fp16) with GradScaler throughout.

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
    steps_per_epoch = train_size // batch_size
    total_steps = steps_per_epoch * epochs

    console.print(f"  Batch: {batch_size} x {grad_accum} accum = {effective_batch} effective")
    console.print(f"  LR: {lr}, warmup: {warmup_steps}")
    console.print(f"  Epochs: {epochs} ({total_steps:,} total steps)")
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

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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
    global_step = 0
    best_val_loss = float("inf")
    training_log = []

    console.print("[bold green]Starting training...[/bold green]\n")
    train_start = time.perf_counter()

    for epoch in range(epochs):
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
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            # Logging
            if global_step % logging_steps == 0:
                avg_loss = epoch_loss / epoch_steps
                avg_lm = epoch_lm_loss / epoch_steps
                avg_aux = epoch_aux_loss / epoch_steps
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.perf_counter() - epoch_start
                tokens_per_sec = epoch_steps * batch_size * model_config.max_seq_len / elapsed
                vram = torch.cuda.max_memory_allocated() / 1e9

                stats = out.get("layer_stats", [])
                avg_ff_sparsity = sum(s.get("ff_sparsity", 0) for s in stats) / max(len(stats), 1)
                exit_layer = out.get("exit_layer", model_config.n_layers)

                log_entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "lm_loss": avg_lm,
                    "aux_loss": avg_aux,
                    "lr": current_lr,
                    "tokens_per_sec": tokens_per_sec,
                    "vram_gb": vram,
                    "ff_sparsity": avg_ff_sparsity,
                    "exit_layer": exit_layer,
                }
                training_log.append(log_entry)

                console.print(
                    f"  step {global_step:>6}/{total_steps} | "
                    f"epoch {epoch+1} | "
                    f"loss {avg_loss:.4f} (lm {avg_lm:.4f} + aux {avg_aux:.4f}) | "
                    f"lr {current_lr:.2e} | "
                    f"{tokens_per_sec:.0f} tok/s | "
                    f"vram {vram:.1f}GB | "
                    f"ff_sp {avg_ff_sparsity:.2f} | "
                    f"exit {exit_layer}/{model_config.n_layers}"
                )

                # Check for NaN
                if not math.isfinite(avg_loss):
                    console.print("[red]ERROR: NaN/Inf loss detected! Stopping.[/red]")
                    # Save what we have
                    with open(output_dir / "training_log.json", "w") as f:
                        json.dump(training_log, f, indent=2)
                    sys.exit(1)

            # Eval
            if global_step % eval_steps == 0:
                val_loss = evaluate(model, val_loader, model_config)
                val_ppl = math.exp(min(val_loss, 20))  # Cap to avoid overflow
                console.print(
                    f"  [cyan]EVAL step {global_step}: val_loss={val_loss:.4f}, val_ppl={val_ppl:.1f}[/cyan]"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, model_config, output_dir, global_step, val_loss)
                    console.print(f"  [green]New best! Saved to {output_dir}[/green]")
                model.train()

            # Save checkpoint
            if global_step % save_steps == 0:
                save_checkpoint(
                    model, model_config,
                    output_dir / f"step-{global_step}",
                    global_step, epoch_loss / epoch_steps,
                )

        # End of epoch
        epoch_elapsed = time.perf_counter() - epoch_start
        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        val_loss = evaluate(model, val_loader, model_config)
        val_ppl = math.exp(min(val_loss, 20))

        console.print(
            f"\n[bold]Epoch {epoch+1}/{epochs}[/bold] | "
            f"train_loss: {avg_train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_ppl: {val_ppl:.1f} | "
            f"time: {epoch_elapsed/3600:.1f}h"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, model_config, output_dir, global_step, val_loss)
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

    # Save training log
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # Save final checkpoint
    save_checkpoint(model, model_config, output_dir, global_step, best_val_loss)

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


def save_checkpoint(model, config, output_dir, step, val_loss):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    config.save(output_dir / "leanformer_config.json")
    meta = {"step": step, "val_loss": val_loss, "val_ppl": math.exp(min(val_loss, 20))}
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
