"""
Train LeanFormer at scale on OpenWebText with mixed precision.

Uses the tokenized dataset saved by download_openwebtext.py.
CUDA mixed precision (fp16) with GradScaler throughout.

Usage:
    python -m leanformer.scripts.train_scale
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

from ..model.config import LeanFormerConfig
from ..model.leanformer import LeanFormer

console = Console()

CONFIG_PATH = "configs/scale.yaml"
DATA_DIR = Path("data/openwebtext-500k-tokenized")


def main():
    if not torch.cuda.is_available():
        console.print("[red]ERROR: CUDA not available.[/red]")
        sys.exit(1)

    if not DATA_DIR.exists():
        console.print(f"[red]ERROR: Tokenized dataset not found at {DATA_DIR}[/red]")
        console.print("Run: python -m leanformer.scripts.download_openwebtext")
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Build model
    model_config = LeanFormerConfig(**cfg["model"])
    model = LeanFormer(model_config).cuda()

    param_count = sum(p.numel() for p in model.parameters())
    dense_equiv = model_config.dense_equivalent_estimate()

    console.print(f"[bold cyan]LeanFormer Scale Training[/bold cyan]")
    console.print(f"  Parameters:  {param_count:,}")
    console.print(f"  Dense equiv: {dense_equiv:,} ({dense_equiv/param_count:.1f}x compression)")
    console.print(f"  Device:      {torch.cuda.get_device_name(0)}")
    console.print()

    # Load tokenized dataset
    console.print("Loading tokenized OpenWebText...")
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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2, pin_memory=True)

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
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.perf_counter() - epoch_start
                tokens_per_sec = epoch_steps * batch_size * model_config.max_seq_len / elapsed
                vram = torch.cuda.max_memory_allocated() / 1e9

                stats = out.get("layer_stats", [])
                avg_ff_sparsity = sum(s.get("ff_sparsity", 0) for s in stats) / max(len(stats), 1)
                exit_layer = out.get("exit_layer", model_config.n_layers)

                log_entry = {
                    "step": global_step,
                    "loss": avg_loss,
                    "lr": current_lr,
                    "tokens_per_sec": tokens_per_sec,
                    "vram_gb": vram,
                    "ff_sparsity": avg_ff_sparsity,
                    "exit_layer": exit_layer,
                }
                training_log.append(log_entry)

                console.print(
                    f"  step {global_step:>6}/{total_steps} | "
                    f"loss {avg_loss:.4f} | "
                    f"lr {current_lr:.2e} | "
                    f"{tokens_per_sec:.0f} tok/s | "
                    f"vram {vram:.1f}GB | "
                    f"ff_sp {avg_ff_sparsity:.2f} | "
                    f"exit {exit_layer}/{model_config.n_layers}"
                )

                # Check for NaN
                if not math.isfinite(avg_loss):
                    console.print("[red]ERROR: NaN/Inf loss detected! Stopping.[/red]")
                    sys.exit(1)

            # Eval
            if global_step % eval_steps == 0:
                val_loss = evaluate(model, val_loader, model_config)
                val_ppl = math.exp(val_loss)
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
                save_checkpoint(model, model_config, output_dir / f"step-{global_step}", global_step, epoch_loss / epoch_steps)

        # End of epoch
        epoch_elapsed = time.perf_counter() - epoch_start
        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        val_loss = evaluate(model, val_loader, model_config)
        val_ppl = math.exp(val_loss)

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

    # Final
    total_time = time.perf_counter() - train_start
    final_ppl = math.exp(best_val_loss)

    console.print(f"[bold green]Training complete![/bold green]")
    console.print(f"  Total time:  {total_time/3600:.1f}h")
    console.print(f"  Best val loss: {best_val_loss:.4f}")
    console.print(f"  Best val ppl:  {final_ppl:.1f}")
    console.print(f"  Checkpoint:    {output_dir}")

    # Save training log
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # Save final checkpoint
    save_checkpoint(model, model_config, output_dir, global_step, best_val_loss)


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
    meta = {"step": step, "val_loss": val_loss, "val_ppl": math.exp(val_loss)}
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
