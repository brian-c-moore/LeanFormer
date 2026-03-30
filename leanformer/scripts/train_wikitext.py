"""
Train LeanFormer on WikiText-2 with rich progress output.

Uses the locally cached WikiText-2 dataset (downloaded once to data/wikitext-2/).
Concatenated sequences — no padding waste, every token is useful.

Usage:
    python -m leanformer.scripts.train_wikitext configs/tiny.yaml
    python -m leanformer.scripts.train_wikitext configs/small.yaml --epochs 5
"""

import sys
import os
import time
import math
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from ..model.config import LeanFormerConfig
from ..model.leanformer import LeanFormer
from ..data.wikitext import WikiTextLoader

console = Console()


def train(config_path: str, epochs_override: int | None = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Build model
    model_config = LeanFormerConfig(**cfg["model"])
    model = LeanFormer(model_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    dense_equiv = model_config.dense_equivalent_estimate()

    console.print(Panel.fit(
        f"[bold cyan]LeanFormer Training[/bold cyan]\n"
        f"Config: {config_path}\n"
        f"Device: {device}\n"
        f"Parameters: {param_count:,} (dense equiv: {dense_equiv:,}, {dense_equiv/param_count:.1f}x compression)",
        border_style="cyan",
    ))

    # Load WikiText-2
    max_seq_len = model_config.max_seq_len
    loader = WikiTextLoader(max_seq_len=max_seq_len)

    console.print("Loading WikiText-2 dataset...")
    train_data = loader.get_concatenated_split("train")
    val_data = loader.get_concatenated_split("validation")
    console.print(f"  Train: {len(train_data)} sequences ({len(train_data) * max_seq_len:,} tokens)")
    console.print(f"  Val:   {len(val_data)} sequences ({len(val_data) * max_seq_len:,} tokens)")

    # Training config
    train_cfg = cfg.get("training", {})
    batch_size = train_cfg.get("batch_size", 8)
    grad_accum = train_cfg.get("gradient_accumulation", 4)
    lr = train_cfg.get("lr", 3e-4)
    warmup_steps = train_cfg.get("warmup_steps", 100)
    epochs = epochs_override or train_cfg.get("epochs", 3)
    output_dir = Path(train_cfg.get("output_dir", "./checkpoints/wikitext2"))

    effective_batch = batch_size * grad_accum
    steps_per_epoch = len(train_data) // batch_size
    total_steps = steps_per_epoch * epochs

    console.print(f"\n  Batch size: {batch_size} x {grad_accum} accum = {effective_batch} effective")
    console.print(f"  Learning rate: {lr}")
    console.print(f"  Epochs: {epochs} ({total_steps:,} steps, {steps_per_epoch:,}/epoch)")
    console.print()

    # DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Cosine schedule with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    global_step = 0
    best_val_loss = float("inf")
    train_losses = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start = time.perf_counter()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids, labels=labels, training=True)
            loss = out["loss"] / grad_accum
            loss.backward()

            epoch_loss += out["loss"].item()
            epoch_steps += 1
            global_step += 1

            # Gradient accumulation step
            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Log every 50 steps
            if global_step % 50 == 0:
                avg_loss = epoch_loss / epoch_steps
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.perf_counter() - epoch_start
                tokens_per_sec = epoch_steps * batch_size * max_seq_len / elapsed

                # Collect efficiency stats from last forward pass
                stats = out.get("layer_stats", [])
                avg_sparsity = sum(s.get("ff_sparsity", 0) for s in stats) / max(len(stats), 1)

                console.print(
                    f"  [dim]step {global_step:>5}/{total_steps} | "
                    f"loss {avg_loss:.4f} | "
                    f"lr {current_lr:.2e} | "
                    f"{tokens_per_sec:.0f} tok/s | "
                    f"ff_sparsity {avg_sparsity:.2f}[/dim]"
                )

            train_losses.append(out["loss"].item())

        # End of epoch
        epoch_elapsed = time.perf_counter() - epoch_start
        avg_train_loss = epoch_loss / max(epoch_steps, 1)

        # Validation
        val_loss = evaluate(model, val_loader, device)
        val_ppl = math.exp(val_loss)

        console.print(
            f"\n[bold]Epoch {epoch+1}/{epochs}[/bold] | "
            f"train_loss: {avg_train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_ppl: {val_ppl:.1f} | "
            f"time: {epoch_elapsed:.0f}s"
        )

        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, model_config, output_dir, epoch, val_loss)
            console.print(f"  [green]New best! Saved checkpoint to {output_dir}[/green]")
        console.print()

    # Final summary
    final_ppl = math.exp(best_val_loss)
    console.print(Panel.fit(
        f"[bold green]Training Complete[/bold green]\n"
        f"Best validation loss: {best_val_loss:.4f}\n"
        f"Best validation perplexity: {final_ppl:.1f}\n"
        f"Checkpoint saved to: {output_dir}",
        border_style="green",
    ))

    return model, best_val_loss


def evaluate(model, val_loader, device):
    """Compute average validation loss."""
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids, labels=labels, training=False)
            total_loss += out["lm_loss"].item()
            total_steps += 1

    model.train()
    return total_loss / max(total_steps, 1)


def save_checkpoint(model, config, output_dir, epoch, val_loss):
    """Save model weights and config."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    config.save(output_dir / "leanformer_config.json")

    # Save training metadata
    import json
    meta = {"epoch": epoch, "val_loss": val_loss, "val_ppl": math.exp(val_loss)}
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m leanformer.scripts.train_wikitext <config.yaml> [--epochs N]")
        sys.exit(1)

    config_path = sys.argv[1]
    epochs = None
    if "--epochs" in sys.argv:
        idx = sys.argv.index("--epochs")
        epochs = int(sys.argv[idx + 1])

    train(config_path, epochs_override=epochs)


if __name__ == "__main__":
    main()
