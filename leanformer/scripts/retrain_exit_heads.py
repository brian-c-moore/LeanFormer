"""
Retrain exit heads only with frozen base weights and increased exit_threshold.

Freezes all model weights except the depth controller's exit_head,
then trains for 500 steps with only the exit loss.
"""

import sys
import torch
import math
from pathlib import Path
from datasets import load_from_disk

from rich.console import Console
from ..model.config import LeanFormerConfig
from ..model.leanformer import LeanFormer

console = Console()

CHECKPOINT_DIR = Path("checkpoints/scale")
DATA_DIR = Path("data/openwebtext-500k-tokenized")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = LeanFormerConfig.load(CHECKPOINT_DIR / "leanformer_config.json")
    model = LeanFormer(config)
    model.load_state_dict(torch.load(
        CHECKPOINT_DIR / "pytorch_model.bin", map_location=device, weights_only=True,
    ))
    model.to(device)

    # Increase exit threshold
    model.depth_controller.threshold = 0.05
    console.print(f"Exit threshold set to: {model.depth_controller.threshold}")

    # Freeze everything except exit head
    for name, param in model.named_parameters():
        if "exit_head" in name:
            param.requires_grad_(True)
            console.print(f"  Trainable: {name}")
        else:
            param.requires_grad_(False)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"  Trainable params: {trainable_count:,} (exit head only)")

    # Load data
    dataset = load_from_disk(str(DATA_DIR)).with_format("torch")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3,
    )

    model.train()
    console.print("\nRetraining exit heads (500 steps)...")

    for step in range(500):
        idx = step % len(dataset)
        batch = dataset[idx]
        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        labels = batch["labels"].unsqueeze(0).to(device)

        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            out = model(input_ids, labels=labels, training=True)
            # We only care about the exit loss component
            loss = out["aux_loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            console.print(f"  step {step+1}/500 | exit_loss: {loss.item():.4f}")

    # Save updated model (only exit heads changed)
    model.depth_controller.threshold = 0.05  # Persist the new threshold
    # Update config
    config.exit_threshold = 0.05
    torch.save(model.state_dict(), CHECKPOINT_DIR / "pytorch_model.bin")
    config.save(CHECKPOINT_DIR / "leanformer_config.json")
    console.print(f"\nSaved updated model to {CHECKPOINT_DIR}")

    # Quick test
    model.eval()
    exit_layers = []
    with torch.no_grad():
        for i in range(100):
            input_ids = dataset[i]["input_ids"].unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                out = model(input_ids, training=False)
            exit_layers.append(out["exit_layer"])

    mean_depth = sum(exit_layers) / len(exit_layers)
    unique = len(set(exit_layers))
    console.print(f"\nPost-retrain depth check (100 samples):")
    console.print(f"  Mean depth: {mean_depth:.1f} / {config.n_layers}")
    console.print(f"  Unique depths: {unique}")
    console.print(f"  Distribution: {dict(sorted(((k, exit_layers.count(k)) for k in set(exit_layers))))}")

    if mean_depth < config.n_layers:
        console.print(f"  [green]Early exit is now working![/green]")
    else:
        console.print(f"  [yellow]Still no early exit. May need more aggressive threshold or longer retraining.[/yellow]")


if __name__ == "__main__":
    main()
