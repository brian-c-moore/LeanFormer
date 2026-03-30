"""
Validate adaptive depth: measure exit layer distribution.

Runs 1000 samples and analyzes depth behavior:
- Distribution histogram
- Mean/median/min/max depth
- Simple vs complex prompt comparison

Usage:
    python -m leanformer.scripts.validate_depth
"""

import sys
import torch
import json
from pathlib import Path
from collections import Counter
from datasets import load_from_disk
from transformers import AutoTokenizer

from rich.console import Console
from rich.table import Table
from rich import box

from ..model.config import LeanFormerConfig
from ..model.leanformer import LeanFormer

console = Console()

CHECKPOINT_DIR = Path("checkpoints/scale")
DATA_DIR = Path("data/openwebtext-500k-tokenized")


def main():
    if not (CHECKPOINT_DIR / "pytorch_model.bin").exists():
        console.print("[red]No checkpoint found.[/red]")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = LeanFormerConfig.load(CHECKPOINT_DIR / "leanformer_config.json")
    model = LeanFormer(config)
    model.load_state_dict(torch.load(
        CHECKPOINT_DIR / "pytorch_model.bin", map_location=device, weights_only=True,
    ))
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # --- Depth distribution on 1000 samples ---
    console.print("[bold]Adaptive Depth Distribution (1000 samples)[/bold]\n")

    if DATA_DIR.exists():
        dataset = load_from_disk(str(DATA_DIR)).with_format("torch")
    else:
        from ..data.wikitext import WikiTextLoader
        loader = WikiTextLoader(max_seq_len=config.max_seq_len)
        dataset = loader.get_concatenated_split("validation")

    exit_layers = []
    n_samples = min(1000, len(dataset))

    with torch.no_grad():
        for i in range(n_samples):
            input_ids = dataset[i]["input_ids"].unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                out = model(input_ids, training=False)
            exit_layers.append(out["exit_layer"])

    # Statistics
    mean_depth = sum(exit_layers) / len(exit_layers)
    sorted_depths = sorted(exit_layers)
    median_depth = sorted_depths[len(sorted_depths) // 2]
    min_depth = min(exit_layers)
    max_depth = max(exit_layers)
    unique_depths = len(set(exit_layers))

    console.print(f"  Mean depth:   {mean_depth:.1f} / {config.n_layers}")
    console.print(f"  Median depth: {median_depth} / {config.n_layers}")
    console.print(f"  Min depth:    {min_depth}")
    console.print(f"  Max depth:    {max_depth}")
    console.print(f"  Unique depths: {unique_depths}")
    console.print()

    # Histogram
    counts = Counter(exit_layers)
    table = Table(title="Exit Layer Histogram", box=box.SIMPLE)
    table.add_column("Layer", justify="center")
    table.add_column("Count", justify="right")
    table.add_column("Fraction", justify="right")
    table.add_column("Bar")

    for layer in range(1, config.n_layers + 1):
        c = counts.get(layer, 0)
        frac = c / n_samples
        bar_len = int(50 * frac)
        table.add_row(str(layer), str(c), f"{frac:.1%}", "#" * bar_len)

    console.print(table)

    # --- Simple vs Complex comparison ---
    console.print("\n[bold]Depth vs Complexity[/bold]\n")

    simple_prompts = [
        "The cat sat on the",
        "It was a nice day",
        "Hello how are you",
        "The dog is big",
        "I like to eat",
    ] * 20  # 100 simple

    complex_prompts = []
    # Use long validation sequences as "complex"
    for i in range(100):
        if i < len(dataset):
            tokens = dataset[i]["input_ids"]
            text = tokenizer.decode(tokens[:64])
            complex_prompts.append(text)

    simple_depths = []
    for prompt in simple_prompts[:100]:
        ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=config.max_seq_len).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda"):
            out = model(ids, training=False)
        simple_depths.append(out["exit_layer"])

    complex_depths = []
    for prompt in complex_prompts[:100]:
        ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=config.max_seq_len).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda"):
            out = model(ids, training=False)
        complex_depths.append(out["exit_layer"])

    simple_mean = sum(simple_depths) / max(len(simple_depths), 1)
    complex_mean = sum(complex_depths) / max(len(complex_depths), 1)

    console.print(f"  Simple prompts mean depth:  {simple_mean:.1f} / {config.n_layers}")
    console.print(f"  Complex prompts mean depth: {complex_mean:.1f} / {config.n_layers}")

    if simple_mean < complex_mean:
        console.print(f"  [green]Simple inputs exit {complex_mean - simple_mean:.1f} layers earlier on average.[/green]")
    elif simple_mean == complex_mean:
        console.print(f"  [yellow]No depth difference — exit classifier may need more training.[/yellow]")
    else:
        console.print(f"  [yellow]Unexpected: simple prompts use more depth. Exit classifier may need tuning.[/yellow]")

    # Save results
    results = {
        "n_samples": n_samples,
        "mean_depth": mean_depth,
        "median_depth": median_depth,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "unique_depths": unique_depths,
        "histogram": dict(counts),
        "simple_mean_depth": simple_mean,
        "complex_mean_depth": complex_mean,
    }
    with open(CHECKPOINT_DIR / "depth_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n  Results saved to {CHECKPOINT_DIR / 'depth_analysis.json'}")

    # Check pass/fail criteria
    console.print("\n[bold]Exit Criteria Check[/bold]")
    passed = True
    if mean_depth >= config.n_layers:
        console.print(f"  [red]FAIL: Mean depth = {mean_depth:.1f} (no early exit happening)[/red]")
        console.print(f"  Recommendation: increase exit_threshold to 0.05, retrain exit heads")
        passed = False
    else:
        console.print(f"  [green]PASS: Mean depth = {mean_depth:.1f} < {config.n_layers}[/green]")

    if unique_depths < 3:
        console.print(f"  [red]FAIL: Only {unique_depths} unique depths (need >= 3)[/red]")
        passed = False
    else:
        console.print(f"  [green]PASS: {unique_depths} unique depths observed[/green]")

    if not passed:
        console.print("\n[yellow]Adaptive depth did not meet criteria. See recommendations above.[/yellow]")


if __name__ == "__main__":
    main()
