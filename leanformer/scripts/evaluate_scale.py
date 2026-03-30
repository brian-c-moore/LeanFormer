"""
Evaluate the trained scale model: perplexity, generation, efficiency stats.

Usage:
    python -m leanformer.scripts.evaluate_scale
"""

import sys
import math
import torch
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_from_disk

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
        console.print("[red]No checkpoint found. Run training first.[/red]")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    config = LeanFormerConfig.load(CHECKPOINT_DIR / "leanformer_config.json")
    model = LeanFormer(config)
    model.load_state_dict(torch.load(
        CHECKPOINT_DIR / "pytorch_model.bin", map_location=device, weights_only=True,
    ))
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    param_count = sum(p.numel() for p in model.parameters())
    dense_equiv = config.dense_equivalent_estimate()

    console.print(f"[bold cyan]LeanFormer Scale Evaluation[/bold cyan]")
    console.print(f"  Parameters:  {param_count:,} ({dense_equiv/param_count:.1f}x compression)")
    console.print(f"  Device:      {device}")
    console.print()

    # --- Validation perplexity ---
    console.print("[bold]Validation Perplexity[/bold]")
    if DATA_DIR.exists():
        from torch.utils.data import DataLoader, random_split
        dataset = load_from_disk(str(DATA_DIR)).with_format("torch")
        val_size = int(len(dataset) * 0.05)
        _, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, drop_last=True)

        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 200:
                    break
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                with torch.amp.autocast("cuda", enabled=device == "cuda"):
                    out = model(input_ids, labels=labels, training=False)
                seq_len = input_ids.shape[1] - 1
                total_loss += out["lm_loss"].item() * seq_len * input_ids.shape[0]
                total_tokens += seq_len * input_ids.shape[0]

        ppl = math.exp(total_loss / total_tokens)
        console.print(f"  Perplexity: [bold]{ppl:.1f}[/bold]")
    else:
        console.print("  [yellow]Tokenized dataset not found — skipping perplexity[/yellow]")
    console.print()

    # --- Inference efficiency ---
    console.print("[bold]Inference Efficiency[/bold]")
    sample = torch.randint(0, 50257, (1, config.max_seq_len)).to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda"):
        out = model(sample, training=False)

    table = Table(title="Per-Layer Efficiency", box=box.SIMPLE)
    table.add_column("Layer", justify="center")
    table.add_column("Attn Sparsity", justify="right")
    table.add_column("FF Sparsity", justify="right")
    table.add_column("Active Neurons", justify="right")

    for s in out["layer_stats"]:
        table.add_row(
            str(s["layer_idx"]),
            f"{s['attention_sparsity']:.0%}",
            f"{s['ff_sparsity']:.0%}",
            f"{s['ff_active_neurons']}/{s['ff_total_neurons']}",
        )
    console.print(table)
    console.print(f"  Exit layer: {out['exit_layer']}/{config.n_layers}")
    console.print()

    # --- Generation samples ---
    console.print("[bold]Generation Samples (100 tokens each)[/bold]\n")
    prompts = [
        "The history of",
        "In the early morning",
        "Scientists discovered that",
        "The government announced",
        "Once upon a time",
    ]

    for prompt_text in prompts:
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda"):
            gen_ids, stats = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_k=50)
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        avg_depth = sum(s["depth_utilization"] for s in stats) / len(stats) if stats else 1.0
        console.print(f"  [cyan]Prompt:[/cyan] {prompt_text}")
        console.print(f"  [green]Output:[/green] {text}")
        console.print(f"  [dim]depth_util={avg_depth:.0%}, {len(stats)} tokens[/dim]\n")

    console.print("[bold green]Evaluation complete.[/bold green]")


if __name__ == "__main__":
    main()
