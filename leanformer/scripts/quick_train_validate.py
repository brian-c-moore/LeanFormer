"""
Quick-train a LeanFormer model on WikiText-2 for pipeline validation.

This produces a checkpoint sufficient for testing the forge, router,
and integration pipeline, not for quality results. Use train_reasoning.py
for the full training run.

Usage:
    python -m leanformer.scripts.quick_train_validate [--steps 500]
"""

import sys
import time
import math
import json
import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from .. import DEFAULT_TOKENIZER
from ..model.config import LeanFormerConfig
from ..model.leanformer import LeanFormer
from ..knowledge_plane.dfs import compute_model_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--output-dir", default="checkpoints/reasoning_core", help="Output dir")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    # Reasoning core config
    config = LeanFormerConfig(
        d_model=768, n_heads=12, n_layers=12, d_ff=3072,
        max_seq_len=512, attention_rank=96, ff_rank=96,
        screening_rank=24, attention_top_k=96,
        ff_gate_rank=24, ff_sparsity_target=0.8,
        min_depth=4, exit_threshold=0.03, dropout=0.1,
    )
    model = LeanFormer(config).cuda()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    # Load WikiText-2
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir="data/wikitext-2")
    ds = ds.filter(lambda x: len(x["text"]) > 50)

    print(f"Tokenizing {len(ds)} samples...")
    all_ids = []
    for ex in ds:
        enc = tokenizer(ex["text"], truncation=True, max_length=512, padding="max_length")
        all_ids.append({"input_ids": enc["input_ids"], "labels": enc["input_ids"]})

    tok_ds = Dataset.from_list(all_ids).with_format("torch")
    print(f"Ready: {len(tok_ds)} samples")

    loader = DataLoader(tok_ds, batch_size=2, shuffle=True, drop_last=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    start = time.time()
    for step, batch in enumerate(loader):
        if step >= args.steps:
            break
        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()
        with torch.amp.autocast("cuda"):
            out = model(input_ids, labels=labels, training=True)
            loss = out["loss"]
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step % 100 == 0:
            print(f"  step {step}: loss={loss.item():.4f}")

    elapsed = time.time() - start
    print(f"Training: {elapsed:.0f}s, final loss: {loss.item():.4f}")

    # Save checkpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    config.save(output_dir / "leanformer_config.json")
    meta = {"step": args.steps, "val_loss": loss.item(), "note": f"quick_wikitext2_{args.steps}steps"}
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Computing model hash...")
    model_hash = compute_model_hash(model)
    with open(output_dir / "model_hash.txt", "w") as f:
        f.write(model_hash)
    print(f"Hash: {model_hash[:16]}...")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
