"""
Forge all three domain fact banks into validated deltas.

Requires a trained checkpoint at checkpoints/reasoning_core/.

Usage:
    python -m leanformer.scripts.forge_all_domains [--facts-per-domain 20] [--batch-size 10] [--max-steps 50]

For pipeline validation (quick, 500-step checkpoint):
    python -m leanformer.scripts.forge_all_domains --facts-per-domain 20 --max-steps 50

For full forge (trained reasoning core):
    python -m leanformer.scripts.forge_all_domains --facts-per-domain 200 --max-steps 200
"""

import sys
import json
import argparse
import shutil
import torch
from pathlib import Path
from transformers import GPT2TokenizerFast

from ..model.config import LeanFormerConfig
from ..model.leanformer import LeanFormer
from ..knowledge_plane.registry import DeltaRegistry
from ..knowledge_plane.forge import KnowledgeForge, load_fact_bank


def main():
    parser = argparse.ArgumentParser(description="Forge all domains")
    parser.add_argument("--checkpoint", default="checkpoints/reasoning_core", help="Model checkpoint dir")
    parser.add_argument("--facts-per-domain", type=int, default=200, help="Max facts per domain")
    parser.add_argument("--batch-size", type=int, default=10, help="Facts per delta batch")
    parser.add_argument("--max-steps", type=int, default=200, help="Encoding optimization steps")
    parser.add_argument("--lr", type=float, default=1e-2, help="Encoding learning rate")
    parser.add_argument("--delta-rank", type=int, default=16, help="Delta rank")
    parser.add_argument("--validation-threshold", type=float, default=0.3, help="Min improvement fraction")
    parser.add_argument("--output-dir", default="deltas", help="Output directory")
    parser.add_argument("--clean", action="store_true", help="Clean existing deltas before forging")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    # Clean if requested
    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        for item in output_dir.iterdir():
            if item.name == ".gitkeep":
                continue
            if item.is_dir():
                shutil.rmtree(str(item))
            else:
                item.unlink()
        print("Cleaned existing deltas.")

    # Load model
    ckpt = Path(args.checkpoint)
    if not (ckpt / "pytorch_model.bin").exists():
        print(f"ERROR: Checkpoint not found at {ckpt}")
        print("Run: python -m leanformer.scripts.quick_train_validate")
        sys.exit(1)

    config = LeanFormerConfig.load(ckpt / "leanformer_config.json")
    model = LeanFormer(config).cuda()
    model.load_state_dict(torch.load(ckpt / "pytorch_model.bin", map_location="cuda", weights_only=True))
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model_hash = (ckpt / "model_hash.txt").read_text().strip()
    print(f"Model hash: {model_hash[:16]}...")

    # Create registry
    registry = DeltaRegistry(
        base_model_hash=model_hash, d_model=config.d_model,
        n_layers=config.n_layers, delta_rank=args.delta_rank,
    )

    # Create forge
    forge = KnowledgeForge(
        model=model, tokenizer=tokenizer, registry=registry,
        delta_rank=args.delta_rank, learning_rate=args.lr,
        max_steps=args.max_steps, validation_threshold=args.validation_threshold,
        device="cuda",
    )

    # Forge each domain
    all_deltas = []
    domains = ["chemistry", "cs", "general"]
    for domain in domains:
        path = Path(f"leanformer/data/domains/{domain}.json")
        if not path.exists():
            print(f"WARNING: Fact bank not found: {path}")
            continue
        facts_by_cat = load_fact_bank(str(path), tokenizer)
        facts = list(facts_by_cat.values())[0][:args.facts_per_domain]
        print(f"\n{'='*60}")
        print(f"Forging: {domain} ({len(facts)} facts)")
        print(f"{'='*60}")
        deltas = forge.forge_domain(facts, domain, batch_size=args.batch_size)
        all_deltas.extend(deltas)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    registry.save(str(output_dir / "registry.json"))
    for d in all_deltas:
        dp = output_dir / d.category / f"{d.delta_id}.delta"
        dp.parent.mkdir(parents=True, exist_ok=True)
        d.save(str(dp))

    # Summary
    print(f"\n{'='*60}")
    print(f"FORGE SUMMARY")
    print(f"{'='*60}")
    print(f"Total deltas: {registry.count}")
    print(f"Capacity remaining: {registry.total_remaining_capacity()}")
    for cat in domains:
        n = len(registry.get_deltas_by_category(cat))
        print(f"  {cat}: {n} deltas")
    print(f"Output: {output_dir}")
    print(f"Registry: {output_dir / 'registry.json'}")

    # Save summary
    summary = {
        "total_deltas": registry.count,
        "capacity_remaining": registry.total_remaining_capacity(),
        "per_domain": {cat: len(registry.get_deltas_by_category(cat)) for cat in domains},
        "config": {
            "facts_per_domain": args.facts_per_domain,
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "delta_rank": args.delta_rank,
            "validation_threshold": args.validation_threshold,
        },
    }
    with open(output_dir / "forge_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
