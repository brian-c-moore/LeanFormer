"""
CLI for forging a domain knowledge set into validated deltas.

Usage:
    python -m leanformer.scripts.forge_domain \
        --model-checkpoint checkpoints/reasoning_core \
        --fact-bank data/domains/chemistry.json \
        --domain chemistry \
        --output-dir deltas/chemistry/ \
        --registry-path deltas/registry.json \
        --delta-rank 16 \
        --batch-size 10
"""

import argparse
import json
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer

from .. import DEFAULT_TOKENIZER
from ..model.config import LeanFormerConfig
from ..model.leanformer import LeanFormer
from ..knowledge_plane.dfs import compute_model_hash
from ..knowledge_plane.registry import DeltaRegistry
from ..knowledge_plane.forge import KnowledgeForge, load_fact_bank


def main():
    parser = argparse.ArgumentParser(description="Forge domain knowledge into deltas")
    parser.add_argument("--model-checkpoint", required=True, help="Path to model checkpoint dir")
    parser.add_argument("--fact-bank", required=True, help="Path to fact bank JSON")
    parser.add_argument("--domain", required=True, help="Domain name")
    parser.add_argument("--output-dir", required=True, help="Output directory for .delta files")
    parser.add_argument("--registry-path", required=True, help="Path to registry JSON")
    parser.add_argument("--delta-rank", type=int, default=16, help="Delta rank")
    parser.add_argument("--batch-size", type=int, default=10, help="Facts per delta batch")
    parser.add_argument("--max-steps", type=int, default=200, help="Encoding optimization steps")
    parser.add_argument("--lr", type=float, default=1e-2, help="Encoding learning rate")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    # Load model
    ckpt_dir = Path(args.model_checkpoint)
    config = LeanFormerConfig.load(ckpt_dir / "leanformer_config.json")
    model = LeanFormer(config).cuda()
    state_dict = torch.load(ckpt_dir / "pytorch_model.bin", map_location="cuda", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {ckpt_dir}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
    tokenizer.pad_token = tokenizer.eos_token

    # Model hash
    model_hash_path = ckpt_dir / "model_hash.txt"
    if model_hash_path.exists():
        model_hash = model_hash_path.read_text().strip()
    else:
        print("Computing model hash...")
        model_hash = compute_model_hash(model)
        print(f"Model hash: {model_hash[:16]}...")

    # Registry
    registry_path = Path(args.registry_path)
    if registry_path.exists():
        registry = DeltaRegistry.load(str(registry_path))
        print(f"Loaded registry with {registry.count} existing deltas")
    else:
        registry = DeltaRegistry(
            base_model_hash=model_hash,
            d_model=config.d_model,
            n_layers=config.n_layers,
            delta_rank=args.delta_rank,
        )
        print("Created new registry")

    # Load facts
    facts_by_category = load_fact_bank(args.fact_bank, tokenizer)
    total_facts = sum(len(v) for v in facts_by_category.values())
    print(f"Loaded {total_facts} facts across {len(facts_by_category)} categories")

    # Create forge
    forge = KnowledgeForge(
        model=model,
        tokenizer=tokenizer,
        registry=registry,
        delta_rank=args.delta_rank,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        device="cuda",
    )

    # Forge each category
    all_deltas = []
    for category, facts in facts_by_category.items():
        print(f"\n{'='*60}")
        print(f"Forging domain: {category} ({len(facts)} facts)")
        print(f"{'='*60}")
        deltas = forge.forge_domain(facts, category, batch_size=args.batch_size)
        all_deltas.extend(deltas)

    # Save deltas
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for delta in all_deltas:
        delta_path = output_dir / f"{delta.delta_id}.delta"
        delta.save(str(delta_path))

    # Save registry
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry.save(str(registry_path))

    # Summary
    print(f"\n{'='*60}")
    print(f"Forge Summary")
    print(f"{'='*60}")
    print(f"Domain: {args.domain}")
    print(f"Total facts: {total_facts}")
    print(f"Deltas forged: {len(all_deltas)}")
    print(f"Registry total: {registry.count}")
    print(f"Remaining capacity: {registry.total_remaining_capacity()}")
    print(f"Output: {output_dir}")
    print(f"Registry: {registry_path}")


if __name__ == "__main__":
    main()
