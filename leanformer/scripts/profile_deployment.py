"""
Profile a trained LeanFormer checkpoint for deployment.

Produces quality-tiered configurations (Full, Standard, Efficient, Minimal)
with latency measurements and a deployment manifest.

Usage:
    python -m leanformer.scripts.profile_deployment --checkpoint checkpoints/reasoning_core
"""

import argparse
import json
import sys
import torch
from pathlib import Path

from ..model.config import LeanFormerConfig
from ..model.leanformer import LeanFormer
from ..training.param_groups import build_param_groups, load_registry
from ..training.deployment import DeploymentProfiler


def main():
    parser = argparse.ArgumentParser(description="Profile LeanFormer deployment tiers")
    parser.add_argument("--checkpoint", default="checkpoints/reasoning_core", help="Checkpoint directory")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for profiling")
    parser.add_argument("--num-runs", type=int, default=20, help="Timing runs per tier")
    parser.add_argument("--output", default=None, help="Output manifest path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.checkpoint)

    if not (ckpt / "pytorch_model.bin").exists():
        print(f"ERROR: Checkpoint not found at {ckpt}")
        sys.exit(1)

    print(f"Loading model from {ckpt}...")
    config = LeanFormerConfig.load(ckpt / "leanformer_config.json")
    model = LeanFormer(config)
    model.load_state_dict(torch.load(ckpt / "pytorch_model.bin", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    registry = load_registry()
    groups = build_param_groups(model, registry)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model: {total_params:,} params on {device}")
    print(f"Profiling with seq_len={args.seq_len}, {args.num_runs} runs per tier\n")

    profiler = DeploymentProfiler(model, groups)
    input_ids = torch.randint(0, config.vocab_size, (1, args.seq_len), device=device)

    tiers = profiler.profile_all_tiers(input_ids, num_runs=args.num_runs)

    # Print results
    print(f"{'Tier':<15} {'Params':>12} {'Latency':>12} {'vs Full':>10} {'Groups':>8}")
    print("-" * 60)
    full_latency = tiers[0].latency_ms if tiers else 1.0
    for tier in tiers:
        speedup = f"{full_latency / tier.latency_ms:.2f}x" if tier.latency_ms else "N/A"
        print(f"{tier.name:<15} {tier.param_count:>12,} {tier.latency_ms:>10.1f}ms {speedup:>10} {len(tier.active_groups):>8}")

    # Get audit hash if available
    audit_hash = None
    audit_path = ckpt / "audit.jsonl"
    if audit_path.exists():
        from ..training.audit import AuditSink
        sink = AuditSink(audit_path)
        audit_hash = sink.current_hash

    manifest = profiler.generate_manifest(
        tiers,
        checkpoint_path=str(ckpt),
        audit_hash=audit_hash,
    )

    output_path = args.output or str(ckpt / "deployment_manifest.json")
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {output_path}")


if __name__ == "__main__":
    main()
