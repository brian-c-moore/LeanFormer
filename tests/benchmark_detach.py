"""
Benchmark: does detaching frozen parameters actually reduce backward time?

Compares wall-clock per step with all params trainable vs. groups frozen
via requires_grad=False + LowRankLinear detach optimization.

Usage:
    python tests/benchmark_detach.py
"""

import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from leanformer.model.config import LeanFormerConfig
from leanformer.model.leanformer import LeanFormer
from leanformer.training.param_groups import build_param_groups, load_registry


def time_steps(model, config, device, n_steps=50, label=""):
    """Time n_steps of forward+backward, return avg ms/step."""
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    # Warmup
    for _ in range(3):
        x = torch.randint(0, config.vocab_size, (2, config.max_seq_len), device=device)
        out = model(x, labels=x, training=True)
        out["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_steps):
        x = torch.randint(0, config.vocab_size, (2, config.max_seq_len), device=device)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        out = model(x, labels=x, training=True)
        out["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()

        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    print(f"  {label:40s} {avg:8.1f} ms/step  (std {std:.1f})")
    return avg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_steps = 80 if device == "cuda" else 30

    print(f"Device: {device}")
    print(f"Steps per benchmark: {n_steps}\n")

    # Scale model to device capabilities
    if device == "cuda":
        config = LeanFormerConfig(
            vocab_size=32000, d_model=512, n_heads=8, n_layers=12, d_ff=2048,
            max_seq_len=256, attention_rank=32, ff_rank=32,
            screening_rank=8, attention_top_k=64, ff_gate_rank=8, dropout=0.0,
        )
    else:
        config = LeanFormerConfig(
            vocab_size=1000, d_model=128, n_heads=4, n_layers=4, d_ff=512,
            max_seq_len=64, attention_rank=16, ff_rank=16,
            screening_rank=4, attention_top_k=16, ff_gate_rank=4, dropout=0.0,
        )

    registry = load_registry()

    # === Benchmark 1: All params trainable (baseline) ===
    model = LeanFormer(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} params\n")

    baseline_ms = time_steps(model, config, device, n_steps, "All params trainable")

    # === Benchmark 2: L1+ frozen (hierarchy L0-only) ===
    model2 = LeanFormer(config).to(device)
    groups = build_param_groups(model2, registry)
    for gid, group in groups.items():
        if group.hierarchy_level > 0:
            group.set_requires_grad(False)

    trainable_l0 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    frozen_frac = 1 - trainable_l0 / total_params
    l0_ms = time_steps(model2, config, device, n_steps,
                       f"L0 only ({frozen_frac:.0%} frozen)")

    # === Benchmark 3: L0+L1 only (L2+L3 frozen) ===
    model3 = LeanFormer(config).to(device)
    groups3 = build_param_groups(model3, registry)
    for gid, group in groups3.items():
        if group.hierarchy_level > 1:
            group.set_requires_grad(False)

    trainable_l01 = sum(p.numel() for p in model3.parameters() if p.requires_grad)
    frozen_frac2 = 1 - trainable_l01 / total_params
    l01_ms = time_steps(model3, config, device, n_steps,
                        f"L0+L1 ({frozen_frac2:.0%} frozen)")

    # === Benchmark 4: Only exit_classifier trainable (extreme) ===
    model4 = LeanFormer(config).to(device)
    groups4 = build_param_groups(model4, registry)
    for gid, group in groups4.items():
        if gid != "exit_classifier":
            group.set_requires_grad(False)

    trainable_exit = sum(p.numel() for p in model4.parameters() if p.requires_grad)
    frozen_frac3 = 1 - trainable_exit / total_params
    exit_ms = time_steps(model4, config, device, n_steps,
                         f"exit_classifier only ({frozen_frac3:.0%} frozen)")

    # === Results ===
    print(f"\n{'='*60}")
    print(f"  DETACH OPTIMIZATION BENCHMARK")
    print(f"{'='*60}")
    print(f"\n  {'Config':<40s} {'ms/step':>10} {'Speedup':>10}")
    print(f"  {'-'*60}")
    print(f"  {'All params trainable':<40s} {baseline_ms:>10.1f} {'1.00x':>10}")
    print(f"  {'L0 only':<40s} {l0_ms:>10.1f} {baseline_ms/l0_ms:>9.2f}x")
    print(f"  {'L0+L1':<40s} {l01_ms:>10.1f} {baseline_ms/l01_ms:>9.2f}x")
    print(f"  {'exit_classifier only':<40s} {exit_ms:>10.1f} {baseline_ms/exit_ms:>9.2f}x")

    # Verdict
    print()
    if l0_ms < baseline_ms * 0.9:
        print(f"  CONFIRMED: Freezing L1+ saves {(1-l0_ms/baseline_ms)*100:.0f}% wall-clock per step")
    elif l0_ms < baseline_ms * 0.98:
        print(f"  MARGINAL: {(1-l0_ms/baseline_ms)*100:.1f}% savings — detach helps but forward dominates")
    else:
        print(f"  NO SAVINGS: {(1-l0_ms/baseline_ms)*100:.1f}% — detach optimization is not working")

    if exit_ms < baseline_ms * 0.5:
        print(f"  EXTREME: Freezing all but exit_classifier saves {(1-exit_ms/baseline_ms)*100:.0f}%")
    print()


if __name__ == "__main__":
    main()
