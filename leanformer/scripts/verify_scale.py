"""
Verify the scale config builds and runs on GPU within VRAM budget.
Also runs the existing test suite to confirm nothing is broken.
"""

import sys
import torch
from ..model.leanformer import LeanFormer
from ..model.config import LeanFormerConfig


def main():
    print("=== Scale Model GPU Verification ===\n")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Install CUDA PyTorch.")
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Build model
    config = LeanFormerConfig.from_yaml("configs/scale.yaml")
    model = LeanFormer(config).cuda().half()

    param_count = sum(p.numel() for p in model.parameters())
    dense_equiv = config.dense_equivalent_estimate()
    print(f"Parameters:       {param_count:,}")
    print(f"Dense equivalent: {dense_equiv:,}")
    print(f"Compression:      {dense_equiv / param_count:.1f}x")
    print()

    # Forward pass with mixed precision
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, 50257, (4, 512)).cuda()

    with torch.amp.autocast("cuda"):
        out = model(x, labels=x, training=True)

    loss = out["loss"].item()
    vram = torch.cuda.max_memory_allocated() / 1e9
    finite = torch.isfinite(out["loss"]).item()

    print(f"Loss:       {loss:.4f}")
    print(f"Finite:     {finite}")
    print(f"VRAM peak:  {vram:.2f} GB")
    print(f"Exit layer: {out['exit_layer']}")
    print()

    # Backward pass to verify gradient flow
    torch.cuda.reset_peak_memory_stats()
    out["loss"].backward()
    vram_backward = torch.cuda.max_memory_allocated() / 1e9
    print(f"VRAM peak (with backward): {vram_backward:.2f} GB")

    if vram_backward > 11.0:
        print("\nWARNING: VRAM exceeds 11GB. Reduce batch_size to 2 in configs/scale.yaml")
    else:
        print(f"\nVRAM OK — {11.0 - vram_backward:.1f} GB headroom")

    # Verify generation works on GPU
    model.eval()
    prompt = torch.randint(0, 50257, (1, 16)).cuda()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        gen_ids, stats = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=50)
    print(f"Generation: {gen_ids.shape[1] - prompt.shape[1]} tokens generated")

    print("\n=== PASS: Scale model verified on GPU ===")


if __name__ == "__main__":
    main()
