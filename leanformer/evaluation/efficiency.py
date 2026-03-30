"""Measure active parameters, FLOPs, and compression for demo/benchmarking."""

import torch
from ..model.leanformer import LeanFormer
from ..model.low_rank import LowRankLinear


def measure_efficiency(model: LeanFormer, sample_input: torch.Tensor) -> dict:
    """
    Run a forward pass and collect comprehensive efficiency metrics.
    Designed for demo output — shows exactly what LeanFormer is doing.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(sample_input, training=False)

    stats = outputs.get("layer_stats", [])

    # Per-layer metrics
    layer_metrics = []
    for s in stats:
        layer_metrics.append({
            "layer": s.get("layer_idx", "?"),
            "attention_sparsity": s.get("attention_sparsity", 0),
            "ff_sparsity": s.get("ff_sparsity", 0),
            "ff_active_neurons": s.get("ff_active_neurons", 0),
            "ff_total_neurons": s.get("ff_total_neurons", 0),
        })

    # Model-level metrics
    model_stats = model.get_efficiency_stats()

    return {
        "model": model_stats,
        "inference": {
            "exit_layer": outputs["exit_layer"],
            "depth_utilization": outputs["depth_utilization"],
            "layers_skipped": model.config.n_layers - outputs["exit_layer"],
        },
        "per_layer": layer_metrics,
    }
