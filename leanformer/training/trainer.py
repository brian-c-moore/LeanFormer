"""
LeanFormerTrainer — extends HuggingFace Trainer with efficiency metrics.

Logs the key metrics that demonstrate LeanFormer's innovations are working:
- Feed-forward sparsity (gate effectiveness)
- Attention sparsity (screening effectiveness)
- Exit layer / depth utilization (adaptive depth)
- Auxiliary losses (gate + exit training)
"""

from transformers import Trainer
import torch


class LeanFormerTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
            training=True,
        )
        loss = outputs["loss"]

        # Log efficiency metrics periodically
        if self.state.global_step % 100 == 0 and self.state.global_step > 0:
            self._log_efficiency(outputs)

        return (loss, outputs) if return_outputs else loss

    def _log_efficiency(self, outputs):
        stats = outputs.get("layer_stats", [])
        if not stats:
            return

        # Aggregate per-layer stats
        avg_attn_sparsity = _mean(s.get("attention_sparsity", 0) for s in stats)
        avg_ff_sparsity = _mean(s.get("ff_sparsity", 0) for s in stats)
        exit_layer = outputs.get("exit_layer", len(stats))
        depth_util = outputs.get("depth_utilization", 1.0)

        metrics = {
            "efficiency/avg_attention_sparsity": avg_attn_sparsity,
            "efficiency/avg_ff_sparsity": avg_ff_sparsity,
            "efficiency/exit_layer": exit_layer,
            "efficiency/depth_utilization": depth_util,
        }

        if "lm_loss" in outputs:
            metrics["loss/lm_loss"] = outputs["lm_loss"].item()
        if "aux_loss" in outputs:
            metrics["loss/aux_loss"] = outputs["aux_loss"].item()

        self.log(metrics)


def _mean(iterable):
    items = list(iterable)
    return sum(items) / max(len(items), 1)
