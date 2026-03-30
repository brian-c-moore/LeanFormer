"""Training callbacks for LeanFormer."""

from transformers import TrainerCallback
from pathlib import Path
import json


class EfficiencyLogCallback(TrainerCallback):
    """Logs a summary of efficiency metrics at each evaluation step."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            efficiency_keys = [k for k in metrics if k.startswith("efficiency/")]
            if efficiency_keys:
                print("\n--- Efficiency Summary ---")
                for k in efficiency_keys:
                    print(f"  {k}: {metrics[k]:.4f}")
                print("--------------------------\n")


class CheckpointConfigCallback(TrainerCallback):
    """Saves the LeanFormerConfig alongside each checkpoint."""

    def __init__(self, config):
        self.config = config

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            self.config.save(checkpoint_dir / "leanformer_config.json")
