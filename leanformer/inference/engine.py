"""
LeanFormerInference — inference engine with compute budget governance.

Wraps the model with tokenization, generation, and efficiency tracking.
Designed to produce demo-quality output showing what the architecture is doing.
"""

import torch
import time
from pathlib import Path
from transformers import AutoTokenizer

from ..model.leanformer import LeanFormer
from ..model.config import LeanFormerConfig
from leanformer import DEFAULT_TOKENIZER


class LeanFormerInference:

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        model_path = Path(model_path)

        # Load config
        config_path = model_path / "leanformer_config.json"
        if config_path.exists():
            self.config = LeanFormerConfig.load(config_path)
        else:
            raise FileNotFoundError(f"No config found at {config_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = LeanFormer(self.config)
        weights_path = model_path / "pytorch_model.bin"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            self.model.load_state_dict(state_dict)

        self.model.to(device)
        self.model.eval()

        # Warmup
        self._warmup()

    def _warmup(self):
        dummy = torch.zeros(1, 16, dtype=torch.long, device=self.device)
        with torch.no_grad():
            self.model(dummy)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> dict:
        """Generate text and return result with efficiency metrics."""
        start_time = time.perf_counter()

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output_ids, step_stats = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        elapsed = time.perf_counter() - start_time
        new_tokens = output_ids.shape[1] - input_ids.shape[1]

        response_text = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        # Aggregate efficiency stats across generation steps
        avg_depth_util = (
            sum(s["depth_utilization"] for s in step_stats) / len(step_stats)
            if step_stats else 1.0
        )
        avg_exit_layer = (
            sum(s["exit_layer"] for s in step_stats) / len(step_stats)
            if step_stats else self.config.n_layers
        )

        return {
            "text": response_text,
            "tokens_generated": new_tokens,
            "time_seconds": elapsed,
            "tokens_per_second": new_tokens / max(elapsed, 1e-8),
            "avg_depth_utilization": avg_depth_util,
            "avg_exit_layer": avg_exit_layer,
            "total_layers": self.config.n_layers,
        }
