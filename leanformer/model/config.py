"""LeanFormer configuration — all hyperparameters in one place."""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path


@dataclass
class LeanFormerConfig:
    # Core architecture
    vocab_size: int = 50257          # GPT-2 tokenizer default
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 2048
    max_seq_len: int = 1024

    # Low-rank parameters — the core innovation
    attention_rank: int = 32         # rank for Q, K, V, O projections
    ff_rank: int = 32                # rank for feed-forward up/down/gate projections

    # Sparse attention parameters
    screening_rank: int = 8          # rank for the cheap screening projection
    attention_top_k: int = 64        # candidates per query in screening pass

    # Sparse activation parameters
    ff_gate_rank: int = 8            # rank for the cheap gate predictor
    ff_sparsity_target: float = 0.8  # target fraction of neurons to skip at inference

    # Adaptive depth parameters
    min_depth: int = 2               # always run at least this many layers
    exit_threshold: float = 0.01     # convergence threshold for early exit

    # Training
    dropout: float = 0.1

    # Delta belief system
    delta_rank: int = 4              # rank for per-belief low-rank deltas
    max_active_deltas: int = 4       # max deltas applied per forward pass
    delta_encoding_steps: int = 30   # gradient steps when encoding a new belief
    delta_encoding_lr: float = 1e-2  # learning rate for belief encoding

    def total_parameters_estimate(self) -> int:
        """Rough estimate of total parameter count."""
        attn_params = 4 * (self.d_model + self.d_model) * self.attention_rank
        screen_params = 2 * (self.d_model + self.d_model // 4) * self.screening_rank
        ff_params = 3 * (self.d_model + self.d_ff) * self.ff_rank
        gate_params = (self.d_model + self.d_ff) * self.ff_gate_rank
        depth_params = self.d_model * self.d_model // 4 + self.d_model // 4 + self.d_model
        per_layer = attn_params + screen_params + ff_params + gate_params + depth_params
        embedding = self.vocab_size * self.d_model
        position = self.max_seq_len * self.d_model
        return per_layer * self.n_layers + embedding + position

    def dense_equivalent_estimate(self) -> int:
        """What a standard dense transformer with same dimensions would cost."""
        attn_params = 4 * self.d_model * self.d_model  # Q, K, V, O
        ff_params = 3 * self.d_model * self.d_ff        # up, down, gate (SwiGLU)
        per_layer = attn_params + ff_params
        embedding = self.vocab_size * self.d_model
        position = self.max_seq_len * self.d_model
        return per_layer * self.n_layers + embedding + position

    def compression_ratio(self) -> float:
        """How much smaller we are than the dense equivalent."""
        return self.dense_equivalent_estimate() / max(self.total_parameters_estimate(), 1)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "LeanFormerConfig":
        with open(path) as f:
            return cls(**json.load(f))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LeanFormerConfig":
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg["model"])
