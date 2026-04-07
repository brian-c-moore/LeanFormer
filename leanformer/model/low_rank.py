"""
LowRankLinear — the base weight primitive for LeanFormer.

Every linear layer in the model stores weights as two low-rank factors (A x B)
instead of a dense matrix. This is the foundational innovation — not a post-hoc
compression but the native representation from initialization.

For in=out=4096, rank=64:
    Standard Linear: 16,777,216 parameters
    LowRankLinear:     524,288 parameters (32x reduction)
"""

import torch
import torch.nn as nn
import math


class LowRankLinear(nn.Module):
    """A linear layer stored as two low-rank factors: y = x @ A @ B + bias."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, min(in_features, out_features))

        # A projects input -> rank space, B projects rank space -> output
        self.A = nn.Parameter(torch.empty(in_features, self.rank))
        self.B = nn.Parameter(torch.empty(self.rank, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self._init_weights()

    def _init_weights(self):
        # A gets Kaiming init for signal flow; B starts at zero so the layer
        # begins as identity-like and learns deviations (LoRA insight)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Detach frozen factors so autograd skips this branch during backward
        A = self.A if self.A.requires_grad else self.A.detach()
        B = self.B if self.B.requires_grad else self.B.detach()
        out = x @ A @ B
        if self.bias is not None:
            bias = self.bias if self.bias.requires_grad else self.bias.detach()
            out = out + bias
        return out

    def effective_parameters(self) -> int:
        """Number of parameters actually stored."""
        count = self.in_features * self.rank + self.rank * self.out_features
        if self.bias is not None:
            count += self.out_features
        return count

    def dense_parameters(self) -> int:
        """What a full-rank nn.Linear would cost."""
        count = self.in_features * self.out_features
        if self.bias is not None:
            count += self.out_features
        return count

    def compression_ratio(self) -> float:
        """How much smaller this is vs full-rank."""
        return self.dense_parameters() / max(self.effective_parameters(), 1)

    @property
    def weight(self) -> torch.Tensor:
        """Materialize full weight matrix for compatibility/analysis."""
        return self.A @ self.B

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, compression={self.compression_ratio():.1f}x, "
            f"bias={self.bias is not None}"
        )
