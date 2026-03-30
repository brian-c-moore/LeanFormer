"""
BeliefEncoder — encodes facts into sparse weight deltas via gradient-based learning.

Process:
1. Freeze all base weights
2. Attach trainable low-rank delta factors (dA, dB) to each LowRankLinear
3. Construct a next-token-prediction loss from the fact text
4. Run N gradient steps (only delta factors are trainable)
5. Detach the trained deltas into a BeliefDelta
6. Compute a semantic routing embedding by running the fact through the frozen base

This is mechanistically transparent: the model's own learning dynamics determine
which weight regions the fact activates. No extra architecture needed.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from ..model.leanformer import LeanFormer
from ..model.low_rank import LowRankLinear
from .delta_system import BeliefDelta


class BeliefEncoder:
    """Encodes text facts into BeliefDelta objects."""

    def __init__(
        self,
        model: LeanFormer,
        tokenizer: AutoTokenizer | None = None,
        delta_rank: int = 4,
        n_steps: int = 30,
        lr: float = 1e-2,
    ):
        self.model = model
        self.delta_rank = delta_rank
        self.n_steps = n_steps
        self.lr = lr

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

    def encode(self, fact_text: str, belief_name: str) -> BeliefDelta:
        """
        Encode a fact into a BeliefDelta.

        Args:
            fact_text: The fact to encode, e.g. "The capital of France is Paris"
            belief_name: Unique name for this belief

        Returns:
            BeliefDelta with trained low-rank deltas and routing embedding
        """
        device = next(self.model.parameters()).device

        # Tokenize the fact
        tokens = self.tokenizer.encode(fact_text, return_tensors="pt").to(device)
        if tokens.shape[1] > self.model.config.max_seq_len:
            tokens = tokens[:, :self.model.config.max_seq_len]

        # Step 1: Compute the routing embedding from the frozen base model
        routing_embedding = self._compute_routing_embedding(tokens)

        # Step 2: Attach trainable delta factors and train them
        layer_deltas = self._train_deltas(tokens)

        return BeliefDelta(
            name=belief_name,
            embedding=routing_embedding.detach(),
            layer_deltas=layer_deltas,
            metadata={"fact_text": fact_text, "n_steps": self.n_steps},
        )

    def _compute_routing_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get a semantic embedding for routing by running fact through frozen model."""
        self.model.eval()
        with torch.no_grad():
            positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
            hidden = self.model.token_embedding(tokens) + self.model.position_embedding(positions)
            # Mean-pool over sequence
            return hidden.mean(dim=(0, 1))  # (d_model,)

    def _train_deltas(
        self,
        tokens: torch.Tensor,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """
        Attach temporary delta factors, train them on the fact, collect results.
        """
        device = tokens.device

        # Freeze all base parameters
        for param in self.model.parameters():
            param.requires_grad_(False)

        # Attach trainable delta factors to each LowRankLinear in each layer
        delta_params = {}  # key -> (dA_param, dB_param)
        trainable = []

        for layer_idx, layer in enumerate(self.model.layers):
            targets = self._get_layer_targets(layer)
            for target_name, module in targets.items():
                key = f"layer_{layer_idx}_{target_name}"
                # Initialize both factors with small random values so gradients
                # flow through from the start (unlike base weights which use
                # A-random/B-zero — here we need both non-zero for fast learning)
                dA = nn.Parameter(torch.randn(
                    module.in_features, self.delta_rank, device=device
                ) * 0.01)
                dB = nn.Parameter(torch.randn(
                    self.delta_rank, module.out_features, device=device
                ) * 0.01)
                delta_params[key] = (dA, dB)
                trainable.extend([dA, dB])

        # Train delta parameters
        optimizer = torch.optim.Adam(trainable, lr=self.lr)
        labels = tokens.clone()

        self.model.train()
        for step in range(self.n_steps):
            # Forward pass with delta contributions
            logits = self._forward_with_temp_deltas(tokens, delta_params)

            # Next-token prediction loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.model.config.vocab_size),
                shift_labels.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Collect trained deltas (detached)
        result = {}
        for key, (dA, dB) in delta_params.items():
            result[key] = (dA.detach().clone(), dB.detach().clone())

        # Restore base parameters to trainable
        for param in self.model.parameters():
            param.requires_grad_(True)

        self.model.eval()
        return result

    def _get_layer_targets(self, layer) -> dict[str, LowRankLinear]:
        """Get all LowRankLinear modules in a layer that should receive deltas."""
        targets = {}
        # Attention projections
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            module = getattr(layer.attn, name, None)
            if isinstance(module, LowRankLinear):
                targets[name] = module
        # Feed-forward projections
        for name in ["up_proj", "gate_proj", "down_proj"]:
            module = getattr(layer.ff, name, None)
            if isinstance(module, LowRankLinear):
                targets[name] = module
        return targets

    def _forward_with_temp_deltas(
        self,
        input_ids: torch.Tensor,
        delta_params: dict[str, tuple[nn.Parameter, nn.Parameter]],
    ) -> torch.Tensor:
        """
        Run a forward pass adding temporary delta contributions.
        This is a training-time forward that allows gradients to flow through deltas.
        """
        B, T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(T, device=device).unsqueeze(0)
        x = self.model.token_embedding(input_ids) + self.model.position_embedding(positions)

        mask = self.model.causal_mask[:, :, :T, :T]

        for i, layer in enumerate(self.model.layers):
            # Attention with deltas
            normed = layer.norm1(x)

            # Get delta contributions for this layer's attention
            attn_out = self._attn_with_deltas(layer.attn, normed, mask, i, delta_params)
            x = x + attn_out

            # FFN with deltas
            normed = layer.norm2(x)
            ff_out = self._ff_with_deltas(layer.ff, normed, i, delta_params)
            x = x + ff_out

        x = self.model.norm(x)
        logits = self.model.lm_head(x)
        return logits

    def _apply_delta_to_linear(
        self,
        module: LowRankLinear,
        x: torch.Tensor,
        layer_idx: int,
        target_name: str,
        delta_params: dict,
    ) -> torch.Tensor:
        """Apply a LowRankLinear module plus its delta contribution."""
        out = module(x)
        key = f"layer_{layer_idx}_{target_name}"
        if key in delta_params:
            dA, dB = delta_params[key]
            out = out + x @ dA @ dB
        return out

    def _attn_with_deltas(self, attn, x, mask, layer_idx, delta_params):
        """Run attention with delta contributions on Q, K, V, O projections."""
        from einops import rearrange

        B, T, D = x.shape
        k = min(attn.top_k, T)

        # Screening pass (no deltas — screening is cheap and doesn't need modification)
        q_s = attn.q_screen(x)
        k_s = attn.k_screen(x)
        screen_scores = torch.bmm(q_s, k_s.transpose(-2, -1)) * attn.screen_scale
        if mask is not None:
            screen_mask = mask.squeeze(0).squeeze(0)
            screen_scores = screen_scores.masked_fill(screen_mask[:T, :T] == 0, float("-inf"))
        _, top_k_indices = torch.topk(screen_scores, k, dim=-1)

        # Full projections with deltas
        Q = rearrange(
            self._apply_delta_to_linear(attn.q_proj, x, layer_idx, "q_proj", delta_params),
            "b t (h d) -> b h t d", h=attn.n_heads,
        )
        K = rearrange(
            self._apply_delta_to_linear(attn.k_proj, x, layer_idx, "k_proj", delta_params),
            "b t (h d) -> b h t d", h=attn.n_heads,
        )
        V = rearrange(
            self._apply_delta_to_linear(attn.v_proj, x, layer_idx, "v_proj", delta_params),
            "b t (h d) -> b h t d", h=attn.n_heads,
        )

        # Sparse attention over top-K candidates
        idx = top_k_indices.unsqueeze(1).unsqueeze(-1).expand(B, attn.n_heads, T, k, attn.d_head)
        K_sparse = torch.gather(K.unsqueeze(2).expand(B, attn.n_heads, T, T, attn.d_head), 3, idx)
        V_sparse = torch.gather(V.unsqueeze(2).expand(B, attn.n_heads, T, T, attn.d_head), 3, idx)

        scores = (Q.unsqueeze(3) * K_sparse).sum(-1) * attn.scale
        attn_weights = torch.softmax(scores, dim=-1)
        out = (attn_weights.unsqueeze(3) @ V_sparse).squeeze(3)

        out = rearrange(out, "b h t d -> b t (h d)")
        out = self._apply_delta_to_linear(attn.out_proj, out, layer_idx, "out_proj", delta_params)
        return out

    def _ff_with_deltas(self, ff, x, layer_idx, delta_params):
        """Run feed-forward with delta contributions."""
        up = self._apply_delta_to_linear(ff.up_proj, x, layer_idx, "up_proj", delta_params)
        gate = ff.act(self._apply_delta_to_linear(ff.gate_proj, x, layer_idx, "gate_proj", delta_params))
        hidden = up * gate
        out = self._apply_delta_to_linear(ff.down_proj, hidden, layer_idx, "down_proj", delta_params)
        return out
