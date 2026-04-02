"""
LeanFormer — the top-level model.

A transformer with four structural innovations over standard architectures:
1. Low-rank weight factorization from initialization
2. Two-pass sparse attention (cheap screening + exact computation)
3. Gated sparse feed-forward (predict-then-compute)
4. Adaptive computation depth (early exit on convergence)

The result: compute proportional to what the input requires, not the theoretical max.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

from .config import LeanFormerConfig
from .low_rank import LowRankLinear
from .attention import TwoPassSparseAttention
from .feedforward import GatedFeedForward
from .depth_controller import DepthController


class LeanFormerLayer(nn.Module):
    """Single transformer layer with sparse attention + gated FFN."""

    def __init__(self, config: LeanFormerConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.attn = TwoPassSparseAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            rank=config.attention_rank,
            screening_rank=config.screening_rank,
            top_k=config.attention_top_k,
            dropout=config.dropout,
        )
        self.ff = GatedFeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            rank=config.ff_rank,
            gate_rank=config.ff_gate_rank,
            sparsity_target=config.ff_sparsity_target,
            dropout=config.dropout,
        )
        # Pre-norm architecture (more stable than post-norm)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        training: bool = False,
        layer_deltas: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if layer_deltas:
            attn_out, attn_stats = self._attn_with_deltas(x, mask, layer_deltas)
        else:
            attn_out, attn_stats = self.attn(self.norm1(x), mask)
        x = x + attn_out

        if layer_deltas:
            ff_out, ff_stats = self._ff_with_deltas(x, training, layer_deltas)
        else:
            ff_out, ff_stats = self.ff(self.norm2(x), training=training)
        x = x + ff_out

        n_active = len(layer_deltas) if layer_deltas else 0
        stats = {**attn_stats, **ff_stats, "layer_idx": self.layer_idx, "active_deltas": n_active}
        return x, stats

    def _attn_with_deltas(self, x, mask, layer_deltas):
        """Run attention with delta contributions."""
        from einops import rearrange
        from ..beliefs.delta_system import forward_with_deltas

        normed = self.norm1(x)
        attn = self.attn
        B, T, D = normed.shape
        k = min(attn.top_k, T)

        # Screening (no deltas needed)
        q_s = attn.q_screen(normed)
        k_s = attn.k_screen(normed)
        screen_scores = torch.bmm(q_s, k_s.transpose(-2, -1)) * attn.screen_scale
        if mask is not None:
            screen_mask = mask.squeeze(0).squeeze(0)
            screen_scores = screen_scores.masked_fill(screen_mask[:T, :T] == 0, float("-inf"))
        _, top_k_indices = torch.topk(screen_scores, k, dim=-1)

        # Projections with deltas
        Q = rearrange(forward_with_deltas(attn.q_proj, normed, layer_deltas.get("q_proj")), "b t (h d) -> b h t d", h=attn.n_heads)
        K = rearrange(forward_with_deltas(attn.k_proj, normed, layer_deltas.get("k_proj")), "b t (h d) -> b h t d", h=attn.n_heads)
        V = rearrange(forward_with_deltas(attn.v_proj, normed, layer_deltas.get("v_proj")), "b t (h d) -> b h t d", h=attn.n_heads)

        # Sparse attention
        idx = top_k_indices.unsqueeze(1).unsqueeze(-1).expand(B, attn.n_heads, T, k, attn.d_head)
        K_sparse = torch.gather(K.unsqueeze(2).expand(B, attn.n_heads, T, T, attn.d_head), 3, idx)
        V_sparse = torch.gather(V.unsqueeze(2).expand(B, attn.n_heads, T, T, attn.d_head), 3, idx)

        scores = (Q.unsqueeze(3) * K_sparse).sum(-1) * attn.scale
        attn_weights = torch.softmax(scores, dim=-1)
        out = (attn_weights.unsqueeze(3) @ V_sparse).squeeze(3)
        out = rearrange(out, "b h t d -> b t (h d)")
        out = forward_with_deltas(attn.out_proj, out, layer_deltas.get("out_proj"))

        stats = {"attention_sparsity": 1.0 - (k / T) if T > 0 else 0.0, "attention_candidates": k, "attention_total_keys": T}
        return out, stats

    def _ff_with_deltas(self, x, training, layer_deltas):
        """Run feed-forward with delta contributions."""
        from ..beliefs.delta_system import forward_with_deltas

        normed = self.norm2(x)
        ff = self.ff
        up = forward_with_deltas(ff.up_proj, normed, layer_deltas.get("up_proj"))
        gate = ff.act(forward_with_deltas(ff.gate_proj, normed, layer_deltas.get("gate_proj")))
        hidden = up * gate
        out = forward_with_deltas(ff.down_proj, hidden, layer_deltas.get("down_proj"))
        stats = {"ff_sparsity": 0.0, "ff_active_neurons": ff.d_ff, "ff_total_neurons": ff.d_ff}
        return out, stats


class LeanFormer(nn.Module):

    def __init__(self, config: LeanFormerConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            LeanFormerLayer(config, i) for i in range(config.n_layers)
        ])

        # Depth controller governs early exit
        self.depth_controller = DepthController(
            d_model=config.d_model,
            min_depth=config.min_depth,
            threshold=config.exit_threshold,
        )

        # Output
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = LowRankLinear(
            config.d_model,
            config.vocab_size,
            rank=config.attention_rank,
            bias=False,
        )

        # Causal mask — registered as buffer so it moves with the model
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0))

        # Optional delta system (set via KnowledgeStore.attach_to_model)
        self.delta_registry = None
        self.delta_router = None

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        training: bool = False,
        active_deltas: list | None = None,
    ) -> dict:
        B, T = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Route to relevant deltas if registry is attached
        if active_deltas is None and self.delta_registry is not None and self.delta_router is not None and not training:
            active_deltas = self.delta_router(x, self.delta_registry)

        # Build per-layer delta map
        delta_map = None
        if active_deltas:
            from ..beliefs.delta_system import build_delta_map
            delta_map = build_delta_map(active_deltas)

        # Causal mask
        mask = self.causal_mask[:, :, :T, :T]

        # Forward through layers with adaptive depth
        all_stats = []
        prev_hidden = x.clone()
        exit_layer = self.config.n_layers
        gate_losses = []
        exit_losses = []

        for i, layer in enumerate(self.layers):
            layer_deltas = delta_map.get(i) if delta_map else None
            x, stats = layer(x, mask, training=training, layer_deltas=layer_deltas)
            all_stats.append(stats)

            if training and "gate_loss" in stats:
                gate_losses.append(stats["gate_loss"])

            # Check for early exit
            should_exit, exit_conf = self.depth_controller.should_exit(
                x, prev_hidden, i, training=training
            )

            if training:
                target_depth = int(self.config.n_layers * 0.75)
                exit_loss = self.depth_controller.compute_exit_loss(
                    x, target_depth, i
                )
                exit_losses.append(exit_loss)

            if should_exit:
                exit_layer = i + 1
                break

            prev_hidden = x.clone()

        # Output projection
        x = self.norm(x)
        logits = self.lm_head(x)

        result = {
            "logits": logits,
            "exit_layer": exit_layer,
            "depth_utilization": exit_layer / self.config.n_layers,
            "layer_stats": all_stats,
            "active_deltas": len(active_deltas) if active_deltas else 0,
        }

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Auxiliary losses
            aux_loss = torch.tensor(0.0, device=device)
            if gate_losses:
                aux_loss = aux_loss + torch.stack(gate_losses).mean() * 0.01
            if exit_losses:
                aux_loss = aux_loss + torch.stack(exit_losses).mean() * 0.01

            result["loss"] = lm_loss + aux_loss
            result["lm_loss"] = lm_loss
            result["aux_loss"] = aux_loss

        return result

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = 2,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Autoregressive generation. Returns (token_ids, per_step_stats).

        Args:
            eos_token_id: Token ID that terminates generation. Default 2
                (Mistral </s>). Set to -1 to disable early stopping.
        """
        self.eval()
        generated = input_ids.clone()
        step_stats = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = generated[:, -self.config.max_seq_len:]
                out = self.forward(context, training=False)
                logits = out["logits"][:, -1, :]

                step_stats.append({
                    "exit_layer": out["exit_layer"],
                    "depth_utilization": out["depth_utilization"],
                })

                # Temperature
                logits = logits / max(temperature, 1e-8)

                # Top-K
                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, k)
                    logits[logits < values[:, [-1]]] = float("-inf")

                # Top-P (nucleus)
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    # Remove tokens with cumulative prob above threshold
                    sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[sorted_mask] = float("-inf")
                    # Scatter back to original indexing
                    logits = logits.scatter(1, sorted_idx, sorted_logits)

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                if eos_token_id >= 0 and next_token.item() == eos_token_id:
                    break

        return generated, step_stats

    def get_efficiency_stats(self) -> dict:
        """Aggregate model-level efficiency stats for demo/logging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        dense_equiv = self.config.dense_equivalent_estimate()

        # Collect compression stats from all LowRankLinear modules
        lr_modules = [m for m in self.modules() if isinstance(m, LowRankLinear)]
        avg_compression = (
            sum(m.compression_ratio() for m in lr_modules) / len(lr_modules)
            if lr_modules else 1.0
        )

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "dense_equivalent": dense_equiv,
            "parameter_compression": dense_equiv / max(total_params, 1),
            "avg_layer_compression": avg_compression,
            "num_low_rank_modules": len(lr_modules),
            "n_layers": self.config.n_layers,
            "d_model": self.config.d_model,
        }
