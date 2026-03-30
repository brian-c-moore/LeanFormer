"""
Knowledge Runtime: the serving component that manages active deltas,
routes queries, and applies composed deltas at inference time.
"""

import time
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from .router import KnowledgePlaneRouter
from .registry import DeltaRegistry


@dataclass
class InferenceResult:
    """Result of a knowledge-augmented inference call."""
    logits: torch.Tensor
    generated_text: str
    active_deltas: List[Tuple[str, float]]  # (delta_id, routing_score)
    categories_activated: List[str]
    composition_layers: List[int]           # Which layers had deltas applied
    inference_time_ms: float
    base_only: bool                         # Whether any deltas were active


class KnowledgeRuntime:
    """
    Manages inference with Knowledge Plane integration.

    Per-request flow:
    1. Embed incoming query using model's own embeddings
    2. Route to relevant delta subset via KnowledgePlaneRouter
    3. Compose active deltas additively
    4. Apply composed deltas at inference time via hooks
    5. Generate response
    6. Return result with full provenance
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        registry: DeltaRegistry,
        top_k: int = 5,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.router = KnowledgePlaneRouter(registry, top_k=top_k)
        self.registry = registry
        self.device = device

        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def embed_query(self, text: str) -> torch.Tensor:
        """Compute query embedding using model's own embedding layer."""
        tokens = self.tokenizer.encode(text, return_tensors="pt")
        tokens = tokens.to(self.device)
        with torch.no_grad():
            positions = torch.arange(tokens.shape[1], device=self.device).unsqueeze(0)
            emb = self.model.token_embedding(tokens) + self.model.position_embedding(positions)
            return emb.mean(dim=1).squeeze().cpu()

    def infer(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        use_knowledge: bool = True,
    ) -> InferenceResult:
        """
        Run inference with optional knowledge plane routing.

        If use_knowledge=True:
          1. Embed prompt
          2. Route to relevant deltas
          3. Compose and apply deltas via hooks
          4. Generate

        If use_knowledge=False:
          Generate with base model only (for comparison)
        """
        start = time.time()

        tokens = self.tokenizer.encode(prompt, return_tensors="pt")
        tokens = tokens.to(self.device)

        hooks = []
        active_deltas = []
        categories = []
        comp_layers = []
        logits = None

        try:
            if use_knowledge:
                query_emb = self.embed_query(prompt)
                composed, routing = self.router.route_and_compose(query_emb)
                active_deltas = routing
                categories = list(set(
                    self.registry.registered_deltas[did].category
                    for did, _ in routing
                    if did in self.registry.registered_deltas
                ))
                comp_layers = sorted(composed.keys())

                # Apply composed deltas via hooks
                for layer_idx, update in composed.items():
                    update_dev = update.to(self.device)
                    layer = self.model.layers[layer_idx]

                    def make_hook(upd):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                x = output[0]
                                return (x + x @ upd,) + output[1:]
                            else:
                                return output + output @ upd
                        return hook_fn

                    h = layer.register_forward_hook(make_hook(update_dev))
                    hooks.append(h)

            # Generate
            with torch.no_grad():
                generated_ids = tokens.clone()
                for _ in range(max_new_tokens):
                    with torch.amp.autocast("cuda"):
                        out = self.model(generated_ids, training=False)
                    logits = out['logits'] if isinstance(out, dict) else out
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)

                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

            generated_text = self.tokenizer.decode(
                generated_ids[0, tokens.shape[1]:],
                skip_special_tokens=True
            )

        finally:
            for h in hooks:
                h.remove()

        elapsed = (time.time() - start) * 1000

        return InferenceResult(
            logits=logits.cpu() if logits is not None else torch.tensor([]),
            generated_text=generated_text,
            active_deltas=active_deltas,
            categories_activated=categories,
            composition_layers=comp_layers,
            inference_time_ms=elapsed,
            base_only=not use_knowledge or len(active_deltas) == 0,
        )

    def verify_base_weight_integrity(self, expected_hash: str) -> bool:
        """Verify base model weights haven't been modified."""
        from .dfs import compute_model_hash
        current = compute_model_hash(self.model)
        return current == expected_hash
