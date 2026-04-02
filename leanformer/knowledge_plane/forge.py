"""
Knowledge Forge: converts knowledge artifacts into DFS-conformant deltas.

The forge pipeline:
1. Parse knowledge source (JSON facts)
2. For each fact, compute routing embedding
3. Consult registry for available subspace
4. Encode fact as targeted-layer low-rank delta
5. Validate: does applying the delta improve target token rank?
6. Check orthogonality against registry
7. If validation passes, produce .delta file
8. If validation fails, retry with wider layer targeting

The forge is a quality gate. Deltas that don't pass validation
don't enter the Knowledge Plane.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import torch
import json
from pathlib import Path

from .dfs import DeltaFormatSpec, compute_model_hash
from .registry import DeltaRegistry, orthogonality_score


@dataclass
class Fact:
    """A single fact to be encoded as a belief delta."""
    prompt: str           # The input prompt
    target: str           # The target completion (e.g., " Paris")
    target_token_id: int  # GPT-2 token ID for the target
    description: str      # Human-readable description
    category: str         # Domain category
    domain_tags: List[str]


def build_layer_strategies(n_layers: int) -> list:
    """Build progressive layer targeting strategies based on model depth.

    Starts with a narrow band around the middle layers (where semantic
    representations are richest), progressively widens, and falls back
    to all layers as a last resort.
    """
    mid = n_layers // 2
    strategies = []

    # Strategy 1: narrow middle band (5 layers centered)
    half = 2
    start = max(0, mid - half)
    end = min(n_layers, mid + half + 1)
    strategies.append(list(range(start, end)))

    # Strategy 2: wider middle band (7 layers)
    half = 3
    start = max(0, mid - half)
    end = min(n_layers, mid + half + 1)
    strategies.append(list(range(start, end)))

    # Strategy 3: broad middle band (~60% of layers)
    half = n_layers // 3
    start = max(0, mid - half)
    end = min(n_layers, mid + half + 1)
    strategies.append(list(range(start, end)))

    # Strategy 4: all layers (last resort)
    strategies.append(list(range(n_layers)))

    # Deduplicate while preserving the invariant that the last strategy
    # is always "all layers". Remove earlier duplicates, not later ones.
    all_layers = list(range(n_layers))
    seen = []
    for s in strategies:
        if s not in seen:
            seen.append(s)
    # Ensure the all-layers fallback is always last
    if seen[-1] != all_layers:
        if all_layers in seen:
            seen.remove(all_layers)
        seen.append(all_layers)
    return seen


# Default strategies for backward compatibility (12-layer models)
LAYER_STRATEGIES = build_layer_strategies(12)


class KnowledgeForge:
    """
    Converts facts into orthogonality-validated targeted-layer deltas.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        registry: DeltaRegistry,
        delta_rank: int = 16,
        learning_rate: float = 1e-2,
        max_steps: int = 200,
        validation_threshold: float = 0.5,  # Min fraction of facts that must improve
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry
        self.delta_rank = delta_rank
        self.lr = learning_rate
        self.max_steps = max_steps
        self.validation_threshold = validation_threshold
        self.device = device

        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False

        # Compute base model hash once
        self.base_model_hash = registry.base_model_hash

    def compute_routing_embedding(self, facts: List[Fact]) -> torch.Tensor:
        """
        Compute a routing embedding for a set of related facts.
        Uses the model's own embeddings - average the hidden state
        of the prompt tokens across all facts in this set.
        """
        embeddings = []
        for fact in facts:
            tokens = self.tokenizer.encode(fact.prompt, return_tensors="pt")
            tokens = tokens.to(self.device)
            with torch.no_grad():
                positions = torch.arange(tokens.shape[1], device=self.device).unsqueeze(0)
                emb = self.model.token_embedding(tokens) + self.model.position_embedding(positions)
                avg = emb.mean(dim=1).squeeze()      # (d_model,)
                embeddings.append(avg)

        # Average across all facts
        stacked = torch.stack(embeddings)
        return stacked.mean(dim=0).cpu()

    def encode_delta(
        self,
        facts: List[Fact],
        target_layers: List[int],
    ) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Encode a set of facts into low-rank delta factors for specified layers.

        Uses gradient-based optimization:
        1. Create trainable A and B matrices for each target layer
        2. For each fact, compute the loss = -log_softmax(logits)[target_token_id]
        3. Optimize A, B to minimize total loss across all facts
        4. Return the learned factors and the subspace basis

        Returns:
            (factors_dict, subspace_basis)
        """
        d_model = self.registry.d_model

        # Create trainable delta parameters
        delta_params = {}
        trainable = []
        for layer_idx in target_layers:
            A = torch.randn(d_model, self.delta_rank, device=self.device) * 0.01
            B = torch.randn(self.delta_rank, d_model, device=self.device) * 0.01
            A.requires_grad = True
            B.requires_grad = True
            delta_params[layer_idx] = (A, B)
            trainable.extend([A, B])

        optimizer = torch.optim.Adam(trainable, lr=self.lr)

        # Prepare fact inputs
        fact_inputs = []
        for fact in facts:
            tokens = self.tokenizer.encode(fact.prompt, return_tensors="pt")
            tokens = tokens.to(self.device)
            fact_inputs.append((tokens, fact.target_token_id))

        # Training loop
        for step in range(self.max_steps):
            total_loss = 0.0
            for tokens, target_id in fact_inputs:
                logits = self._forward_with_delta(tokens, delta_params)
                last_logits = logits[0, -1, :]  # (vocab_size,)
                log_probs = torch.log_softmax(last_logits, dim=-1)
                loss = -log_probs[target_id]
                total_loss += loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Extract factors and compute subspace basis
        factors = {}
        for layer_idx, (A, B) in delta_params.items():
            factors[layer_idx] = (A.detach().cpu(), B.detach().cpu())

        # Subspace basis = orthonormal basis of column space of A
        all_A = torch.cat([A.detach().cpu() for A, B in factors.values()], dim=1)
        Q, R = torch.linalg.qr(all_A)
        subspace_basis = Q[:, :self.delta_rank]

        return factors, subspace_basis

    def _forward_with_delta(
        self,
        tokens: torch.Tensor,
        delta_params: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Run the model's forward pass with delta factors injected.
        Uses PyTorch hooks to inject deltas without modifying model code.
        """
        hooks = []

        def make_hook(A, B):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    x = output[0]
                    delta = x @ A @ B
                    return (x + delta,) + output[1:]
                else:
                    delta = output @ A @ B
                    return output + delta
            return hook_fn

        for layer_idx, (A, B) in delta_params.items():
            layer = self.model.layers[layer_idx]
            h = layer.register_forward_hook(make_hook(A, B))
            hooks.append(h)

        try:
            with torch.amp.autocast("cuda"):
                out = self.model(tokens, training=False)
            logits = out['logits'] if isinstance(out, dict) else out
        finally:
            for h in hooks:
                h.remove()

        return logits

    def validate_delta(
        self,
        facts: List[Fact],
        factors: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict:
        """
        Validate that the delta actually improves target token ranks.

        For each fact:
        1. Run model WITHOUT delta, record target token rank
        2. Run model WITH delta, record target token rank
        3. Delta is valid if rank improves for >threshold fraction

        Returns validation results dict.
        """
        results = {
            "total_facts": len(facts),
            "improved": 0,
            "degraded": 0,
            "unchanged": 0,
            "per_fact": [],
        }

        # Move factors to device
        device_factors = {
            k: (A.to(self.device), B.to(self.device))
            for k, (A, B) in factors.items()
        }

        for fact in facts:
            tokens = self.tokenizer.encode(fact.prompt, return_tensors="pt")
            tokens = tokens.to(self.device)

            # Baseline (no delta)
            with torch.no_grad():
                out_base = self.model(tokens, training=False)
                logits_base = out_base['logits'] if isinstance(out_base, dict) else out_base
                probs_base = torch.softmax(logits_base[0, -1, :], dim=-1)
                sorted_base = torch.argsort(probs_base, descending=True)
                rank_before = (sorted_base == fact.target_token_id).nonzero().item()

            # With delta
            with torch.no_grad():
                logits_delta = self._forward_with_delta(tokens, device_factors)
                probs_delta = torch.softmax(logits_delta[0, -1, :], dim=-1)
                sorted_delta = torch.argsort(probs_delta, descending=True)
                rank_after = (sorted_delta == fact.target_token_id).nonzero().item()

            if rank_after < rank_before:
                results["improved"] += 1
            elif rank_after > rank_before:
                results["degraded"] += 1
            else:
                results["unchanged"] += 1

            results["per_fact"].append({
                "prompt": fact.prompt,
                "target": fact.target,
                "rank_before": rank_before,
                "rank_after": rank_after,
                "improved": rank_after < rank_before,
            })

        results["success_rate"] = results["improved"] / max(results["total_facts"], 1)
        results["passed"] = results["success_rate"] >= self.validation_threshold
        return results

    def forge_domain(
        self,
        facts: List[Fact],
        domain_name: str,
        batch_size: int = 10,
    ) -> List[DeltaFormatSpec]:
        """
        Forge an entire domain of facts into validated deltas.

        Facts are processed in batches. Each batch produces one delta
        that encodes batch_size facts. This amortizes delta overhead
        across related facts.

        Returns list of successfully forged deltas.
        """
        forged = []

        for i in range(0, len(facts), batch_size):
            batch = facts[i : i + batch_size]
            batch_idx = i // batch_size
            print(f"\n[Forge] Domain '{domain_name}' batch {batch_idx + 1} "
                  f"({len(batch)} facts)")

            delta = self._forge_batch(batch, domain_name, batch_idx)
            if delta is not None:
                forged.append(delta)
                print(f"  -> Delta forged and registered "
                      f"(layers={delta.target_layers}, "
                      f"params={delta.param_count:,})")
            else:
                print(f"  -> Batch failed all layer strategies")

        n_batches = (len(facts) + batch_size - 1) // batch_size
        print(f"\n[Forge] Domain '{domain_name}': {len(forged)}/{n_batches} "
              f"deltas forged successfully")
        return forged

    def _forge_batch(
        self,
        facts: List[Fact],
        domain_name: str,
        batch_idx: int,
    ) -> Optional[DeltaFormatSpec]:
        """
        Try to forge a batch of facts, progressively widening layer targeting
        if validation fails.
        """
        strategies = build_layer_strategies(self.registry.n_layers)
        for attempt, target_layers in enumerate(strategies):
            print(f"  Attempt {attempt + 1}: layers {target_layers}")

            # Check registry capacity for these layers
            capacity = self.registry.remaining_capacity_estimate()
            can_fit = all(capacity.get(l, 0) > 0 for l in target_layers)
            if not can_fit:
                print(f"    Insufficient capacity at target layers")
                continue

            # Encode
            factors, subspace_basis = self.encode_delta(facts, target_layers)

            # Validate
            results = self.validate_delta(facts, factors)
            print(f"    Validation: {results['success_rate']:.0%} improved "
                  f"({results['improved']}/{results['total_facts']})")

            if not results["passed"]:
                continue

            # Compute routing embedding
            embedding = self.compute_routing_embedding(facts)

            # Build DFS
            delta = DeltaFormatSpec(
                delta_id=DeltaFormatSpec.create_id(),
                version=1,
                created_at=DeltaFormatSpec.timestamp(),
                source=f"forge:{domain_name}:batch_{batch_idx}",
                embedding=embedding,
                category=domain_name,
                description=f"{domain_name} facts batch {batch_idx}",
                domain_tags=[domain_name],
                target_layers=target_layers,
                factors=factors,
                delta_rank=self.delta_rank,
                param_count=0,
                subspace_basis=subspace_basis,
                orthogonality_score=0.0,
                conflicting_delta_ids=[],
                base_model_hash=self.base_model_hash,
                d_model=self.registry.d_model,
                n_layers=self.registry.n_layers,
                confidence=results["success_rate"],
                validation_results=results,
                forge_attempts=attempt + 1,
            )
            delta.param_count = delta.compute_param_count()

            # Register (checks orthogonality)
            success, msg = self.registry.register(delta)
            if success:
                return delta
            else:
                print(f"    Registry rejected: {msg}")
                continue

        return None  # All strategies failed


def load_fact_bank(path: str, tokenizer) -> Dict[str, List[Fact]]:
    """
    Load a fact bank JSON file and convert to Fact objects.

    Expected format:
    {
      "category_name": [
        {"prompt": "...", "target": "...", "fact": "..."},
        ...
      ]
    }

    Verifies each target token exists in the tokenizer vocabulary.
    """
    with open(path) as f:
        raw = json.load(f)

    facts_by_category = {}
    for category, items in raw.items():
        facts = []
        for item in items:
            # Tokenize the target and take the first token
            target_tokens = tokenizer.encode(item["target"])
            if not target_tokens:
                print(f"  Warning: target '{item['target']}' "
                      f"produces no tokens, skipping")
                continue
            target_token_id = target_tokens[0]

            facts.append(Fact(
                prompt=item["prompt"],
                target=item["target"],
                target_token_id=target_token_id,
                description=item.get("fact", item["prompt"] + item["target"]),
                category=category,
                domain_tags=[category],
            ))
        facts_by_category[category] = facts

    return facts_by_category
