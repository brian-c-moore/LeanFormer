"""
Knowledge Store — user-facing API for the delta-based belief system.

Wraps BeliefEncoder + DeltaRegistry into a clean interface:
    store.add("capital_france", "The capital of France is Paris")
    store.update("capital_france", "The capital of France is Lyon")
    store.remove("capital_france")
    store.query(query_embedding, top_k=5)

Adding a fact: encodes it as a sparse weight delta, registers it.
Updating a fact: re-encodes and replaces the delta.
Removing a fact: deletes the delta, restores original behavior.
Base weights are NEVER modified.

This is ResourceRegistry for knowledge — the authoritative source of domain data,
separate from the processing logic that consumes it.
"""

import torch
from dataclasses import dataclass, field
from transformers import AutoTokenizer

from ..model.leanformer import LeanFormer
from .delta_system import DeltaRegistry, DeltaRouter, BeliefDelta
from .belief_encoder import BeliefEncoder


@dataclass
class KnowledgeEntry:
    """A single piece of factual knowledge with its delta."""
    key: str
    content: str
    source: str = ""
    version: int = 1


class KnowledgeStore:
    """
    User-facing API for managing beliefs/knowledge in a LeanFormer model.

    Usage:
        store = KnowledgeStore(model)
        store.add("capital_france", "The capital of France is Paris")
        store.attach_to_model(model)  # enables automatic routing during inference
    """

    def __init__(
        self,
        model: LeanFormer,
        tokenizer: AutoTokenizer | None = None,
        delta_rank: int | None = None,
        encoding_steps: int | None = None,
        encoding_lr: float | None = None,
    ):
        config = model.config
        self.encoder = BeliefEncoder(
            model=model,
            tokenizer=tokenizer,
            delta_rank=delta_rank or config.delta_rank,
            n_steps=encoding_steps or config.delta_encoding_steps,
            lr=encoding_lr or config.delta_encoding_lr,
        )
        self.registry = DeltaRegistry()
        self.entries: dict[str, KnowledgeEntry] = {}
        self.model = model

    def add(self, key: str, fact: str, source: str = "") -> dict:
        """
        Encode a fact as a belief delta and register it.

        Returns info dict with registration details.
        """
        delta = self.encoder.encode(fact, key)
        info = self.registry.register(delta)

        self.entries[key] = KnowledgeEntry(
            key=key, content=fact, source=source, version=1,
        )

        info["fact"] = fact
        info["key"] = key
        return info

    def update(self, key: str, new_fact: str) -> dict:
        """
        Re-encode a fact and replace its delta. Base weights unchanged.
        """
        if key not in self.entries:
            raise KeyError(f"No entry with key '{key}' to update")

        new_delta = self.encoder.encode(new_fact, key)
        info = self.registry.update(key, new_delta)

        self.entries[key].content = new_fact
        self.entries[key].version += 1

        info["fact"] = new_fact
        info["version"] = self.entries[key].version
        return info

    def remove(self, key: str) -> bool:
        """Remove a fact and its delta. Restores original model behavior."""
        if key not in self.entries:
            return False
        self.registry.remove(key)
        del self.entries[key]
        return True

    def query(self, query_embedding: torch.Tensor, top_k: int = 5) -> list[dict]:
        """Retrieve relevant facts by embedding similarity."""
        deltas = self.registry.query(query_embedding, top_k=top_k)
        results = []
        for delta in deltas:
            if delta.name in self.entries:
                entry = self.entries[delta.name]
                results.append({
                    "key": entry.key,
                    "content": entry.content,
                    "source": entry.source,
                    "version": entry.version,
                    "parameters": delta.total_parameters(),
                })
        return results

    def attach_to_model(self, model: LeanFormer | None = None):
        """
        Attach the registry and router to a model so the forward pass
        automatically routes to relevant deltas during inference.
        """
        target = model or self.model
        target.delta_registry = self.registry
        target.delta_router = DeltaRouter(
            d_model=target.config.d_model,
            max_active_deltas=target.config.max_active_deltas,
        )

    def detach_from_model(self, model: LeanFormer | None = None):
        """Remove delta routing from the model."""
        target = model or self.model
        target.delta_registry = None
        target.delta_router = None

    def list_facts(self) -> list[dict]:
        """List all stored facts."""
        return [
            {
                "key": e.key,
                "content": e.content,
                "source": e.source,
                "version": e.version,
                "parameters": self.registry.deltas[e.key].total_parameters()
                if e.key in self.registry.deltas else 0,
            }
            for e in self.entries.values()
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, key: str) -> bool:
        return key in self.entries
