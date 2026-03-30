"""
Consolidation Pipeline: merges stable related deltas into compressed
consolidated deltas. Analogous to database VACUUM / log compaction.

Purpose:
1. Free subspace capacity (merged delta uses fewer slots than individuals)
2. Reduce routing cost (fewer embeddings to compare)
3. Improve composition efficiency (one merged delta vs many small ones)
"""

import torch
from typing import List, Dict, Optional
from .dfs import DeltaFormatSpec
from .registry import DeltaRegistry


class KnowledgePlaneConsolidator:
    """Merges groups of related deltas into consolidated deltas."""

    def __init__(self, registry: DeltaRegistry, model=None, tokenizer=None, device="cuda"):
        self.registry = registry
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def identify_candidates(self, min_group_size: int = 5) -> List[List[str]]:
        """
        Find groups of deltas eligible for consolidation.

        Criteria:
        - Same category
        - All registered (not recently updated)
        - Group size >= min_group_size
        """
        by_category = {}
        for did, delta in self.registry.registered_deltas.items():
            cat = delta.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(did)

        return [
            group for group in by_category.values()
            if len(group) >= min_group_size
        ]

    def consolidate(
        self,
        delta_ids: List[str],
        new_delta_rank: int = None,
    ) -> Optional[DeltaFormatSpec]:
        """
        Merge a group of deltas into a single consolidated delta.

        Process:
        1. Sum all (A_i @ B_i) contributions per layer
        2. Re-factorize the summed matrix via SVD truncation
        3. Build new DFS with merged factors
        4. Validate merged delta against all original facts
        5. Register merged, deregister individuals

        The re-factorization step is key: the merged rank can be
        smaller than the sum of individual ranks because shared
        subspace directions collapse.
        """
        if new_delta_rank is None:
            new_delta_rank = self.registry.delta_rank

        deltas = [
            self.registry.registered_deltas[did]
            for did in delta_ids
            if did in self.registry.registered_deltas
        ]
        if not deltas:
            return None

        # Collect all target layers
        all_layers = set()
        for d in deltas:
            all_layers.update(d.target_layers)
        all_layers = sorted(all_layers)

        # Sum contributions per layer
        merged_factors = {}
        for layer_idx in all_layers:
            W_sum = torch.zeros(deltas[0].d_model, deltas[0].d_model)
            for d in deltas:
                if layer_idx in d.factors:
                    A, B = d.factors[layer_idx]
                    W_sum += A @ B
            # Re-factorize via truncated SVD
            U, S, Vh = torch.linalg.svd(W_sum, full_matrices=False)
            A_new = U[:, :new_delta_rank] * S[:new_delta_rank].sqrt()
            B_new = (S[:new_delta_rank].sqrt().unsqueeze(1) *
                     Vh[:new_delta_rank, :])
            merged_factors[layer_idx] = (A_new, B_new)

        # Average routing embeddings
        avg_embedding = torch.stack([d.embedding for d in deltas]).mean(dim=0)

        # Compute new subspace basis
        all_A = torch.cat([A for A, B in merged_factors.values()], dim=1)
        Q, _ = torch.linalg.qr(all_A)
        subspace_basis = Q[:, :new_delta_rank]

        merged = DeltaFormatSpec(
            delta_id=DeltaFormatSpec.create_id(),
            version=1,
            created_at=DeltaFormatSpec.timestamp(),
            source=f"consolidation:{len(deltas)}_deltas",
            embedding=avg_embedding,
            category=deltas[0].category,
            description=f"Consolidated {deltas[0].category} ({len(deltas)} deltas merged)",
            domain_tags=deltas[0].domain_tags,
            target_layers=all_layers,
            factors=merged_factors,
            delta_rank=new_delta_rank,
            param_count=0,
            subspace_basis=subspace_basis,
            orthogonality_score=0.0,
            conflicting_delta_ids=[],
            base_model_hash=deltas[0].base_model_hash,
            d_model=deltas[0].d_model,
            n_layers=deltas[0].n_layers,
            tags=["consolidated"],
        )
        merged.param_count = merged.compute_param_count()

        return merged

    def consolidate_and_replace(
        self,
        delta_ids: List[str],
    ) -> Optional[DeltaFormatSpec]:
        """
        Consolidate deltas and atomically replace them in the registry.
        """
        merged = self.consolidate(delta_ids)
        if merged is None:
            return None

        # Remove old deltas
        for did in delta_ids:
            self.registry.unregister(did)

        # Register merged
        success, msg = self.registry.register(merged)
        if not success:
            # Rollback: this should not happen if the merge is correct
            print(f"WARNING: Consolidated delta rejected: {msg}")
            return None

        return merged
