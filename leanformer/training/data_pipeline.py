"""
Data Pipeline

Governed data ingestion with sample quality scoring, tiered sampling,
LSH deduplication, and quality gating.
"""

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler


@dataclass
class DataPipelineConfig:
    """Configuration for the governed data pipeline."""
    # Difficulty tier thresholds (on per-sample loss)
    mastered_threshold: float = 1.0
    struggling_threshold: float = 4.0
    failing_threshold: float = 8.0

    # Tier sampling fractions
    tier_fractions: dict[int, float] = field(default_factory=lambda: {
        0: 0.10,  # Mastered
        1: 0.30,  # Learning
        2: 0.40,  # Struggling
        3: 0.20,  # Failing
    })

    # Deduplication
    dedup_threshold: float = 0.95  # Cosine similarity threshold for near-dupes

    # Quality gate
    quality_floor: float = 0.1   # Min loss (below = already mastered, skip)
    quality_ceiling: float = 20.0  # Max loss (above = garbage/OOD)

    # Re-scoring
    rescore_interval: int = 1000  # Steps between re-scoring
    rescore_fraction: float = 0.1  # Fraction of each tier to re-score


class SampleScorer:
    """Scores dataset samples by difficulty and quality.

    Runs a forward pass to compute per-sample loss, then assigns
    each sample to a difficulty tier.
    """

    def __init__(self, config: DataPipelineConfig | None = None):
        self.config = config or DataPipelineConfig()

    @torch.no_grad()
    def score_samples(
        self,
        model: nn.Module,
        dataset: Dataset,
        indices: list[int] | None = None,
        batch_size: int = 32,
        max_samples: int | None = None,
    ) -> dict[str, Any]:
        """Score a dataset (or subset) by difficulty.

        Returns dict with:
            - scores: dict[int, float] mapping sample index -> loss
            - tiers: dict[int, list[int]] mapping tier -> sample indices
            - excluded: list[int] samples excluded by quality gate
        """
        model.eval()
        if indices is None:
            indices = list(range(len(dataset)))
        if max_samples:
            indices = indices[:max_samples]

        scores: dict[int, float] = {}
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = self._collate(dataset, batch_indices)
            if batch is None:
                continue

            device = next(model.parameters()).device
            input_ids = batch["input_ids"].to(device)
            labels = batch.get("labels", input_ids.clone()).to(device)

            result = model(input_ids, labels=labels, training=False)
            if "loss" in result:
                # Per-sample loss
                logits = result["logits"]
                for i, idx in enumerate(batch_indices):
                    sample_logits = logits[i:i+1]
                    sample_labels = labels[i:i+1]
                    loss = nn.functional.cross_entropy(
                        sample_logits[:, :-1, :].contiguous().view(-1, sample_logits.shape[-1]),
                        sample_labels[:, 1:].contiguous().view(-1),
                        ignore_index=-100,
                    )
                    scores[idx] = loss.item()

        # Assign tiers
        tiers: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
        excluded: list[int] = []

        for idx, loss in scores.items():
            # Quality gate
            if loss < self.config.quality_floor or loss > self.config.quality_ceiling:
                excluded.append(idx)
                continue

            if loss < self.config.mastered_threshold:
                tiers[0].append(idx)
            elif loss < self.config.struggling_threshold:
                tiers[1].append(idx)
            elif loss < self.config.failing_threshold:
                tiers[2].append(idx)
            else:
                tiers[3].append(idx)

        model.train()
        return {"scores": scores, "tiers": tiers, "excluded": excluded}

    def _collate(self, dataset: Dataset, indices: list[int]) -> dict | None:
        """Collate a batch of samples from the dataset."""
        samples = []
        for idx in indices:
            item = dataset[idx]
            if isinstance(item, dict):
                samples.append(item)
            elif isinstance(item, torch.Tensor):
                samples.append({"input_ids": item})
            else:
                return None

        if not samples:
            return None

        # Stack tensors
        result = {}
        for key in samples[0]:
            tensors = [s[key] for s in samples if key in s]
            if tensors and isinstance(tensors[0], torch.Tensor):
                # Pad to same length
                max_len = max(t.shape[0] for t in tensors)
                padded = [
                    nn.functional.pad(t, (0, max_len - t.shape[0]), value=-100 if key == "labels" else 0)
                    for t in tensors
                ]
                result[key] = torch.stack(padded)

        return result if result else None


class TieredSampler(Sampler):
    """Samples from tiered difficulty buckets according to configured fractions.

    Each tier has a sampling fraction governed by Budget<SamplesPerStep>.
    Within each tier, sampling is uniform random.
    """

    def __init__(
        self,
        tiers: dict[int, list[int]],
        batch_size: int,
        tier_fractions: dict[int, float] | None = None,
        num_batches: int | None = None,
    ):
        self.tiers = tiers
        self.batch_size = batch_size
        self.tier_fractions = tier_fractions or {0: 0.10, 1: 0.30, 2: 0.40, 3: 0.20}
        self.num_batches = num_batches or 100

        # Compute per-tier sample counts per batch
        self.tier_counts: dict[int, int] = {}
        remaining = batch_size
        for tier in sorted(self.tier_fractions.keys()):
            if tier == max(self.tier_fractions.keys()):
                self.tier_counts[tier] = remaining
            else:
                count = max(0, int(batch_size * self.tier_fractions[tier]))
                # Can't sample more than the tier has
                count = min(count, len(self.tiers.get(tier, [])))
                self.tier_counts[tier] = count
                remaining -= count

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for tier, count in self.tier_counts.items():
                pool = self.tiers.get(tier, [])
                if pool and count > 0:
                    actual_count = min(count, len(pool))
                    perm = torch.randperm(len(pool))[:actual_count]
                    batch.extend(pool[i] for i in perm.tolist())

            # Fill any shortfall from the largest non-empty tier
            while len(batch) < self.batch_size:
                for tier in [2, 1, 3, 0]:
                    pool = self.tiers.get(tier, [])
                    if pool:
                        idx = torch.randint(0, len(pool), (1,)).item()
                        batch.append(pool[idx])
                        break
                else:
                    break

            yield batch[:self.batch_size]

    def __len__(self):
        return self.num_batches

    def update_tiers(self, new_tiers: dict[int, list[int]]):
        """Update tier assignments (after re-scoring)."""
        self.tiers = new_tiers

    def get_tier_stats(self) -> dict[str, Any]:
        """Return current tier sizes and sampling fractions."""
        return {
            "tier_sizes": {t: len(indices) for t, indices in self.tiers.items()},
            "tier_fractions": dict(self.tier_fractions),
            "tier_counts_per_batch": dict(self.tier_counts),
        }


class Deduplicator:
    """LSH-based near-duplicate detection for training data."""

    def __init__(
        self,
        threshold: float = 0.95,
        num_hashes: int = 64,
        seed: int = 42,
    ):
        self.threshold = threshold
        self.num_hashes = num_hashes
        self.seed = seed
        self._random_planes: torch.Tensor | None = None

    def _get_planes(self, dim: int) -> torch.Tensor:
        """Get or create random hyperplanes for LSH."""
        if self._random_planes is None or self._random_planes.shape[1] != dim:
            gen = torch.Generator()
            gen.manual_seed(self.seed)
            self._random_planes = torch.randn(self.num_hashes, dim, generator=gen)
        return self._random_planes

    @torch.no_grad()
    def compute_hashes(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute LSH binary hashes for a batch of embeddings.

        Args:
            embeddings: [N, dim] tensor of sample embeddings.

        Returns:
            [N, num_hashes] binary hash tensor.
        """
        planes = self._get_planes(embeddings.shape[-1]).to(embeddings.device)
        projections = embeddings @ planes.T  # [N, num_hashes]
        return (projections > 0).int()

    def find_duplicates(
        self,
        embeddings: torch.Tensor,
        max_comparisons: int = 10000,
    ) -> list[tuple[int, int]]:
        """Find near-duplicate pairs using LSH.

        Returns list of (idx_i, idx_j) pairs that are near-duplicates.
        """
        hashes = self.compute_hashes(embeddings)
        n = hashes.shape[0]
        duplicates = []

        # For each sample, check against others with similar hash
        for i in range(min(n, max_comparisons)):
            # Hamming distance to all others
            hamming = (hashes[i:i+1] != hashes).sum(dim=1).float()
            similarity = 1.0 - hamming / self.num_hashes

            # Find near-duplicates (excluding self)
            for j in range(i + 1, n):
                if similarity[j].item() >= self.threshold:
                    duplicates.append((i, j))

        return duplicates

    def deduplicate_indices(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[list[int], list[int]]:
        """Partition indices into unique and duplicate.

        Returns (unique_indices, duplicate_indices).
        Keeps the first occurrence of each duplicate cluster.
        """
        pairs = self.find_duplicates(embeddings)
        duplicates = set()
        for i, j in pairs:
            duplicates.add(j)  # Keep i, remove j

        n = embeddings.shape[0]
        unique = [i for i in range(n) if i not in duplicates]
        return unique, sorted(duplicates)
