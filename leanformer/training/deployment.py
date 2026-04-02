"""
Deployment Profiling

Full evaluation, per-tier latency profiling, and deployment manifest
generation. Produces quality-tiered model configurations.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn


@dataclass
class DeploymentTier:
    """A deployment configuration at a specific quality/cost tradeoff."""
    name: str
    active_levels: list[int]
    active_groups: list[str]
    description: str
    quality_score: float | None = None
    latency_ms: float | None = None
    param_count: int | None = None


class DeploymentProfiler:
    """Profiles model at different deployment configurations.

    Produces a deployment manifest with quality-tiered configurations
    and latency measurements.
    """

    def __init__(
        self,
        model: nn.Module,
        groups: dict[str, Any],  # group_id -> ParamGroup
        eval_fn: Callable[[], dict[str, float]] | None = None,
    ):
        self.model = model
        self.groups = groups
        self.eval_fn = eval_fn

        # Build hierarchy level info
        self.levels: dict[int, list[str]] = {}
        for gid, group in groups.items():
            lvl = group.hierarchy_level
            if lvl not in self.levels:
                self.levels[lvl] = []
            self.levels[lvl].append(gid)

    def build_tiers(self) -> list[DeploymentTier]:
        """Build deployment tiers based on hierarchy levels."""
        all_groups = list(self.groups.keys())
        sorted_levels = sorted(self.levels.keys())

        tiers = []
        # Full: all groups
        tiers.append(DeploymentTier(
            name="full",
            active_levels=sorted_levels,
            active_groups=all_groups,
            description="Maximum capability — all parameter groups active",
        ))

        # Standard: L0+L1+L2
        if len(sorted_levels) > 1:
            std_levels = sorted_levels[:-1] if len(sorted_levels) > 3 else sorted_levels[:3]
            std_groups = [
                gid for gid in all_groups
                if self.groups[gid].hierarchy_level in std_levels
            ]
            tiers.append(DeploymentTier(
                name="standard",
                active_levels=std_levels,
                active_groups=std_groups,
                description="General deployment — L3 specialization omitted",
            ))

        # Efficient: L0+L1
        if len(sorted_levels) > 2:
            eff_levels = sorted_levels[:2]
            eff_groups = [
                gid for gid in all_groups
                if self.groups[gid].hierarchy_level in eff_levels
            ]
            tiers.append(DeploymentTier(
                name="efficient",
                active_levels=eff_levels,
                active_groups=eff_groups,
                description="Latency-constrained — core representations only",
            ))

        # Minimal: L0 only
        min_groups = [
            gid for gid in all_groups
            if self.groups[gid].hierarchy_level == 0
        ]
        if min_groups != all_groups:  # Only add if different from full
            tiers.append(DeploymentTier(
                name="minimal",
                active_levels=[0],
                active_groups=min_groups,
                description="Edge deployment — structural parameters only",
            ))

        # Compute param counts
        for tier in tiers:
            tier.param_count = sum(
                self.groups[gid].param_count for gid in tier.active_groups
            )

        return tiers

    @torch.no_grad()
    def profile_latency(
        self,
        tier: DeploymentTier,
        input_ids: torch.Tensor,
        num_runs: int = 10,
        warmup_runs: int = 3,
    ) -> float:
        """Profile inference latency for a tier configuration.

        Returns average latency in milliseconds.
        """
        self.model.eval()

        # Freeze non-active groups for this tier
        original_requires_grad = {}
        for gid, group in self.groups.items():
            original_requires_grad[gid] = [p.requires_grad for p in group.params]
            if gid not in tier.active_groups:
                group.set_requires_grad(False)

        try:
            # Warmup
            for _ in range(warmup_runs):
                self.model(input_ids, training=False)

            # Timed runs
            start = time.perf_counter()
            for _ in range(num_runs):
                self.model(input_ids, training=False)
            elapsed = (time.perf_counter() - start) / num_runs * 1000  # ms

            return elapsed
        finally:
            # Restore requires_grad state
            for gid, group in self.groups.items():
                for p, rg in zip(group.params, original_requires_grad[gid]):
                    p.requires_grad_(rg)

    def profile_all_tiers(
        self,
        input_ids: torch.Tensor,
        num_runs: int = 10,
    ) -> list[DeploymentTier]:
        """Profile all tiers and optionally evaluate quality."""
        tiers = self.build_tiers()

        for tier in tiers:
            tier.latency_ms = self.profile_latency(tier, input_ids, num_runs)

            if self.eval_fn:
                # Evaluate quality for this tier
                metrics = self.eval_fn()
                # Use perplexity as primary quality score (lower is better)
                if "perplexity" in metrics:
                    tier.quality_score = metrics["perplexity"]
                else:
                    tier.quality_score = sum(metrics.values()) / len(metrics) if metrics else 0.0

        return tiers

    def generate_manifest(
        self,
        tiers: list[DeploymentTier],
        checkpoint_path: str | None = None,
        audit_hash: str | None = None,
    ) -> dict[str, Any]:
        """Generate deployment manifest."""
        manifest = {
            "version": "1.0",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "checkpoint_path": checkpoint_path,
            "audit_hash": audit_hash,
            "total_params": sum(g.param_count for g in self.groups.values()),
            "tiers": [],
        }

        for tier in tiers:
            tier_info = {
                "name": tier.name,
                "description": tier.description,
                "active_levels": tier.active_levels,
                "active_groups": tier.active_groups,
                "param_count": tier.param_count,
                "latency_ms": tier.latency_ms,
                "quality_score": tier.quality_score,
            }
            manifest["tiers"].append(tier_info)

        return manifest

    def select_tier(
        self,
        tiers: list[DeploymentTier],
        max_latency_ms: float | None = None,
        min_quality: float | None = None,
    ) -> DeploymentTier | None:
        """Select optimal tier given constraints."""
        candidates = tiers[:]

        if max_latency_ms is not None:
            candidates = [
                t for t in candidates
                if t.latency_ms is not None and t.latency_ms <= max_latency_ms
            ]

        if min_quality is not None:
            candidates = [
                t for t in candidates
                if t.quality_score is not None and t.quality_score <= min_quality
            ]

        if not candidates:
            return None

        # Prefer the tier with the most active groups (highest quality)
        return max(candidates, key=lambda t: len(t.active_groups))
