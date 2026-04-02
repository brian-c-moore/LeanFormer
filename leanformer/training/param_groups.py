"""
Parameter Group Registry

Maps every parameter tensor in a LeanFormer model to a named group with
hierarchy level assignment. Used by convergence governors, hierarchy
activation, budget allocation, and gradient routing.
"""

import json
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import torch.nn as nn


_REGISTRY_PATH = Path(__file__).parent.parent.parent / "configs" / "parameter_groups.json"


def load_registry(path: Path | str | None = None) -> dict:
    """Load the parameter group registry JSON."""
    path = Path(path) if path else _REGISTRY_PATH
    with open(path) as f:
        return json.load(f)


def build_param_groups(
    model: nn.Module,
    registry: dict | None = None,
) -> dict[str, "ParamGroup"]:
    """Build parameter groups from a model and registry.

    Returns a dict mapping group_id -> ParamGroup.
    Validates that every parameter is assigned to exactly one group.
    """
    if registry is None:
        registry = load_registry()

    groups: dict[str, ParamGroup] = {}
    assigned: dict[str, str] = {}  # param_name -> group_id

    # Build groups from registry
    for group_spec in registry["groups"]:
        gid = group_spec["group_id"]
        groups[gid] = ParamGroup(
            group_id=gid,
            hierarchy_level=group_spec["hierarchy_level"],
            description=group_spec["description"],
            tensor_patterns=group_spec["tensor_patterns"],
            expected_convergence_order=group_spec["expected_convergence_order"],
            dependent_metrics=group_spec["dependent_metrics"],
        )

    # Assign parameters to groups
    all_param_names = set()
    for name, param in model.named_parameters():
        all_param_names.add(name)
        matched_group = None
        for gid, group in groups.items():
            if group.matches(name):
                if matched_group is not None:
                    raise ValueError(
                        f"Parameter '{name}' matches multiple groups: "
                        f"'{matched_group}' and '{gid}'"
                    )
                matched_group = gid
                group.add_param(name, param)
                assigned[name] = gid

        if matched_group is None:
            raise ValueError(
                f"Parameter '{name}' does not match any group in the registry"
            )

    # Validate completeness
    unassigned = all_param_names - set(assigned.keys())
    if unassigned:
        raise ValueError(
            f"Parameters not assigned to any group: {unassigned}"
        )

    return groups


def validate_registry(model: nn.Module, registry: dict | None = None) -> dict[str, Any]:
    """Validate the parameter group registry against a model.

    Returns a summary dict with group stats and validation results.
    """
    groups = build_param_groups(model, registry)

    total_model_params = sum(p.numel() for p in model.parameters())
    total_group_params = sum(g.param_count for g in groups.values())

    # Build hierarchy level summary
    levels: dict[int, dict] = {}
    for g in groups.values():
        lvl = g.hierarchy_level
        if lvl not in levels:
            levels[lvl] = {"groups": [], "total_params": 0}
        levels[lvl]["groups"].append(g.group_id)
        levels[lvl]["total_params"] += g.param_count

    for lvl_info in levels.values():
        lvl_info["fraction_of_model"] = round(
            lvl_info["total_params"] / total_model_params, 4
        )

    result = {
        "total_model_params": total_model_params,
        "total_group_params": total_group_params,
        "params_match": total_model_params == total_group_params,
        "num_groups": len(groups),
        "groups": {
            gid: {
                "hierarchy_level": g.hierarchy_level,
                "param_count": g.param_count,
                "num_tensors": len(g.param_names),
                "fraction_of_model": round(g.param_count / total_model_params, 4),
            }
            for gid, g in groups.items()
        },
        "hierarchy_levels": {
            f"L{lvl}": info for lvl, info in sorted(levels.items())
        },
    }
    return result


class ParamGroup:
    """A named group of parameter tensors with hierarchy level."""

    def __init__(
        self,
        group_id: str,
        hierarchy_level: int,
        description: str,
        tensor_patterns: list[str],
        expected_convergence_order: str,
        dependent_metrics: list[str],
    ):
        self.group_id = group_id
        self.hierarchy_level = hierarchy_level
        self.description = description
        self.tensor_patterns = tensor_patterns
        self.expected_convergence_order = expected_convergence_order
        self.dependent_metrics = dependent_metrics

        # Populated by build_param_groups
        self.param_names: list[str] = []
        self.params: list[nn.Parameter] = []
        self.param_count: int = 0

    def matches(self, param_name: str) -> bool:
        """Check if a parameter name matches any pattern in this group."""
        for pattern in self.tensor_patterns:
            if fnmatch(param_name, pattern):
                return True
        return False

    def add_param(self, name: str, param: nn.Parameter):
        self.param_names.append(name)
        self.params.append(param)
        self.param_count += param.numel()

    def set_requires_grad(self, requires_grad: bool):
        """Toggle gradient computation for all parameters in this group."""
        for p in self.params:
            p.requires_grad_(requires_grad)

    def grad_norm(self) -> float:
        """Compute L2 norm of gradients across all parameters in the group."""
        total = 0.0
        for p in self.params:
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

    def __repr__(self) -> str:
        return (
            f"ParamGroup('{self.group_id}', L{self.hierarchy_level}, "
            f"{self.param_count:,} params, {len(self.param_names)} tensors)"
        )
