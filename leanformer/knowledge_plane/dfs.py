import uuid
import json
import zipfile
import io
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional
import torch
from pathlib import Path


@dataclass
class DeltaFormatSpec:
    """
    The contract between the reasoning core, knowledge forge,
    and knowledge runtime.

    Every belief delta in the system conforms to this format.
    Any tool that produces this format is a valid forge.
    Any system that consumes this format is a valid runtime.
    """

    # === Identity ===
    delta_id: str                          # UUID4 string
    version: int                           # Monotonically increasing
    created_at: str                        # ISO 8601 timestamp string
    source: str                            # Provenance: what produced this delta

    # === Routing ===
    embedding: torch.Tensor                # Shape: (d_model,) - for routing
    category: str                          # Human-readable category tag
    description: str                       # What this delta encodes
    domain_tags: List[str]                 # Fine-grained labels

    # === Content ===
    target_layers: List[int]               # Which layers this delta modifies
    factors: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    # Per-layer: (A_delta, B_delta) low-rank factors
    # A_delta shape: (d_model, delta_rank)
    # B_delta shape: (delta_rank, d_model)
    # Applied as: layer_output += x @ A_delta @ B_delta

    delta_rank: int                        # Rank of the delta factors
    param_count: int                       # Total parameter count

    # === Orthogonality Record ===
    subspace_basis: torch.Tensor           # Shape: (d_model, delta_rank)
    orthogonality_score: float             # Min angle vs existing deltas (0-1)
    conflicting_delta_ids: List[str]       # Deltas with conflicts at forge time

    # === Compatibility ===
    base_model_hash: str                   # SHA-256 of base model weights
    d_model: int                           # Must match base model
    n_layers: int                          # Must match base model
    format_version: str = "2.0"

    # === Validation Record ===
    confidence: float = 0.0                # Encoder confidence (0-1)
    validation_results: Dict = field(default_factory=dict)
    forge_attempts: int = 1

    # === Metadata ===
    tags: List[str] = field(default_factory=list)
    composable_with: List[str] = field(default_factory=list)
    incompatible_with: List[str] = field(default_factory=list)

    @staticmethod
    def create_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def compute_param_count(self) -> int:
        """Recompute param_count from factors."""
        total = 0
        for layer_idx in self.target_layers:
            if layer_idx in self.factors:
                A, B = self.factors[layer_idx]
                total += A.numel() + B.numel()
        return total

    def validate_shapes(self) -> List[str]:
        """
        Validate all tensor shapes are consistent.
        Returns list of error strings (empty = valid).
        """
        errors = []
        if self.embedding.shape != (self.d_model,):
            errors.append(
                f"Embedding shape {self.embedding.shape} != ({self.d_model},)"
            )
        if self.subspace_basis.shape != (self.d_model, self.delta_rank):
            errors.append(
                f"Subspace basis shape {self.subspace_basis.shape} "
                f"!= ({self.d_model}, {self.delta_rank})"
            )
        for layer_idx in self.target_layers:
            if layer_idx not in self.factors:
                errors.append(f"Missing factors for target layer {layer_idx}")
                continue
            A, B = self.factors[layer_idx]
            if A.shape != (self.d_model, self.delta_rank):
                errors.append(
                    f"Layer {layer_idx} A shape {A.shape} "
                    f"!= ({self.d_model}, {self.delta_rank})"
                )
            if B.shape != (self.delta_rank, self.d_model):
                errors.append(
                    f"Layer {layer_idx} B shape {B.shape} "
                    f"!= ({self.delta_rank}, {self.d_model})"
                )
        return errors

    def save(self, path: str):
        """
        Serialize to a .delta file (ZIP archive).

        Contents:
          manifest.json - all non-tensor fields
          embedding.pt - routing embedding
          subspace_basis.pt - orthogonality basis
          factors/{layer_idx}_A.pt - per-layer A factor
          factors/{layer_idx}_B.pt - per-layer B factor
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        manifest = {
            "delta_id": self.delta_id,
            "version": self.version,
            "created_at": self.created_at,
            "source": self.source,
            "category": self.category,
            "description": self.description,
            "domain_tags": self.domain_tags,
            "target_layers": self.target_layers,
            "delta_rank": self.delta_rank,
            "param_count": self.param_count,
            "orthogonality_score": self.orthogonality_score,
            "conflicting_delta_ids": self.conflicting_delta_ids,
            "base_model_hash": self.base_model_hash,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "format_version": self.format_version,
            "confidence": self.confidence,
            "validation_results": self.validation_results,
            "forge_attempts": self.forge_attempts,
            "tags": self.tags,
            "composable_with": self.composable_with,
            "incompatible_with": self.incompatible_with,
        }

        with zipfile.ZipFile(str(path), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

            # Save tensors as .pt files inside the zip
            buf = io.BytesIO()
            torch.save(self.embedding, buf)
            zf.writestr("embedding.pt", buf.getvalue())

            buf = io.BytesIO()
            torch.save(self.subspace_basis, buf)
            zf.writestr("subspace_basis.pt", buf.getvalue())

            for layer_idx in self.target_layers:
                A, B = self.factors[layer_idx]
                buf = io.BytesIO()
                torch.save(A, buf)
                zf.writestr(f"factors/{layer_idx}_A.pt", buf.getvalue())

                buf = io.BytesIO()
                torch.save(B, buf)
                zf.writestr(f"factors/{layer_idx}_B.pt", buf.getvalue())

    @classmethod
    def load(cls, path: str) -> "DeltaFormatSpec":
        """Load a .delta file and reconstruct the DeltaFormatSpec."""
        with zipfile.ZipFile(str(path), "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))

            embedding = torch.load(
                io.BytesIO(zf.read("embedding.pt")), weights_only=True
            )
            subspace_basis = torch.load(
                io.BytesIO(zf.read("subspace_basis.pt")), weights_only=True
            )

            factors = {}
            for layer_idx in manifest["target_layers"]:
                A = torch.load(
                    io.BytesIO(zf.read(f"factors/{layer_idx}_A.pt")),
                    weights_only=True,
                )
                B = torch.load(
                    io.BytesIO(zf.read(f"factors/{layer_idx}_B.pt")),
                    weights_only=True,
                )
                factors[layer_idx] = (A, B)

            return cls(
                delta_id=manifest["delta_id"],
                version=manifest["version"],
                created_at=manifest["created_at"],
                source=manifest["source"],
                embedding=embedding,
                category=manifest["category"],
                description=manifest["description"],
                domain_tags=manifest["domain_tags"],
                target_layers=manifest["target_layers"],
                factors=factors,
                delta_rank=manifest["delta_rank"],
                param_count=manifest["param_count"],
                subspace_basis=subspace_basis,
                orthogonality_score=manifest["orthogonality_score"],
                conflicting_delta_ids=manifest["conflicting_delta_ids"],
                base_model_hash=manifest["base_model_hash"],
                d_model=manifest["d_model"],
                n_layers=manifest["n_layers"],
                format_version=manifest["format_version"],
                confidence=manifest.get("confidence", 0.0),
                validation_results=manifest.get("validation_results", {}),
                forge_attempts=manifest.get("forge_attempts", 1),
                tags=manifest.get("tags", []),
                composable_with=manifest.get("composable_with", []),
                incompatible_with=manifest.get("incompatible_with", []),
            )


def compute_model_hash(model: torch.nn.Module) -> str:
    """
    Compute SHA-256 hash of all model parameters.
    Used to lock base model identity - deltas are only compatible
    with the specific base model they were forged against.
    """
    import hashlib
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()
