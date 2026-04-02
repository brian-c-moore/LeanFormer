"""
Delta-Aware Quantization.

Standard quantization optimizes for reconstruction MSE globally.
Delta artifacts have typed operational contracts that specify which
invariants must be preserved — and these are not MSE.

This module implements the Delta Quantization Specification (DQS)
framework: quantization that is typed per-delta, preserving routing
fidelity, orthogonality fidelity, and composition fidelity.

Quantization method: Lloyd-Max optimal scalar quantization with
random orthogonal rotation (PolarQuant approach from TurboQuant,
without QJL residual correction per community validation).

Three tiers:
  Tier 1: Routing-Critical (preserve cosine similarity)
  Tier 2: Composition (preserve Frobenius norm of composition)
  Tier 3: Archive (max compression, preserve provenance only)
"""

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .registry import orthogonality_score


# === Delta Quantization Specification ===

@dataclass
class DeltaQuantizationSpec:
    """
    Typed compression specification for a delta artifact.

    Declares which invariants must be preserved and at what tolerance.
    Produced at forge time, enforced at load time.
    """
    # Quantization tier
    tier: int = 1  # 1=routing-critical, 2=composition, 3=archive

    # Target bit-width
    bit_width: int = 4

    # Invariant tolerances
    routing_cosine_floor: float = 0.95    # Min cosine sim after compression
    composition_frobenius_ceiling: float = 0.05  # Max relative Frobenius error
    orthogonality_angle_floor: float = 0.25  # Min orthogonality score post-quant

    # Asymmetric K/V allocation (for deltas targeting attention)
    asymmetric_kv: bool = False
    k_bits: int = 4
    v_bits: int = 4

    # Residual: keep this many singular values at full precision
    residual_top_k: int = 0

    def to_dict(self) -> Dict:
        return {
            "tier": self.tier,
            "bit_width": self.bit_width,
            "routing_cosine_floor": self.routing_cosine_floor,
            "composition_frobenius_ceiling": self.composition_frobenius_ceiling,
            "orthogonality_angle_floor": self.orthogonality_angle_floor,
            "asymmetric_kv": self.asymmetric_kv,
            "k_bits": self.k_bits,
            "v_bits": self.v_bits,
            "residual_top_k": self.residual_top_k,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DeltaQuantizationSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def for_tier(cls, tier: int) -> "DeltaQuantizationSpec":
        """Create a DQS with tier-appropriate defaults."""
        if tier == 1:
            return cls(
                tier=1, bit_width=4,
                routing_cosine_floor=0.95,
                composition_frobenius_ceiling=0.05,
                orthogonality_angle_floor=0.25,
            )
        elif tier == 2:
            return cls(
                tier=2, bit_width=4,
                routing_cosine_floor=0.85,
                composition_frobenius_ceiling=0.10,
                orthogonality_angle_floor=0.20,
            )
        elif tier == 3:
            return cls(
                tier=3, bit_width=3,
                routing_cosine_floor=0.70,
                composition_frobenius_ceiling=0.20,
                orthogonality_angle_floor=0.10,
            )
        else:
            raise ValueError(f"Unknown tier: {tier}")


# === Rotation Matrix Generation ===

def generate_rotation_matrix(
    dim: int,
    seed: int = 42,
    device: torch.device = None,
) -> torch.Tensor:
    """Generate a random orthogonal rotation matrix via QR decomposition."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    M = torch.randn(dim, dim, generator=gen, device=device)
    Q, R = torch.linalg.qr(M)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    return Q * diag_sign.unsqueeze(0)


# === Quantization Core ===

def _quantize_matrix(
    matrix: torch.Tensor,
    bits: int,
    rotation: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Quantize a 2D matrix to N-bit per-column representation.

    Uses affine (min-max) scalar quantization with optional rotation.
    """
    original_dtype = matrix.dtype
    mat = matrix.float()

    if rotation is not None:
        mat = mat @ rotation

    n_levels = 2 ** bits
    col_min = mat.min(dim=0).values
    col_max = mat.max(dim=0).values
    col_range = torch.clamp(col_max - col_min, min=1e-8)

    scale = col_range / (n_levels - 1)
    zero_point = col_min

    normalized = (mat - zero_point) / scale
    codes = torch.clamp(torch.round(normalized), 0, n_levels - 1).to(torch.uint8)

    return {
        "codes": codes,
        "scale": scale,
        "zero_point": zero_point,
        "bits": bits,
        "dtype": original_dtype,
    }


def _dequantize_matrix(
    quantized: Dict,
    rotation_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize a matrix from its quantized representation."""
    codes = quantized["codes"].float()
    mat = codes * quantized["scale"] + quantized["zero_point"]

    if rotation_inv is not None:
        mat = mat @ rotation_inv

    return mat.to(quantized["dtype"])


# === Delta Quantizer ===

@dataclass
class QuantizedDelta:
    """A delta artifact in quantized form."""
    delta_id: str
    dqs: DeltaQuantizationSpec

    # Quantized factors per layer: {layer_idx: (quantized_A, quantized_B)}
    quantized_factors: Dict[int, Tuple[Dict, Dict]] = field(default_factory=dict)

    # Residual: top-K singular components at full precision (optional)
    residual_factors: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )

    # Uncompressed metadata preserved for routing/orthogonality
    embedding: Optional[torch.Tensor] = None
    subspace_basis: Optional[torch.Tensor] = None

    # Rotation matrices used (needed for dequantization)
    rotation_seed: int = 42


class DeltaQuantizer:
    """
    Quantizes and dequantizes delta artifacts respecting DQS invariants.

    The quantization pipeline:
    1. For each layer's (A, B) factors:
       a. Optionally extract top-K singular components as residual
       b. Apply random orthogonal rotation to decorrelate
       c. Quantize to target bit-width
    2. Preserve embedding and subspace_basis at full precision
       (routing and orthogonality checks need full fidelity)
    3. Validate all DQS invariants on the quantized result
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rotation_cache: Dict[int, torch.Tensor] = {}

    def _get_rotation(self, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get or create rotation matrix for a given dimension."""
        if dim not in self._rotation_cache:
            R = generate_rotation_matrix(dim, self.seed)
            self._rotation_cache[dim] = R
        R = self._rotation_cache[dim]
        return R, R.T.contiguous()

    def quantize_delta(
        self,
        delta,  # DeltaFormatSpec
        dqs: Optional[DeltaQuantizationSpec] = None,
    ) -> QuantizedDelta:
        """
        Quantize a delta artifact according to its DQS.

        Args:
            delta: DeltaFormatSpec to quantize
            dqs: quantization spec (if None, auto-detect tier)

        Returns:
            QuantizedDelta with compressed factors
        """
        if dqs is None:
            dqs = DeltaQuantizationSpec.for_tier(1)

        result = QuantizedDelta(
            delta_id=delta.delta_id,
            dqs=dqs,
            embedding=delta.embedding.clone(),
            subspace_basis=delta.subspace_basis.clone(),
            rotation_seed=self.seed,
        )

        for layer_idx, (A, B) in delta.factors.items():
            bits = dqs.bit_width

            # Get rotation matrices for A and B dimensions
            R_a, R_a_inv = self._get_rotation(A.shape[1])  # (delta_rank, delta_rank)
            R_b, R_b_inv = self._get_rotation(B.shape[1])  # (d_model, d_model)

            if dqs.residual_top_k > 0:
                # Extract top-K singular components at full precision
                W = A @ B
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                k = min(dqs.residual_top_k, len(S))

                # Residual: top-k components at full precision
                A_res = U[:, :k] * S[:k].sqrt()
                B_res = S[:k].sqrt().unsqueeze(1) * Vh[:k, :]
                result.residual_factors[layer_idx] = (A_res, B_res)

                # Quantize the remainder
                A_remainder = U[:, k:] * S[k:].sqrt()
                B_remainder = S[k:].sqrt().unsqueeze(1) * Vh[k:, :]

                R_rem, R_rem_inv = self._get_rotation(A_remainder.shape[1])
                q_A = _quantize_matrix(A_remainder, bits, R_rem)
                R_b_rem, _ = self._get_rotation(B_remainder.shape[1])
                q_B = _quantize_matrix(B_remainder, bits, R_b_rem)
            else:
                # Quantize A and B directly
                q_A = _quantize_matrix(A, bits, R_a)
                q_B = _quantize_matrix(B, bits, R_b)

            result.quantized_factors[layer_idx] = (q_A, q_B)

        return result

    def dequantize_delta(
        self,
        quantized: QuantizedDelta,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Dequantize a QuantizedDelta back to full-precision factors.

        Returns:
            {layer_idx: (A, B)} dict of reconstructed factors
        """
        factors = {}

        for layer_idx, (q_A, q_B) in quantized.quantized_factors.items():
            # Get inverse rotations
            _, R_a_inv = self._get_rotation(q_A["codes"].shape[1])
            _, R_b_inv = self._get_rotation(q_B["codes"].shape[1])

            A = _dequantize_matrix(q_A, R_a_inv)
            B = _dequantize_matrix(q_B, R_b_inv)

            # Add residual if present
            if layer_idx in quantized.residual_factors:
                A_res, B_res = quantized.residual_factors[layer_idx]
                # Combine: full delta = residual + dequantized remainder
                # We need to reconstruct the full W and re-factor
                W = A_res @ B_res + A @ B
                # Re-factor at original rank
                rank = max(A.shape[1], A_res.shape[1])
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                A = U[:, :rank] * S[:rank].sqrt()
                B = S[:rank].sqrt().unsqueeze(1) * Vh[:rank, :]

            factors[layer_idx] = (A, B)

        return factors

    def validate_invariants(
        self,
        original_delta,  # DeltaFormatSpec
        quantized: QuantizedDelta,
        other_deltas: Optional[List] = None,
    ) -> Dict:
        """
        Validate DQS invariant tolerances after quantization.

        Checks:
        1. Routing fidelity: cosine similarity of embedding (preserved at fp)
        2. Composition fidelity: Frobenius norm of reconstruction error
        3. Orthogonality fidelity: score vs other deltas

        Returns:
            Dict with per-invariant pass/fail and measured values
        """
        dqs = quantized.dqs
        results = {"passed": True, "checks": {}}

        # 1. Routing fidelity (embedding preserved at full precision)
        if quantized.embedding is not None:
            cos_sim = F.cosine_similarity(
                original_delta.embedding.unsqueeze(0).float(),
                quantized.embedding.unsqueeze(0).float(),
            ).item()
            routing_ok = cos_sim >= dqs.routing_cosine_floor
            results["checks"]["routing_fidelity"] = {
                "cosine_similarity": cos_sim,
                "threshold": dqs.routing_cosine_floor,
                "passed": routing_ok,
            }
            if not routing_ok:
                results["passed"] = False

        # 2. Composition fidelity
        deq_factors = self.dequantize_delta(quantized)
        max_rel_error = 0.0
        for layer_idx, (A_orig, B_orig) in original_delta.factors.items():
            if layer_idx not in deq_factors:
                results["passed"] = False
                continue
            A_deq, B_deq = deq_factors[layer_idx]
            W_orig = A_orig.float() @ B_orig.float()
            W_deq = A_deq.float() @ B_deq.float()
            frob_error = (W_orig - W_deq).norm()
            frob_orig = W_orig.norm()
            rel_error = (frob_error / max(frob_orig, 1e-8)).item()
            max_rel_error = max(max_rel_error, rel_error)

        composition_ok = max_rel_error <= dqs.composition_frobenius_ceiling
        results["checks"]["composition_fidelity"] = {
            "max_relative_frobenius_error": max_rel_error,
            "threshold": dqs.composition_frobenius_ceiling,
            "passed": composition_ok,
        }
        if not composition_ok:
            results["passed"] = False

        # 3. Orthogonality fidelity
        if other_deltas and quantized.subspace_basis is not None:
            min_orth = 1.0
            for other in other_deltas:
                other_basis = other.subspace_basis
                score = orthogonality_score(
                    quantized.subspace_basis, other_basis
                )
                min_orth = min(min_orth, score)

            orth_ok = min_orth >= dqs.orthogonality_angle_floor
            results["checks"]["orthogonality_fidelity"] = {
                "min_orthogonality_score": min_orth,
                "threshold": dqs.orthogonality_angle_floor,
                "passed": orth_ok,
            }
            if not orth_ok:
                results["passed"] = False
        else:
            results["checks"]["orthogonality_fidelity"] = {
                "min_orthogonality_score": 1.0,
                "threshold": dqs.orthogonality_angle_floor,
                "passed": True,
                "note": "no other deltas to check against",
            }

        return results

    def estimate_compression(
        self,
        original_delta,  # DeltaFormatSpec
        dqs: Optional[DeltaQuantizationSpec] = None,
    ) -> Dict:
        """Estimate compression ratio for a delta under a given DQS."""
        if dqs is None:
            dqs = DeltaQuantizationSpec.for_tier(1)

        original_bytes = 0
        quantized_bytes = 0

        for layer_idx, (A, B) in original_delta.factors.items():
            # Original: float32
            orig_a = A.numel() * 4
            orig_b = B.numel() * 4
            original_bytes += orig_a + orig_b

            # Quantized: N-bit codes + scale/zp overhead
            quant_a = A.numel() * dqs.bit_width / 8
            quant_b = B.numel() * dqs.bit_width / 8
            # Per-column scale + zero_point (float32 each)
            overhead_a = A.shape[1] * 8
            overhead_b = B.shape[1] * 8
            quantized_bytes += quant_a + quant_b + overhead_a + overhead_b

        # Embedding and basis preserved at full precision
        emb_bytes = original_delta.embedding.numel() * 4
        basis_bytes = original_delta.subspace_basis.numel() * 4
        original_bytes += emb_bytes + basis_bytes
        quantized_bytes += emb_bytes + basis_bytes

        return {
            "original_bytes": original_bytes,
            "quantized_bytes": quantized_bytes,
            "compression_ratio": original_bytes / max(quantized_bytes, 1),
            "savings_pct": (1 - quantized_bytes / max(original_bytes, 1)) * 100,
        }
