"""
FastAPI inference server with Knowledge Plane integration.

Endpoints:
  POST /infer          - inference with knowledge routing
  POST /infer/base     - inference without knowledge (baseline comparison)
  GET  /deltas         - list registered deltas
  GET  /deltas/{id}    - delta details
  GET  /capacity       - registry capacity report
  GET  /health         - health check + base model hash verification
  POST /consolidate    - trigger consolidation for a category

Usage:
    python -m leanformer.knowledge_plane.server \
        --model-checkpoint checkpoints/reasoning_core \
        --registry-path deltas/registry.json
"""

import argparse
import json
import sys
import time
import torch
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from .runtime import KnowledgeRuntime
from .registry import DeltaRegistry
from .consolidation import KnowledgePlaneConsolidator
from .dfs import compute_model_hash


# === Request/Response Models ===

if HAS_FASTAPI:

    class InferRequest(BaseModel):
        prompt: str
        max_new_tokens: int = 50

    class InferResponse(BaseModel):
        generated_text: str
        active_deltas: list
        categories_activated: list
        composition_layers: list
        inference_time_ms: float
        base_only: bool

    class ConsolidateRequest(BaseModel):
        category: str


def create_app(
    model_checkpoint: str,
    registry_path: str,
    model_hash_path: Optional[str] = None,
) -> "FastAPI":
    """Create and configure the FastAPI application."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    from transformers import GPT2TokenizerFast
    from ..model.config import LeanFormerConfig
    from ..model.leanformer import LeanFormer

    app = FastAPI(title="LeanFormer Knowledge Plane Server", version="3.0")

    # Load model
    ckpt_dir = Path(model_checkpoint)
    config = LeanFormerConfig.load(ckpt_dir / "leanformer_config.json")
    model = LeanFormer(config).cuda()
    state_dict = torch.load(
        ckpt_dir / "pytorch_model.bin",
        map_location="cuda",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load registry
    registry = DeltaRegistry.load(registry_path)

    # Load model hash
    expected_hash = None
    if model_hash_path and Path(model_hash_path).exists():
        expected_hash = Path(model_hash_path).read_text().strip()

    # Create runtime
    runtime = KnowledgeRuntime(
        model=model,
        tokenizer=tokenizer,
        registry=registry,
        top_k=5,
        device="cuda",
    )

    # Provenance log
    provenance_log = []

    @app.post("/infer", response_model=InferResponse)
    def infer(req: InferRequest):
        result = runtime.infer(req.prompt, max_new_tokens=req.max_new_tokens)

        entry = {
            "timestamp": time.time(),
            "prompt": req.prompt,
            "active_deltas": result.active_deltas,
            "categories": result.categories_activated,
            "layers": result.composition_layers,
            "time_ms": result.inference_time_ms,
        }
        provenance_log.append(entry)

        return InferResponse(
            generated_text=result.generated_text,
            active_deltas=[(did, score) for did, score in result.active_deltas],
            categories_activated=result.categories_activated,
            composition_layers=result.composition_layers,
            inference_time_ms=result.inference_time_ms,
            base_only=result.base_only,
        )

    @app.post("/infer/base", response_model=InferResponse)
    def infer_base(req: InferRequest):
        result = runtime.infer(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            use_knowledge=False,
        )
        return InferResponse(
            generated_text=result.generated_text,
            active_deltas=[],
            categories_activated=[],
            composition_layers=[],
            inference_time_ms=result.inference_time_ms,
            base_only=True,
        )

    @app.get("/deltas")
    def list_deltas():
        deltas = []
        for did, d in registry.registered_deltas.items():
            deltas.append({
                "delta_id": did,
                "category": d.category,
                "description": d.description,
                "target_layers": d.target_layers,
                "param_count": d.param_count,
                "orthogonality_score": d.orthogonality_score,
                "confidence": d.confidence,
            })
        return {"count": len(deltas), "deltas": deltas}

    @app.get("/deltas/{delta_id}")
    def get_delta(delta_id: str):
        if delta_id not in registry.registered_deltas:
            raise HTTPException(404, f"Delta {delta_id} not found")
        d = registry.registered_deltas[delta_id]
        return {
            "delta_id": d.delta_id,
            "version": d.version,
            "category": d.category,
            "description": d.description,
            "target_layers": d.target_layers,
            "delta_rank": d.delta_rank,
            "param_count": d.param_count,
            "orthogonality_score": d.orthogonality_score,
            "confidence": d.confidence,
            "tags": d.tags,
            "composable_with": d.composable_with,
            "incompatible_with": d.incompatible_with,
        }

    @app.get("/capacity")
    def capacity():
        remaining = registry.remaining_capacity_estimate()
        return {
            "registered_deltas": registry.count,
            "theoretical_max_per_layer": registry.theoretical_max_per_layer,
            "total_remaining": registry.total_remaining_capacity(),
            "per_layer_remaining": remaining,
        }

    @app.get("/health")
    def health():
        result = {"status": "ok", "model_loaded": True, "registry_count": registry.count}
        if expected_hash:
            result["base_hash_verified"] = runtime.verify_base_weight_integrity(expected_hash)
        return result

    @app.post("/consolidate")
    def consolidate(req: ConsolidateRequest):
        consolidator = KnowledgePlaneConsolidator(registry, model, tokenizer)
        deltas_in_cat = [
            did for did, d in registry.registered_deltas.items()
            if d.category == req.category
        ]
        if len(deltas_in_cat) < 2:
            raise HTTPException(400, f"Not enough deltas in category '{req.category}'")

        merged = consolidator.consolidate_and_replace(deltas_in_cat)
        if merged is None:
            raise HTTPException(500, "Consolidation failed")

        return {
            "merged_delta_id": merged.delta_id,
            "deltas_merged": len(deltas_in_cat),
            "new_registry_count": registry.count,
            "capacity_remaining": registry.total_remaining_capacity(),
        }

    @app.get("/provenance")
    def provenance():
        return {"entries": len(provenance_log), "log": provenance_log[-100:]}

    return app


def main():
    if not HAS_FASTAPI:
        print("ERROR: FastAPI not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--registry-path", required=True)
    parser.add_argument("--model-hash-path", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(
        args.model_checkpoint,
        args.registry_path,
        args.model_hash_path,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
