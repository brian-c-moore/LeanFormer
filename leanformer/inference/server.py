"""FastAPI inference server for LeanFormer."""

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from .engine import LeanFormerInference

app = FastAPI(title="LeanFormer Inference Server")
engine: LeanFormerInference | None = None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    time_seconds: float
    tokens_per_second: float
    avg_depth_utilization: float
    avg_exit_layer: float
    total_layers: int


@app.on_event("startup")
async def startup():
    global engine
    engine = LeanFormerInference("./checkpoints/small")


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    result = engine.generate(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
    )
    return GenerateResponse(**result)


@app.get("/stats")
async def stats():
    return engine.model.get_efficiency_stats()


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
