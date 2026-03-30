"""Inference server entry point."""

import sys


def main():
    import uvicorn
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints/small"
    # Override the default model path via environment variable
    import os
    os.environ["LEANFORMER_MODEL_PATH"] = model_path
    uvicorn.run("leanformer.inference.server:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
