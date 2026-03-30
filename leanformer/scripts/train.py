"""Training entry point for LeanFormer."""

import sys
import yaml
import torch
from pathlib import Path
from transformers import TrainingArguments

from ..model.leanformer import LeanFormer
from ..model.config import LeanFormerConfig
from ..data.pipeline import DataPipeline
from ..training.trainer import LeanFormerTrainer
from ..training.callbacks import EfficiencyLogCallback, CheckpointConfigCallback


def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Build model
    model_config = LeanFormerConfig(**cfg["model"])
    model = LeanFormer(model_config)

    param_count = sum(p.numel() for p in model.parameters())
    dense_equiv = model_config.dense_equivalent_estimate()
    print(f"LeanFormer initialized:")
    print(f"  Parameters:       {param_count:,}")
    print(f"  Dense equivalent: {dense_equiv:,}")
    print(f"  Compression:      {dense_equiv / param_count:.1f}x")

    # Build dataset
    pipeline = DataPipeline(
        tokenizer_name=cfg["data"]["tokenizer"],
        max_seq_len=model_config.max_seq_len,
    )
    train_dataset = pipeline.build_training_dataset(
        samples_per_source=cfg["data"]["samples_per_source"],
    )

    # Training arguments
    output_dir = cfg["training"]["output_dir"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation"],
        learning_rate=cfg["training"]["lr"],
        warmup_steps=cfg["training"]["warmup_steps"],
        lr_scheduler_type="cosine",
        logging_steps=100,
        save_steps=1000,
        fp16=torch.cuda.is_available(),
        report_to=["wandb"] if cfg["training"].get("use_wandb", True) else [],
        run_name=cfg["training"].get("run_name", "leanformer"),
    )

    trainer = LeanFormerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[
            EfficiencyLogCallback(),
            CheckpointConfigCallback(model_config),
        ],
    )

    trainer.train()

    # Save final model + config
    trainer.save_model(output_dir)
    model_config.save(Path(output_dir) / "leanformer_config.json")
    print(f"Training complete. Model saved to {output_dir}")


def main():
    if len(sys.argv) < 2:
        print("Usage: leanformer-train <config.yaml>")
        sys.exit(1)
    train(sys.argv[1])


if __name__ == "__main__":
    main()
