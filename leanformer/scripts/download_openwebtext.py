"""
Download and tokenize OpenWebText (500K samples) for scale training.
Saves tokenized dataset to disk for reuse.

This is a one-time operation. The tokenized dataset is saved to
data/openwebtext-500k-tokenized/ and loaded directly during training.
"""

import sys
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

from .. import DEFAULT_TOKENIZER


OUTPUT_DIR = Path("data/openwebtext-500k-tokenized")
MAX_SAMPLES = 500_000
MAX_SEQ_LEN = 512


def main():
    print("=== OpenWebText Download & Tokenization ===\n")

    if OUTPUT_DIR.exists():
        print(f"Tokenized dataset already exists at {OUTPUT_DIR}")
        print("Delete it manually if you want to re-download.")
        sys.exit(0)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer: GPT-2 (vocab size: {tokenizer.vocab_size})")

    # Download
    print(f"\nDownloading OpenWebText ({MAX_SAMPLES:,} samples)...")
    print("This may take a while on first download (~40GB full dataset).")
    dataset = load_dataset("openwebtext", split="train")
    print(f"Full dataset size: {len(dataset):,} samples")

    dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
    print(f"Selected: {len(dataset):,} samples")

    # Tokenize
    print(f"\nTokenizing (max_seq_len={MAX_SEQ_LEN})...")

    def tokenize(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(
        tokenize,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        num_proc=1,  # Single process to avoid Windows file locking issues
        desc="Tokenizing",
    )

    # Save to disk
    print(f"\nSaving to {OUTPUT_DIR}...")
    tokenized.save_to_disk(str(OUTPUT_DIR))

    print(f"\nDataset size: {len(tokenized):,} samples")
    print(f"Tokens: ~{len(tokenized) * MAX_SEQ_LEN / 1e6:.0f}M")
    print(f"Saved to: {OUTPUT_DIR}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
