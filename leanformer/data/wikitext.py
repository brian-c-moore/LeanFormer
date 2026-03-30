"""
WikiText-2 data loader — the standard validation dataset for LeanFormer.

WikiText-2 is a standard language modeling benchmark (~2M training tokens).
It is downloaded once to data/wikitext-2/ and reused for all validation,
testing, and demo purposes.

Usage:
    loader = WikiTextLoader()
    train_ds = loader.get_tokenized_split("train")
    val_ds = loader.get_tokenized_split("validation")
"""

from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import torch

from leanformer import DEFAULT_TOKENIZER


# Default cache location relative to project root
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "wikitext-2"


class WikiTextLoader:
    """Loads and tokenizes WikiText-2 from a local cache."""

    def __init__(
        self,
        tokenizer_name: str = DEFAULT_TOKENIZER,
        max_seq_len: int = 256,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_seq_len = max_seq_len
        self.cache_dir = Path(cache_dir)

    def _load_raw(self, split: str) -> Dataset:
        """Load raw WikiText-2 from cache (downloads on first call)."""
        return load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split=split,
            cache_dir=str(self.cache_dir),
        )

    def get_tokenized_split(
        self,
        split: str = "train",
        max_samples: int | None = None,
    ) -> Dataset:
        """
        Get a tokenized split ready for training/evaluation.

        Returns a HuggingFace Dataset with columns:
            input_ids: list[int] (length max_seq_len)
            attention_mask: list[int]
            labels: list[int] (same as input_ids for causal LM)
        """
        raw = self._load_raw(split)

        # Filter out empty lines and section headers
        raw = raw.filter(
            lambda x: len(x["text"].strip()) > 20 and not x["text"].strip().startswith("="),
            desc=f"Filtering {split}",
        )

        if max_samples:
            raw = raw.select(range(min(max_samples, len(raw))))

        tokenizer = self.tokenizer
        max_len = self.max_seq_len

        def tokenize(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized = raw.map(
            tokenize,
            batched=True,
            remove_columns=raw.column_names,
            desc=f"Tokenizing {split}",
        )

        tokenized = tokenized.with_format("torch")
        return tokenized

    def get_concatenated_split(
        self,
        split: str = "train",
        max_samples: int | None = None,
    ) -> Dataset:
        """
        Concatenate all text into a single stream and chunk into fixed-length
        sequences. This is the standard approach for LM training — no padding
        waste, every token is useful.
        """
        raw = self._load_raw(split)

        # Filter empty lines
        raw = raw.filter(lambda x: len(x["text"].strip()) > 0)

        # Concatenate all text
        all_text = " ".join(raw["text"])

        # Tokenize the full stream
        tokens = self.tokenizer.encode(all_text)

        # Chunk into fixed-length sequences
        chunks = []
        for i in range(0, len(tokens) - self.max_seq_len, self.max_seq_len):
            chunk = tokens[i : i + self.max_seq_len]
            chunks.append({"input_ids": chunk, "labels": chunk})

        if max_samples:
            chunks = chunks[:max_samples]

        ds = Dataset.from_list(chunks)
        ds = ds.with_format("torch")
        return ds

    def info(self) -> dict:
        """Return dataset statistics."""
        train = self._load_raw("train")
        val = self._load_raw("validation")
        test = self._load_raw("test")

        # Count total tokens
        all_train_text = " ".join(t for t in train["text"] if t.strip())
        train_tokens = len(self.tokenizer.encode(all_train_text))

        return {
            "dataset": "WikiText-2 (wikitext-2-raw-v1)",
            "train_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
            "approx_train_tokens": train_tokens,
            "tokenizer": DEFAULT_TOKENIZER,
            "max_seq_len": self.max_seq_len,
            "cache_dir": str(self.cache_dir),
        }
