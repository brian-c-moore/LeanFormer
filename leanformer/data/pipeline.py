"""
DataPipeline — download, preprocess, and tokenize training data from HuggingFace.

Corpus composition:
- OpenWebText: web text, conversational and factual (40%)
- Wikipedia English: factual knowledge, structured (20%)
- BookCorpus: long-form reasoning, narrative (20%)
- Stack Exchange: Q&A, reasoning, technical (10%)
"""

from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer

from leanformer import DEFAULT_TOKENIZER


DATASET_CONFIGS = {
    "openwebtext": {
        "path": "openwebtext",
        "split": "train",
        "text_field": "text",
        "weight": 0.4,
    },
    "wikipedia": {
        "path": "wikipedia",
        "name": "20220301.en",
        "split": "train",
        "text_field": "text",
        "weight": 0.2,
    },
    "bookcorpus": {
        "path": "bookcorpus",
        "split": "train",
        "text_field": "text",
        "weight": 0.2,
    },
    "stack_exchange": {
        "path": "HuggingFaceH4/stack-exchange-preferences",
        "split": "train",
        "text_field": "question",
        "weight": 0.1,
    },
}


class DataPipeline:

    def __init__(
        self,
        tokenizer_name: str = DEFAULT_TOKENIZER,
        max_seq_len: int = 1024,
        cache_dir: str = "./data/cache",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_seq_len = max_seq_len
        self.cache_dir = cache_dir

    def load_and_tokenize(self, dataset_name: str, max_samples: int | None = None):
        """Load a single dataset and tokenize it."""
        config = DATASET_CONFIGS[dataset_name]

        kwargs = {
            "path": config["path"],
            "split": config["split"],
            "cache_dir": self.cache_dir,
            "trust_remote_code": True,
        }
        if "name" in config:
            kwargs["name"] = config["name"]

        dataset = load_dataset(**kwargs)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        text_field = config["text_field"]
        max_len = self.max_seq_len
        tokenizer = self.tokenizer

        def tokenize(examples):
            texts = examples[text_field]
            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=4,
            desc=f"Tokenizing {dataset_name}",
        )

    def build_training_dataset(self, samples_per_source: int = 100_000):
        """Build the interleaved training dataset from all sources."""
        datasets = []
        for name, config in DATASET_CONFIGS.items():
            print(f"Loading {name}...")
            ds = self.load_and_tokenize(name, max_samples=samples_per_source)
            ds = ds.with_format("torch")
            datasets.append(ds)

        weights = [DATASET_CONFIGS[n]["weight"] for n in DATASET_CONFIGS]
        combined = interleave_datasets(datasets, probabilities=weights, seed=42)
        return combined
