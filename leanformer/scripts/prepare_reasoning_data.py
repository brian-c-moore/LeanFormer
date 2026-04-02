"""
Download, filter, tokenize, and interleave reasoning-focused training data.

Produces a single tokenized dataset at ./data/reasoning-core-tokenized/
with ~250M tokens at max_seq_len=512.

Data sources (knowledge-naive — structural reasoning, not factual):
  - Code: bigcode/starcoderdata (Python, JavaScript) — requires HF_TOKEN
  - Math: open-web-math/open-web-math
  - Scientific: scientific_papers (arxiv, abstracts)
  - Prose: wikipedia (long articles as quality proxy)
  - Instruction: Open-Orca/SlimOrca

Set HF_TOKEN environment variable for gated datasets:
    export HF_TOKEN=your_token_here

Usage:
    python -m leanformer.scripts.prepare_reasoning_data
"""

import sys
import gc
import shutil
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from transformers import AutoTokenizer

from .. import DEFAULT_TOKENIZER

OUTPUT_DIR = Path("data/reasoning-core-tokenized")
MAX_SEQ_LEN = 512
TEMP_DIR = Path("data/_reasoning_temp")

# Target samples per source (~488K total for ~250M tokens)
TARGETS = {
    "code": 146_000,       # 30%
    "math": 73_000,        # 15%
    "science": 73_000,     # 15%
    "prose": 122_000,      # 25%
    "instruction": 73_000, # 15%
}


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_and_save(dataset, tokenizer, text_field, name, target_samples):
    """Tokenize a dataset and save to temp dir. Returns path."""
    print(f"\n  Tokenizing {name} ({len(dataset):,} samples, field='{text_field}')...")

    n = min(target_samples, len(dataset))
    dataset = dataset.select(range(n))

    # Tokenize inline to avoid multiprocess serialization issues on Windows
    all_ids = []
    for i, ex in enumerate(dataset):
        enc = tokenizer(ex[text_field], truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
        all_ids.append({"input_ids": enc["input_ids"], "labels": enc["input_ids"]})
        if i % 30000 == 0 and i > 0:
            print(f"    Tokenized {i:,}...")

    tokenized = Dataset.from_list(all_ids)
    save_path = TEMP_DIR / name
    tokenized.save_to_disk(str(save_path))
    print(f"  Saved {len(tokenized):,} samples to {save_path}")
    return save_path


def load_code(tokenizer):
    """Source 1: Code (30%) — Python + JavaScript from starcoderdata."""
    print("\n=== Source 1: Code (30%) ===")

    # Try starcoderdata (needs HF_TOKEN + dataset access approval)
    try:
        print("  Loading bigcode/starcoderdata (Python)...")
        ds = load_dataset("bigcode/starcoderdata", data_dir="python", split="train", streaming=True)
        samples = []
        for i, example in enumerate(ds):
            if len(samples) >= 100_000:
                break
            content = example.get("content", "")
            if len(content) > 100:
                samples.append({"text": content})
            if i % 20000 == 0 and i > 0:
                print(f"    {len(samples):,} Python samples from {i:,} scanned...")
        print(f"  Python: {len(samples):,} samples")

        print("  Loading bigcode/starcoderdata (JavaScript)...")
        ds_js = load_dataset("bigcode/starcoderdata", data_dir="javascript", split="train", streaming=True)
        for i, example in enumerate(ds_js):
            if len(samples) >= TARGETS["code"]:
                break
            content = example.get("content", "")
            if len(content) > 100:
                samples.append({"text": content})
            if i % 20000 == 0 and i > 0:
                print(f"    Total {len(samples):,} samples from {i:,} JS scanned...")
        print(f"  Total code: {len(samples):,} samples")

        dataset = Dataset.from_list(samples[:TARGETS["code"]])
        return tokenize_and_save(dataset, tokenizer, "text", "code", TARGETS["code"])

    except Exception as e:
        print(f"  starcoderdata failed: {e}")
        print("  Falling back to openwebtext code-like samples...")
        print("  TIP: Set HF_TOKEN and accept terms at https://huggingface.co/datasets/bigcode/starcoderdata")
        ds = load_dataset("openwebtext", split="train", streaming=True)
        samples = []
        for i, example in enumerate(ds):
            if len(samples) >= TARGETS["code"]:
                break
            text = example.get("text", "")
            if any(kw in text for kw in ["def ", "function ", "class ", "import "]):
                samples.append({"text": text})
            if i > 2_000_000:
                break
        dataset = Dataset.from_list(samples[:TARGETS["code"]])
        return tokenize_and_save(dataset, tokenizer, "text", "code", TARGETS["code"])


def load_math(tokenizer):
    """Source 2: Math (15%)."""
    print("\n=== Source 2: Math (15%) ===")
    print("  Loading open-web-math/open-web-math...")
    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    samples = []
    for i, example in enumerate(ds):
        if len(samples) >= TARGETS["math"]:
            break
        text = example.get("text", "")
        if len(text) > 100:
            samples.append({"text": text})
        if i > 500_000:
            break
    print(f"  Collected {len(samples):,} math samples")
    dataset = Dataset.from_list(samples)
    return tokenize_and_save(dataset, tokenizer, "text", "math", TARGETS["math"])


def load_science(tokenizer):
    """Source 3: Scientific papers (15%) — arxiv abstracts."""
    print("\n=== Source 3: Scientific (15%) ===")

    # Try multiple arxiv abstract sources
    sources = [
        ("ccdv/arxiv-summarization", {"split": "train", "streaming": True}),
        ("togethercomputer/RedPajama-Data-1T-Sample", {"split": "train", "streaming": True}),
    ]

    for name, kwargs in sources:
        try:
            print(f"  Trying {name}...")
            ds = load_dataset(name, **kwargs)
            samples = []
            for i, example in enumerate(ds):
                if len(samples) >= TARGETS["science"]:
                    break
                # Try common abstract/text field names
                text = (example.get("abstract", "") or
                        example.get("summary", "") or
                        example.get("text", ""))
                if len(text) > 100:
                    # For RedPajama, filter for arxiv-like content
                    if name == "togethercomputer/RedPajama-Data-1T-Sample":
                        meta = example.get("meta", {})
                        if isinstance(meta, str):
                            if "arxiv" not in meta.lower():
                                continue
                        elif isinstance(meta, dict):
                            if "arxiv" not in str(meta).lower():
                                continue
                    samples.append({"text": text[:4000]})
                if i > 1_000_000:
                    break
            if len(samples) >= TARGETS["science"] // 2:
                print(f"  Collected {len(samples):,} science samples from {name}")
                dataset = Dataset.from_list(samples[:TARGETS["science"]])
                return tokenize_and_save(dataset, tokenizer, "text", "science", TARGETS["science"])
            else:
                print(f"  Only got {len(samples):,} — trying next source")
        except Exception as e:
            print(f"  {name} failed: {e}")

    # Final fallback: use Wikipedia science articles
    print("  Falling back to Wikipedia science articles...")
    ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    samples = []
    sci_keywords = ["experiment", "hypothesis", "molecular", "quantum", "theorem",
                    "equation", "physics", "biology", "chemistry", "scientific"]
    for i, example in enumerate(ds):
        if len(samples) >= TARGETS["science"]:
            break
        text = example.get("text", "")
        if any(kw in text.lower() for kw in sci_keywords) and len(text) > 500:
            samples.append({"text": text[:4000]})
        if i > 3_000_000:
            break
    print(f"  Collected {len(samples):,} science samples from Wikipedia")
    dataset = Dataset.from_list(samples[:TARGETS["science"]])
    return tokenize_and_save(dataset, tokenizer, "text", "science", TARGETS["science"])


def load_prose(tokenizer):
    """Source 4: High-quality prose (25%) — long Wikipedia articles."""
    print("\n=== Source 4: Prose (25%) ===")

    sources = [
        ("wikimedia/wikipedia", {"name": "20231101.en", "split": "train", "streaming": True}),
        ("openwebtext", {"split": "train", "streaming": True}),
    ]

    for name, kwargs in sources:
        try:
            print(f"  Trying {name}...")
            ds = load_dataset(name, **kwargs)
            samples = []
            for i, example in enumerate(ds):
                if len(samples) >= TARGETS["prose"]:
                    break
                text = example.get("text", "")
                if len(text) > 2000:
                    samples.append({"text": text[:4000]})
                if i > 2_000_000:
                    break
            if len(samples) >= TARGETS["prose"] // 2:
                print(f"  Collected {len(samples):,} prose samples from {name}")
                dataset = Dataset.from_list(samples[:TARGETS["prose"]])
                return tokenize_and_save(dataset, tokenizer, "text", "prose", TARGETS["prose"])
            else:
                print(f"  Only got {len(samples):,} — trying next source")
        except Exception as e:
            print(f"  {name} failed: {e}")

    # Last resort: openwebtext long articles
    print("  Falling back to openwebtext long articles...")
    ds = load_dataset("openwebtext", split="train", streaming=True)
    samples = []
    for i, example in enumerate(ds):
        if len(samples) >= TARGETS["prose"]:
            break
        text = example.get("text", "")
        if len(text) > 1000:
            samples.append({"text": text})
    dataset = Dataset.from_list(samples[:TARGETS["prose"]])
    return tokenize_and_save(dataset, tokenizer, "text", "prose", TARGETS["prose"])


def load_instruction(tokenizer):
    """Source 5: Instruction-following (15%) — SlimOrca."""
    print("\n=== Source 5: Instruction (15%) ===")
    print("  Loading Open-Orca/SlimOrca...")
    ds = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
    samples = []
    for i, example in enumerate(ds):
        if len(samples) >= TARGETS["instruction"]:
            break
        convos = example.get("conversations", [])
        if convos:
            text = "\n\n".join(c.get("value", "") for c in convos)
            if len(text) > 100:
                samples.append({"text": text})
        if i > 500_000:
            break
    print(f"  Collected {len(samples):,} instruction samples")
    dataset = Dataset.from_list(samples)
    return tokenize_and_save(dataset, tokenizer, "text", "instruction", TARGETS["instruction"])


def main():
    print("=== Reasoning Core Data Preparation ===\n")

    if OUTPUT_DIR.exists() and not TEMP_DIR.exists():
        print(f"Tokenized dataset already exists at {OUTPUT_DIR}")
        print("Delete it manually if you want to re-prepare.")
        sys.exit(0)
    if OUTPUT_DIR.exists() and TEMP_DIR.exists():
        print(f"Resuming — temp sources exist, will rebuild final dataset.")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer()
    print(f"Tokenizer: {DEFAULT_TOKENIZER} (vocab size: {tokenizer.vocab_size})")

    source_paths = []
    loaders = [
        ("code", load_code),
        ("math", load_math),
        ("science", load_science),
        ("prose", load_prose),
        ("instruction", load_instruction),
    ]

    for name, loader_fn in loaders:
        existing_path = TEMP_DIR / name
        if existing_path.exists():
            print(f"\n  Skipping {name} — already saved at {existing_path}")
            source_paths.append((name, existing_path))
            continue
        path = loader_fn(tokenizer)
        source_paths.append((name, path))
        gc.collect()

    # Combine all sources
    print("\n=== Combining all sources ===")
    all_datasets = []
    total_samples = 0
    for name, path in source_paths:
        ds = load_from_disk(str(path))
        print(f"  {name}: {len(ds):,} samples")
        all_datasets.append(ds)
        total_samples += len(ds)

    combined = concatenate_datasets(all_datasets)
    combined = combined.shuffle(seed=42)
    print(f"\n  Total: {len(combined):,} samples")
    print(f"  Estimated tokens: ~{len(combined) * MAX_SEQ_LEN / 1e6:.0f}M")

    print(f"\n  Saving to {OUTPUT_DIR}...")
    if OUTPUT_DIR.exists():
        shutil.rmtree(str(OUTPUT_DIR))
    combined.save_to_disk(str(OUTPUT_DIR))

    print("  Cleaning up temp files...")
    shutil.rmtree(str(TEMP_DIR), ignore_errors=True)

    print(f"\n=== Done ===")
    print(f"  Samples: {len(combined):,}")
    print(f"  Tokens:  ~{len(combined) * MAX_SEQ_LEN / 1e6:.0f}M")
    print(f"  Saved:   {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
