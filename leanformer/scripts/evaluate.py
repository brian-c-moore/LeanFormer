"""
Evaluate a trained LeanFormer checkpoint using lm-eval-harness.

Supports DCLM CORE tasks (HellaSwag, ARC-Easy, MMLU, COPA, etc.)
and any other task supported by lm-eval-harness.

Usage:
    python -m leanformer.scripts.evaluate --checkpoint checkpoints/reasoning_core
    python -m leanformer.scripts.evaluate --checkpoint checkpoints/reasoning_core --tasks hellaswag,arc_easy
    python -m leanformer.scripts.evaluate --checkpoint checkpoints/reasoning_core --tasks mmlu --num_fewshot 5
"""

import argparse
import json
import sys
from pathlib import Path

import torch

import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from ..model.config import LeanFormerConfig
from ..model.leanformer import LeanFormer
from .. import DEFAULT_TOKENIZER


CORE_TASKS = [
    "hellaswag",
    "arc_easy",
    "copa",
    "piqa",
    "winogrande",
    "openbookqa",
    "boolq",
]


@register_model("leanformer")
class LeanFormerHarnessModel(LM):
    """lm-eval-harness adapter for LeanFormer."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/reasoning_core",
        device: str = "cuda",
        batch_size: int = 1,
    ):
        super().__init__()
        from transformers import AutoTokenizer

        ckpt = Path(checkpoint_path)
        config = LeanFormerConfig.load(ckpt / "leanformer_config.json")
        self.model = LeanFormer(config)
        weights = torch.load(ckpt / "pytorch_model.bin", map_location=device, weights_only=True)
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._device = device
        self._batch_size = batch_size
        self._config = config

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._config.max_seq_len

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, left_truncate_len: int = None, add_special_tokens: bool = None) -> list[int]:
        encoding = self.tokenizer.encode(string)
        if left_truncate_len is not None:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        """Compute log-likelihood for each (context, continuation) pair."""
        import torch.nn.functional as F

        results = []
        for context_enc, continuation_enc in requests:
            input_ids = torch.tensor(
                [context_enc + continuation_enc], dtype=torch.long, device=self._device
            )

            # Truncate to max_length
            if input_ids.shape[1] > self._config.max_seq_len:
                input_ids = input_ids[:, -self._config.max_seq_len:]

            with torch.no_grad():
                out = self.model(input_ids, training=False)
                logits = out["logits"]  # [1, seq_len, vocab_size]

            # Compute log-probs for the continuation tokens only
            log_probs = F.log_softmax(logits, dim=-1)

            # The continuation starts at position len(context_enc) in the input
            # But we need to align: logits[t] predicts token[t+1]
            ctx_len = len(context_enc)
            cont_len = len(continuation_enc)

            # Handle truncation
            total_len = input_ids.shape[1]
            if ctx_len + cont_len > total_len:
                ctx_len = total_len - cont_len

            cont_log_probs = 0.0
            is_greedy = True
            for i in range(cont_len):
                pos = ctx_len + i - 1  # logits at pos predict token at pos+1
                if pos < 0:
                    continue
                token_id = continuation_enc[i]
                token_logprob = log_probs[0, pos, token_id].item()
                cont_log_probs += token_logprob

                # Check if this token was the greedy choice
                if logits[0, pos].argmax().item() != token_id:
                    is_greedy = False

            results.append((cont_log_probs, is_greedy))

        return results

    def loglikelihood(self, requests, disable_tqdm=False):
        new_reqs = []
        for req in requests:
            context, continuation = req.args
            context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation)
            new_reqs.append((context_enc, continuation_enc))
        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        results = []
        for req in requests:
            (string,) = req.args
            encoding = self.tok_encode(string)
            # Compute log-likelihood of the entire string
            results_for_req = self._loglikelihood_tokens(
                [([], encoding)], disable_tqdm=disable_tqdm
            )
            results.append(results_for_req[0][0])  # Just the log-prob, not is_greedy
        return results

    def generate_until(self, requests, disable_tqdm=False):
        results = []
        for req in requests:
            context, gen_kwargs = req.args, req.kwargs
            if isinstance(context, tuple):
                context = context[0]

            until = gen_kwargs.get("until", [self.tokenizer.eos_token])
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            input_ids = torch.tensor(
                [self.tok_encode(context)], dtype=torch.long, device=self._device
            )

            with torch.no_grad():
                output_ids, _ = self.model.generate(
                    input_ids,
                    max_new_tokens=max_gen,
                    temperature=0.0,  # Greedy for eval
                    top_k=1,
                )

            generated = self.tok_decode(output_ids[0, input_ids.shape[1]:].tolist())

            # Truncate at stop sequences
            for stop in until:
                if stop in generated:
                    generated = generated[:generated.index(stop)]

            results.append(generated)

        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LeanFormer with lm-eval-harness")
    parser.add_argument("--checkpoint", default="checkpoints/reasoning_core", help="Checkpoint directory")
    parser.add_argument("--tasks", default=",".join(CORE_TASKS), help="Comma-separated task names")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--output", default=None, help="Output JSON path (default: checkpoint_dir/eval_results.json)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not (ckpt / "pytorch_model.bin").exists():
        print(f"ERROR: Checkpoint not found at {ckpt}")
        sys.exit(1)

    print(f"Loading model from {ckpt}...")
    model = LeanFormerHarnessModel(
        checkpoint_path=str(ckpt),
        device=args.device,
        batch_size=args.batch_size,
    )

    tasks = args.tasks.split(",")
    print(f"Running evaluation: {tasks}")
    print(f"Few-shot: {args.num_fewshot}")

    results = lm_eval.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
    )

    # Print results
    print(f"\n{'='*60}")
    print("  EVALUATION RESULTS")
    print(f"{'='*60}\n")

    for task_name, task_results in results.get("results", {}).items():
        print(f"  {task_name}:")
        for metric, value in sorted(task_results.items()):
            if isinstance(value, float):
                print(f"    {metric}: {value:.4f}")
            elif metric != "alias":
                print(f"    {metric}: {value}")
        print()

    # Save results
    output_path = args.output or str(ckpt / "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
