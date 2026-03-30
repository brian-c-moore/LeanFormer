"""Wrapper around lm-eval-harness for standard benchmark evaluation."""


def evaluate_model(model_path: str, tasks: list[str] | None = None):
    """
    Run standard benchmarks via lm-eval-harness.

    Requires: pip install lm-eval
    """
    # from lm_eval import evaluator
    # results = evaluator.simple_evaluate(model=..., tasks=tasks)
    raise NotImplementedError(
        "Benchmark evaluation requires a trained model and lm-eval installed. "
        "Install lm-eval and configure the HuggingFace-compatible model wrapper."
    )
