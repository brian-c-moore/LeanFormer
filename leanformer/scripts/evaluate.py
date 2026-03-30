"""Evaluation entry point."""

import sys


def main():
    print("Evaluation requires a trained model checkpoint.")
    print("Usage: leanformer-evaluate <checkpoint_path>")
    if len(sys.argv) < 2:
        sys.exit(1)
    from ..evaluation.harness import evaluate_model
    evaluate_model(sys.argv[1])


if __name__ == "__main__":
    main()
