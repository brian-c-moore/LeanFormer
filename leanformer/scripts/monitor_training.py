"""
Monitor training progress. Prints a status report every hour until training completes.

Usage:
    python -m leanformer.scripts.monitor_training
"""

import time
import subprocess
import sys
from pathlib import Path


CHECKPOINT_DIR = Path("checkpoints/scale")
LOG_FILE = CHECKPOINT_DIR / "training_log.json"


def get_latest_log():
    """Read the last few lines of the training output."""
    # Find the most recent training output file
    import glob
    task_dir = Path.home() / "AppData/Local/Temp/claude"
    candidates = list(task_dir.rglob("bqvj3lnh1.output"))
    if not candidates:
        return "No training output found."
    output_file = candidates[0]
    with open(output_file) as f:
        lines = f.readlines()
    return "".join(lines[-20:])


def get_gpu_status():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "GPU status unavailable"


def main():
    print("=== LeanFormer Training Monitor ===")
    print("Checking every hour. Ctrl+C to stop.\n")

    hour = 0
    while True:
        hour += 1
        print(f"\n{'='*60}")
        print(f"Hour {hour} — {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        # Check if training is done
        if LOG_FILE.exists():
            print("Training log found — training may be complete.")
            import json
            with open(LOG_FILE) as f:
                log = json.load(f)
            if log:
                last = log[-1]
                print(f"Last logged step: {last.get('step', '?')}")
                print(f"Last loss: {last.get('loss', '?')}")
            break

        print(get_latest_log())
        print(f"GPU: {get_gpu_status()}")

        # Wait 1 hour
        try:
            time.sleep(3600)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break


if __name__ == "__main__":
    main()
