"""Download StreamArena datasets from the Hugging Face Hub into ./datasets.

Usage:
    python download.py                # download everything
    python download.py classification # download a single task only
"""

import sys

from huggingface_hub import snapshot_download

REPO_ID = "techynilesh/streamarena"
TASKS = ["classification", "regression", "clustering", "anomaly_detection"]


def main() -> None:
    task = sys.argv[1] if len(sys.argv) > 1 else None
    allow_patterns = [f"{task}/**"] if task else None

    if task and task not in TASKS:
        raise SystemExit(f"Unknown task '{task}'. Choose from: {', '.join(TASKS)}")

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir="datasets",
        allow_patterns=allow_patterns,
    )
    print(f"Downloaded to {path}")


if __name__ == "__main__":
    main()
