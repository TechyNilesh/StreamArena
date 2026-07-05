"""Validate community submission JSONs before they reach the leaderboard.

Usage:
    python3 benchmarks/validate_submission.py submissions/MyAlgorithm/
    python3 benchmarks/validate_submission.py submissions/          # everything

Exits non-zero on any invalid file (used by CI on submission PRs).
"""

import json
import sys
from pathlib import Path

TASK_METRICS = {
    "classification": {"accuracy", "kappa", "kappa_t", "kappa_m"},
    "regression": {"rmse", "mae", "r2"},
    "clustering": {"ari_mean", "ari_last"},
    "anomaly_detection": {"auc"},
}
BOUNDS = {"accuracy": (0, 100), "auc": (0, 1), "ari_mean": (-1, 1), "ari_last": (-1, 1)}


def check(path):
    errors = []
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"invalid JSON: {e}"]

    for field in ("task", "algorithm", "dataset", "seed", "metrics", "wallclock_s", "submission", "hyperparameters"):
        if field not in record:
            errors.append(f"missing field '{field}'")
    if errors:
        return errors

    hp = record["hyperparameters"]
    if not isinstance(hp, dict) or not hp:
        errors.append(
            "hyperparameters must be a non-empty object listing every "
            "hyperparameter of the algorithm as name: value"
        )
    elif "_defaults" in hp:
        errors.append(
            "hyperparameters must spell out the full configuration — "
            "'_defaults' shorthand is not accepted (defaults change between versions)"
        )

    task = record["task"]
    if task not in TASK_METRICS:
        errors.append(f"unknown task '{task}'")
    else:
        missing = TASK_METRICS[task] - set(record["metrics"])
        if missing:
            errors.append(f"metrics missing {sorted(missing)}")
    for metric, value in record.get("metrics", {}).items():
        if value is None:
            continue
        if not isinstance(value, (int, float)):
            errors.append(f"metric '{metric}' is not numeric")
        elif metric in BOUNDS:
            lo, hi = BOUNDS[metric]
            if not lo <= value <= hi:
                errors.append(f"metric '{metric}'={value} outside [{lo}, {hi}]")

    sub = record.get("submission", {})
    for field in ("author", "contact_or_repo"):
        if not sub.get(field):
            errors.append(f"submission block missing '{field}'")
    if not isinstance(record.get("seed"), int):
        errors.append("seed must be an integer")
    return errors


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "submissions")
    files = [p for p in sorted(root.rglob("*.json"))]
    if not files:
        print(f"No submission JSONs under {root} — nothing to validate.")
        return
    bad = 0
    for path in files:
        errors = check(path)
        if errors:
            bad += 1
            print(f"FAIL {path}")
            for e in errors:
                print(f"     - {e}")
        else:
            print(f"ok   {path}")
    print(f"\n{len(files) - bad}/{len(files)} valid")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
