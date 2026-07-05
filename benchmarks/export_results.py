"""Pack per-run result JSONs into one compact baseline_runs.json for the repo.

benchmarks/results/ is gitignored (thousands of files + windowed CSVs), but the
leaderboard site rebuilds in CI, which needs the raw run records. This packs
just the JSON records (no windowed data) into leaderboard/data/baseline_runs.json
— a single committable file.

Usage (typically on the experiment server):
    python3 benchmarks/export_results.py
"""

import argparse
import json
from pathlib import Path

from common import DEFAULT_RESULTS_ROOT, REPO_ROOT

DEFAULT_OUT = REPO_ROOT / "leaderboard" / "data" / "baseline_runs.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    records = []
    for path in sorted(args.results_dir.rglob("*.json")):
        if path.name.endswith(".failed.json"):
            continue
        record = json.loads(path.read_text())
        if record.get("status") != "ok":
            continue
        records.append(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, separators=(",", ":")))
    size_mb = args.out.stat().st_size / 1e6
    print(f"Packed {len(records)} run records into {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
