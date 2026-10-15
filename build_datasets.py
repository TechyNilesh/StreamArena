#!/usr/bin/env python3
"""Build the StreamArena v2 dataset catalog.

    python build_datasets.py --list
    python build_datasets.py synth                      # 32 synthetic datasets (fast, local)
    python build_datasets.py real                       # 26 real datasets from API archives
    python build_datasets.py real --skip-heavy          # everything except the multi-GB archives
    python build_datasets.py real --only weather_temp_berlin grid_carbon_gb
    python build_datasets.py real --only tide_level_battery --start 2024-01-01 --end 2024-03-01   # smoke test

Outputs land in datasets/<task>/<real|synth>/<name>.csv with a sidecar JSON
manifest; completed datasets are skipped unless --force is given.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", nargs="?", choices=["synth", "real", "all"], default="all")
    p.add_argument("--list", action="store_true")
    p.add_argument("--only", nargs="*")
    p.add_argument("--skip-heavy", action="store_true")
    p.add_argument("--start"), p.add_argument("--end")
    p.add_argument("--max-rows", type=int, default=1_000_000)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", type=Path, default=None, help="output root (default datasets/)")
    args = p.parse_args()

    from streamarena.sources import HEAVY, catalog as real_catalog
    from streamarena.synth.catalog import catalog as synth_catalog, N
    from streamarena.synth.catalog import DATASETS_ROOT

    real, synth = real_catalog(), synth_catalog()
    if args.out:
        DATASETS_ROOT = args.out
    if args.list:
        for k, s in synth.items():
            print(f"synth  {s.task:14s} {k}")
        for k, s in real.items():
            print(f"real   {s.contract.task:14s} {k}{'  [heavy]' if k in HEAVY else ''}")
        return

    if args.what in ("synth", "all"):
        for key, stream in synth.items():
            if args.only and key not in args.only:
                continue
            path = DATASETS_ROOT / stream.task / "synth" / f"{key}.csv"
            if path.exists() and not args.force:
                print(f"[skip] {key} exists"); continue
            t = time.time(); stream.to_csv(path, N); print(f"[done] {key} ({time.time() - t:.0f}s)", flush=True)

    if args.what in ("real", "all"):
        for key, src in real.items():
            if args.only and key not in args.only:
                continue
            if args.skip_heavy and key in HEAVY and not args.only:
                print(f"[skip] {key} is heavy (use --only to force)"); continue
            path = DATASETS_ROOT / src.contract.task / "real" / f"{key}.csv"
            if path.exists() and not args.force:
                print(f"[skip] {key} exists"); continue
            print(f"[run ] {key}", flush=True)
            t = time.time()
            try:
                out = src.to_csv(DATASETS_ROOT, start=args.start, end=args.end, max_rows=args.max_rows)
                import pandas as pd
                n = sum(1 for _ in open(out)) - 1
                print(f"[done] {key}: {n:,} rows ({time.time() - t:.0f}s)", flush=True)
            except Exception:
                print(f"[FAIL] {key}\n{traceback.format_exc()}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
