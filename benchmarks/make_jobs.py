"""Emit one shell command per (algorithm script × dataset) so a sweep can be
fanned out with GNU parallel — one independent process per job, resumable.

Usage on the experiment server:
    python3 benchmarks/make_jobs.py | parallel -j 32 --joblog sweep.log
    python3 benchmarks/make_jobs.py --task classification | parallel -j 16
    python3 benchmarks/make_jobs.py --per-seed | parallel -j 64   # finest granularity

Each job skips already-completed runs, so re-piping after a crash only runs
what's missing.
"""

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import STOCHASTIC_SEEDS, TASKS, discover_datasets

BENCH_DIR = Path(__file__).resolve().parent


def algorithm_scripts(task):
    return sorted(
        p for p in (BENCH_DIR / task).glob("*.py") if not p.name.startswith("_")
    )


def script_is_stochastic(path):
    return "stochastic=True" in path.read_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, help="Limit to one task.")
    parser.add_argument("--python", default="python3", help="Python executable to use.")
    parser.add_argument(
        "--per-seed",
        action="store_true",
        help="One job per (script, dataset, seed) instead of per (script, dataset).",
    )
    parser.add_argument(
        "--extra-args", default="", help='Appended to every command, e.g. "--window 500".'
    )
    args = parser.parse_args()

    tasks = [args.task] if args.task else list(TASKS)
    for task in tasks:
        datasets = discover_datasets(task)
        if not datasets:
            print(f"# no datasets for {task} — run download.py first", file=sys.stderr)
            continue
        for script in algorithm_scripts(task):
            seeds = STOCHASTIC_SEEDS if script_is_stochastic(script) else [1]
            rel = script.relative_to(BENCH_DIR.parent)
            for dataset in datasets:
                base = f"{args.python} {rel} --datasets {shlex.quote(dataset)}"
                if args.per_seed:
                    for seed in seeds:
                        print(f"{base} --seeds {seed} {args.extra_args}".rstrip())
                else:
                    print(f"{base} {args.extra_args}".rstrip())


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:  # e.g. `make_jobs.py | head`
        import os

        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
