"""Shared harness for StreamArena baseline benchmarks.

Each algorithm script in benchmarks/<task>/ defines a learner factory and
calls :func:`run`. The harness handles dataset discovery, CLI arguments,
prequential (test-then-train) evaluation via CapyMOA, and one JSON result
file per (dataset, seed) run — with resume support, so an interrupted sweep
can simply be relaunched.

Protocol details live in BENCHMARK.md at the repo root.
"""

import argparse
import json
import platform
import resource
import socket
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = REPO_ROOT / "datasets"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"

TASKS = ("classification", "regression", "clustering", "anomaly_detection")
STOCHASTIC_SEEDS = list(range(1, 11))
DETERMINISTIC_SEEDS = [1]


def discover_datasets(task):
    """Return {dataset_key: csv_path}, e.g. {"real/electricity": Path(...)}."""
    task_dir = DATASETS_ROOT / task
    return {
        str(p.relative_to(task_dir).with_suffix("")): p
        for p in sorted(task_dir.rglob("*.csv"))
    }


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux
    return rss / 1e6 if sys.platform == "darwin" else rss / 1e3


def result_paths(results_dir, task, algorithm, dataset_key, seed):
    stem = f"{dataset_key.replace('/', '__')}__seed{seed}"
    out_dir = Path(results_dir) / task / algorithm
    return out_dir / f"{stem}.json", out_dir / f"{stem}.windowed.csv"


def _parse_args(task, algorithm, stochastic):
    parser = argparse.ArgumentParser(
        description=f"Run {algorithm} on all StreamArena {task} datasets "
        "(prequential evaluation, see BENCHMARK.md)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset keys to run (e.g. real/electricity). Default: all.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=STOCHASTIC_SEEDS if stochastic else DETERMINISTIC_SEEDS,
        help="Random seeds to run per dataset.",
    )
    parser.add_argument("--window", type=int, default=1000, help="Windowed-metrics window size.")
    parser.add_argument("--max-instances", type=int, default=None, help="Cap instances (smoke tests).")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--force", action="store_true", help="Re-run even if a result file exists.")
    parser.add_argument("--list", action="store_true", help="List dataset keys and exit.")
    return parser.parse_args()


def count_rows(csv_path):
    with open(csv_path) as f:
        return sum(1 for _ in f) - 1


def run(task, algorithm, make_learner, stochastic=False, max_rows=None):
    """Entry point called by every per-algorithm script.

    :param make_learner: callable (schema, seed) -> CapyMOA learner.
    :param stochastic: seeds 1-10 by default instead of just 1.
    :param max_rows: skip datasets with more instances than this — a
        documented coverage cap for algorithms too slow for huge streams
        (see BENCHMARK.md).
    """
    assert task in TASKS, f"unknown task {task}"
    args = _parse_args(task, algorithm, stochastic)

    datasets = discover_datasets(task)
    if args.list:
        print("\n".join(datasets))
        return
    if args.datasets:
        missing = [d for d in args.datasets if d not in datasets]
        if missing:
            raise SystemExit(f"Unknown dataset keys: {missing}. Use --list to see options.")
        datasets = {k: datasets[k] for k in args.datasets}
    if not datasets:
        raise SystemExit(f"No CSVs found under {DATASETS_ROOT / task} — run download.py first.")

    for key, csv_path in datasets.items():
        if max_rows is not None:
            n = count_rows(csv_path)
            if n > max_rows:
                print(f"[skip] {key}: {n} rows exceeds {algorithm}'s {max_rows}-row coverage cap")
                continue
        for seed in args.seeds:
            json_path, windowed_path = result_paths(args.results_dir, task, algorithm, key, seed)
            if json_path.exists() and not args.force:
                print(f"[skip] {json_path.relative_to(args.results_dir)} exists")
                continue
            json_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[run ] {task}/{algorithm} on {key} (seed {seed})", flush=True)
            record = {
                "task": task,
                "algorithm": algorithm,
                "dataset": key,
                "seed": seed,
                "window_size": args.window,
                "max_instances": args.max_instances,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
            }
            try:
                metrics, extras, windowed_df = _run_one(
                    task, csv_path, key, make_learner, seed, args.window, args.max_instances
                )
            except Exception:
                record.update(status="failed", error=traceback.format_exc())
                failed_path = json_path.with_suffix(".failed.json")
                failed_path.write_text(json.dumps(record, indent=2))
                print(f"[FAIL] {key} seed {seed} — see {failed_path}", file=sys.stderr, flush=True)
                continue
            record.update(status="ok", metrics=metrics, **extras)
            record["peak_rss_mb"] = round(peak_rss_mb(), 1)
            if windowed_df is not None:
                windowed_df.to_csv(windowed_path, index=False)
            json_path.write_text(json.dumps(record, indent=2))
            print(f"[done] {key} seed {seed}: {metrics}", flush=True)


def _run_one(task, csv_path, key, make_learner, seed, window, max_instances):
    import capymoa
    from capymoa.stream import stream_from_file

    target_type = "numeric" if task == "regression" else "categorical"
    stream = stream_from_file(
        str(csv_path), dataset_name=key.replace("/", "_"), class_index=-1, target_type=target_type
    )
    learner = make_learner(stream.get_schema(), seed)
    extras = {"capymoa_version": capymoa.__version__, "learner": str(learner)}

    if task == "clustering":
        metrics, windowed_df, timing = _run_clustering(stream, learner, window, max_instances)
        extras.update(timing)
        return metrics, extras, windowed_df

    if task == "anomaly_detection":
        from capymoa.evaluation import prequential_evaluation_anomaly

        results = prequential_evaluation_anomaly(
            stream, learner, max_instances=max_instances, window_size=window
        )
        metrics = {"auc": results.cumulative.auc(), "s_auc": results.cumulative.s_auc()}
    else:
        from capymoa.evaluation import prequential_evaluation

        results = prequential_evaluation(
            stream, learner, max_instances=max_instances, window_size=window
        )
        cumulative = results.cumulative
        if task == "classification":
            metrics = {
                "accuracy": cumulative.accuracy(),
                "kappa": cumulative.kappa(),
                "kappa_t": cumulative.kappa_t(),
                "kappa_m": cumulative.kappa_m(),
                "f1_score": cumulative.f1_score(),
            }
        else:
            metrics = {
                "rmse": cumulative.rmse(),
                "mae": cumulative.mae(),
                "rmae": cumulative.rmae(),
                "r2": cumulative.r2(),
                "adjusted_r2": cumulative.adjusted_r2(),
            }
    metrics = {k: (None if v is None else float(v)) for k, v in metrics.items()}
    extras["wallclock_s"] = round(float(results.wallclock()), 3)
    extras["cpu_time_s"] = round(float(results.cpu_time()), 3)
    windowed_df = results.windowed.metrics_per_window()
    return metrics, extras, windowed_df


def _run_clustering(stream, learner, window, max_instances):
    """CapyMOA has no prequential clustering evaluator, so we do a windowed
    loop: train on each instance; every `window` instances assign the window's
    points to the nearest cluster center (macro if available, else micro) and
    score ARI against the ground-truth labels.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score

    t0_wall, t0_cpu = time.perf_counter(), time.process_time()
    xs, ys, rows, seen = [], [], [], 0
    while stream.has_more_instances():
        if max_instances is not None and seen >= max_instances:
            break
        instance = stream.next_instance()
        learner.train(instance)
        xs.append(instance.x)
        ys.append(instance.y_index)
        seen += 1
        if len(xs) == window:
            rows.append(_score_cluster_window(learner, np.asarray(xs, dtype=float), ys, seen, adjusted_rand_score))
            xs, ys = [], []

    aris = [r["ari"] for r in rows if r["ari"] is not None]
    metrics = {
        "ari_mean": float(np.mean(aris)) if aris else None,
        "ari_last": aris[-1] if aris else None,
        "n_windows_scored": len(aris),
        "instances_seen": seen,
    }
    timing = {
        "wallclock_s": round(time.perf_counter() - t0_wall, 3),
        "cpu_time_s": round(time.process_time() - t0_cpu, 3),
    }
    return metrics, pd.DataFrame(rows), timing


def _score_cluster_window(learner, X, ys, seen, ari_fn):
    import numpy as np

    centers = []
    if learner.implements_macro_clusters():
        centers = learner.get_clustering_result().get_centers()
    if len(centers) == 0 and learner.implements_micro_clusters():
        centers = learner.get_micro_clustering_result().get_centers()
    if len(centers) == 0:
        return {"instances_seen": seen, "n_clusters": 0, "ari": None}
    C = np.asarray([np.asarray(c, dtype=float) for c in centers])
    assignments = np.argmin(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1), axis=1)
    return {
        "instances_seen": seen,
        "n_clusters": len(C),
        "ari": float(ari_fn(ys, assignments)),
    }
