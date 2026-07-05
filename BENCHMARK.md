# StreamArena Benchmark Protocol

How baselines are evaluated on StreamArena. All runner code lives in
[`benchmarks/`](benchmarks/) — one thin CLI script per algorithm, backed by a
shared harness ([`benchmarks/common.py`](benchmarks/common.py)) built on
[CapyMOA](https://capymoa.org/) (MOA/Java backend).

## Protocol

- **Prequential (test-then-train) evaluation**: each instance is predicted
  first, then learned from — full stream, original row order, single pass.
  No shuffling, no train/test split.
- **Cumulative + windowed metrics**: one cumulative number per run for
  ranking, plus per-window metrics (default window = 1,000 instances) saved
  as CSV for accuracy-over-time / drift-recovery plots.
- **Seeds**: stochastic learners (forests, patches, isolation-based
  detectors) run with seeds 1–10; deterministic learners run once (seed 1).
  Report mean ± std across seeds.
- **Hyperparameters**: library defaults, no tuning. CapyMOA and dataset
  versions are recorded in every result file.
- **Resources**: wall-clock time, CPU time, and peak RSS are recorded per run.
- **Aggregation across datasets**: average rank per task (never average raw
  metric values across datasets).

### Metrics per task

| Task | Cumulative metrics | Primary |
|---|---|---|
| Classification | accuracy, kappa, kappa-temporal (κt), kappa-M, F1 | **κt** (accuracy alone is misleading on temporally correlated streams — beat the NoChange baseline first) |
| Regression | RMSE, MAE, RMAE, R², adjusted R² | **RMSE** |
| Clustering | windowed ARI vs. ground-truth labels (mean + last window) | **mean ARI** |
| Anomaly detection | ROC-AUC, sliding-window AUC (s-AUC) | **ROC-AUC** |

Clustering note: CapyMOA has no prequential clustering evaluator, so the
harness trains on every instance and, at each window boundary, assigns the
window's points to the nearest cluster center (macro-clusters when the
algorithm provides them, micro-clusters otherwise) and scores ARI against the
held-back labels. Labels are never shown to the clusterer.

Anomaly note: detectors are unsupervised; labels are used only to compute AUC.

Coverage caps (documented, not silent): the two pure-Python detectors are too
slow for the largest streams at default settings. RobustRandomCutForest
(~0.18 s/instance) runs only on datasets ≤ 25,000 instances. Online
Isolation Forest runs everywhere, but datasets over 200,000 instances use
seeds 1–3 instead of 1–10. HalfSpaceTrees (Java) covers all datasets at full
seeds. Reduced coverage is visible in the per-algorithm #Datasets column of
the results.

## Baseline algorithms (v0 — 17)

| Task | Naive floors | Classics | Modern ensembles |
|---|---|---|---|
| Classification | MajorityClass, NoChange | NaiveBayes, HoeffdingTree, HoeffdingAdaptiveTree | AdaptiveRandomForest, StreamingRandomPatches |
| Regression | TargetMean | FIMT-DD | SOKNL, AdaptiveRandomForestRegressor |
| Clustering | — | CluStream, CluStream+k-means, DenStream+DBSCAN | — |
| Anomaly detection | — | HalfSpaceTrees | OnlineIsolationForest, RobustRandomCutForest |

## Running

Setup (needs Java 11+ for CapyMOA's MOA backend):

```bash
pip install capymoa scikit-learn pandas
python download.py            # fetch datasets from the Hugging Face Hub
```

Single algorithm, single dataset (smoke test):

```bash
python3 benchmarks/classification/hoeffding_tree.py --datasets real/electricity
python3 benchmarks/classification/hoeffding_tree.py --list        # see dataset keys
python3 benchmarks/anomaly_detection/half_space_trees.py --seeds 1 2 3
```

Full sweep on a many-core server with [GNU parallel](https://www.gnu.org/software/parallel/):

```bash
# one process per (algorithm × dataset); resumable — completed runs are skipped
python3 benchmarks/make_jobs.py | parallel -j 32 --joblog sweep.log

# finer granularity (one job per seed), or a single task:
python3 benchmarks/make_jobs.py --per-seed | parallel -j 64
python3 benchmarks/make_jobs.py --task anomaly_detection | parallel -j 16
```

Runs are independent processes; a crash costs one cell of the matrix. Re-run
the same pipeline to fill in whatever is missing (`--force` re-runs). For very
large streams, give the JVM more heap: `export CAPYMOA_JVM_ARGS="-Xmx8g"`.

Long-running sweeps over SSH survive disconnects with tmux:

```bash
tmux new -s streamarena
python3 benchmarks/make_jobs.py | parallel -j 32 --joblog sweep.log
# detach: Ctrl-b d — reattach later: tmux attach -t streamarena
```

## Results format

One JSON per (task, algorithm, dataset, seed) under `benchmarks/results/`:

```
benchmarks/results/classification/AdaptiveRandomForest/real__electricity__seed1.json
benchmarks/results/classification/AdaptiveRandomForest/real__electricity__seed1.windowed.csv
```

Each JSON records: task, algorithm, dataset, seed, cumulative metrics,
wall-clock / CPU time, peak RSS, CapyMOA + Python versions, hostname, and
timestamp. Failures are written as `*.failed.json` with the traceback and
don't block the rest of the sweep.
