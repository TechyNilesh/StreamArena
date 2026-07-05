# Submitting results to the StreamArena leaderboard

Run your algorithm under the [StreamArena protocol](../BENCHMARK.md), then open a
Pull Request adding your result files under `submissions/<YourAlgorithmName>/`.
When the PR merges, the [leaderboard](https://techynilesh.github.io/StreamArena/)
rebuilds automatically — no backend, no accounts.

## The easy path

If your algorithm has a CapyMOA-compatible API, copy any script in
`benchmarks/<task>/`, swap in your learner, and run it. The files it writes to
`benchmarks/results/<task>/<YourAlgorithmName>/` **are** the submission format —
copy them (the `.json` files, not the `.windowed.csv`) into
`submissions/<YourAlgorithmName>/` and open the PR.

## The format

One JSON file per (dataset, seed) run:

```json
{
  "task": "classification",
  "algorithm": "MyCoolTree",
  "dataset": "real/electricity",
  "seed": 1,
  "status": "ok",
  "window_size": 1000,
  "metrics": {
    "accuracy": 88.1, "kappa": 75.2, "kappa_t": 35.0, "kappa_m": 70.1, "f1_score": 87.9
  },
  "wallclock_s": 41.2,
  "cpu_time_s": 39.8,
  "hyperparameters": {
    "grace_period": 200,
    "split_confidence": 1e-7,
    "ensemble_size": 100,
    "_defaults": "mycooltree 1.2.0 defaults unless listed"
  },
  "submission": {
    "author": "Jane Doe",
    "contact_or_repo": "https://github.com/janedoe/mycooltree",
    "paper": "https://arxiv.org/abs/...",
    "library_versions": {"capymoa": "0.13.0"},
    "dataset_revision": "techynilesh/streamarena@main",
    "notes": "anything the fields above don't capture"
  }
}
```

Required fields: `task`, `algorithm`, `dataset`, `seed`, `metrics`, `wallclock_s`,
`hyperparameters`, and a `submission` block with at least `author` and
`contact_or_repo`. The `submission` block is what marks your rows with the
*community* badge.

**`hyperparameters`** is a machine-readable object documenting the single
configuration used across all datasets of the task (the protocol forbids
per-dataset tuning, so there is exactly one). List every parameter you set
explicitly as `name: value`; parameters left at library defaults may be
summarized with the `_defaults` key naming the library and version whose
defaults apply (e.g. `"capymoa 0.13.0 defaults"`). If you ran pure defaults,
`{"_defaults": "capymoa 0.13.0 defaults"}` is a complete, valid value —
but the field itself must be present so reproducibility is never ambiguous.

`metrics` must contain the task's metrics:

| task | required metrics |
|---|---|
| `classification` | `accuracy`, `kappa`, `kappa_t`, `kappa_m` |
| `regression` | `rmse`, `mae`, `r2` |
| `clustering` | `ari_mean`, `ari_last` |
| `anomaly_detection` | `auc` |

## The rules

- **Protocol**: prequential (test-then-train), full stream, original order, single
  pass, no tuning per dataset (one hyperparameter configuration across all
  datasets of a task). See [BENCHMARK.md](../BENCHMARK.md).
- **Seeds**: 1–10 for stochastic algorithms, seed 1 for deterministic ones.
  Partial dataset coverage is fine — the leaderboard ranks you only where you ran.
- **Datasets**: use the current files from the
  [Hugging Face Hub](https://huggingface.co/datasets/techynilesh/streamarena);
  dataset keys are the harness keys (`real/electricity`, `6_cardio`, …) —
  `python3 benchmarks/classification/hoeffding_tree.py --list` prints them.
- **Honesty**: results are currently accepted on trust with provenance recorded
  in git history. Submissions may be spot-checked by re-running; irreproducible
  entries are removed.

Validate before opening the PR:

```bash
python3 benchmarks/validate_submission.py submissions/MyCoolTree/
```
