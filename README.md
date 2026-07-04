<div align="center">
  <img src="assets/streamarena_logo.png?v=3" width="600" alt="StreamArena Logo"/>

  <p align="center"><strong>A Living Benchmark for Machine Learning on Streaming Data</strong></p>

  <br/>

  ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
  ![Hugging Face](https://img.shields.io/badge/-Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
  ![Streaming ML](https://img.shields.io/badge/-Streaming%20ML-0EA5E9?style=flat-square)
  ![Anomaly Detection](https://img.shields.io/badge/-Anomaly%20Detection-EF4444?style=flat-square)
  ![Classification](https://img.shields.io/badge/-Classification-16A34A?style=flat-square)
  ![Regression](https://img.shields.io/badge/-Regression-F59E0B?style=flat-square)
  ![Clustering](https://img.shields.io/badge/-Clustering-EC4899?style=flat-square)
  ![License MIT](https://img.shields.io/badge/-MIT%20License-000000?style=flat-square)
</div>

---

StreamArena aggregates datasets for **stream learning** — classification, regression, clustering,
and anomaly detection under concept drift — into one consistently organized, task-first collection.
It plays the same role for streaming/online ML that [TabArena](https://github.com/autogluon/tabarena)
plays for tabular ML: a single place to find curated, ready-to-use datasets instead of hunting through
individual paper repos.

The datasets here were consolidated from several independent research codebases, deduplicated where
the same dataset appeared in multiple sources, and reorganized by **task** rather than by source paper.
Every dataset is stored as a single unified format — **CSV** — chosen because it's what the
streaming-ML ecosystem (River's `stream.iter_csv`, MOA, scikit-multiflow's `FileStream`) actually
consumes row-by-row, unlike batch/columnar formats.

## 📂 Structure

```
StreamArena/
├── assets/
│   └── streamarena_logo.png
├── datasets/                 # downloaded via download.py, not committed to git
│   ├── classification/
│   │   ├── real/               # real-world streams (electricity, forest cover, airlines, ...)
│   │   └── synth/               # synthetic drift generators (SEA, RBF, Hyperplane, Agrawal, Madelon, ...)
│   ├── regression/
│   │   ├── real/               # housing, wages, sensor/physical measurements, ...
│   │   └── synth/               # Friedman & Hyperplane synthetic generators
│   ├── clustering/
│   │   ├── real/               # real-world streams reused from classification
│   │   └── synth/               # synthetic drift streams + blobs
│   └── anomaly_detection/     # ODDS/ADBench-style outlier detection sets (all real-world)
├── benchmarks/               # baseline benchmark harness (CapyMOA) — see BENCHMARK.md
│   ├── common.py             # shared prequential-evaluation runner
│   ├── make_jobs.py          # emit per-(algorithm × dataset) jobs for GNU parallel
│   ├── classification/       # one CLI script per baseline algorithm
│   ├── regression/
│   ├── clustering/
│   └── anomaly_detection/
├── examples/
│   ├── load_dataset.py       # one loader function for every dataset in the collection
│   ├── river_usage.py        # run a River model on a StreamArena dataset
│   └── capymoa_usage.py      # run a CapyMOA model on a StreamArena dataset
├── download.py                # pulls datasets/ from Hugging Face Hub
├── BENCHMARK.md               # evaluation protocol + how to run the baselines
└── README.md
```

## 📊 Datasets

See [`DATASETS.md`](DATASETS.md) for the full per-dataset table — exact instance/feature/class counts
computed directly from each file, plus a best-effort source attribution (UCI, OpenML, DELVE, MOA/River
generators, ODDS/ADBench, etc.) for every dataset.

All files are `.csv`. Anomaly-detection files hold feature columns plus a trailing `label` column;
everything else follows the same feature-columns-plus-target convention. Every task except
anomaly detection (which is entirely real-world benchmark data) is split into `real/` and `synth/`.

| Task | Count | Notes |
|---|---:|---|
| **Classification** | 43 files (23 real + 20 synthetic) | `real/`: electricity, forest cover, airlines, poker, weather, KDD-99, insects, Nomao, MNIST, Usenet, Gisette, Dota, Spambase, HAR, etc. `synth/`: classic drift generators (SEA, RBF, Hyperplane, Agrawal, Madelon) |
| **Regression** | 30 files (25 real + 5 synthetic) | `real/`: housing (king's county, california, miami, brazilian), wages, sensor/physical (sarcos, naval propulsion, superconductivity, kin8nm), and more. `synth/`: Friedman & Hyperplane generators |
| **Clustering** | 13 files (6 real + 7 synthetic) | Streaming clustering benchmarks — reuses classification drift streams plus a dedicated synthetic blobs set |
| **Anomaly Detection** | 51 files | ODDS/ADBench-style outlier detection collection (annthyroid, mnist, shuttle, satellite, mammography, etc.) — all real-world, no `real/`/`synth/` split |

## ⚡ Quickstart

Dataset files (~5GB) are hosted on the [Hugging Face Hub](https://huggingface.co/datasets/techynilesh/streamarena)
rather than committed to this repo. Download them first:

```bash
pip install huggingface_hub
python download.py                # download everything into ./datasets
python download.py classification # or just one task
```

Then load any dataset — it's always just a CSV:

```python
import pandas as pd

df = pd.read_csv("datasets/classification/real/electricity.csv")
print(df.shape)
```

See [`examples/load_dataset.py`](examples/load_dataset.py).

### Using it with River or CapyMOA

Since every dataset is plain CSV, it plugs directly into the two most common Python streaming-ML
libraries — no conversion needed.

```python
# River
import pandas as pd
from river import metrics, stream, tree

path = "datasets/classification/real/electricity.csv"
sample = pd.read_csv(path, nrows=100)
target = sample.columns[-1]
# Convert only numeric feature columns to float; categorical/string columns
# (e.g. in adult.csv) pass through as-is — River trees handle them natively.
converters = {
    c: float for c in sample.columns[:-1] if pd.api.types.is_numeric_dtype(sample[c])
}

dataset = stream.iter_csv(path, target=target, converters=converters)
model = tree.HoeffdingTreeClassifier()
metric = metrics.Accuracy()

for x, y in dataset:
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    metric.update(y, y_pred)

print(metric)
```

```python
# CapyMOA (requires a working JVM — Java 11+)
from capymoa.classifier import HoeffdingTree
from capymoa.evaluation import prequential_evaluation
from capymoa.stream import stream_from_file

stream = stream_from_file(
    "datasets/classification/real/electricity.csv",
    dataset_name="Electricity",
    class_index=-1,       # StreamArena's convention: label is the trailing column
    target_type="categorical",
)
learner = HoeffdingTree(schema=stream.get_schema())
results = prequential_evaluation(stream, learner)
print("accuracy:", results.cumulative.accuracy())
```

See [`examples/river_usage.py`](examples/river_usage.py) and
[`examples/capymoa_usage.py`](examples/capymoa_usage.py) for the full runnable scripts.

## 🏁 Benchmarks

StreamArena ships a prequential (test-then-train) benchmark harness built on
[CapyMOA](https://capymoa.org/), with 17 baseline algorithms across the four tasks — one CLI
script per algorithm so sweeps parallelize trivially on a many-core machine:

```bash
pip install capymoa scikit-learn pandas
python3 benchmarks/classification/hoeffding_tree.py --datasets real/electricity   # one run
python3 benchmarks/make_jobs.py | parallel -j 32                                   # full sweep
```

See [`BENCHMARK.md`](BENCHMARK.md) for the protocol (metrics per task, seeds, windowed
evaluation) and the full baseline list.

## 🛣️ Roadmap

- [x] Evaluation harness — prequential protocol + per-algorithm runner scripts ([BENCHMARK.md](BENCHMARK.md))
- [ ] Baseline results for the full dataset × algorithm matrix
- [ ] Public leaderboard
- [ ] Unified `StreamArenaDataset` loader API across formats
- [ ] Dataset cards documenting drift type, size, and source per dataset

## 📄 Citation

If you use StreamArena in your research, please cite it as below (see also [CITATION.cff](CITATION.cff)):

```bibtex
@misc{verma2026streamarena,
  title  = {StreamArena: A Living Benchmark for Machine Learning on Streaming Data},
  author = {Verma, Nilesh},
  year   = {2026},
  url    = {https://github.com/TechyNilesh/StreamArena}
}
```

Please also cite the original dataset sources where applicable.

## 📜 License

See [LICENSE](LICENSE). Individual datasets retain their original licenses/terms from their
respective sources — check before redistribution.
