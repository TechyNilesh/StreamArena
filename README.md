<div align="center">
  <img src="assets/streamarena_logo.png?v=2" width="600" alt="StreamArena Logo"/>

  <p align="center"><strong>A Curated Benchmark Collection for Machine Learning on Streaming Data</strong></p>

  [🤗 Datasets on Hugging Face](https://huggingface.co/datasets/techynilesh/streamarena)

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

## 📂 Structure

```
StreamArena/
├── assets/
│   └── streamarena_logo.png
├── datasets/                 # downloaded via download.py, not committed to git
│   ├── classification/       # drift-labeled synthetic streams + real-world streams
│   ├── regression/           # streaming/tabular regression sets (.arff)
│   ├── clustering/           # streaming clustering sets (reuses classification streams + synthetic blobs)
│   └── anomaly_detection/    # ODDS/ADBench-style outlier detection sets (.npz)
├── examples/
│   └── load_dataset.py       # minimal loader for each task/file type
├── download.py                # pulls datasets/ from Hugging Face Hub
└── README.md
```

## 📊 Datasets

| Task | Count | Formats | Notes |
|---|---:|---|---|
| **Classification** | 60 files + 12 sub-collections | `.csv`, `.arff` | Includes a 100-file drift-labeled synthetic stream set (`synthetic_streams/`, tagged sudden/gradual/incremental/recurring), classic drift benchmarks (SEA, RBF, Hyperplane, Agrawal), real-world streams (electricity, forest cover, airlines, poker, weather, KDD-99, insects), recurring-concept-drift ensembles (MNIST, Usenet, Gisette, Madelon, Dota, Spambase, RTG), and raw UCI HAR sensor data |
| **Regression** | 30 files | `.arff` | Streaming/tabular regression sets: Friedman synthetics, housing (king's county, california, miami, brazilian), sensor/physical (sarcos, naval propulsion, superconductivity, kin8nm), and more |
| **Clustering** | 13 files | `.csv` | Streaming clustering benchmarks — reuses classification drift streams plus a dedicated synthetic blobs set |
| **Anomaly Detection** | 51 files | `.npz` | ODDS/ADBench-style outlier detection collection (annthyroid, mnist, shuttle, satellite, mammography, etc.) |

Each `.npz` anomaly-detection file bundles `X`/`y` arrays. Each `.arff` file is Weka-format and loads
directly with `scipy.io.arff` or `liac-arff`. Synthetic drift streams are plain CSVs with a trailing
label column.

## ⚡ Quickstart

Dataset files (~6.7GB) are hosted on the [Hugging Face Hub](https://huggingface.co/datasets/techynilesh/streamarena)
rather than committed to this repo. Download them first:

```bash
pip install huggingface_hub
python download.py                # download everything into ./datasets
python download.py classification # or just one task
```

Then load any dataset:

```python
import pandas as pd

df = pd.read_csv("datasets/classification/electricity.csv")
print(df.shape)
```

See [`examples/load_dataset.py`](examples/load_dataset.py) for loaders covering every format in the
collection (`.csv`, `.arff`, `.npz`, and the raw UCI HAR signal directory).

## 🛣️ Roadmap

- [ ] Standardized train/test drift-aware splits per dataset
- [ ] Unified `StreamArenaDataset` loader API across formats
- [ ] Reference stream-learning baselines (Hoeffding Tree, ARF, HAT, streaming k-means, HS-Trees, etc.)
- [ ] Leaderboard + evaluation harness (accuracy/prequential error, ARI, ROC-AUC by task)
- [ ] Dataset cards documenting drift type, size, and source per dataset

## 📄 Citation

A citation entry will be added once StreamArena has an accompanying paper. Until then, please cite
the original dataset sources where applicable.

## 📜 License

See [LICENSE](LICENSE). Individual datasets retain their original licenses/terms from their
respective sources — check before redistribution.
