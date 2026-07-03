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
├── examples/
│   └── load_dataset.py       # one loader function for every dataset in the collection
├── download.py                # pulls datasets/ from Hugging Face Hub
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
| **Classification** | 44 files (23 real + 21 synthetic) | `real/`: electricity, forest cover, airlines, poker, weather, KDD-99, insects, Nomao, MNIST, Usenet, Gisette, Dota, Spambase, HAR, etc. `synth/`: classic drift generators (SEA, RBF, Hyperplane, Agrawal, Madelon) |
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

## 🛣️ Roadmap

- [ ] Standardized train/test drift-aware splits per dataset
- [ ] Unified `StreamArenaDataset` loader API across formats
- [ ] Reference stream-learning baselines (Hoeffding Tree, ARF, HAT, streaming k-means, HS-Trees, etc.)
- [ ] Leaderboard + evaluation harness (accuracy/prequential error, ARI, ROC-AUC by task)
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
