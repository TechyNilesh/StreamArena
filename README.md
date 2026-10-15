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

**🏆 Live leaderboard: [techynilesh.github.io/StreamArena](https://techynilesh.github.io/StreamArena/)** ·
[Dataset catalog](https://techynilesh.github.io/StreamArena/datasets.html) ·
[Submit your results](submissions/README.md)

StreamArena is a **deployment-oriented benchmark for stream learning** (classification and
regression under concept drift). Instead of re-using decades-old static files, every real-world
dataset is built from the multi-year archive of a **public open API** (weather, air quality, power
grids, rivers, tides, space weather, earthquakes, bike/taxi demand, flights, crypto markets,
Wikipedia edits), with a declared feature contract and a delayed label. The same source classes run
in **live mode** against the API for deployment tests. Synthetic streams come from generators
re-implemented here that double as infinite, timestamped live sources.

## 📂 Structure

```
StreamArena/
├── streamarena/
│   ├── synth/                 # 16 generators (9 classic clf, 4 classic reg, 3 new API-like simulators)
│   │   ├── base.py            #   Generator + DriftStream (abrupt / gradual / incremental / recurring)
│   │   ├── classification.py  #   SEA, Agrawal, STAGGER, Hyperplane, RandomRBF, RandomTree, LED, Waveform, Mixed
│   │   ├── regression.py      #   Friedman, HyperplaneRegression, Mv, Planes2D
│   │   ├── simulated.py       #   SeasonalSensor, RegimeSwitchingWalk, PoissonDemand
│   │   └── catalog.py         #   the 32 fixed synthetic datasets (pinned seeds)
│   └── sources/               # 26 real-world datasets from public APIs (history + live modes)
│       ├── base.py            #   Source, FeatureContract, HTTP helpers
│       ├── open_meteo.py      #   weather (ERA5) + air quality (CAMS)
│       ├── usgs_water.py      #   river discharge / flood  (USGS NWIS)
│       ├── noaa_coops.py      #   tide level              (NOAA CO-OPS)
│       ├── energy.py          #   GB carbon intensity, DE load, Agile prices
│       ├── space_weather.py   #   geomagnetic storms      (GFZ Kp / Hp30)
│       ├── usgs_quake.py      #   earthquake significance (USGS ComCat)
│       ├── mobility.py        #   Citi Bike demand / rider type, NYC taxi demand
│       ├── binance.py         #   BTC 1-minute returns / direction
│       ├── bts_flights.py     #   US flight delays        (BTS On-Time)
│       └── wikimedia.py       #   Wikipedia edit reverts  (mediawiki_history dumps)
├── build_datasets.py          # builds datasets/ from generators + API archives
├── datasets/                  # built locally / hosted on Hugging Face, not committed
│   ├── classification/{real,synth}/
│   └── regression/{real,synth}/
├── benchmarks/                # prequential harness (CapyMOA) — see BENCHMARK.md
├── site/, submissions/, leaderboard/   # public leaderboard + community results
└── download.py                # pulls prebuilt datasets/ from the Hugging Face Hub
```

## 📊 Datasets (58)

| | Classification | Regression |
|---|---:|---:|
| **Real (API archives)** | 12 | 14 |
| **Synthetic (2 drift configs × 16 generators)** | 18 | 14 |

Every CSV has a leading `timestamp` column (UTC, for event-time replay), feature columns, and a
trailing `target`. Real datasets are capped at 1M rows by uniform thinning so the stream still spans
the whole archive period; a sidecar `<name>.json` records span, horizon, lags and provenance.
Features are **observed values only** (plus lags and calendar terms); targets are the value `horizon`
steps ahead, so labels arrive with exactly the delay a deployed model would see. Full per-dataset
tables with sources live in [`DATASETS.md`](DATASETS.md).

Build it yourself (synthetic takes a few minutes; light API archives ~10 min; the five
multi-GB archives — Citi Bike, NYC taxi, BTS flights, Wikipedia — are flagged `heavy`):

```bash
pip install numpy pandas pyarrow requests
python build_datasets.py --list
python build_datasets.py synth
python build_datasets.py real --skip-heavy
python build_datasets.py real --only flight_delay_bts         # one heavy source, on a server
```

Or pull the prebuilt copy: `python download.py`.

### Live mode

Every real source and every generator is also a live stream:

```python
from streamarena.sources import catalog
src = catalog()["grid_carbon_gb"]
for kind, ts, payload in src.live():          # ("instance", ts, x) now, ("label", ts, y) later
    ...

from streamarena.synth import SEA, DriftStream
stream = DriftStream(SEA(concept=1), SEA(concept=3), kind="gradual", position=50_000)
for ts, x, y in stream.live(rate=100):         # 100 instances / second, wall-clock timestamps
    ...
```

### Using it with River or CapyMOA

Since every dataset is plain CSV, it plugs directly into the common Python streaming-ML libraries —
drop the `timestamp` column (or let `benchmarks/common.py` do it) and stream the rest.

## 🏁 Benchmarks

StreamArena ships a prequential (test-then-train) benchmark harness built on
[CapyMOA](https://capymoa.org/), with baseline algorithms for classification and regression — one CLI
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
- [x] Baseline results for the full dataset × algorithm matrix ([LEADERBOARD.md](LEADERBOARD.md))
- [x] Public leaderboard with community submissions — [techynilesh.github.io/StreamArena](https://techynilesh.github.io/StreamArena/)
- [x] Real-world datasets rebuilt from public API archives; generators rewritten with live mode
- [ ] Event-time replayer with delayed labels and deadline accounting
- [ ] Edge resource envelopes (RAM/CPU profiles) and cloud latency injection
- [ ] Language-neutral learner protocol + C++ SDK and baselines
- [ ] Baseline sweep on the new catalog and leaderboard refresh

## 📄 Citation

If you use StreamArena in your research, please cite it as below (see also [CITATION.cff](CITATION.cff)):

```bibtex
@misc{verma2026streamarena,
  title  = {StreamArena: A Living Benchmark for Machine Learning on Streaming Data},
  author = {Verma, Nilesh and Bifet, Albert and Pfahringer, Bernhard and Bahri, Maroua},
  year   = {2026},
  url    = {https://github.com/TechyNilesh/StreamArena}
}
```

Please also cite the original dataset sources where applicable.

## 📜 License

See [LICENSE](LICENSE). Individual datasets retain their original licenses/terms from their
respective sources — check before redistribution.
