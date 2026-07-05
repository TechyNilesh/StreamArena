"""RobustRandomCutForest on all StreamArena anomaly_detection datasets (prequential evaluation).

Coverage cap: RRCF is a pure-Python implementation at ~0.18 s/instance with
default settings, so datasets over 25,000 instances are skipped (a single
seed on the 619k-instance donors set would take ~31 hours). The cap is
visible in results as reduced #Datasets coverage — see BENCHMARK.md.

Usage:
    python3 benchmarks/anomaly_detection/robust_random_cut_forest.py                 # all datasets, default seeds
    python3 benchmarks/anomaly_detection/robust_random_cut_forest.py --datasets 6_cardio
    python3 benchmarks/anomaly_detection/robust_random_cut_forest.py --list

See BENCHMARK.md for the protocol; results land in benchmarks/results/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import run


def make_learner(schema, seed):
    from capymoa.anomaly import RobustRandomCutForest

    return RobustRandomCutForest(schema=schema, random_state=seed)


if __name__ == "__main__":
    run(
        task="anomaly_detection",
        algorithm="RobustRandomCutForest",
        make_learner=make_learner,
        stochastic=True,
        max_rows=25_000,
    )
