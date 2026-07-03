"""HalfSpaceTrees on all StreamArena anomaly_detection datasets (prequential evaluation).

Usage:
    python3 benchmarks/anomaly_detection/half_space_trees.py                 # all datasets, default seeds
    python3 benchmarks/anomaly_detection/half_space_trees.py --datasets 6_cardio
    python3 benchmarks/anomaly_detection/half_space_trees.py --list

See BENCHMARK.md for the protocol; results land in benchmarks/results/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import run


def make_learner(schema, seed):
    from capymoa.anomaly import HalfSpaceTrees

    return HalfSpaceTrees(schema=schema, random_seed=seed)


if __name__ == "__main__":
    run(
        task="anomaly_detection",
        algorithm="HalfSpaceTrees",
        make_learner=make_learner,
        stochastic=True,
    )
