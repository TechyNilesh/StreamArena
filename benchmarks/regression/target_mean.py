"""TargetMean on all StreamArena regression datasets (prequential evaluation).

Usage:
    python3 benchmarks/regression/target_mean.py                 # all datasets, default seeds
    python3 benchmarks/regression/target_mean.py --datasets real/kin8nm
    python3 benchmarks/regression/target_mean.py --list

See BENCHMARK.md for the protocol; results land in benchmarks/results/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import run


def make_learner(schema, seed):
    from capymoa.regressor import TargetMean

    return TargetMean(schema=schema)


if __name__ == "__main__":
    run(
        task="regression",
        algorithm="TargetMean",
        make_learner=make_learner,
        stochastic=False,
    )
