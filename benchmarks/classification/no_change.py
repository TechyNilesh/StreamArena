"""NoChange on all StreamArena classification datasets (prequential evaluation).

Usage:
    python3 benchmarks/classification/no_change.py                 # all datasets, default seeds
    python3 benchmarks/classification/no_change.py --datasets real/electricity
    python3 benchmarks/classification/no_change.py --list

See BENCHMARK.md for the protocol; results land in benchmarks/results/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import run


def make_learner(schema, seed):
    from capymoa.classifier import NoChange

    return NoChange(schema=schema)


if __name__ == "__main__":
    run(
        task="classification",
        algorithm="NoChange",
        make_learner=make_learner,
        stochastic=False,
    )
