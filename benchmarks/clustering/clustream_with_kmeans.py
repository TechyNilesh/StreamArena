"""ClustreamKMeans on all StreamArena clustering datasets (prequential evaluation).

Usage:
    python3 benchmarks/clustering/clustream_with_kmeans.py                 # all datasets, default seeds
    python3 benchmarks/clustering/clustream_with_kmeans.py --datasets synth/synth_blobs_sudden
    python3 benchmarks/clustering/clustream_with_kmeans.py --list

See BENCHMARK.md for the protocol; results land in benchmarks/results/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import run


def make_learner(schema, seed):
    from capymoa.clusterers import Clustream_with_kmeans

    return Clustream_with_kmeans(schema=schema)


if __name__ == "__main__":
    run(
        task="clustering",
        algorithm="ClustreamKMeans",
        make_learner=make_learner,
        stochastic=False,
    )
