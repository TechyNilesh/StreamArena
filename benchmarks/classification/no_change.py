"""NoChange on all StreamArena classification datasets (prequential evaluation).

Predicts the previously seen label — the naive floor that any real learner
must beat on temporally correlated streams (this is what kappa-temporal
measures). Implemented in Python rather than MOA because MOA's NoChange
indexes its vote array by raw class value and crashes with
ArrayIndexOutOfBoundsException on datasets whose labels are not 0-based
(e.g. shuttle, nomao, real_sensor).

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
    import numpy as np
    from capymoa.base import Classifier

    class NoChange(Classifier):
        def __init__(self, schema=None, random_seed=1):
            super().__init__(schema=schema, random_seed=random_seed)
            self._last_label_index = None

        def train(self, instance):
            self._last_label_index = instance.y_index

        def predict(self, instance):
            return self._last_label_index

        def predict_proba(self, instance):
            if self._last_label_index is None:
                return None
            proba = np.zeros(self.schema.get_num_classes())
            proba[self._last_label_index] = 1.0
            return proba

        def __str__(self):
            return "NoChange"

    return NoChange(schema=schema, random_seed=seed)


if __name__ == "__main__":
    run(
        task="classification",
        algorithm="NoChange",
        make_learner=make_learner,
        stochastic=False,
    )
