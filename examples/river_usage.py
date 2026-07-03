"""Use a StreamArena dataset with River, the most common Python
streaming-ML library. It reads the CSVs directly; no conversion needed.

    pip install river pandas
"""

from pathlib import Path

import pandas as pd
from river import metrics, stream, tree

DATASET = Path(__file__).resolve().parent.parent / "datasets" / "classification" / "real" / "electricity.csv"


def main():
    # Build target/converters by sniffing dtypes from a sample instead of
    # hardcoding column names, since they differ across StreamArena datasets.
    # Only numeric feature columns get float converters; categorical/string
    # columns (e.g. in adult.csv) pass through as-is — River trees handle
    # them natively. The label also stays unconverted.
    sample = pd.read_csv(DATASET, nrows=100)
    target = sample.columns[-1]
    converters = {
        c: float for c in sample.columns[:-1] if pd.api.types.is_numeric_dtype(sample[c])
    }

    dataset = stream.iter_csv(DATASET, target=target, converters=converters)

    model = tree.HoeffdingTreeClassifier()
    metric = metrics.Accuracy()

    for x, y in dataset:
        y_pred = model.predict_one(x)
        model.learn_one(x, y)
        metric.update(y, y_pred)

    print("River —", metric)


if __name__ == "__main__":
    main()
