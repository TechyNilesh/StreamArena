"""Use a StreamArena dataset with River and CapyMOA — the two most common
Python streaming-ML libraries. Both read the CSVs directly; no conversion needed.

    pip install river capymoa
"""

from pathlib import Path

DATASET = Path(__file__).resolve().parent.parent / "datasets" / "classification" / "real" / "electricity.csv"


def river_example():
    import pandas as pd
    from river import metrics, stream, tree

    # Build target/converters from the header instead of hardcoding column
    # names, since they differ across StreamArena datasets.
    columns = pd.read_csv(DATASET, nrows=0).columns.tolist()
    target = columns[-1]
    converters = {c: float for c in columns if c != target}
    converters[target] = int

    dataset = stream.iter_csv(DATASET, target=target, converters=converters)

    model = tree.HoeffdingTreeClassifier()
    metric = metrics.Accuracy()

    for x, y in dataset:
        y_pred = model.predict_one(x)
        model.learn_one(x, y)
        metric.update(y, y_pred)

    print("River  —", metric)


def capymoa_example():
    from capymoa.classifier import HoeffdingTree
    from capymoa.evaluation import prequential_evaluation
    from capymoa.stream import stream_from_file

    # class_index=-1 matches StreamArena's convention of a trailing label column.
    stream_ = stream_from_file(str(DATASET), dataset_name="Electricity", class_index=-1, target_type="categorical")

    learner = HoeffdingTree(schema=stream_.get_schema())
    results = prequential_evaluation(stream_, learner)

    print("CapyMOA —", "accuracy:", results.cumulative.accuracy())


if __name__ == "__main__":
    river_example()
    capymoa_example()
