"""Use a StreamArena dataset with CapyMOA, the Python API for MOA.
It reads the CSVs directly; no conversion needed.

Requires a working JVM (Java 11+):

    pip install capymoa
"""

from pathlib import Path

from capymoa.classifier import HoeffdingTree
from capymoa.evaluation import prequential_evaluation
from capymoa.stream import stream_from_file

DATASET = Path(__file__).resolve().parent.parent / "datasets" / "classification" / "real" / "electricity.csv"


def main():
    # class_index=-1 matches StreamArena's convention of a trailing label column.
    stream = stream_from_file(str(DATASET), dataset_name="Electricity", class_index=-1, target_type="categorical")

    learner = HoeffdingTree(schema=stream.get_schema())
    results = prequential_evaluation(stream, learner)

    print("CapyMOA —", "accuracy:", results.cumulative.accuracy())


if __name__ == "__main__":
    main()
