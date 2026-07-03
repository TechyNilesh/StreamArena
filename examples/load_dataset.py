"""Minimal loaders covering every file format found in StreamArena/datasets."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

DATASETS_ROOT = Path(__file__).resolve().parent.parent / "datasets"


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_arff(path: Path) -> pd.DataFrame:
    data, _meta = arff.loadarff(path)
    return pd.DataFrame(data)


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    npz = np.load(path)
    return npz["X"], npz["y"]


if __name__ == "__main__":
    electricity = load_csv(DATASETS_ROOT / "classification" / "electricity.csv")
    print("classification/electricity.csv:", electricity.shape)

    kin8nm = load_arff(DATASETS_ROOT / "regression" / "kin8nm.arff")
    print("regression/kin8nm.arff:", kin8nm.shape)

    X, y = load_npz(DATASETS_ROOT / "anomaly_detection" / "6_cardio.npz")
    print("anomaly_detection/6_cardio.npz:", X.shape, y.shape)

    blobs = load_csv(DATASETS_ROOT / "clustering" / "synthetic_blobs_100k_samples_5features_8clusters.csv")
    print("clustering/synthetic_blobs_...csv:", blobs.shape)
