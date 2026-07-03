"""Every dataset in StreamArena is a plain CSV — one format, one loader.

Layout:
    classification/real/...      classification/synth/...
    regression/*.csv
    clustering/*.csv
    anomaly_detection/*.csv      (feature columns + a trailing `label` column)
"""

from pathlib import Path

import pandas as pd

DATASETS_ROOT = Path(__file__).resolve().parent.parent / "datasets"


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    electricity = load_csv(DATASETS_ROOT / "classification" / "real" / "electricity.csv")
    print("classification/real/electricity.csv:", electricity.shape)

    sea = load_csv(DATASETS_ROOT / "classification" / "synth" / "sea_high_abrupt_drift.csv")
    print("classification/synth/sea_high_abrupt_drift.csv:", sea.shape)

    kin8nm = load_csv(DATASETS_ROOT / "regression" / "kin8nm.csv")
    print("regression/kin8nm.csv:", kin8nm.shape)

    cardio = load_csv(DATASETS_ROOT / "anomaly_detection" / "6_cardio.csv")
    print("anomaly_detection/6_cardio.csv:", cardio.shape)

    blobs = load_csv(DATASETS_ROOT / "clustering" / "synthetic_blobs_100k_samples_5features_8clusters.csv")
    print("clustering/synthetic_blobs_...csv:", blobs.shape)
