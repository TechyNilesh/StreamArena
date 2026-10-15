"""Fixed synthetic benchmark configurations.

Every generator ships two datasets of ``N`` instances: ``abrupt`` (one hard
concept switch at the midpoint) and ``gradual`` (sigmoid mix over 10% of the
stream centred at the midpoint). Seeds are pinned, so re-running this module
reproduces the published CSVs byte-for-byte.

    python -m streamarena.synth.catalog            # write all 32 datasets
    python -m streamarena.synth.catalog --list
    python -m streamarena.synth.catalog --only sea_abrupt --n 10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import GENERATORS, DriftStream

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = REPO_ROOT / "datasets"
N = 500_000
DRIFTS = {"abrupt": dict(kind="abrupt"), "gradual": dict(kind="gradual", width=N // 10)}

# name -> (generator, kwargs before drift, kwargs after drift)
BASE = {
    # classification
    "sea": ("SEA", dict(concept=1), dict(concept=3)),
    "agrawal": ("Agrawal", dict(concept=1, perturbation=0.05), dict(concept=4, perturbation=0.05)),
    "stagger": ("STAGGER", dict(concept=1), dict(concept=2)),
    "hyperplane": ("Hyperplane", dict(n_features=10, concept=1, noise=0.05), dict(n_features=10, concept=2, noise=0.05)),
    "rbf": ("RandomRBF", dict(n_features=10, n_classes=4, n_centroids=50, concept=1), dict(n_features=10, n_classes=4, n_centroids=50, concept=2)),
    "randomtree": ("RandomTree", dict(n_classes=3, concept=1), dict(n_classes=3, concept=2)),
    "led": ("LED", dict(noise=0.1, concept=1), dict(noise=0.1, concept=4)),
    "waveform": ("Waveform", dict(n_irrelevant=19, concept=1), dict(n_irrelevant=19, concept=2)),
    "mixed": ("Mixed", dict(concept=1), dict(concept=2)),
    # regression
    "friedman": ("Friedman", dict(concept=1), dict(concept=2)),
    "hyperplane_reg": ("HyperplaneRegression", dict(n_features=10, concept=1), dict(n_features=10, concept=2)),
    "mv": ("Mv", dict(concept=1), dict(concept=2)),
    "planes2d": ("Planes2D", dict(concept=1), dict(concept=2)),
    # API-like simulators
    "seasonal_sensor": ("SeasonalSensor", dict(concept=1, degradation=0.0), dict(concept=2, degradation=1e-5)),
    "regime_walk": ("RegimeSwitchingWalk", dict(target="return", concept=1), dict(target="return", concept=2)),
    "poisson_demand": ("PoissonDemand", dict(concept=1), dict(concept=2)),
}


def build(name: str, drift: str, n: int = N, seed: int = 1) -> DriftStream:
    gname, before, after = BASE[name]
    cls = GENERATORS[gname]
    a, b = cls(seed=seed, **before), cls(seed=seed + 1, **after)
    return DriftStream(a, b, position=n // 2, period=n // 2, seed=seed, **DRIFTS[drift])


def catalog() -> dict[str, DriftStream]:
    return {f"{name}_{drift}": build(name, drift) for name in BASE for drift in DRIFTS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=N)
    p.add_argument("--only", nargs="*")
    p.add_argument("--list", action="store_true")
    p.add_argument("--out", type=Path, default=DATASETS_ROOT)
    args = p.parse_args()
    cat = catalog()
    if args.list:
        print("\n".join(cat))
        return
    manifest = {}
    for key, stream in cat.items():
        if args.only and key not in args.only:
            continue
        task_dir = args.out / stream.task / "synth"
        path = stream.to_csv(task_dir / f"{key}.csv", args.n)
        manifest[key] = {"path": str(path.relative_to(args.out)), "n": args.n, **stream.describe()}
        print(f"[done] {path.relative_to(args.out)}", flush=True)
    (args.out / "synth_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
