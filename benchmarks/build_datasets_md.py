#!/usr/bin/env python3
"""Regenerate DATASETS.md from the built catalog (sidecar manifests + CSV headers).

    python3 benchmarks/build_datasets_md.py
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "datasets"


def count_rows(p):
    with open(p) as f:
        return sum(1 for _ in f) - 1


def header(p):
    with open(p) as f:
        return next(csv.reader(f))


def classes(p):
    seen = set()
    with open(p) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            seen.add(row[-1])
    return len(seen)


def real_rows(task):
    from streamarena.sources import catalog
    cat = catalog()
    out = []
    for p in sorted((ROOT / task / "real").glob("*.csv")):
        m = json.loads(p.with_suffix(".json").read_text()) if p.with_suffix(".json").exists() else {}
        src = cat.get(p.stem)
        n_feat = len(header(p)) - 2
        extra = f"{classes(p)}" if task == "classification" else f"h={m.get('horizon', '?')}"
        out.append(f"| {p.stem} | {count_rows(p):,} | {n_feat} | {extra} | {m.get('start','')[:10]} → {m.get('end','')[:10]} | {m.get('provenance', getattr(src, 'provenance', ''))} |")
    return out


def synth_rows(task):
    man = json.loads((ROOT / "synth_manifest.json").read_text()) if (ROOT / "synth_manifest.json").exists() else {}
    out = []
    for p in sorted((ROOT / task / "synth").glob("*.csv")):
        m = man.get(p.stem, {})
        n_feat = len(header(p)) - 1
        extra = f"{classes(p)}" if task == "classification" else "—"
        out.append(f"| {p.stem} | {count_rows(p):,} | {n_feat} | {extra} | {m.get('drift','')} @ {m.get('position','')} | {m.get('before',{}).get('generator','')} |")
    return out


def main():
    lines = ["# StreamArena — Dataset Tables", "",
             "Counts are computed directly from each built CSV. Real datasets are capped at 1M rows by uniform",
             "thinning; `h` is the prediction horizon in steps of the stream's native cadence. Every file has a leading",
             "`timestamp` column (not a feature) and a trailing `target`. Sources are cited in `paper/references.bib`.", ""]
    for task in ("classification", "regression"):
        lines += [f"## {task.capitalize()}", "", "### Real-world (API archives)", "",
                  "| Dataset | #Instances | #Features | " + ("#Classes" if task == "classification" else "Horizon") + " | Span | Source |",
                  "|---|---:|---:|---:|---|---|", *real_rows(task), "",
                  "### Synthetic", "",
                  "| Dataset | #Instances | #Features | " + ("#Classes" if task == "classification" else "—") + " | Drift | Generator |",
                  "|---|---:|---:|---:|---|---|", *synth_rows(task), ""]
    (REPO / "DATASETS.md").write_text("\n".join(lines))
    print("wrote DATASETS.md")


if __name__ == "__main__":
    main()
