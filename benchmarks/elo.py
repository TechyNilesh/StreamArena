"""Elo ratings from per-dataset scores, following TabArena / Chatbot Arena.

Protocol (TabArena, Erickson et al. 2025, §2.3; code: tabrepo ``bencheval/elo_utils.py``):

* every pair of algorithms "battles" once per dataset; the one with the better
  primary metric wins, equal scores are a tie counted as half a win each;
* every dataset contributes total weight 1, so no dataset dominates;
* ratings are the Bradley-Terry maximum-likelihood fit of those battles,
  mapped to the chess scale: a 400-point gap means 10:1 expected odds;
* the scale is anchored so a named reference algorithm sits at 1000;
* 95% intervals come from bootstrapping datasets (100 rounds by default).

Algorithms that skipped a dataset simply have no battles there, so partial
coverage is handled without imputation. Pure Python (no numpy) because the
leaderboard is rebuilt in a bare CI container.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

SCALE = 400.0
INIT_RATING = 1000.0
LOG_BASE = math.log(10.0)


def battles_from_scores(scores, higher_better):
    """{dataset: {algorithm: value}} -> [(dataset, a, b, outcome_a)] with
    outcome_a in {1.0, 0.5, 0.0}. One battle per unordered pair per dataset."""
    out = []
    for dataset, by_algo in scores.items():
        algos = sorted(by_algo)
        for i, a in enumerate(algos):
            for b in algos[i + 1:]:
                va, vb = by_algo[a], by_algo[b]
                if va == vb:
                    outcome = 0.5
                elif (va > vb) == higher_better:
                    outcome = 1.0
                else:
                    outcome = 0.0
                out.append((dataset, a, b, outcome))
    return out


def bradley_terry(battles, weights=None, tol=1e-9, max_iter=10_000):
    """MM fit (Hunter 2004) of Bradley-Terry strengths.

    ``weights`` maps dataset -> multiplicity (bootstrap counts); each dataset's
    battles are further divided by its pair count so a dataset weighs 1 in total,
    whatever the number of algorithms that ran on it.
    Returns {algorithm: log-strength}, centred at mean 0.
    """
    per_dataset = defaultdict(int)
    for dataset, _, _, _ in battles:
        per_dataset[dataset] += 1
    wins = defaultdict(float)          # weighted wins (ties as half)
    pair_n = defaultdict(float)        # weighted comparisons per unordered pair
    algos = set()
    for dataset, a, b, outcome in battles:
        w = (weights.get(dataset, 0.0) if weights is not None else 1.0) / per_dataset[dataset]
        if w == 0.0:
            continue
        algos.update((a, b))
        wins[a] += w * outcome
        wins[b] += w * (1.0 - outcome)
        pair_n[(a, b)] += w
    algos = sorted(algos)
    if len(algos) < 2:
        return {a: 0.0 for a in algos}
    neighbours = defaultdict(list)
    for (a, b), n in pair_n.items():
        neighbours[a].append((b, n))
        neighbours[b].append((a, n))
    # An algorithm that never scores (or never loses) a point has an infinite MLE;
    # clamp with a tiny prior pseudo-count so the fit stays finite, as the
    # near-unregularised logistic fit (C=1e6) does in TabArena.
    eps = 1e-6
    p = {a: 1.0 for a in algos}
    for _ in range(max_iter):
        new = {}
        for a in algos:
            denom = sum(n / (p[a] + p[b]) for b, n in neighbours[a])
            new[a] = (wins[a] + eps) / (denom + eps)
        gm = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        new = {a: v / gm for a, v in new.items()}
        delta = max(abs(math.log(new[a]) - math.log(p[a])) for a in algos)
        p = new
        if delta < tol:
            break
    return {a: math.log(v) for a, v in p.items()}


def to_elo(log_strength, anchor=None):
    """log-strength -> Elo. If ``anchor`` is present it is pinned to 1000;
    otherwise the field averages 1000 (Chatbot Arena's convention)."""
    elo = {a: SCALE * s / LOG_BASE + INIT_RATING for a, s in log_strength.items()}
    if anchor is not None and anchor in elo:
        shift = INIT_RATING - elo[anchor]
        elo = {a: v + shift for a, v in elo.items()}
    return elo


def elo_ratings(scores, higher_better, anchor=None, n_bootstrap=100, seed=0):
    """Point estimate plus bootstrap 95% interval.

    Returns {algorithm: {"elo", "elo_low", "elo_high"}}; interval bounds are
    None when ``n_bootstrap`` is 0.
    """
    battles = battles_from_scores(scores, higher_better)
    point = to_elo(bradley_terry(battles), anchor)
    result = {a: {"elo": v, "elo_low": None, "elo_high": None} for a, v in point.items()}
    if n_bootstrap and len(scores) > 1:
        datasets = sorted(scores)
        rng = random.Random(seed)
        draws = defaultdict(list)
        for _ in range(n_bootstrap):
            counts = defaultdict(float)
            for _ in datasets:
                counts[rng.choice(datasets)] += 1.0
            for a, v in to_elo(bradley_terry(battles, counts), anchor).items():
                draws[a].append(v)
        for a, values in draws.items():
            if a not in result:
                continue
            values.sort()
            lo = values[max(0, int(math.floor(0.025 * (len(values) - 1))))]
            hi = values[min(len(values) - 1, int(math.ceil(0.975 * (len(values) - 1))))]
            result[a]["elo_low"], result[a]["elo_high"] = lo, hi
    return result
