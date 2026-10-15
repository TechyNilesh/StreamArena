"""Classic concept-drift classification generators, re-implemented from scratch.

Each generator takes a ``concept`` argument selecting one of its classic
concept functions (or a re-seeded concept for randomised generators), so the
:class:`~streamarena.synth.base.DriftStream` wrapper can mix two concepts of
the same generator. References for each concept are in the class docstrings.
"""

from __future__ import annotations

import math

import numpy as np

from .base import Generator


class SEA(Generator):
    """SEA concepts (Street & Kim, 2001).

    Three features in U(0, 10); only the first two matter: ``y = 1`` iff
    ``x1 + x2 <= theta`` with ``theta`` in ``(8, 9, 7, 9.5)`` for concepts 1-4.
    ``noise`` is the probability that the label is flipped.
    """

    task = "classification"
    n_classes = 2
    THRESHOLDS = {1: 8.0, 2: 9.0, 3: 7.0, 4: 9.5}

    def __init__(self, concept: int = 1, noise: float = 0.1, seed: int = 1):
        super().__init__(seed)
        self.theta = self.THRESHOLDS[concept]
        self.noise = noise

    feature_names = ["x1", "x2", "x3"]

    def _next(self):
        x = self.rng.uniform(0, 10, 3)
        y = int(x[0] + x[1] <= self.theta)
        if self.rng.random() < self.noise:
            y = 1 - y
        return dict(zip(self.feature_names, np.round(x, 4))), y


class Agrawal(Generator):
    """Agrawal loan-approval concepts (Agrawal, Imielinski & Swami, 1993).

    Nine attributes (salary, commission, age, elevel, car, zipcode, hvalue,
    hyears, loan) and ten classification functions, following the original
    paper / MOA's AgrawalGenerator. ``perturbation`` adds relative noise to the
    numeric attributes after the label is computed.
    """

    task = "classification"
    n_classes = 2
    feature_names = ["salary", "commission", "age", "elevel", "car", "zipcode", "hvalue", "hyears", "loan"]

    def __init__(self, concept: int = 1, perturbation: float = 0.0, seed: int = 1):
        super().__init__(seed)
        assert 1 <= concept <= 10
        self.concept = concept
        self.perturbation = perturbation

    def _label(self, s, c, a, e, car, z, h, hy, l):
        f = self.concept
        if f == 1:
            return int(a < 40 or a >= 60)
        if f == 2:
            return int((a < 40 and 50_000 <= s <= 100_000) or (40 <= a < 60 and 75_000 <= s <= 125_000) or (a >= 60 and 25_000 <= s <= 75_000))
        if f == 3:
            return int((a < 40 and e in (0, 1)) or (40 <= a < 60 and e in (1, 2, 3)) or (a >= 60 and e in (2, 3, 4)))
        if f == 4:
            return int((a < 40 and (e in (0, 1)) == (50_000 <= s <= 100_000)) or (40 <= a < 60 and (e in (1, 2, 3)) == (75_000 <= s <= 125_000)) or (a >= 60 and (e in (2, 3, 4)) == (25_000 <= s <= 75_000)))
        if f == 5:
            return int((a < 40 and ((50_000 <= s <= 100_000) == (100_000 <= l <= 300_000))) or (40 <= a < 60 and ((75_000 <= s <= 125_000) == (200_000 <= l <= 400_000))) or (a >= 60 and ((25_000 <= s <= 75_000) == (300_000 <= l <= 500_000))))
        if f == 6:
            t = s + c
            return int((a < 40 and 50_000 <= t <= 100_000) or (40 <= a < 60 and 75_000 <= t <= 125_000) or (a >= 60 and 25_000 <= t <= 75_000))
        if f == 7:
            return int(2 * (s + c) / 3 - l / 5 - 20_000 > 0)
        if f == 8:
            return int(2 * (s + c) / 3 - 5000 * e - 20_000 > 0)
        if f == 9:
            return int(2 * (s + c) / 3 - 5000 * e - l / 5 - 10_000 > 0)
        equity = h * hy / 10 if hy > 20 else 0
        return int(2 * (s + c) / 3 - 5000 * e + equity / 5 - 10_000 > 0)

    def _next(self):
        r = self.rng
        salary = r.uniform(20_000, 150_000)
        commission = 0.0 if salary >= 75_000 else r.uniform(10_000, 75_000)
        age = int(r.integers(20, 81))
        elevel = int(r.integers(0, 5))
        car = int(r.integers(1, 21))
        zipcode = int(r.integers(0, 9))
        hvalue = (9 - zipcode) * 100_000 * r.uniform(0.5, 1.5)
        hyears = int(r.integers(1, 31))
        loan = r.uniform(0, 500_000)
        y = self._label(salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan)
        if self.perturbation > 0:
            p = self.perturbation
            salary += r.uniform(-p, p) * 130_000
            commission += r.uniform(-p, p) * 65_000 if commission else 0
            age += int(r.uniform(-p, p) * 60)
            hvalue += r.uniform(-p, p) * 450_000
            hyears += int(r.uniform(-p, p) * 29)
            loan += r.uniform(-p, p) * 500_000
        x = dict(salary=round(salary, 2), commission=round(commission, 2), age=age, elevel=elevel, car=car,
                 zipcode=zipcode, hvalue=round(hvalue, 2), hyears=hyears, loan=round(loan, 2))
        return x, y


class STAGGER(Generator):
    """STAGGER concepts (Schlimmer & Granger, 1986).

    Three nominal attributes: size {small, medium, large}, color {red, green,
    blue}, shape {circle, square, triangle}. Concept 1: size=small AND
    color=red; concept 2: color=green OR shape=circle; concept 3: size=medium
    OR size=large. Attributes are integer-coded 0-2.
    """

    task = "classification"
    n_classes = 2
    feature_names = ["size", "color", "shape"]

    def __init__(self, concept: int = 1, balance: bool = False, seed: int = 1):
        super().__init__(seed)
        assert concept in (1, 2, 3)
        self.concept, self.balance = concept, balance
        self._want = 0

    def _label(self, size, color, shape):
        if self.concept == 1:
            return int(size == 0 and color == 0)
        if self.concept == 2:
            return int(color == 1 or shape == 0)
        return int(size in (1, 2))

    def _next(self):
        while True:
            size, color, shape = (int(v) for v in self.rng.integers(0, 3, 3))
            y = self._label(size, color, shape)
            if not self.balance or y == self._want:
                self._want = 1 - self._want
                return dict(size=size, color=color, shape=shape), y


class Hyperplane(Generator):
    """Rotating hyperplane (Hulten, Spencer & Domingos, 2001).

    ``n_features`` in U(0, 1); label is 1 iff ``sum(w_i x_i) >= sum(w_i)/2``.
    ``n_drift_features`` weights change by ``mag_change`` per instance
    (intrinsic incremental drift, sign flipped with prob ``sigma``). ``concept``
    just re-seeds the initial weights so two concepts can be mixed abruptly.
    """

    task = "classification"
    n_classes = 2

    def __init__(self, n_features: int = 10, n_drift_features: int = 2, mag_change: float = 0.0,
                 noise: float = 0.05, sigma: float = 0.1, concept: int = 1, seed: int = 1):
        super().__init__(seed)
        self.n, self.k, self.mag, self.noise, self.sigma = n_features, n_drift_features, mag_change, noise, sigma
        wrng = np.random.default_rng(1000 * seed + concept)
        self.w = wrng.uniform(0, 1, n_features)
        self.dir = np.ones(n_features)

    @property
    def feature_names(self):
        return [f"x{i + 1}" for i in range(self.n)]

    def _next(self):
        x = self.rng.uniform(0, 1, self.n)
        y = int(float(x @ self.w) >= self.w.sum() / 2)
        if self.rng.random() < self.noise:
            y = 1 - y
        if self.mag:
            self.w[: self.k] += self.dir[: self.k] * self.mag
            flip = self.rng.random(self.k) < self.sigma
            self.dir[: self.k][flip] *= -1
        return dict(zip(self.feature_names, np.round(x, 5))), y


class RandomRBF(Generator):
    """Random radial basis function stream (MOA, Bifet et al., 2010).

    ``n_centroids`` Gaussian blobs with random centre, class and std in U(0,1).
    ``speed`` moves centroids each instance (intrinsic drift, MOA's
    RandomRBFGeneratorDrift). ``concept`` re-seeds the centroid layout.
    """

    task = "classification"

    def __init__(self, n_features: int = 10, n_classes: int = 2, n_centroids: int = 50, speed: float = 0.0,
                 n_drift_centroids: int = 50, concept: int = 1, seed: int = 1):
        super().__init__(seed)
        self.n, self.n_classes, self.m, self.speed, self.kd = n_features, n_classes, n_centroids, speed, n_drift_centroids
        crng = np.random.default_rng(1000 * seed + concept)
        self.centres = crng.uniform(0, 1, (n_centroids, n_features))
        self.labels = crng.integers(0, n_classes, n_centroids)
        self.stds = crng.uniform(0, 1, n_centroids)
        w = crng.uniform(0, 1, n_centroids)
        self.weights = w / w.sum()
        d = crng.normal(size=(n_centroids, n_features))
        self.dirs = d / np.linalg.norm(d, axis=1, keepdims=True)

    @property
    def feature_names(self):
        return [f"x{i + 1}" for i in range(self.n)]

    def _next(self):
        c = self.rng.choice(self.m, p=self.weights)
        v = self.rng.normal(size=self.n)
        v *= self.rng.normal() * self.stds[c] / np.linalg.norm(v)
        x = self.centres[c] + v
        if self.speed:
            self.centres[: self.kd] += self.dirs[: self.kd] * self.speed
            out = (self.centres[: self.kd] < 0) | (self.centres[: self.kd] > 1)
            self.dirs[: self.kd][out] *= -1
            self.centres[: self.kd] = np.clip(self.centres[: self.kd], 0, 1)
        return dict(zip(self.feature_names, np.round(x, 5))), int(self.labels[c])


class RandomTree(Generator):
    """Random tree stream (MOA, Bifet et al., 2010; after Domingos & Hulten, 2000).

    A random decision tree over ``n_numeric`` numeric and ``n_nominal``
    nominal (``n_values`` each) attributes labels each uniformly random
    instance. ``concept`` re-seeds the tree.
    """

    task = "classification"

    def __init__(self, n_numeric: int = 5, n_nominal: int = 5, n_values: int = 5, n_classes: int = 2,
                 max_depth: int = 5, min_depth: int = 3, leaf_prob: float = 0.15, concept: int = 1, seed: int = 1):
        super().__init__(seed)
        self.nn, self.nc, self.nv, self.n_classes = n_numeric, n_nominal, n_values, n_classes
        trng = np.random.default_rng(1000 * seed + concept)
        self.tree = self._build(trng, 0, max_depth, min_depth, leaf_prob, [(0.0, 1.0)] * n_numeric, list(range(n_nominal)))

    @property
    def feature_names(self):
        return [f"num{i + 1}" for i in range(self.nn)] + [f"nom{i + 1}" for i in range(self.nc)]

    def _build(self, r, depth, max_depth, min_depth, leaf_prob, ranges, nominals):
        if depth >= max_depth or (depth >= min_depth and r.random() < leaf_prob) or (not nominals and not ranges):
            return ("leaf", int(r.integers(0, self.n_classes)))
        if nominals and (not ranges or r.random() < 0.5):
            a = nominals[int(r.integers(len(nominals)))]
            rest = [n for n in nominals if n != a]
            return ("nom", a, [self._build(r, depth + 1, max_depth, min_depth, leaf_prob, ranges, rest) for _ in range(self.nv)])
        a = int(r.integers(len(ranges)))
        lo, hi = ranges[a]
        split = r.uniform(lo, hi)
        left, right = list(ranges), list(ranges)
        left[a], right[a] = (lo, split), (split, hi)
        return ("num", a, split,
                self._build(r, depth + 1, max_depth, min_depth, leaf_prob, left, nominals),
                self._build(r, depth + 1, max_depth, min_depth, leaf_prob, right, nominals))

    def _classify(self, node, num, nom):
        while node[0] != "leaf":
            if node[0] == "nom":
                node = node[2][nom[node[1]]]
            else:
                node = node[3] if num[node[1]] < node[2] else node[4]
        return node[1]

    def _next(self):
        num = self.rng.uniform(0, 1, self.nn)
        nom = self.rng.integers(0, self.nv, self.nc)
        y = self._classify(self.tree, num, nom)
        x = dict(zip(self.feature_names, [*np.round(num, 5), *map(int, nom)]))
        return x, y


class LED(Generator):
    """LED digit display (Breiman et al., 1984; MOA LEDGenerator / LEDGeneratorDrift).

    Seven segment bits for digits 0-9, each flipped with probability
    ``noise``, plus 17 irrelevant random bits. ``n_swapped`` swaps relevant
    segments with irrelevant attributes (MOA's drift mechanism) — set via
    ``concept`` so concepts differ by which attributes carry the signal.
    """

    task = "classification"
    n_classes = 10
    SEGMENTS = np.array([
        [1, 1, 1, 0, 1, 1, 1], [0, 0, 1, 0, 0, 1, 0], [1, 0, 1, 1, 1, 0, 1], [1, 0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 1, 1], [1, 1, 0, 1, 1, 1, 1], [1, 0, 1, 0, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1],
    ])

    def __init__(self, noise: float = 0.1, n_irrelevant: int = 17, concept: int = 1, seed: int = 1):
        super().__init__(seed)
        self.noise, self.ni = noise, n_irrelevant
        self.perm = np.arange(7 + n_irrelevant)
        n_swapped = concept - 1  # concept 1 = no swap, concept k swaps k-1 segments
        if n_swapped:
            for i in range(min(n_swapped, 7, n_irrelevant)):
                self.perm[i], self.perm[7 + i] = self.perm[7 + i], self.perm[i]

    @property
    def feature_names(self):
        return [f"a{i + 1}" for i in range(7 + self.ni)]

    def _next(self):
        d = int(self.rng.integers(0, 10))
        seg = self.SEGMENTS[d] ^ (self.rng.random(7) < self.noise)
        bits = np.concatenate([seg.astype(int), self.rng.integers(0, 2, self.ni)])
        return dict(zip(self.feature_names, map(int, bits[self.perm]))), d


class Waveform(Generator):
    """Waveform (Breiman et al., 1984; MOA WaveformGenerator).

    Three classes, each a random convex combination of two of three base
    triangular waves sampled at 21 points plus N(0,1) noise. ``n_irrelevant``
    adds pure-noise attributes (MOA's WaveformGenerator -n). ``concept``
    permutes which attribute positions carry the waves (WaveformGeneratorDrift).
    """

    task = "classification"
    n_classes = 3

    def __init__(self, n_irrelevant: int = 0, concept: int = 1, seed: int = 1):
        super().__init__(seed)
        self.ni = n_irrelevant
        t = np.arange(21)
        self.waves = np.array([np.maximum(6 - np.abs(t - 6), 0), np.maximum(6 - np.abs(t - 10), 0), np.maximum(6 - np.abs(t - 14), 0)], float)
        self.pairs = [(0, 1), (1, 2), (0, 2)]
        n = 21 + n_irrelevant
        self.perm = np.arange(n) if concept == 1 else np.random.default_rng(1000 * seed + concept).permutation(n)

    @property
    def feature_names(self):
        return [f"x{i + 1}" for i in range(21 + self.ni)]

    def _next(self):
        c = int(self.rng.integers(0, 3))
        i, j = self.pairs[c]
        u = self.rng.random()
        x = u * self.waves[i] + (1 - u) * self.waves[j] + self.rng.normal(size=21)
        if self.ni:
            x = np.concatenate([x, self.rng.normal(size=self.ni)])
        return dict(zip(self.feature_names, np.round(x[self.perm], 4))), c


class Mixed(Generator):
    """Mixed concept (Gama et al., 2004).

    Two boolean attributes ``v, w`` and two numeric ``x, y`` in U(0,1). Class
    is 1 if at least two of ``v``, ``w`` and ``y < 0.5 + 0.3 sin(3 pi x)``
    hold. ``concept=2`` is the inverted concept.
    """

    task = "classification"
    n_classes = 2
    feature_names = ["v", "w", "x", "y"]

    def __init__(self, concept: int = 1, seed: int = 1):
        super().__init__(seed)
        self.invert = concept == 2

    def _next(self):
        v, w = (int(b) for b in self.rng.integers(0, 2, 2))
        x, y = self.rng.uniform(0, 1, 2)
        z = int(y < 0.5 + 0.3 * math.sin(3 * math.pi * x))
        label = int(v + w + z >= 2)
        if self.invert:
            label = 1 - label
        return dict(v=v, w=w, x=round(float(x), 5), y=round(float(y), 5)), label


CLASSIFICATION_GENERATORS = {g.__name__: g for g in (SEA, Agrawal, STAGGER, Hyperplane, RandomRBF, RandomTree, LED, Waveform, Mixed)}
