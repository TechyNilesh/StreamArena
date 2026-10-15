"""Classic regression generators, re-implemented from scratch."""

from __future__ import annotations

import math

import numpy as np

from .base import Generator


class Friedman(Generator):
    """Friedman #1 (Friedman, 1991).

    Ten features in U(0,1), only five matter:
    ``y = 10 sin(pi x1 x2) + 20 (x3 - 0.5)^2 + 10 x4 + 5 x5 + N(0,1)``.
    ``concept`` selects which five features carry the signal (concept 1 =
    x1..x5, concept 2 = x6..x10, concept 3 = x3..x7), mirroring the
    local/global feature-swap drifts of River's FriedmanDrift.
    """

    task = "regression"

    def __init__(self, concept: int = 1, noise: float = 1.0, seed: int = 1):
        super().__init__(seed)
        self.idx = {1: [0, 1, 2, 3, 4], 2: [5, 6, 7, 8, 9], 3: [2, 3, 4, 5, 6]}[concept]
        self.noise = noise

    feature_names = [f"x{i + 1}" for i in range(10)]

    def _next(self):
        x = self.rng.uniform(0, 1, 10)
        a, b, c, d, e = x[self.idx]
        y = 10 * math.sin(math.pi * a * b) + 20 * (c - 0.5) ** 2 + 10 * d + 5 * e + self.rng.normal(0, self.noise)
        return dict(zip(self.feature_names, np.round(x, 5))), round(float(y), 5)


class HyperplaneRegression(Generator):
    """Linear hyperplane target (MOA HyperPlaneRegression).

    ``y = sum(w_i x_i) + N(0, noise)`` with ``x`` in U(0,1). ``mag_change``
    gives intrinsic incremental drift of the first ``n_drift_features``
    weights; ``concept`` re-seeds the weights.
    """

    task = "regression"

    def __init__(self, n_features: int = 10, n_drift_features: int = 2, mag_change: float = 0.0,
                 noise: float = 0.1, sigma: float = 0.1, concept: int = 1, seed: int = 1):
        super().__init__(seed)
        self.n, self.k, self.mag, self.noise, self.sigma = n_features, n_drift_features, mag_change, noise, sigma
        self.w = np.random.default_rng(1000 * seed + concept).uniform(-1, 1, n_features)
        self.dir = np.ones(n_features)

    @property
    def feature_names(self):
        return [f"x{i + 1}" for i in range(self.n)]

    def _next(self):
        x = self.rng.uniform(0, 1, self.n)
        y = float(x @ self.w) + self.rng.normal(0, self.noise)
        if self.mag:
            self.w[: self.k] += self.dir[: self.k] * self.mag
            flip = self.rng.random(self.k) < self.sigma
            self.dir[: self.k][flip] *= -1
        return dict(zip(self.feature_names, np.round(x, 5))), round(y, 5)


class Mv(Generator):
    """Mv artificial regression (Torgo's "Mv" set, as in River ``synth.Mv``).

    Ten mixed-type attributes with dependencies between them and a piecewise
    target. Nominal attributes are integer-coded. ``concept=2`` swaps the two
    arms of the main rule so the drift wrapper can mix the concepts.
    """

    task = "regression"
    feature_names = [f"x{i + 1}" for i in range(10)]

    def __init__(self, concept: int = 1, seed: int = 1):
        super().__init__(seed)
        self.swap = concept == 2

    def _next(self):
        r = self.rng
        x1 = r.uniform(-5, 5)
        x2 = r.uniform(-15, -10)
        x3 = 1 if x1 > 0 else int(r.choice([0, 2, 3]))        # 0 red, 1 green, 2 brown, 3 blue
        x4 = -0.5 * x1 if x3 == 2 else r.uniform(-1, 1)
        x5 = x2 + 1 if x3 == 0 else r.uniform(-1, 1)
        x6 = r.uniform(0, 1) * 10 if x3 in (1, 2) else r.uniform(-1, 1)
        x7 = 1 if x6 >= 5 else 0                               # yes / no
        x8 = int(r.integers(0, 3))                             # normal / large / small
        x9 = r.uniform(-1, 1) if x8 == 0 else r.uniform(-5, 5)
        x10 = r.uniform(-1, 1)
        arm_a = x3 == 1
        if self.swap:
            arm_a = not arm_a
        if arm_a:
            y = x4 + x5 + x6 * 0.5 + x9
        else:
            y = 2 * x1 - 3 * x4 + x10 * 5
        if x7 == 1:
            y += 3
        if x8 == 1:
            y *= 1.5
        y += r.normal(0, 0.5)
        x = dict(zip(self.feature_names, [round(x1, 5), round(x2, 5), x3, round(x4, 5), round(x5, 5), round(x6, 5), x7, x8, round(x9, 5), round(x10, 5)]))
        return x, round(float(y), 5)


class Planes2D(Generator):
    """2-D planes (Breiman et al., 1984; River ``synth.Planes2D``).

    ``x1`` in {-1, 1}, ``x2..x10`` in U(-1, 1).
    ``y = 3 + 3 x2 + 2 x3 + x4 + N(0,1)`` if ``x1 = 1`` else
    ``y = -3 + 3 x5 + 2 x6 + x7 + N(0,1)``. ``concept=2`` negates ``x1``'s
    role, i.e. the two planes trade places.
    """

    task = "regression"
    feature_names = [f"x{i + 1}" for i in range(10)]

    def __init__(self, concept: int = 1, seed: int = 1):
        super().__init__(seed)
        self.flip = concept == 2

    def _next(self):
        x1 = int(self.rng.choice([-1, 1]))
        rest = self.rng.uniform(-1, 1, 9)
        s = -x1 if self.flip else x1
        if s == 1:
            y = 3 + 3 * rest[0] + 2 * rest[1] + rest[2]
        else:
            y = -3 + 3 * rest[3] + 2 * rest[4] + rest[5]
        y += self.rng.normal()
        return dict(zip(self.feature_names, [x1, *np.round(rest, 5)])), round(float(y), 5)


REGRESSION_GENERATORS = {g.__name__: g for g in (Friedman, HyperplaneRegression, Mv, Planes2D)}
