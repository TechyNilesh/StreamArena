"""Base classes for synthetic stream generators.

Design
------
A :class:`Generator` is an infinite iterator of ``(x, y)`` pairs where ``x`` is
an ordered dict of features and ``y`` is the target (int class index or float).
Everything is driven by a seeded ``numpy.random.Generator`` so a (generator,
seed) pair is fully reproducible.

Two ways to consume one:

* **dataset mode** — :meth:`Generator.to_csv` writes ``n`` instances to a CSV
  with the StreamArena convention (feature columns, then ``target``).
* **live mode** — :meth:`Generator.live` yields instances at ``rate`` per second
  with wall-clock timestamps, so the same generator can feed the deployment
  simulator exactly like a real API source.

Concept drift is *not* baked into individual generators. :class:`DriftStream`
mixes any two generators (or a generator and a re-parameterised copy of
itself) with abrupt, gradual, incremental or recurring transitions, so every
generator shares one drift vocabulary.
"""

from __future__ import annotations

import csv
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import numpy as np

Instance = tuple[dict, object]


class Generator(ABC):
    """Infinite, seeded stream of ``(features, target)``."""

    #: "classification" or "regression"
    task: str = "classification"
    #: number of classes for classification generators
    n_classes: int | None = None

    def __init__(self, seed: int = 1):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # -- subclass API -------------------------------------------------------
    @abstractmethod
    def _next(self) -> Instance:
        """Produce one instance. Must only draw randomness from ``self.rng``."""

    @property
    @abstractmethod
    def feature_names(self) -> list[str]: ...

    # -- iteration ----------------------------------------------------------
    def __iter__(self) -> Iterator[Instance]:
        while True:
            yield self._next()

    def take(self, n: int) -> list[Instance]:
        it = iter(self)
        return [next(it) for _ in range(n)]

    def live(self, rate: float = 10.0, max_instances: int | None = None):
        """Yield ``(timestamp, x, y)`` at ``rate`` instances/second (wall-clock).

        This is what makes a synthetic generator behave like a live API: the
        deployment simulator can apply latency and resource envelopes to it
        without knowing it is synthetic.
        """
        period = 1.0 / rate
        nxt = time.monotonic()
        for i, (x, y) in enumerate(self):
            if max_instances is not None and i >= max_instances:
                return
            now = time.monotonic()
            if now < nxt:
                time.sleep(nxt - now)
            nxt += period
            yield time.time(), x, y

    # -- dataset mode -------------------------------------------------------
    def to_csv(self, path: str | Path, n: int, target_name: str = "target") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        it = iter(self)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([*self.feature_names, target_name])
            for _ in range(n):
                x, y = next(it)
                w.writerow([*x.values(), y])
        return path

    def describe(self) -> dict:
        return {
            "generator": type(self).__name__,
            "task": self.task,
            "n_classes": self.n_classes,
            "n_features": len(self.feature_names),
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


class DriftStream(Generator):
    """Mix two generators with a configurable transition.

    ``kind``:

    * ``"abrupt"``     — switch from ``a`` to ``b`` at ``position``.
    * ``"gradual"``    — sigmoid mixing over ``width`` instances centred at
                         ``position`` (the MOA/River ConceptDriftStream rule).
    * ``"recurring"``  — alternate a/b every ``period`` instances, abruptly.
    * ``"incremental"``— like gradual but the mixture never saturates at 1:
                         after the window the probability of ``b`` stays at
                         ``plateau`` (default 1.0 → identical to gradual).

    Multiple drifts can be chained by nesting ``DriftStream(DriftStream(a, b), c)``.
    """

    def __init__(
        self,
        a: Generator,
        b: Generator,
        kind: str = "abrupt",
        position: int = 50_000,
        width: int = 10_000,
        period: int = 50_000,
        plateau: float = 1.0,
        seed: int = 1,
    ):
        super().__init__(seed)
        assert a.task == b.task, "both streams must share a task"
        assert a.feature_names == b.feature_names, "both streams must share a feature space"
        assert kind in {"abrupt", "gradual", "recurring", "incremental"}
        self.a, self.b = a, b
        self.kind, self.position, self.width, self.period, self.plateau = kind, position, width, period, plateau
        self.task = a.task
        self.n_classes = a.n_classes
        self._ia, self._ib = iter(a), iter(b)
        self._t = 0

    @property
    def feature_names(self):
        return self.a.feature_names

    def p_b(self, t: int) -> float:
        """Probability that instance ``t`` is drawn from stream ``b``."""
        if self.kind == "abrupt":
            return float(t >= self.position)
        if self.kind == "recurring":
            return float((t // self.period) % 2 == 1)
        # gradual / incremental: sigmoid with slope -4/width (MOA convention)
        z = -4.0 * (t - self.position) / self.width
        p = 1.0 / (1.0 + math.exp(z))
        return p * self.plateau if self.kind == "incremental" else p

    def _next(self) -> Instance:
        use_b = self.rng.random() < self.p_b(self._t)
        self._t += 1
        return next(self._ib if use_b else self._ia)

    def describe(self):
        d = super().describe()
        d.update(drift=self.kind, position=self.position, width=self.width, period=self.period,
                 before=self.a.describe(), after=self.b.describe())
        return d
