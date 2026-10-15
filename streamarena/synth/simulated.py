"""API-like synthetic simulators (new in StreamArena v2).

The classic generators produce i.i.d. instances with no notion of time. Real
API streams do not look like that: they are autocorrelated, seasonal, bursty
and their labels arrive later. These three simulators reproduce those
properties with known ground truth so results on them can be compared with
the real-API datasets they imitate:

* :class:`SeasonalSensor`      ~ weather / air-quality / tide sensors
* :class:`RegimeSwitchingWalk` ~ prices / energy markets
* :class:`PoissonDemand`       ~ bike-share / taxi demand counts

All three expose the same *feature contract* the real sources use: the raw
observation, ``n_lags`` lagged observations, and calendar features
(sin/cos of hour-of-day and day-of-year). The target is the observation
``horizon`` steps ahead, so labels are delayed exactly as in deployment.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .base import Generator

HOURS_PER_YEAR = 8766.0


def _calendar(step_hours: float, t: int) -> dict:
    h = t * step_hours
    hod = (h % 24) / 24
    doy = (h % HOURS_PER_YEAR) / HOURS_PER_YEAR
    dow = ((h // 24) % 7) / 7
    return {
        "hour_sin": round(math.sin(2 * math.pi * hod), 5), "hour_cos": round(math.cos(2 * math.pi * hod), 5),
        "dow_sin": round(math.sin(2 * math.pi * dow), 5), "dow_cos": round(math.cos(2 * math.pi * dow), 5),
        "doy_sin": round(math.sin(2 * math.pi * doy), 5), "doy_cos": round(math.cos(2 * math.pi * doy), 5),
    }


class _Lagged(Generator):
    """Shared machinery: keep ``n_lags`` past values and emit target at ``horizon``."""

    def __init__(self, n_lags: int, horizon: int, step_hours: float, seed: int):
        super().__init__(seed)
        self.n_lags, self.horizon, self.step_hours = n_lags, horizon, step_hours
        self._t = 0
        self._hist = deque(maxlen=n_lags + 1)
        self._future = deque()
        # warm-up so lags and the first horizon targets exist
        for _ in range(n_lags + horizon):
            self._future.append(self._observe())

    def _observe(self) -> float:
        raise NotImplementedError

    def _extra(self) -> dict:
        return {}

    def _next(self):
        # value observed now is the oldest in the future buffer; append a new one
        self._future.append(self._observe())
        v = self._future.popleft()
        self._hist.append(v)
        x = {"value": round(v, 5)}
        hist = list(self._hist)
        for k in range(1, self.n_lags + 1):
            x[f"lag{k}"] = round(hist[-1 - k], 5) if len(hist) > k else round(hist[0], 5)
        x.update(_calendar(self.step_hours, self._t))
        x.update(self._extra())
        y = round(self._future[self.horizon - 1], 5)
        self._t += 1
        return x, y

    @property
    def feature_names(self):
        names = ["value"] + [f"lag{k}" for k in range(1, self.n_lags + 1)]
        names += ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]
        return names + list(self._extra().keys())


class SeasonalSensor(_Lagged):
    """Autocorrelated sensor with daily and yearly cycles and slow degradation.

    ``value_t = mean + A_year sin(2 pi t/yr) + A_day sin(2 pi t/day + phase)
    + AR(1) noise + bias_t``, where ``bias_t`` grows by ``degradation`` per step
    (sensor calibration drift) and ``dropout`` is the probability the reading
    is stuck at the previous value (a stuck sensor). ``concept`` 1 vs 2
    changes the seasonal amplitudes and phase (e.g. a relocated sensor), for
    use with the drift wrapper.
    """

    task = "regression"

    def __init__(self, mean: float = 15.0, amp_year: float = 10.0, amp_day: float = 5.0, ar: float = 0.8,
                 noise: float = 1.0, degradation: float = 0.0, dropout: float = 0.0, horizon: int = 1,
                 n_lags: int = 6, step_hours: float = 1.0, concept: int = 1, seed: int = 1):
        self.mean, self.ay, self.ad, self.ar, self.noise = mean, amp_year, amp_day, ar, noise
        self.degradation, self.dropout = degradation, dropout
        self.phase = 0.0 if concept == 1 else math.pi / 2
        if concept == 2:
            self.ay, self.ad = amp_year * 0.5, amp_day * 2.0
        self._e = 0.0
        self._bias = 0.0
        self._prev = mean
        self._k = 0
        super().__init__(n_lags, horizon, step_hours, seed)

    def _observe(self):
        h = self._k * self.step_hours
        self._k += 1
        self._e = self.ar * self._e + self.rng.normal(0, self.noise)
        self._bias += self.degradation
        season = self.ay * math.sin(2 * math.pi * h / HOURS_PER_YEAR) + self.ad * math.sin(2 * math.pi * h / 24 + self.phase)
        v = self.mean + season + self._e + self._bias
        if self.rng.random() < self.dropout:
            v = self._prev
        self._prev = v
        return v


class RegimeSwitchingWalk(_Lagged):
    """Log-price random walk whose drift and volatility follow a hidden Markov chain.

    Regimes: calm (low vol, mild positive drift), volatile (high vol, zero
    drift), crash (high vol, negative drift). ``target`` is ``"return"``
    (regression: log return over ``horizon`` steps) or ``"direction"``
    (classification: 1 if that return is positive). Features are the last
    ``n_lags`` returns, a realised-volatility estimate and calendar terms.
    ``concept=2`` doubles the regime persistence and swaps the drift signs,
    i.e. a structural market change.
    """

    def __init__(self, target: str = "return", horizon: int = 1, n_lags: int = 10, step_hours: float = 1 / 60,
                 concept: int = 1, seed: int = 1):
        assert target in ("return", "direction")
        self.task = "regression" if target == "return" else "classification"
        self.n_classes = None if target == "return" else 2
        self.target_kind = target
        self.mu = np.array([2e-5, 0.0, -2e-4])
        self.sig = np.array([5e-4, 2e-3, 4e-3])
        stay = 0.999 if concept == 1 else 0.9995
        if concept == 2:
            self.mu = -self.mu
        self.P = np.full((3, 3), (1 - stay) / 2)
        np.fill_diagonal(self.P, stay)
        self._regime = 0
        self._logp = math.log(100.0)
        self._rets = deque(maxlen=60)
        super().__init__(n_lags, horizon, step_hours, seed)

    def _observe(self):
        self._regime = int(self.rng.choice(3, p=self.P[self._regime]))
        r = self.rng.normal(self.mu[self._regime], self.sig[self._regime])
        self._logp += r
        self._rets.append(r)
        return r  # the "value" stream is the per-step log return

    def _extra(self):
        rv = float(np.std(self._rets)) if len(self._rets) > 1 else 0.0
        return {"realised_vol": round(rv, 7), "log_price": round(self._logp, 5)}

    def _next(self):
        x, y = super()._next()
        if self.horizon > 1:  # multi-step return = sum of next horizon returns
            y = round(float(sum(list(self._future)[: self.horizon])), 7)
        if self.target_kind == "direction":
            y = int(y > 0)
        return x, y


class PoissonDemand(_Lagged):
    """Hourly demand counts at one station: ``count ~ Poisson(lambda_t)``.

    ``lambda_t = base * hour_profile[h] * weekday_profile[d] * weather_t``
    where ``weather_t`` is a hidden AR(1) multiplier (rain suppresses
    demand) and ``base`` grows by ``growth`` per year (slow trend). Target is
    the count ``horizon`` hours ahead. ``concept=2`` shifts the hour profile
    by six hours (e.g. a commuter station turning into a leisure station).
    """

    task = "regression"

    def __init__(self, base: float = 20.0, growth: float = 0.1, horizon: int = 1, n_lags: int = 24,
                 concept: int = 1, seed: int = 1):
        self.base, self.growth = base, growth
        hours = np.arange(24)
        profile = 0.3 + np.exp(-((hours - 8) ** 2) / 8) + 1.2 * np.exp(-((hours - 18) ** 2) / 10)
        self.hour_profile = np.roll(profile, 6 if concept == 2 else 0)
        self.day_profile = np.array([1.0, 1.0, 1.0, 1.0, 1.1, 0.8, 0.7])
        self._w = 0.0
        self._k = 0
        super().__init__(n_lags, horizon, 1.0, seed)

    def _observe(self):
        h = self._k
        self._k += 1
        self._w = 0.9 * self._w + self.rng.normal(0, 0.2)
        weather = math.exp(min(self._w, 0.0))  # only bad weather suppresses
        trend = 1 + self.growth * (h / HOURS_PER_YEAR)
        lam = self.base * trend * self.hour_profile[h % 24] * self.day_profile[(h // 24) % 7] * weather
        return float(self.rng.poisson(lam))


SIMULATED_GENERATORS = {g.__name__: g for g in (SeasonalSensor, RegimeSwitchingWalk, PoissonDemand)}
