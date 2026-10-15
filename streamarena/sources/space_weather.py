"""Geomagnetic activity from GFZ Potsdam (no key, CC BY 4.0).

https://kp.gfz.de/app/json/ serves the definitive Kp (3-hourly, since 1932)
and the half-hourly Hp30 index (since 1985). Live values come from the same
endpoint (nowcast status). Label: a geomagnetic storm (Kp >= 5, i.e. NOAA
G1 or stronger) begins within the next 3 hours.
"""

from __future__ import annotations

import pandas as pd

from .base import FeatureContract, Source, get

URL = "https://kp.gfz.de/app/json/"


class GeomagneticStorm(Source):
    provenance = "GFZ German Research Centre for Geosciences, Kp/Hp30 indices, CC BY 4.0"
    start = "2000-01-01"
    poll_seconds = 1800

    def __init__(self, storm_threshold: float = 5.0, lookahead_steps: int = 6):
        self.name = "space_storm_kp"
        self.thr, self.ahead = storm_threshold, lookahead_steps
        self.contract = FeatureContract(features=["hp30", "kp"], target_col="storm_ahead", horizon=1, n_lags=12,
                                        lag_cols=["hp30"], to_class=lambda s: s.astype("Int64"), label_is_current=True)

    def _fetch(self, index: str, s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
        j = get(URL, params=dict(start=s.strftime("%Y-%m-%dT%H:%M:%SZ"), end=e.strftime("%Y-%m-%dT%H:%M:%SZ"), index=index)).json()
        return pd.DataFrame({"ts": pd.to_datetime(j["datetime"], utc=True), index.lower(): j[index]})

    def _assemble(self, hp: pd.DataFrame, kp: pd.DataFrame) -> pd.DataFrame:
        hp = hp.drop_duplicates("ts").set_index("ts").sort_index()
        kp = kp.drop_duplicates("ts").set_index("ts").sort_index().reindex(hp.index, method="ffill")
        df = hp.join(kp).reset_index()
        future_max = df["hp30"][::-1].rolling(self.ahead, min_periods=1).max()[::-1].shift(-1)
        df["storm_ahead"] = ((future_max >= self.thr) & (df["hp30"] < self.thr)).astype(float)
        return df.dropna()

    def fetch_history(self, start, end):
        hp, kp = [], []
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        while s < e:
            n = min(s + pd.DateOffset(years=1), e)
            hp.append(self._fetch("Hp30", s, n))
            kp.append(self._fetch("Kp", s, n))
            s = n
        return self._assemble(pd.concat(hp), pd.concat(kp))

    def fetch_latest(self):
        now = pd.Timestamp.utcnow()
        s = now - pd.Timedelta(days=3)
        return self._assemble(self._fetch("Hp30", s, now), self._fetch("Kp", s, now))
