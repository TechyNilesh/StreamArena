"""USGS earthquake catalog (ComCat) via the FDSN event web service (no key).

https://earthquake.usgs.gov/fdsnws/event/1/ — one row per event, M2.5+
worldwide. This is an *event stream*: features describe the current event
(location, depth, time since previous event, recent activity) and the label
is whether the event itself is M4.5+ (the USGS "significant for alerting"
threshold). Magnitude is never a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureContract, Source, get, month_ranges

URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
FEATURES = ["latitude", "longitude", "depth", "secs_since_prev", "count_1d", "count_7d", "max_mag_7d", "mean_mag_30"]


class EarthquakeSignificance(Source):
    provenance = "USGS Earthquake Hazards Program ComCat, FDSN event service, public domain"
    start = "2010-01-01"
    poll_seconds = 300

    def __init__(self, min_magnitude: float = 2.5, significant: float = 4.5):
        self.min_mag, self.sig = min_magnitude, significant
        self.name = "quake_significant"
        self.contract = FeatureContract(features=FEATURES, target_col="significant", n_lags=5, lag_cols=["mag"],
                                        to_class=lambda s: s.astype("Int64"), label_is_current=True)

    def _query(self, s, e) -> pd.DataFrame:
        r = get(URL, params=dict(format="csv", starttime=s.date().isoformat(), endtime=e.date().isoformat(), minmagnitude=self.min_mag, orderby="time-asc"))
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), usecols=["time", "latitude", "longitude", "depth", "mag"])
        return df.rename(columns={"time": "ts"})

    def _derive(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.dropna(subset=["mag", "depth"]).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        t = df["ts"]
        df["secs_since_prev"] = t.diff().dt.total_seconds().fillna(0)
        s = pd.Series(1.0, index=t)
        df["count_1d"] = s.rolling("1D").sum().values - 1
        df["count_7d"] = s.rolling("7D").sum().values - 1
        # rolling stats of *previous* magnitudes only (shifted so the current magnitude is not leaked)
        prev = pd.Series(df["mag"].shift(1).values, index=t)
        df["max_mag_7d"] = prev.rolling("7D", min_periods=1).max().fillna(self.min_mag).values
        df["mean_mag_30"] = df["mag"].shift(1).rolling(30, min_periods=1).mean().fillna(self.min_mag)
        df["significant"] = (df["mag"] >= self.sig).astype(float)
        return df

    def fetch_history(self, start, end):
        frames = [self._query(s, e) for s, e in month_ranges(start, end)]
        df = self._derive(pd.concat(frames, ignore_index=True))
        # magnitude lags are of previous events, so the current event's mag must not appear as a feature:
        # FeatureContract adds mag_lag1..5 from the shifted column and does not expose "mag" itself.
        return df

    def fetch_latest(self):
        now = pd.Timestamp.utcnow()
        return self._derive(self._query(now - pd.Timedelta(days=30), now + pd.Timedelta(days=1)))
