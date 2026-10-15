"""NOAA CO-OPS Tides & Currents water levels (no key).

https://api.tidesandcurrents.noaa.gov/api/prod/ — 6-minute verified water
levels; the API serves at most 31 days of 6-minute data per call. Default
station 8518750 is The Battery, New York.
"""

from __future__ import annotations

import pandas as pd

from .base import FeatureContract, Source, get

URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"


class NOAATides(Source):
    provenance = "NOAA CO-OPS Tides & Currents API, public domain"
    start = "2016-01-01"
    poll_seconds = 360

    def __init__(self, station: str = "8518750", horizon: int = 10):
        self.station = station
        self.name = "tide_level_battery"
        self.contract = FeatureContract(features=["water_level", "sigma"], target_col="water_level", horizon=horizon, n_lags=12)

    def _call(self, **params) -> pd.DataFrame:
        base = dict(station=self.station, product="water_level", datum="MLLW", units="metric", time_zone="gmt", format="json")
        r = get(URL, params={**base, **params})
        data = r.json().get("data", [])
        if not data:
            return pd.DataFrame(columns=["ts", "water_level", "sigma"])
        df = pd.DataFrame(data).rename(columns={"t": "ts", "v": "water_level", "s": "sigma"})[["ts", "water_level", "sigma"]]
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["water_level"] = pd.to_numeric(df["water_level"], errors="coerce")
        df["sigma"] = pd.to_numeric(df["sigma"], errors="coerce")
        return df

    def fetch_history(self, start, end):
        frames = []
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        while s < e:
            n = min(s + pd.Timedelta(days=31), e)
            frames.append(self._call(begin_date=s.strftime("%Y%m%d"), end_date=(n - pd.Timedelta(days=1)).strftime("%Y%m%d")))
            s = n
        df = pd.concat(frames, ignore_index=True).drop_duplicates("ts").set_index("ts").sort_index()
        return df.resample("6min").mean().interpolate(limit=10).dropna().reset_index()

    def fetch_latest(self):
        return self._call(date="recent")
