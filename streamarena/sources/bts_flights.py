"""BTS Airline On-Time Performance (no key).

https://www.transtats.bts.gov/ — monthly zipped CSVs of every scheduled US
domestic flight by reporting carriers. Event stream ordered by scheduled
departure; label is arrival delay >= 15 minutes (``ArrDel15``), available
only after the flight lands. Features are known at scheduling time only.
"""

from __future__ import annotations

import zipfile

import pandas as pd

from .base import CACHE_ROOT, FeatureContract, Source, cached_download

URL = "https://www.transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip"
USECOLS = ["FlightDate", "CRSDepTime", "CRSArrTime", "Reporting_Airline", "OriginAirportID", "DestAirportID", "Distance", "CRSElapsedTime", "ArrDel15", "Cancelled"]


class FlightDelays(Source):
    provenance = "US Bureau of Transportation Statistics, Reporting Carrier On-Time Performance, public domain"
    start = "2016-01-01"

    def __init__(self, years: range = range(2016, 2026)):
        self.years = years
        self.name = "flight_delay_bts"
        self.contract = FeatureContract(features=["carrier", "origin", "dest", "distance", "sched_elapsed", "sched_dep_min", "sched_arr_min"],
                                        target_col="delayed", n_lags=0, to_class=lambda s: s.astype("Int64"), label_is_current=True)

    def fetch_history(self, start, end):
        frames = []
        for y in self.years:
            for m in range(1, 13):
                if pd.Timestamp(f"{y}-{m:02d}-01") >= pd.Timestamp(end):
                    break
                try:
                    path = cached_download(URL.format(y=y, m=m), CACHE_ROOT / "bts", f"bts_{y}_{m:02d}.zip")
                    with zipfile.ZipFile(path) as z:
                        csv = next(n for n in z.namelist() if n.endswith(".csv"))
                        df = pd.read_csv(z.open(csv), usecols=USECOLS, low_memory=False)
                except Exception as e:
                    print(f"  [bts] {y}-{m:02d}: unavailable ({e})", flush=True)
                    continue
                df = df[(df["Cancelled"] == 0)].dropna(subset=["ArrDel15"])
                dep = df["CRSDepTime"].astype(int).clip(0, 2359)
                out = pd.DataFrame({
                    "ts": pd.to_datetime(df["FlightDate"]) + pd.to_timedelta(dep // 100, unit="h") + pd.to_timedelta(dep % 100, unit="m"),
                    "carrier": df["Reporting_Airline"].astype("category").cat.codes if False else df["Reporting_Airline"],
                    "origin": df["OriginAirportID"], "dest": df["DestAirportID"], "distance": df["Distance"],
                    "sched_elapsed": df["CRSElapsedTime"], "sched_dep_min": (dep // 100) * 60 + dep % 100,
                    "sched_arr_min": (df["CRSArrTime"].astype(int) // 100) * 60 + df["CRSArrTime"].astype(int) % 100,
                    "delayed": df["ArrDel15"].astype(float),
                })
                frames.append(out.iloc[::10])  # thin at read time; final stream thinned to 1M rows
                print(f"  [bts] {y}-{m:02d}: {len(df):,} flights", flush=True)
        df = pd.concat(frames, ignore_index=True)
        df["carrier"] = df["carrier"].astype("category").cat.codes
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize("UTC")  # local scheduled times, treated as a global ordering key
        return df[(df["ts"] >= pd.Timestamp(start, tz="UTC")) & (df["ts"] < pd.Timestamp(end, tz="UTC"))]
