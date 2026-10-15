"""Urban mobility archives (no key).

* Citi Bike NYC system data: https://s3.amazonaws.com/tripdata/ — yearly
  zips to 2023, monthly zips from 2024. Two streams: hourly pickups per
  station (regression) and per-trip rider type (classification).
* NYC TLC yellow taxi trip records (monthly Parquet):
  https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page — hourly
  pickups per taxi zone (regression).

Both are large; downloads are cached under datasets/.cache and processed in
chunks. Year ranges are configurable so a laptop can build a small slice.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .base import CACHE_ROOT, FeatureContract, Source, cached_download, get

TRIPDATA = "https://s3.amazonaws.com/tripdata/"
TLC = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{ym}.parquet"


def _tripdata_keys() -> list[str]:
    xml = get(TRIPDATA).text
    return re.findall(r"<Key>([^<]+)</Key>", xml)


def _iter_citibike_csvs(years: range):
    """Yield (name, file-like) for every trip CSV in the archive for ``years``."""
    keys = _tripdata_keys()
    for key in keys:
        m = re.match(r"(\d{4})(\d{2})?-citibike-tripdata", key)
        if not m or int(m.group(1)) not in years or not key.endswith(".zip"):
            continue
        path = cached_download(TRIPDATA + key, CACHE_ROOT / "citibike")
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith(".zip"):  # yearly zips contain monthly zips
                    with zipfile.ZipFile(io.BytesIO(z.read(n))) as z2:
                        for n2 in z2.namelist():
                            if n2.endswith(".csv") and "__MACOSX" not in n2:
                                yield n2, z2.open(n2)
                elif n.endswith(".csv") and "__MACOSX" not in n:
                    yield n, z.open(n)


_COLS = {  # old schema -> new schema
    "starttime": "started_at", "stoptime": "ended_at", "start station id": "start_station_id",
    "end station id": "end_station_id", "usertype": "member_casual", "tripduration": "tripduration",
    "start station latitude": "start_lat", "start station longitude": "start_lng",
}


def _read_trips(f, usecols) -> pd.DataFrame:
    df = pd.read_csv(f, low_memory=False)
    df = df.rename(columns={c: _COLS.get(c.strip().lower(), c.strip().lower()) for c in df.columns})
    if "member_casual" in df and df["member_casual"].dtype == object:
        df["member_casual"] = df["member_casual"].replace({"Subscriber": "member", "Customer": "casual"})
    keep = [c for c in usecols if c in df]
    return df[keep]


class CitiBikeDemand(Source):
    provenance = "Citi Bike NYC System Data (Lyft), NYCBS Data Use Policy"
    start = "2016-01-01"

    def __init__(self, years: range = range(2016, 2026), top_stations: int = 50, horizon: int = 1):
        self.years, self.top = years, top_stations
        self.name = "bike_demand_nyc"
        self.contract = FeatureContract(features=["pickups", "dropoffs"], target_col="pickups", horizon=horizon, n_lags=24,
                                        lag_cols=["pickups"], static=["station"], group_col="station")

    def fetch_history(self, start, end):
        counts = []
        for name, f in _iter_citibike_csvs(self.years):
            df = _read_trips(f, ["started_at", "ended_at", "start_station_id", "end_station_id"])
            st = pd.to_datetime(df["started_at"], errors="coerce", format="mixed").dt.floor("h")
            en = pd.to_datetime(df["ended_at"], errors="coerce", format="mixed").dt.floor("h")
            p = pd.DataFrame({"ts": st, "station": df["start_station_id"].astype(str)}).dropna().value_counts().rename("pickups")
            d = pd.DataFrame({"ts": en, "station": df["end_station_id"].astype(str)}).dropna().value_counts().rename("dropoffs")
            counts.append(pd.concat([p, d], axis=1).fillna(0).reset_index())
            print(f"  [citibike] {name}: {len(df):,} trips", flush=True)
        c = pd.concat(counts, ignore_index=True).groupby(["ts", "station"], as_index=False).sum()
        top = c.groupby("station")["pickups"].sum().nlargest(self.top).index
        c = c[c["station"].isin(top)]
        # complete the hourly grid per station (hours with zero trips are real zeros)
        full = c.set_index(["station", "ts"]).unstack("station").fillna(0).stack("station").reset_index()
        full["ts"] = pd.to_datetime(full["ts"], utc=True)
        return full[(full["ts"] >= pd.Timestamp(start, tz="UTC")) & (full["ts"] < pd.Timestamp(end, tz="UTC"))]


class CitiBikeRiderType(Source):
    provenance = "Citi Bike NYC System Data (Lyft), NYCBS Data Use Policy"
    start = "2016-01-01"

    def __init__(self, years: range = range(2016, 2026)):
        self.years = years
        self.name = "bike_rider_type_nyc"
        self.contract = FeatureContract(features=["start_station", "end_station", "duration_s", "start_lat", "start_lng", "same_station"],
                                        target_col="is_member", n_lags=0, to_class=lambda s: s.astype("Int64"), label_is_current=True)

    def fetch_history(self, start, end):
        frames = []
        for name, f in _iter_citibike_csvs(self.years):
            df = _read_trips(f, ["started_at", "ended_at", "start_station_id", "end_station_id", "member_casual", "start_lat", "start_lng"])
            df = df.dropna(subset=["member_casual"])
            st = pd.to_datetime(df["started_at"], errors="coerce", format="mixed")
            en = pd.to_datetime(df["ended_at"], errors="coerce", format="mixed")
            out = pd.DataFrame({
                "ts": st, "duration_s": (en - st).dt.total_seconds(),
                "start_station": pd.to_numeric(df["start_station_id"].astype(str).str.replace(r"\.\d+$", "", regex=True), errors="coerce"),
                "end_station": pd.to_numeric(df["end_station_id"].astype(str).str.replace(r"\.\d+$", "", regex=True), errors="coerce"),
                "start_lat": df.get("start_lat"), "start_lng": df.get("start_lng"),
                "is_member": (df["member_casual"] == "member").astype(float),
            }).dropna()
            out["same_station"] = (out["start_station"] == out["end_station"]).astype(int)
            frames.append(out.iloc[::20])  # thin at read time; the final stream is thinned again to 1M rows
            print(f"  [citibike] {name}: {len(df):,} trips", flush=True)
        df = pd.concat(frames, ignore_index=True)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df[(df["ts"] >= pd.Timestamp(start, tz="UTC")) & (df["ts"] < pd.Timestamp(end, tz="UTC"))]


class TaxiDemandNYC(Source):
    provenance = "NYC Taxi & Limousine Commission trip record data, public"
    start = "2016-01-01"

    def __init__(self, years: range = range(2016, 2026), top_zones: int = 30, horizon: int = 1):
        self.years, self.top = years, top_zones
        self.name = "taxi_demand_nyc"
        self.contract = FeatureContract(features=["pickups", "dropoffs", "mean_fare"], target_col="pickups", horizon=horizon,
                                        n_lags=24, lag_cols=["pickups"], static=["zone"], group_col="zone")

    def fetch_history(self, start, end):
        counts = []
        for y in self.years:
            for m in range(1, 13):
                ym = f"{y}-{m:02d}"
                if pd.Timestamp(ym + "-01") >= pd.Timestamp(end):
                    break
                try:
                    path = cached_download(TLC.format(ym=ym), CACHE_ROOT / "tlc")
                except Exception as e:
                    print(f"  [tlc] {ym}: unavailable ({e})", flush=True)
                    continue
                df = pd.read_parquet(path, columns=["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID", "fare_amount"])
                df = df[(df["tpep_pickup_datetime"] >= pd.Timestamp(ym + "-01")) & (df["tpep_pickup_datetime"] < pd.Timestamp(ym + "-01") + pd.offsets.MonthBegin(1))]
                pu = df.assign(ts=df["tpep_pickup_datetime"].dt.floor("h")).groupby(["ts", "PULocationID"]).agg(pickups=("fare_amount", "size"), mean_fare=("fare_amount", "mean"))
                do = df.assign(ts=df["tpep_dropoff_datetime"].dt.floor("h")).groupby(["ts", "DOLocationID"]).size().rename("dropoffs")
                pu.index.names = do.index.names = ["ts", "zone"]
                counts.append(pd.concat([pu, do], axis=1).fillna(0).reset_index())
                print(f"  [tlc] {ym}: {len(df):,} trips", flush=True)
        c = pd.concat(counts, ignore_index=True).groupby(["ts", "zone"], as_index=False).sum()
        top = c.groupby("zone")["pickups"].sum().nlargest(self.top).index
        c = c[c["zone"].isin(top)]
        full = c.set_index(["zone", "ts"]).unstack("zone").fillna(0).stack("zone").reset_index()
        full["ts"] = pd.to_datetime(full["ts"]).dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
        full = full.dropna(subset=["ts"])
        return full[(full["ts"] >= pd.Timestamp(start, tz="UTC")) & (full["ts"] < pd.Timestamp(end, tz="UTC"))]
