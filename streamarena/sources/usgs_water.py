"""USGS National Water Information System instantaneous values (no key).

https://waterservices.usgs.gov/docs/instantaneous-values/ — 15-minute gauge
readings. Default site 01646500 is the Potomac River at Little Falls near
Washington, DC (discharge 00060 in ft3/s, gauge height 00065 in ft). The NWS
flood stage at Little Falls is 10 ft.
"""

from __future__ import annotations

import pandas as pd

from .base import FeatureContract, Source, get

IV_URL = "https://nwis.waterservices.usgs.gov/nwis/iv/"


def _parse(payload: dict) -> pd.DataFrame:
    frames = []
    for ts in payload["value"]["timeSeries"]:
        code = ts["variable"]["variableCode"][0]["value"]
        vals = ts["values"][0]["value"]
        if not vals:
            continue
        df = pd.DataFrame(vals)[["dateTime", "value"]].rename(columns={"dateTime": "ts", "value": code})
        df[code] = pd.to_numeric(df[code], errors="coerce")
        frames.append(df.set_index("ts"))
    if not frames:
        return pd.DataFrame(columns=["ts"])
    out = pd.concat(frames, axis=1).reset_index()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out


class USGSWater(Source):
    provenance = "USGS NWIS Instantaneous Values web service, public domain"
    start = "2016-01-01"
    poll_seconds = 900

    def __init__(self, site: str = "01646500", flood: bool = False, flood_stage_ft: float = 10.0, horizon: int = 4):
        self.site = site
        self.name = f"flood_potomac" if flood else "river_discharge_potomac"
        if flood:
            # label: gauge height reaches flood stage within the next 6 h (24 x 15-min steps)
            self.contract = FeatureContract(
                features=["00060", "00065"], target_col="flood_ahead", horizon=1, n_lags=8, lag_cols=["00065", "00060"],
                to_class=lambda s: s.astype("Int64"), label_is_current=True,
            )
        else:
            self.contract = FeatureContract(features=["00060", "00065"], target_col="00060", horizon=horizon, n_lags=8, lag_cols=["00060", "00065"])
        self.flood, self.flood_stage = flood, flood_stage_ft

    def _regularise(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index("ts").sort_index()
        df = df[~df.index.duplicated()].resample("15min").mean().interpolate(limit=8).dropna().reset_index()
        if self.flood:
            ahead = df["00065"][::-1].rolling(24, min_periods=1).max()[::-1].shift(-1)
            df["flood_ahead"] = (ahead >= self.flood_stage).astype(float)
            df = df.dropna()
        return df

    def fetch_history(self, start, end):
        frames = []
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        while s < e:
            n = min(s + pd.DateOffset(months=6), e)
            r = get(IV_URL, params=dict(format="json", sites=self.site, parameterCd="00060,00065",
                                        startDT=s.date().isoformat(), endDT=(n - pd.Timedelta(days=1)).date().isoformat()))
            frames.append(_parse(r.json()))
            s = n
        return self._regularise(pd.concat(frames, ignore_index=True).drop_duplicates("ts"))

    def fetch_latest(self):
        r = get(IV_URL, params=dict(format="json", sites=self.site, parameterCd="00060,00065", period="P2D"))
        return self._regularise(_parse(r.json()))
