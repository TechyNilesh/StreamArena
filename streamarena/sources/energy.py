"""Power-grid and energy-market streams (all keyless).

* UK Carbon Intensity API (National Energy System Operator), 30-min
  settlement periods since 2017-09: https://carbon-intensity.github.io/api-definitions/
* Energy-Charts (Fraunhofer ISE), 15-min German public net electricity
  generation and load since 2015: https://api.energy-charts.info
* Octopus Energy Agile tariff half-hourly unit rates since 2018:
  https://developer.octopus.energy/rest/
"""

from __future__ import annotations

import time

import pandas as pd

from .base import FeatureContract, Source, get

CI_URL = "https://api.carbonintensity.org.uk"
CI_INDEX = {"very low": 0, "low": 1, "moderate": 2, "high": 3, "very high": 4}
FUELS = ["biomass", "coal", "imports", "gas", "nuclear", "other", "hydro", "solar", "wind"]


class CarbonIntensityGB(Source):
    provenance = "NESO Carbon Intensity API, CC BY 4.0"
    start = "2018-01-01"
    poll_seconds = 1800

    def __init__(self, classify_index: bool = False, horizon: int = 1):
        self.name = "grid_carbon_index_gb" if classify_index else "grid_carbon_gb"
        self.contract = FeatureContract(
            features=["actual", *FUELS], target_col="index" if classify_index else "actual", horizon=horizon,
            n_lags=8, lag_cols=["actual"], to_class=(lambda s: s.astype("Int64")) if classify_index else None,
        )

    def _window(self, s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
        f = lambda t: t.strftime("%Y-%m-%dT%H:%MZ")
        inten = get(f"{CI_URL}/intensity/{f(s)}/{f(e)}").json().get("data", [])
        gen = get(f"{CI_URL}/generation/{f(s)}/{f(e)}").json().get("data", [])
        if not inten or not gen:  # windows before the generation-mix series starts, or API gaps
            return pd.DataFrame(columns=["ts", "actual", "forecast", "index", *FUELS])
        a = pd.DataFrame([{"ts": d["from"], "actual": d.get("intensity", {}).get("actual"),
                           "forecast": d.get("intensity", {}).get("forecast"),
                           "index": CI_INDEX.get(d.get("intensity", {}).get("index"))} for d in inten])
        g = pd.DataFrame([{"ts": d["from"], **{m["fuel"]: m["perc"] for m in d["generationmix"]}} for d in gen])
        return a.merge(g, on="ts", how="inner")

    def fetch_history(self, start, end):
        frames = []
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        while s < e:
            n = min(s + pd.Timedelta(days=14), e)
            frames.append(self._window(s, n))
            s = n
        df = pd.concat(frames, ignore_index=True).drop_duplicates("ts")
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
        # the "index" band published by the API is derived from the *forecast*; recompute it from actuals
        df["index"] = pd.cut(df["actual"], [-1, 50, 150, 250, 350, 1e9], labels=False)
        return df.dropna(subset=["actual"])

    def fetch_latest(self):
        now = pd.Timestamp.utcnow().floor("30min")
        df = self._window(now - pd.Timedelta(days=2), now)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["index"] = pd.cut(pd.to_numeric(df["actual"], errors="coerce"), [-1, 50, 150, 250, 350, 1e9], labels=False)
        return df.dropna(subset=["actual"])


class EnergyChartsLoad(Source):
    provenance = "Energy-Charts (Fraunhofer ISE) public_power API, CC BY 4.0"
    start = "2016-01-01"
    poll_seconds = 900
    KEEP = {"Load": "load", "Residual load": "residual_load", "Wind onshore": "wind_onshore", "Wind offshore": "wind_offshore",
            "Solar": "solar", "Nuclear": "nuclear", "Fossil gas": "gas", "Fossil hard coal": "hard_coal",
            "Fossil brown coal / lignite": "lignite", "Hydro Run-of-River": "hydro", "Biomass": "biomass"}

    def __init__(self, country: str = "de", horizon: int = 4):
        self.country = country
        self.name = f"grid_load_{country}"
        self.contract = FeatureContract(features=list(self.KEEP.values()), target_col="load", horizon=horizon, n_lags=8, lag_cols=["load"])

    def _call(self, s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
        d = get("https://api.energy-charts.info/public_power", params=dict(country=self.country, start=s.date().isoformat(), end=e.date().isoformat())).json()
        df = pd.DataFrame({"ts": pd.to_datetime(d["unix_seconds"], unit="s", utc=True)})
        for p in d["production_types"]:
            if p["name"] in self.KEEP:
                df[self.KEEP[p["name"]]] = p["data"]
        return df

    def fetch_history(self, start, end):
        frames = []
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        while s < e:
            n = min(s + pd.DateOffset(years=1), e)
            frames.append(self._call(s, n - pd.Timedelta(days=1)))
            time.sleep(3.0)  # Energy-Charts rate-limits bursts (HTTP 429)
            s = n
        df = pd.concat(frames, ignore_index=True).drop_duplicates("ts").set_index("ts").sort_index()
        for c in self.KEEP.values():
            if c not in df:
                df[c] = float("nan")
        return df.resample("15min").mean().interpolate(limit=8).dropna(subset=["load"]).fillna(0.0).reset_index()

    def fetch_latest(self):
        now = pd.Timestamp.utcnow()
        return self._call(now - pd.Timedelta(days=2), now)


class OctopusAgile(Source):
    provenance = "Octopus Energy public tariff API (Agile Octopus, region C London)"
    start = "2018-02-21"
    poll_seconds = 1800
    PRODUCTS = ["AGILE-18-02-21", "AGILE-22-07-22", "AGILE-22-08-31", "AGILE-FLEX-22-11-25", "AGILE-23-12-06", "AGILE-24-04-03", "AGILE-24-10-01"]

    def __init__(self, region: str = "C", horizon: int = 48):
        self.region = region
        self.name = "energy_price_agile"
        self.contract = FeatureContract(features=["price"], target_col="price", horizon=horizon, n_lags=48)

    def _rates(self, product: str, s: str, e: str) -> list[dict]:
        url = f"https://api.octopus.energy/v1/products/{product}/electricity-tariffs/E-1R-{product}-{self.region}/standard-unit-rates/"
        params = dict(period_from=s, period_to=e, page_size=1500)
        rows = []
        while url:
            try:
                r = get(url, params=params, retries=2)
            except Exception:
                break
            j = r.json()
            rows += j.get("results", [])
            url, params = j.get("next"), None
        return rows

    def fetch_history(self, start, end):
        rows = []
        for p in self.PRODUCTS:
            rows += self._rates(p, f"{start}T00:00Z", f"{end}T00:00Z")
        df = pd.DataFrame(rows).rename(columns={"valid_from": "ts", "value_inc_vat": "price"})[["ts", "price"]]
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        # several products overlap in time; keep the newest product's price per slot
        df = df.drop_duplicates("ts", keep="last").set_index("ts").sort_index()
        return df.resample("30min").mean().interpolate(limit=4).dropna().reset_index()

    def fetch_latest(self):
        now = pd.Timestamp.utcnow()
        rows = self._rates(self.PRODUCTS[-1], (now - pd.Timedelta(days=3)).strftime("%Y-%m-%dT%H:%MZ"), now.strftime("%Y-%m-%dT%H:%MZ"))
        df = pd.DataFrame(rows).rename(columns={"valid_from": "ts", "value_inc_vat": "price"})[["ts", "price"]]
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df.sort_values("ts")
