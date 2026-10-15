"""Open-Meteo weather archive and CAMS air-quality archive (no API key).

* Weather: https://archive-api.open-meteo.com (ERA5-based reanalysis, hourly,
  1940-present). Zippenfenig, P. (2023) Open-Meteo.com, doi:10.5281/zenodo.7970649.
* Air quality: https://air-quality-api.open-meteo.com (CAMS, hourly, 2013-present).

Only *observed* (reanalysis) variables are used as features; the target is the
same variable ``horizon`` hours later.
"""

from __future__ import annotations

import time

import pandas as pd

from .base import FeatureContract, Source, get

CITIES = {
    "auckland": (-36.85, 174.76),
    "berlin": (52.52, 13.41),
    "phoenix": (33.45, -112.07),
    "hamburg": (53.55, 9.99),
    "delhi": (28.61, 77.21),
    "losangeles": (34.05, -118.24),
}
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "surface_pressure",
                "precipitation", "cloud_cover", "wind_speed_10m", "wind_direction_10m"]
AQ_VARS = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "european_aqi"]


def _fetch(base: str, lat: float, lon: float, variables: list[str], start: str, end: str, step_years: int = 1) -> pd.DataFrame:
    frames = []
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    while s < e:
        n = min(s + pd.DateOffset(years=step_years), e)
        r = get(base, params=dict(latitude=lat, longitude=lon, start_date=s.date().isoformat(),
                                  end_date=(n - pd.Timedelta(days=1)).date().isoformat(),
                                  hourly=",".join(variables), timezone="UTC"))
        h = r.json()["hourly"]
        frames.append(pd.DataFrame(h).rename(columns={"time": "ts"}))
        s = n
        time.sleep(1.0)  # stay well inside Open-Meteo's free-tier per-minute limit
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.drop_duplicates("ts")


class OpenMeteoWeather(Source):
    provenance = "Open-Meteo historical weather API (ERA5), Zippenfenig 2023, CC BY 4.0"
    start = "2016-01-01"

    def __init__(self, city: str, target: str = "temperature_2m", horizon: int = 1, classify_rain: bool = False):
        self.city = city
        self.lat, self.lon = CITIES[city]
        kind = "rain" if classify_rain else {"temperature_2m": "temp", "wind_speed_10m": "wind"}.get(target, target)
        self.name = f"weather_{kind}_{city}"
        self.contract = FeatureContract(
            features=WEATHER_VARS, target_col="precipitation" if classify_rain else target, horizon=horizon,
            n_lags=6, lag_cols=[target] if not classify_rain else ["precipitation", "cloud_cover", "surface_pressure"],
            to_class=(lambda s: (s > 0.1).astype("Int64")) if classify_rain else None,
        )

    def fetch_history(self, start, end):
        return _fetch("https://archive-api.open-meteo.com/v1/archive", self.lat, self.lon, WEATHER_VARS, start, end)

    def fetch_latest(self):
        r = get("https://api.open-meteo.com/v1/forecast", params=dict(latitude=self.lat, longitude=self.lon,
                hourly=",".join(WEATHER_VARS), past_days=2, forecast_days=1, timezone="UTC"))
        h = r.json()["hourly"]
        df = pd.DataFrame(h).rename(columns={"time": "ts"})
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df[df["ts"] <= pd.Timestamp.utcnow()]  # observed hours only


def _aqi_band(s: pd.Series) -> pd.Series:
    # European AQI bands: good <20, fair <40, moderate <60, poor <80, very poor <100, extremely poor >=100
    return pd.cut(s, [-1, 20, 40, 60, 80, 100, 1e9], labels=False).astype("Int64")


class OpenMeteoAirQuality(Source):
    provenance = "Open-Meteo air-quality API (CAMS global/European reanalysis), CC BY 4.0"
    start = "2015-01-01"

    def __init__(self, city: str, classify_aqi: bool = False, horizon: int = 1):
        self.city = city
        self.lat, self.lon = CITIES[city]
        self.name = f"airquality_{'aqi' if classify_aqi else 'pm25'}_{city}"
        self.contract = FeatureContract(
            features=AQ_VARS, target_col="european_aqi" if classify_aqi else "pm2_5", horizon=horizon,
            n_lags=6, lag_cols=["pm2_5", "pm10"], to_class=_aqi_band if classify_aqi else None,
        )

    def fetch_history(self, start, end):
        return _fetch("https://air-quality-api.open-meteo.com/v1/air-quality", self.lat, self.lon, AQ_VARS, start, end, step_years=1)

    def fetch_latest(self):
        r = get("https://air-quality-api.open-meteo.com/v1/air-quality", params=dict(latitude=self.lat, longitude=self.lon,
                hourly=",".join(AQ_VARS), past_days=2, forecast_days=1, timezone="UTC"))
        df = pd.DataFrame(r.json()["hourly"]).rename(columns={"time": "ts"})
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df[df["ts"] <= pd.Timestamp.utcnow()]
