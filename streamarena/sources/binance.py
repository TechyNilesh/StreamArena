"""Binance public market data (no key).

Historical 1-minute klines from https://data.binance.vision (monthly zips);
live data from the keyless market-data host data-stream.binance.vision /
data-api.binance.vision. Features: last ``n_lags`` 1-minute log returns,
realised volatility over 60 minutes, volume and trade count. Targets: the
next-minute log return (regression) or its sign (classification).
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from .base import CACHE_ROOT, FeatureContract, Source, cached_download, get, month_ranges

BULK = "https://data.binance.vision/data/spot/monthly/klines/{sym}/1m/{sym}-1m-{ym}.zip"
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


class BinanceKlines(Source):
    provenance = "Binance public market data archive (data.binance.vision)"
    start = "2018-01-01"
    poll_seconds = 60

    def __init__(self, symbol: str = "BTCUSDT", direction: bool = False, horizon: int = 1):
        self.symbol = symbol
        self.name = f"crypto_{'direction' if direction else 'btc_1m'}" if symbol == "BTCUSDT" else f"crypto_{symbol.lower()}_{'direction' if direction else '1m'}"
        self.contract = FeatureContract(features=["log_return", "realised_vol_60", "log_volume", "trades", "hl_range"],
                                        target_col="log_return", horizon=horizon, n_lags=10,
                                        to_class=(lambda s: (s > 0).astype("Int64")) if direction else None)

    @staticmethod
    def _derive(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
        df["log_return"] = np.log(df["close"]).diff()
        df["realised_vol_60"] = df["log_return"].rolling(60).std()
        df["log_volume"] = np.log1p(df["volume"])
        df["hl_range"] = (df["high"] - df["low"]) / df["close"]
        return df.dropna()

    def fetch_history(self, start, end):
        frames = []
        for s, _ in month_ranges(start, end):
            ym = s.strftime("%Y-%m")
            try:
                path = cached_download(BULK.format(sym=self.symbol, ym=ym), CACHE_ROOT / "binance")
            except Exception as e:
                print(f"  [binance] {ym}: unavailable ({e})", flush=True)
                continue
            with zipfile.ZipFile(path) as z:
                raw = pd.read_csv(z.open(z.namelist()[0]), header=None, names=KLINE_COLS)
            if raw["open_time"].iloc[0] > 1e14:  # microseconds since 2025-01
                raw["open_time"] //= 1000
            frames.append(raw[["open_time", "open", "high", "low", "close", "volume", "trades"]])
        df = pd.concat(frames, ignore_index=True)
        df["ts"] = pd.to_datetime(df.pop("open_time"), unit="ms", utc=True)
        return self._derive(df)

    def fetch_latest(self):
        r = get("https://data-api.binance.vision/api/v3/klines", params=dict(symbol=self.symbol, interval="1m", limit=200))
        raw = pd.DataFrame(r.json(), columns=KLINE_COLS)
        for c in ("open", "high", "low", "close", "volume"):
            raw[c] = raw[c].astype(float)
        raw["ts"] = pd.to_datetime(raw["open_time"], unit="ms", utc=True)
        return self._derive(raw[["ts", "open", "high", "low", "close", "volume", "trades"]].iloc[:-1])  # drop the open candle
