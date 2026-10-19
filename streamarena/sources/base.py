"""Base classes for real-world stream sources built on public open APIs.

Every source has two modes:

* ``history`` — download the multi-year archive the API exposes and turn it
  into a fixed, reproducible benchmark CSV (feature columns + ``target``).
  This is what the leaderboard runs on.
* ``live``    — poll the same API and emit instances as they appear, with
  labels resolved later when the future value becomes known. This feeds the
  deployment simulator.

Both modes share one :class:`FeatureContract`: which raw columns are
features, how many lags to add, whether to add calendar features, and how the
target is derived (value ``horizon`` steps ahead, optionally transformed into
a class). The contract is the dataset's documentation: it says exactly which
information a learner had at prediction time.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = REPO_ROOT / "datasets"
CACHE_ROOT = REPO_ROOT / "datasets" / ".cache"
USER_AGENT = "StreamArena/2.0 (https://github.com/techynilesh/StreamArena)"
MAX_ROWS = 1_000_000


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT


def get(url: str, params: dict | None = None, retries: int = 8, timeout: int = 120, **kw) -> requests.Response:
    """GET with exponential back-off; honours 429 Retry-After."""
    delay = 2.0
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=timeout, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("Retry-After", delay))
                time.sleep(max(wait, 5.0 if r.status_code == 429 else delay))
                delay *= 2
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"gave up on {url}")


def cached_download(url: str, cache_dir: Path, name: str | None = None) -> Path:
    """Download ``url`` to ``cache_dir`` once (resumable sweeps re-use it)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (name or url.rsplit("/", 1)[-1])
    if path.exists() and path.stat().st_size > 0:
        return path
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.part")  # unique: builders may share a file
    with _session.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if path.exists() and path.stat().st_size > 0:  # another process finished first
        tmp.unlink()
    else:
        os.replace(tmp, path)
    return path


def month_ranges(start: str, end: str):
    """Yield (first_day, last_day) date pairs, month by month."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    cur = s
    while cur < e:
        nxt = min((cur + pd.offsets.MonthBegin(1)).normalize(), e)
        yield cur, nxt
        cur = nxt


# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------


@dataclass
class FeatureContract:
    """How a raw timestamped table becomes a stream-learning dataset."""

    #: raw columns exposed as features (observed values only — never forecasts)
    features: list[str]
    #: column the target is derived from
    target_col: str
    #: steps ahead (in rows of the regular grid) the target refers to
    horizon: int = 1
    #: number of lagged copies of each column in ``lag_cols`` (default: target_col)
    n_lags: int = 6
    lag_cols: list[str] | None = None
    #: add sin/cos hour-of-day, day-of-week, day-of-year from the timestamp
    calendar: bool = True
    #: optional map from target value -> class label (makes it classification)
    to_class: Callable[[pd.Series], pd.Series] | None = None
    #: for event streams (one row per event, no regular grid): the label is a
    #: property of the *current* row rather than a future one
    label_is_current: bool = False
    #: static/categorical columns passed through untouched (e.g. station id)
    static: list[str] = field(default_factory=list)
    #: rows grouped by this column get their own lags/targets (per-station streams)
    group_col: str | None = None
    #: name of the timestamp column in the raw table
    time_col: str = "ts"

    @property
    def task(self) -> str:
        return "classification" if self.to_class is not None else "regression"

    def apply(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col], utc=True)
        sort_cols = [self.group_col, self.time_col] if self.group_col else [self.time_col]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        lag_cols = self.lag_cols or [self.target_col]
        g = df.groupby(self.group_col, sort=False) if self.group_col else None
        out = pd.DataFrame(index=df.index)
        for c in self.static:
            out[c] = df[c]
        for c in self.features:
            out[c] = df[c]
        for c in lag_cols:
            for k in range(1, self.n_lags + 1):
                out[f"{c}_lag{k}"] = (g[c].shift(k) if g is not None else df[c].shift(k))
        if self.calendar:
            t = df[self.time_col]
            hod = (t.dt.hour + t.dt.minute / 60) / 24
            dow = t.dt.dayofweek / 7
            doy = (t.dt.dayofyear - 1) / 366
            for name, v in (("hour", hod), ("dow", dow), ("doy", doy)):
                out[f"{name}_sin"] = np.sin(2 * np.pi * v)
                out[f"{name}_cos"] = np.cos(2 * np.pi * v)
        if self.label_is_current:
            y = df[self.target_col]
        else:
            y = g[self.target_col].shift(-self.horizon) if g is not None else df[self.target_col].shift(-self.horizon)
        out["target"] = self.to_class(y) if self.to_class is not None else y
        out.insert(0, "timestamp", df[self.time_col].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
        out = out.dropna().reset_index(drop=True)
        # stream order is time order even for grouped (per-station) streams
        out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)
        return out


def thin_to(df: pd.DataFrame, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """Keep every k-th row so the stream still spans the whole period."""
    if len(df) <= max_rows:
        return df
    k = int(np.ceil(len(df) / max_rows))
    return df.iloc[::k].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


class Source(ABC):
    """A real-world stream. Subclasses implement the two fetchers."""

    #: dataset key, e.g. "weather_temp_berlin"
    name: str
    #: citation / provenance (goes into the manifest)
    provenance: str
    #: default archive span
    start: str = "2016-01-01"
    end: str = "2026-10-01"
    #: seconds between polls in live mode
    poll_seconds: int = 300

    contract: FeatureContract

    @abstractmethod
    def fetch_history(self, start: str, end: str) -> pd.DataFrame:
        """Return a raw table with a ``ts`` column covering [start, end)."""

    def fetch_latest(self) -> pd.DataFrame:
        """Return the most recent raw rows (live mode). Default: not supported."""
        raise NotImplementedError(f"{self.name} has no live mode")

    # -- history mode -------------------------------------------------------
    def build(self, start: str | None = None, end: str | None = None, max_rows: int = MAX_ROWS) -> pd.DataFrame:
        raw = self.fetch_history(start or self.start, end or self.end)
        df = self.contract.apply(raw)
        return thin_to(df, max_rows)

    def to_csv(self, out_root: Path = DATASETS_ROOT, **kw) -> Path:
        df = self.build(**kw)
        path = out_root / self.contract.task / "real" / f"{self.name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, float_format="%.6g")
        manifest = {
            "name": self.name, "task": self.contract.task, "rows": len(df), "columns": list(df.columns),
            "start": df["timestamp"].iloc[0], "end": df["timestamp"].iloc[-1], "horizon": self.contract.horizon,
            "n_lags": self.contract.n_lags, "provenance": self.provenance, "built": time.strftime("%Y-%m-%d"),
        }
        path.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
        return path

    # -- live mode ----------------------------------------------------------
    def live(self, max_polls: int | None = None):
        """Yield ``("instance", ts, x)`` and later ``("label", ts, y)`` events.

        Labels are emitted once the future value the target refers to has
        been observed, so a learner sees exactly the delay it would in
        production.
        """
        c = self.contract
        seen: set = set()
        pending: dict = {}
        polls = 0
        while max_polls is None or polls < max_polls:
            polls += 1
            raw = self.fetch_latest()
            df = c.apply(raw.assign(**{c.target_col: raw[c.target_col]}))
            # apply() drops rows whose target is unknown; recompute features without the target
            feats = c.apply(raw.assign(**{c.target_col: raw[c.target_col].bfill().ffill()}))
            for _, row in feats.iterrows():
                ts = row["timestamp"]
                if ts in seen:
                    continue
                seen.add(ts)
                x = row.drop(["timestamp", "target"]).to_dict()
                pending[ts] = x
                yield "instance", ts, x
            for _, row in df.iterrows():
                ts = row["timestamp"]
                if ts in pending:
                    pending.pop(ts)
                    yield "label", ts, row["target"]
            time.sleep(self.poll_seconds)
