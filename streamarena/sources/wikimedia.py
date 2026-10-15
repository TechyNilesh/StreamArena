"""Wikimedia edit history (no key).

Historical: the monthly ``mediawiki_history`` dumps
(https://dumps.wikimedia.org/other/mediawiki_history/), one TSV per wiki
with every revision and the ``revision_is_identity_reverted`` flag. Live: the
EventStreams ``recentchange`` SSE feed. Default wiki is ``simplewiki``
(Simple English Wikipedia, ~1 GB compressed) so the build fits on a laptop;
``enwiki`` works the same way but is ~100x larger.

Label: the edit is reverted (identity revert) within 48 hours — a standard
proxy for damaging edits. Features are what is known at save time.
"""

from __future__ import annotations

import bz2
import re

import numpy as np
import pandas as pd

from .base import CACHE_ROOT, FeatureContract, Source, cached_download, get

INDEX = "https://dumps.wikimedia.org/other/mediawiki_history/"
SCHEMA = ("wiki_db,event_log_id,event_entity,event_type,event_timestamp,event_comment,event_user_id,event_user_central_id,"
          "event_user_text_historical,event_user_text,event_user_blocks_historical,event_user_blocks,event_user_groups_historical,"
          "event_user_groups,event_user_is_bot_by_historical,event_user_is_bot_by,event_user_is_created_by_self,"
          "event_user_is_created_by_system,event_user_is_created_by_peer,event_user_is_anonymous,event_user_is_temporary,"
          "event_user_is_permanent,event_user_is_cross_wiki,event_user_registration_timestamp,event_user_creation_timestamp,"
          "event_user_first_edit_timestamp,event_user_revision_count,event_user_seconds_since_previous_revision,page_id,"
          "page_title_historical,page_title,page_namespace_historical,page_namespace_is_content_historical,page_namespace,"
          "page_namespace_is_content,page_is_redirect,page_is_deleted,page_creation_timestamp,page_first_edit_timestamp,"
          "page_revision_count,page_seconds_since_previous_revision,user_id,user_central_id,user_text_historical,user_text,"
          "user_blocks_historical,user_blocks,user_groups_historical,user_groups,user_is_bot_by_historical,user_is_bot_by,"
          "user_is_created_by_self,user_is_created_by_system,user_is_created_by_peer,user_is_anonymous,user_is_temporary,"
          "user_is_permanent,user_registration_timestamp,user_creation_timestamp,user_first_edit_timestamp,revision_id,"
          "revision_parent_id,revision_minor_edit,revision_deleted_parts,revision_deleted_parts_are_suppressed,"
          "revision_text_bytes,revision_text_bytes_diff,revision_text_sha1,revision_content_model,revision_content_format,"
          "revision_is_deleted_by_page_deletion,revision_deleted_by_page_deletion_timestamp,revision_is_identity_reverted,"
          "revision_first_identity_reverting_revision_id,revision_seconds_to_identity_revert,revision_is_identity_revert,"
          "revision_is_from_before_page_creation,revision_tags").split(",")
COL = {c: i for i, c in enumerate(SCHEMA)}
KEEP = ["event_timestamp", "event_user_is_anonymous", "event_user_is_bot_by", "event_user_revision_count",
        "event_user_seconds_since_previous_revision", "page_namespace", "page_revision_count",
        "page_seconds_since_previous_revision", "revision_minor_edit", "revision_text_bytes", "revision_text_bytes_diff",
        "event_comment", "revision_is_identity_reverted", "revision_seconds_to_identity_revert", "event_type", "event_entity"]
FEATURES = ["is_anonymous", "is_bot", "user_edit_count", "user_secs_since_prev", "namespace", "page_edit_count",
            "page_secs_since_prev", "minor_edit", "text_bytes", "bytes_diff", "comment_len", "has_comment"]


class WikipediaReverts(Source):
    provenance = "Wikimedia Foundation mediawiki_history dumps, CC0"
    start = "2016-01-01"
    poll_seconds = 60

    def __init__(self, wiki: str = "simplewiki", revert_window_s: int = 48 * 3600):
        self.wiki, self.window = wiki, revert_window_s
        self.name = f"wiki_revert_{wiki.replace('wiki', '')}"
        self.contract = FeatureContract(features=FEATURES, target_col="reverted", n_lags=0,
                                        to_class=lambda s: s.astype("Int64"), label_is_current=True)

    def _latest_dump(self) -> str:
        months = re.findall(r'href="(\d{4}-\d{2})/"', get(INDEX).text)
        return sorted(months)[-1]

    def fetch_history(self, start, end):
        month = self._latest_dump()
        listing = get(f"{INDEX}{month}/{self.wiki}/").text
        files = re.findall(r'href="([^"]+\.tsv\.bz2)"', listing)
        s_ts, e_ts = pd.Timestamp(start), pd.Timestamp(end)
        frames = []
        for fn in sorted(files):
            # yearly/monthly files are named <dump>.<wiki>.<YYYY>[-MM].tsv.bz2 ; all-time files carry everything
            m = re.search(r"\.(\d{4})(?:-(\d{2}))?\.tsv\.bz2$", fn)
            if m and (int(m.group(1)) < s_ts.year or int(m.group(1)) > e_ts.year):
                continue
            path = cached_download(f"{INDEX}{month}/{self.wiki}/{fn}", CACHE_ROOT / "wikimedia")
            idx = [COL[c] for c in KEEP]
            rows = []
            with bz2.open(path, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    if len(p) < len(SCHEMA) or p[COL["event_entity"]] != "revision" or p[COL["event_type"]] != "create":
                        continue
                    rows.append([p[i] for i in idx])
            df = pd.DataFrame(rows, columns=KEEP)
            frames.append(self._derive(df))
            print(f"  [wikimedia] {fn}: {len(df):,} revisions", flush=True)
        df = pd.concat(frames, ignore_index=True)
        return df[(df["ts"] >= s_ts.tz_localize("UTC")) & (df["ts"] < e_ts.tz_localize("UTC"))]

    def _derive(self, df: pd.DataFrame) -> pd.DataFrame:
        num = lambda c: pd.to_numeric(df[c], errors="coerce")
        tf = lambda c: (df[c].str.lower() == "true").astype(int)
        secs_to_revert = num("revision_seconds_to_identity_revert")
        out = pd.DataFrame({
            "ts": pd.to_datetime(df["event_timestamp"], errors="coerce", utc=True),
            "is_anonymous": tf("event_user_is_anonymous"),
            "is_bot": (df["event_user_is_bot_by"].str.len() > 0).astype(int),
            "user_edit_count": num("event_user_revision_count").fillna(0),
            "user_secs_since_prev": num("event_user_seconds_since_previous_revision").fillna(-1),
            "namespace": num("page_namespace").fillna(0),
            "page_edit_count": num("page_revision_count").fillna(0),
            "page_secs_since_prev": num("page_seconds_since_previous_revision").fillna(-1),
            "minor_edit": tf("revision_minor_edit"),
            "text_bytes": num("revision_text_bytes").fillna(0),
            "bytes_diff": num("revision_text_bytes_diff").fillna(0),
            "comment_len": df["event_comment"].str.len().fillna(0),
            "has_comment": (df["event_comment"].str.len() > 0).astype(int),
            "reverted": ((df["revision_is_identity_reverted"].str.lower() == "true") & (secs_to_revert <= self.window)).astype(float),
        })
        return out.dropna(subset=["ts"]).sort_values("ts")

    def fetch_latest(self):
        # EventStreams carries no revert flag; live labels are resolved by polling the revision later.
        raise NotImplementedError("live mode for wiki reverts needs the EventStreams + revision-poll adapter (TODO)")
