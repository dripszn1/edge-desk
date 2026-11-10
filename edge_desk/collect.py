"""Collector: pull every open market in scope from Kalshi's public API and save one snapshot.

Read-only. Never places an order. No API key needed.
Output: data/snapshots/<YYYY-MM-DD>/<HHMMSS>Z.parquet (UTC), one row per market.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from . import config

ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = ROOT / "data" / "snapshots"

session = requests.Session()
session.headers["User-Agent"] = "edge-desk/0.1 (read-only research)"


def _get(path: str, params: dict | None = None, tries: int = 4) -> dict:
    for i in range(tries):
        r = session.get(config.KALSHI_BASE + path, params=params, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** i)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def fee_multiplier(series: str) -> float:
    try:
        return float(_get(f"/series/{series}")["series"].get("fee_multiplier") or 1.0)
    except Exception:
        return 1.0


def open_events(series: str) -> list[dict]:
    events, cursor = [], None
    while True:
        params = {"series_ticker": series, "status": "open", "with_nested_markets": "true", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        j = _get("/events", params)
        events += j.get("events", [])
        cursor = j.get("cursor")
        if not cursor:
            return events
        time.sleep(0.1)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def flatten(league: str, series: str, mult: float, ev: dict, ts: str) -> list[dict]:
    rows = []
    for m in ev.get("markets", []):
        cs = m.get("custom_strike") or {}
        rows.append({
            "snapshot_ts": ts,
            "league": league,
            "series": series,
            "event_ticker": ev.get("event_ticker"),
            "event_title": ev.get("title"),
            "event_sub": ev.get("sub_title"),
            "ticker": m.get("ticker"),
            "title": m.get("title"),
            "yes_sub_title": m.get("yes_sub_title"),
            "strike_type": m.get("strike_type"),
            "floor_strike": _f(m.get("floor_strike")),
            "cap_strike": _f(m.get("cap_strike")),
            "team_key": "|".join(str(v) for v in cs.values()) or None,
            "yes_bid": _f(m.get("yes_bid_dollars")),
            "yes_ask": _f(m.get("yes_ask_dollars")),
            "yes_bid_size": _f(m.get("yes_bid_size_fp")),
            "yes_ask_size": _f(m.get("yes_ask_size_fp")),
            "no_bid": _f(m.get("no_bid_dollars")),
            "no_ask": _f(m.get("no_ask_dollars")),
            "last_price": _f(m.get("last_price_dollars")),
            "volume": _f(m.get("volume_fp")),
            "volume_24h": _f(m.get("volume_24h_fp")),
            "open_interest": _f(m.get("open_interest_fp")),
            "status": m.get("status"),
            "result": m.get("result") or None,
            "occurrence_datetime": m.get("occurrence_datetime"),
            "close_time": m.get("close_time"),
            "fee_multiplier": mult,
        })
    return rows


def collect() -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, errors = [], []
    for league, series_list in config.SERIES.items():
        for s in series_list:
            try:
                mult = fee_multiplier(s)
                for ev in open_events(s):
                    rows += flatten(league, s, mult, ev, ts)
            except Exception as e:  # log the gap, keep going
                errors.append(f"{s}: {e}")
    df = pd.DataFrame(rows)
    out = SNAP_DIR / now.strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_parquet(out / f"{now.strftime('%H%M%S')}Z.parquet", index=False)
    log = ROOT / "data" / "collector_log.csv"
    new = not log.exists()
    with log.open("a") as fh:
        if new:
            fh.write("snapshot_ts,rows,errors\n")
        fh.write(f"{ts},{len(df)},\"{' ; '.join(errors)}\"\n")
    print(f"{ts}: {len(df)} markets, {len(errors)} errors")
    for e in errors:
        print("  ERROR", e, file=sys.stderr)
    return df


if __name__ == "__main__":
    collect()
