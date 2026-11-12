"""Kalshi 1-minute candles for graded contracts (#39a), so Auto Sell fills aren't missed between snapshots.

GET {KALSHI_BASE}/series/{series}/markets/{ticker}/candlesticks?start_ts&end_ts&period_interval=1
Each candle: end_period_ts, yes_bid {open, low, high, close}, yes_ask {...} in cents (or *_dollars
strings on newer responses). Cached per ticker in data/candles/<ticker>.csv as
end_ts, yes_bid_hi, yes_bid_lo, yes_ask_hi, yes_ask_lo (0-1); fetch state in data/candles/_state.json.

Only runs where Kalshi is reachable (GitHub Actions); anywhere else the first request fails and
refresh() returns quietly. If the endpoint needs a key (401/403) that's recorded in the state as
"needs_key" and nothing more is tried until the next day. Never stores a candle whose minute
ends after `now`.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DIR = DATA / "candles"
STATE = DIR / "_state.json"
CHUNK = pd.Timedelta(hours=12)          # <= 720 one-minute candles per request
MAX_TICKERS = 150                       # per refresh, so a weekly step stays quick
LOOKBACK = pd.Timedelta(days=5)         # only bets entered in the last 5 days
PACE_S = 0.12                           # ~8 req/s, under the public read limit (#43)
BACKOFF_S = (1.0, 2.0, 4.0)             # 429 retries (#43)
BUDGET_S = float(os.environ.get("CANDLES_BUDGET_S", "30"))   # live runs weekly every 15 min (#43)
COLS = ["end_ts", "yes_bid_hi", "yes_bid_lo", "yes_ask_hi", "yes_ask_lo"]


def _p(v):
    """Cents int / dollars string / None -> 0-1 float or None."""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x / 100 if x > 1 else x


def _px(d: dict, k: str):
    """`<k>_dollars` is already 0-1; plain `<k>` is cents (so 1 means 1c, not $1)."""
    if d.get(f"{k}_dollars") is not None:
        try:
            return float(d[f"{k}_dollars"])
        except (TypeError, ValueError):
            return None
    v = d.get(k)
    try:
        return None if v is None else float(v) / 100
    except (TypeError, ValueError):
        return None


def parse(j: dict, now) -> pd.DataFrame:
    now = pd.Timestamp(now)
    rows = []
    for c in (j or {}).get("candlesticks") or []:
        end = pd.Timestamp(int(c["end_period_ts"]), unit="s", tz="UTC")
        if end > now:
            continue                                  # an unfinished / future minute: never
        yb, ya = c.get("yes_bid") or {}, c.get("yes_ask") or {}
        hi = lambda d: _px(d, "high")   # noqa: E731
        lo = lambda d: _px(d, "low")    # noqa: E731
        rows.append({"end_ts": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "yes_bid_hi": hi(yb), "yes_bid_lo": lo(yb),
                     "yes_ask_hi": hi(ya), "yes_ask_lo": lo(ya)})
    return pd.DataFrame(rows, columns=COLS)


def load_state(path: Path | None = None) -> dict:
    try:
        return json.loads((path or STATE).read_text())
    except Exception:  # noqa: BLE001
        return {"tickers": {}}


def _get(path: str, params: dict):
    from . import collect
    return collect.session.get(config.KALSHI_BASE + path, params=params, timeout=30)


def fetch(ticker: str, start, end, now, get=_get, sleep=time.sleep) -> tuple[pd.DataFrame, str | None]:
    """Candles for one ticker over [start, end], in CHUNK requests. Returns (frame, problem).

    #43: requests are paced (PACE_S) and a 429 is retried with backoff (1, 2, 4 s) before giving up."""
    series = ticker.split("-")[0]
    parts, t = [], pd.Timestamp(start)
    end = min(pd.Timestamp(end), pd.Timestamp(now))
    got = lambda: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)  # noqa: E731
    while t < end:
        u = min(t + CHUNK, end)
        for attempt in range(len(BACKOFF_S) + 1):
            r = get(f"/series/{series}/markets/{ticker}/candlesticks",
                    {"start_ts": int(t.timestamp()), "end_ts": int(u.timestamp()), "period_interval": 1})
            sleep(PACE_S)
            code = getattr(r, "status_code", 200)
            if code != 429:
                break
            if attempt < len(BACKOFF_S):
                sleep(BACKOFF_S[attempt])
        if code in (401, 403):
            return pd.DataFrame(columns=COLS), "needs_key"
        if code == 429:
            return got(), "throttled"
        if code >= 400:
            return got(), f"http {code}"
        parts.append(parse(r.json(), now))
        t = u
    return got(), None


def merge(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    try:
        old = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        old = pd.DataFrame(columns=COLS)
    df = pd.concat([old, new], ignore_index=True)
    df = df.drop_duplicates("end_ts", keep="last").sort_values("end_ts")
    path.parent.mkdir(parents=True, exist_ok=True)
    df[COLS].to_csv(path, index=False)
    return df


def todo(bets: pd.DataFrame, state: dict, now) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """(ticker, fetch_from, fetch_to) for bets entered in the last LOOKBACK that aren't finished.

    #43 order: settled tickers first (only they can be graded, and their window ends at settlement,
    so they finish), then the least recently fetched, so pending ones take turns instead of the
    same oldest few using up every run's request allowance."""
    now = pd.Timestamp(now)
    if bets is None or bets.empty:
        return []
    b = bets.copy()
    b["_ts"] = pd.to_datetime(b["snapshot_ts"], utc=True, errors="coerce")
    b = b[b["_ts"].notna() & (b["_ts"] >= now - LOOKBACK) & (b["_ts"] <= now)].sort_values("_ts")
    sts_by = {}
    if "status" in b and "settled_ts" in b:
        for r in b[b["status"] == "settled"].to_dict("records"):
            sts = pd.to_datetime(r.get("settled_ts"), utc=True, errors="coerce")
            if pd.notna(sts):
                sts_by[str(r["leg1"]).split(" ", 1)[-1]] = sts
    rows, seen = [], set()
    for r in b.to_dict("records"):
        tk = str(r["leg1"]).split(" ", 1)[-1]
        if tk in seen:
            continue
        seen.add(tk)
        st = (state.get("tickers") or {}).get(tk) or {}
        if st.get("done"):
            continue
        frm = st.get("fetched_to") or st.get("last_end")
        start = pd.Timestamp(frm) if frm else r["_ts"]
        stop = min(now, sts_by[tk]) if tk in sts_by else now
        rows.append((tk not in sts_by, st.get("fetched_at") or "", r["_ts"], tk, start, stop))
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    return [(tk, a, z) for *_, tk, a, z in rows][:MAX_TICKERS]


def refresh(now=None, bets: pd.DataFrame | None = None, get=_get, state_path: Path | None = None,
            directory: Path | None = None, sleep=time.sleep, budget_s: float | None = None,
            clock=time.monotonic) -> dict:
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    nows = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    budget_s = BUDGET_S if budget_s is None else budget_s
    directory = directory or DIR
    state_path = state_path or (directory / "_state.json")
    state = load_state(state_path)
    if state.get("needs_key_until") and pd.Timestamp(state["needs_key_until"]) > now:
        return {"skipped": "needs_key"}
    if bets is None:
        from . import clv
        frames = []
        for src, p in clv.BET_FILES.items():
            try:
                frames.append(pd.read_csv(p))
            except Exception:  # noqa: BLE001
                continue
        bets = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    jobs = todo(bets, state, now)
    stats = {"todo": len(jobs), "tickers": 0, "candles": 0, "done": 0, "errors": 0, "problem": None}
    t0 = clock()
    for tk, start, stop in jobs:
        if clock() - t0 > budget_s:
            stats["problem"] = "budget"          # the rest wait for the next run (settled ones first)
            break
        try:
            df, problem = fetch(tk, start, stop, now, get, sleep)
        except Exception as e:  # noqa: BLE001 - no network here (cloud workspace): stop quietly
            stats["problem"] = type(e).__name__
            break
        if problem == "needs_key":
            state["needs_key_until"] = (now + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            stats["problem"] = "needs_key"
            break
        if len(df):
            merge(directory / f"{tk}.csv", df)
            stats["candles"] += len(df)
        st = state.setdefault("tickers", {}).setdefault(tk, {})
        st["fetched_at"] = nows
        if len(df):
            st["last_end"] = max(str(df["end_ts"].max()), st.get("last_end") or "")
        if problem == "throttled":
            stats["problem"] = "throttled"       # still 429 after backoff: stop for this run
            break
        if problem:                              # e.g. http 404 on one market: note it, keep going
            st["error"] = problem
            stats["errors"] += 1
            continue
        st.pop("error", None)
        st["fetched_to"] = min(pd.Timestamp(stop), now).strftime("%Y-%m-%dT%H:%M:%SZ")
        if pd.Timestamp(stop) < now:             # window ended at settlement and was read in full
            st["done"] = True
            stats["done"] += 1
        stats["tickers"] += 1
    directory.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=1, sort_keys=True))
    print(f"candles: {stats}")
    return stats


if __name__ == "__main__":
    refresh()
