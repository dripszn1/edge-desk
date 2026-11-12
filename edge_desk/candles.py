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
FINAL_AFTER = pd.Timedelta(hours=8)     # stop refreshing 8h past kickoff-ish (entry + 8h after settle)
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


def parse(j: dict, now) -> pd.DataFrame:
    now = pd.Timestamp(now)
    rows = []
    for c in (j or {}).get("candlesticks") or []:
        end = pd.Timestamp(int(c["end_period_ts"]), unit="s", tz="UTC")
        if end > now:
            continue                                  # an unfinished / future minute: never
        yb, ya = c.get("yes_bid") or {}, c.get("yes_ask") or {}
        hi = lambda d: _p(d.get("high_dollars", d.get("high")))   # noqa: E731
        lo = lambda d: _p(d.get("low_dollars", d.get("low")))     # noqa: E731
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


def fetch(ticker: str, start, end, now, get=_get) -> tuple[pd.DataFrame, str | None]:
    """Candles for one ticker over [start, end], in CHUNK requests. Returns (frame, problem)."""
    series = ticker.split("-")[0]
    parts, t = [], pd.Timestamp(start)
    end = min(pd.Timestamp(end), pd.Timestamp(now))
    while t < end:
        u = min(t + CHUNK, end)
        r = get(f"/series/{series}/markets/{ticker}/candlesticks",
                {"start_ts": int(t.timestamp()), "end_ts": int(u.timestamp()), "period_interval": 1})
        code = getattr(r, "status_code", 200)
        if code in (401, 403):
            return pd.DataFrame(columns=COLS), "needs_key"
        if code == 429:
            return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)), "throttled"
        if code >= 400:
            return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)), f"http {code}"
        parts.append(parse(r.json(), now))
        t = u
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)), None


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


def todo(bets: pd.DataFrame, state: dict, now) -> list[tuple[str, pd.Timestamp]]:
    """(ticker, fetch_from) for bets entered in the last LOOKBACK that aren't finished."""
    now = pd.Timestamp(now)
    out, seen = [], set()
    if bets is None or bets.empty:
        return out
    b = bets.copy()
    b["_ts"] = pd.to_datetime(b["snapshot_ts"], utc=True, errors="coerce")
    b = b[b["_ts"].notna() & (b["_ts"] >= now - LOOKBACK) & (b["_ts"] <= now)].sort_values("_ts")
    for r in b.to_dict("records"):
        tk = str(r["leg1"]).split(" ", 1)[-1]
        if tk in seen:
            continue
        seen.add(tk)
        st = (state.get("tickers") or {}).get(tk) or {}
        if st.get("done"):
            continue
        last = pd.Timestamp(st["last_end"]) if st.get("last_end") else r["_ts"]
        out.append((tk, last))
    return out[:MAX_TICKERS]


def refresh(now=None, bets: pd.DataFrame | None = None, get=_get, state_path: Path | None = None,
            directory: Path | None = None, sleep=time.sleep) -> dict:
    now = pd.Timestamp(now or datetime.now(timezone.utc))
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
                f = pd.read_csv(p)
                frames.append(f)
            except Exception:  # noqa: BLE001
                continue
        bets = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    settled = {}
    if bets is not None and not bets.empty and "status" in bets:
        for r in bets[bets["status"] == "settled"].to_dict("records"):
            settled[str(r["leg1"]).split(" ", 1)[-1]] = r.get("settled_ts")
    stats = {"tickers": 0, "candles": 0, "problem": None}
    for tk, start in todo(bets, state, now):
        try:
            df, problem = fetch(tk, start, now, now, get)
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
        st["last_end"] = (df["end_ts"].max() if len(df) else start.strftime("%Y-%m-%dT%H:%M:%SZ"))
        sts = settled.get(tk)
        if sts and pd.Timestamp(st["last_end"]) >= pd.Timestamp(sts) + pd.Timedelta(minutes=0) \
                and now - pd.Timestamp(sts) > FINAL_AFTER:
            st["done"] = True
        stats["tickers"] += 1
        if problem:
            stats["problem"] = problem
            break
        sleep(0.05)
    directory.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=1, sort_keys=True))
    print(f"candles: {stats}")
    return stats


if __name__ == "__main__":
    refresh()
