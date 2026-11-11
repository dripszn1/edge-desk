"""Closing-price value (#31): did a SHARP shadow / tracked entry beat Kalshi's own pregame close?

Settled records need >= 30 bets and a weekend of games; CLV is readable as soon as a game kicks
off, and is the standard early test of whether entries are well priced.

Per bet (first sighting = entry, as graded):
  close     = mid of the bet's side (YES: yes bid/ask, NO: no bid/ask) in the LAST snapshot taken
              strictly after the entry and strictly before kickoff (Kalshi occurrence - 3h).
              Snapshots stamped after `now` or at/after kickoff are never used (no look-ahead).
              One-sided or crossed quotes (bid <= 0, ask >= 1, bid > ask) are skipped.
  stale     = the last usable pregame quote is more than CLOSE_MAX_GAP before kickoff -> no CLV.
  clv_raw_c = (close_mid - entry price) in cents
  clv_c     = clv_raw_c - entry fee per contract (bet fees / contracts): beat the close after fees.
Statuses: ok | pending (kickoff not reached) | stale | no_close (no quote after the entry).
Output: data/clv_bets.csv; summary() goes into weekly_report.json["clv"] and one digest line.
Archived snapshot weeks (#25) are not pulled back here; those bets read 'no_close'.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SNAP_DIR = DATA / "snapshots"
OUT = DATA / "clv_bets.csv"
BET_FILES = {"shadow": DATA / "sharp_shadow_bets.csv", "tracked": DATA / "sharp_tracked_bets.csv"}
CLOSE_MAX_GAP = pd.Timedelta(hours=2)
OCC_OFFSET = pd.Timedelta(hours=3)
SNAP_COLS = ["snapshot_ts", "ticker", "yes_bid", "yes_ask", "no_bid", "no_ask", "occurrence_datetime"]
COLS = ["bet_id", "source", "league", "leg1", "entry_ts", "price", "fee_c", "kickoff_utc",
        "close_ts", "close_mid", "clv_raw_c", "clv_c", "beat_close", "close_status"]


def _mid(bid, ask):
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return None
    if not (0 < bid <= ask < 1) or math.isnan(bid) or math.isnan(ask):
        return None
    return (bid + ask) / 2


def closing_value(bets: pd.DataFrame, snaps: pd.DataFrame, now=None) -> pd.DataFrame:
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if bets is None or bets.empty:
        return pd.DataFrame(columns=COLS)
    s = snaps.copy() if snaps is not None else pd.DataFrame(columns=SNAP_COLS)
    if s.empty:
        s = pd.DataFrame(columns=SNAP_COLS)
    s["_ts"] = pd.to_datetime(s["snapshot_ts"], utc=True, errors="coerce")
    s = s[s["_ts"].notna() & (s["_ts"] <= now)]
    s["_ko"] = pd.to_datetime(s["occurrence_datetime"], utc=True, errors="coerce") - OCC_OFFSET
    by_t = {t: g.sort_values("_ts") for t, g in s.groupby("ticker")}
    rows = []
    for b in bets.to_dict("records"):
        side, ticker = str(b["leg1"]).split(" ", 1)
        entry = pd.Timestamp(b["snapshot_ts"])
        entry = entry.tz_localize("UTC") if entry.tzinfo is None else entry
        price = float(b["price"])
        n = float(b.get("contracts") or 0)
        fee_c = 100 * float(b.get("fees") or 0) / n if n > 0 else 0.0
        r = {"bet_id": b.get("bet_id"), "source": b.get("source"), "league": b.get("league"),
             "leg1": b["leg1"], "entry_ts": entry.strftime("%Y-%m-%dT%H:%M:%SZ"), "price": price,
             "fee_c": round(fee_c, 4), "kickoff_utc": None, "close_ts": None, "close_mid": None,
             "clv_raw_c": None, "clv_c": None, "beat_close": None, "close_status": "no_close"}
        g = by_t.get(ticker)
        if g is None or g.empty:
            rows.append(r)
            continue
        ko = g["_ko"].dropna()
        if ko.empty:
            rows.append(r)
            continue
        ko = ko.iloc[-1]                              # latest known kickoff (reschedules)
        r["kickoff_utc"] = ko.strftime("%Y-%m-%dT%H:%M:%SZ")
        if now < ko:
            r["close_status"] = "pending"
            rows.append(r)
            continue
        pre = g[(g["_ts"] > entry) & (g["_ts"] < ko)]
        lo, hi = ("yes_bid", "yes_ask") if side.upper() == "YES" else ("no_bid", "no_ask")
        close = None
        for q in pre.iloc[::-1].to_dict("records"):
            m = _mid(q[lo], q[hi])
            if m is not None:
                close = (q["_ts"], m)
                break
        if close is None:
            rows.append(r)
            continue
        if ko - close[0] > CLOSE_MAX_GAP:
            r["close_status"] = "stale"
            rows.append(r)
            continue
        raw = 100 * (close[1] - price)
        r.update(close_ts=close[0].strftime("%Y-%m-%dT%H:%M:%SZ"), close_mid=close[1],
                 clv_raw_c=round(raw, 4), clv_c=round(raw - fee_c, 4), beat_close=bool(raw - fee_c > 0),
                 close_status="ok")
        rows.append(r)
    return pd.DataFrame(rows, columns=COLS)


def _t(x: pd.Series):
    x = pd.Series(x).dropna().astype(float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return None
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2)


def _event(leg1: str) -> str:
    """'YES KXNFLSPREAD-26NOV14AAABBB-AAA3' -> '26NOV14AAABBB' (same game across series)."""
    parts = str(leg1).split(" ", 1)[-1].split("-")
    return parts[1] if len(parts) > 1 else parts[0]


def summary(c: pd.DataFrame) -> dict:
    """{source: {football|cbb: n, avg_clv_c, avg_raw_c, beat_close_pct, t_stat, stale, pending}}."""
    out = {}
    if c is None or c.empty:
        return out
    c = c.copy()
    c["_grp"] = c["league"].map(lambda x: "cbb" if x == "CBB" else "football")
    for src, gs in c.groupby("source"):
        out[src] = {}
        for grp, g in gs.groupby("_grp"):
            ok = g[g["close_status"] == "ok"]
            v = pd.to_numeric(ok["clv_c"], errors="coerce")
            out[src][grp] = {
                "n": int(len(ok)),
                "avg_clv_c": round(float(v.mean()), 2) if len(ok) else None,
                "avg_raw_c": round(float(pd.to_numeric(ok["clv_raw_c"]).mean()), 2) if len(ok) else None,
                "beat_close_pct": round(100 * float(ok["beat_close"].astype(bool).mean()), 1) if len(ok) else None,
                "t_stat": _t(v),
                # rungs of one game move together, so the honest t uses one mean per event
                "events": int(ok["leg1"].map(_event).nunique()) if len(ok) else 0,
                "t_by_event": _t(v.groupby(ok["leg1"].map(_event)).mean()) if len(ok) else None,
                "stale": int((g["close_status"] == "stale").sum()),
                "pending": int((g["close_status"] == "pending").sum()),
                "no_close": int((g["close_status"] == "no_close").sum()),
            }
    return out


FINAL_AFTER = pd.Timedelta(hours=1)


def final_rows(prev: pd.DataFrame, now) -> pd.DataFrame:
    """Cached rows that can no longer change: kickoff more than FINAL_AFTER ago (late-committed
    pregame snapshots have landed by then) and a close already decided (ok / stale). Keeps the
    live loop's 15-min weekly step from re-reading every snapshot day."""
    if prev is None or prev.empty or "close_status" not in prev:
        return pd.DataFrame(columns=COLS)
    ko = pd.to_datetime(prev["kickoff_utc"], utc=True, errors="coerce")
    m = prev["close_status"].isin(["ok", "stale"]) & ko.notna() & (pd.Timestamp(now) - ko > FINAL_AFTER)
    return prev[m].reindex(columns=COLS)


def load_snapshots(tickers: set[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Only the needed tickers/columns from day folders between start and end (UTC dates)."""
    parts = []
    if not tickers or not SNAP_DIR.exists():
        return pd.DataFrame(columns=SNAP_COLS)
    for day in sorted(p for p in SNAP_DIR.iterdir() if p.is_dir()):
        try:
            d = pd.Timestamp(day.name).date()
        except Exception:
            continue
        if d < start.date() or d > end.date():
            continue
        for f in sorted(day.glob("*.parquet")):
            try:
                df = pd.read_parquet(f, columns=SNAP_COLS)
            except Exception:
                continue
            df = df[df["ticker"].isin(tickers)]
            if len(df):
                parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SNAP_COLS)


def run(now=None) -> dict:
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    frames = []
    for src, p in BET_FILES.items():
        try:
            b = pd.read_csv(p)
        except Exception:
            continue
        if len(b):
            b["source"] = src
            frames.append(b)
    if not frames:
        pd.DataFrame(columns=COLS).to_csv(OUT, index=False)
        return {}
    bets = pd.concat(frames, ignore_index=True)
    bets = bets[pd.to_datetime(bets["snapshot_ts"], utc=True, errors="coerce").notna()]
    try:
        prev = pd.read_csv(OUT)
    except Exception:
        prev = pd.DataFrame(columns=COLS)
    done = final_rows(prev, now)
    keep = done[done["bet_id"].isin(bets["bet_id"]) & done["source"].isin(bets["source"])]
    todo = bets[~(bets["bet_id"] + "|" + bets["source"]).isin(keep["bet_id"] + "|" + keep["source"])]
    c = keep
    if len(todo):
        tickers = set(todo["leg1"].astype(str).str.split(" ", n=1).str[1])
        start = pd.to_datetime(todo["snapshot_ts"], utc=True).min()
        snaps = load_snapshots(tickers, start, now)
        c = pd.concat([keep, closing_value(todo, snaps, now)], ignore_index=True)
    c[COLS].to_csv(OUT, index=False)
    s = summary(c)
    print("clv: " + " | ".join(f"{src} {grp} n={v['n']} ({v['events']} games) avg {v['avg_clv_c']}c "
                               f"t/game={v['t_by_event']} "
                               f"(pending {v['pending']}, stale {v['stale']})"
                               for src, d in s.items() for grp, v in d.items()))
    return s


if __name__ == "__main__":
    run()
