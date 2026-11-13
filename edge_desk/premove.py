"""Kalshi-moved-first guard (Emerson OK'd 11-12 evening, from #48's first proposal).

CBB plays whose side had already fallen >= PRE_MOVE_C toward our buy over the 30 min before the scan
lost ~3c each to Kalshi's close (53 plays, t -2.34 by game, 11-12): when Kalshi moves first, someone
there likely has news our sharp pull doesn't, so the "edge" is us being late. Such a play is NOT
alerted; it goes to data/flags/sharp_tracked.csv with reason 'kalshi moved first' and is still
graded, so the guard can be judged (and reverted) on data. Leagues: GUARD_LEAGUES only (football has
no evidence yet; #48 keeps reporting it).

The earlier quote comes from the saved snapshot nearest to (but not after) scan time - PRE_WINDOW,
no older than scan time - 2 x PRE_WINDOW; no such snapshot -> the play passes (fails open, as before).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import clv

PRE_WINDOW = pd.Timedelta(minutes=30)
PRE_MOVE_C = 1.0
GUARD_LEAGUES = ("CBB",)
REASON = "kalshi moved first"
QCOLS = ["ticker", "yes_bid", "yes_ask", "no_bid", "no_ask"]
_CACHE: dict = {}


def _stamp(p: Path) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(f"{p.parent.name}T{p.stem[:2]}:{p.stem[2:4]}:{p.stem[4:6]}Z")
    except Exception:
        return None


def earlier_quotes(ts, snap_dir: Path | None = None) -> pd.DataFrame | None:
    """Quotes from the newest snapshot stamped in [ts - 2*PRE_WINDOW, ts - PRE_WINDOW]."""
    ts = pd.Timestamp(ts)
    lo, hi = ts - 2 * PRE_WINDOW, ts - PRE_WINDOW
    d = snap_dir or clv.SNAP_DIR
    best = None
    for day in {lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")}:
        for f in (d / day).glob("*.parquet") if (d / day).exists() else []:
            s = _stamp(f)
            if s is not None and lo <= s <= hi and (best is None or s > best[0]):
                best = (s, f)
    if best is None:
        return None
    key = str(best[1])
    if key not in _CACHE:
        try:
            q = pd.read_parquet(best[1], columns=QCOLS)
        except Exception:
            return None
        _CACHE.clear()
        _CACHE[key] = q.drop_duplicates("ticker", keep="last").set_index("ticker")
    return _CACHE[key]


def side_mid(row, side: str):
    if row is None:
        return None
    g = row.get if hasattr(row, "get") else (lambda k: row[k])
    if side == "YES":
        return clv._mid(g("yes_bid"), g("yes_ask"))
    return clv._mid(g("no_bid"), g("no_ask"))


def pre_move_c(flag: dict, now_row, earlier: pd.DataFrame | None):
    if earlier is None or now_row is None or flag["ticker"] not in earlier.index:
        return None
    m1, m0 = side_mid(now_row, flag["side"]), side_mid(earlier.loc[flag["ticker"]], flag["side"])
    if m1 is None or m0 is None:
        return None
    return round(100 * (m1 - m0), 2)


def screen(flags: list[dict], df: pd.DataFrame, ts, league: str, log=None, snap_dir: Path | None = None) -> list[dict]:
    """Drop (and log as held back) guarded-league flags whose side Kalshi already moved toward.
    Any error in the guard itself passes the flags through unchanged (never lose a play to a bug)."""
    try:
        return _screen(flags, df, ts, league, log, snap_dir)
    except Exception as e:                      # noqa: BLE001
        print(f"premove: guard skipped ({type(e).__name__})")
        return flags


def _screen(flags, df, ts, league, log=None, snap_dir=None):
    if league not in GUARD_LEAGUES or not flags or ts is None or df is None or "ticker" not in df:
        return flags
    earlier = earlier_quotes(ts, snap_dir)
    if earlier is None:
        return flags
    rows = df.drop_duplicates("ticker", keep="last").set_index("ticker")
    if log is None:
        from .sharp import log_rejected as log
    keep = []
    for f in flags:
        now_row = rows.loc[f["ticker"]] if f["ticker"] in rows.index else None
        mv = pre_move_c(f, now_row, earlier)
        if mv is not None and mv <= -PRE_MOVE_C:
            log({**f, "pre_move_c": mv}, REASON)
            continue
        keep.append(f)
    return keep
