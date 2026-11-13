"""Picked-off check (#48): is our SHARP "edge" really us being late to news?

On 11-12, 3 alerts got "line moved, don't chase" within ~30-45 min of entry. If Kalshi traders react
to news before our next sharp pull, the gap we see is just us being behind. Per alerted (shadow) and
held-back (tracked) SHARP play, first sighting = entry (as graded):

  pull_age_min   minutes between the sharp pull the play was priced on and the entry snapshot.
  pre_move_c     change in the side's Kalshi mid over the PRE_WINDOW before entry (cents; negative =
                 the price was already falling toward our buy, i.e. Kalshi moved first).
  close_line     the same rung's sharp line from the last usable pull before kickoff (never a pull
                 after kickoff or after now).
  line_move_fav  close_line - entry line, signed so + = moved in our favour, - = against us (points).
  clv_c          from data/clv_bets.csv (#31): side mid at Kalshi's pregame close - entry, after fee.

Report: share of plays with an adverse sharp move, CLV split by Kalshi-moved-first vs not and by pull
age bucket, t by game. A bucket with >= MIN_PLAYS plays losing to the close at t <= -2 by game gets a
'PROPOSE ...' line (report only; Emerson decides; nothing in pricing or alerts changes here).
Output: data/pickoff_bets.csv, weekly_report.json["pickoff"].
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from . import clv, odds, premove, sharp

DATA = clv.DATA
OUT = DATA / "pickoff_bets.csv"
FLAG_FILES = {"shadow": DATA / "flags" / "sharp_shadow.csv", "tracked": DATA / "flags" / "sharp_tracked.csv"}
PRE_WINDOW = pd.Timedelta(minutes=30)
PRE_MOVE_C = 1.0            # mid fell >= 1c toward our buy in the 30 min before entry = Kalshi moved first
AGE_BUCKETS = [(0, 10, "0-10 min"), (10, 30, "10-30 min"), (30, 90, "30-90 min"), (90, 10_000, "90+ min")]
MIN_PLAYS, T_BAD = 30, -2.0
COLS = ["bet_id", "source", "league", "leg1", "entry_ts", "kickoff_utc", "odds_fetched_ts", "pull_age_min",
        "age_bucket", "pre_move_c", "kalshi_first", "entry_line", "close_line", "close_pull_ts",
        "line_move_fav", "adverse", "clv_c", "close_status"]
SNAP_EXTRA = ["series", "event_title", "event_sub", "floor_strike"]


class _LineOnly:
    """rung_prob() wants a distribution; we only need the line it reads, so p is a dummy."""
    def p_greater(self, line, strike):
        return 0.5


_DUMMY = {"spread": _LineOnly(), "total": _LineOnly()}


def age_bucket(m) -> str | None:
    try:
        m = float(m)
    except (TypeError, ValueError):
        return None
    if math.isnan(m) or m < 0:
        return None
    for lo, hi, name in AGE_BUCKETS:
        if lo <= m < hi:
            return name
    return None


def side_mid(row: dict, side: str):
    if side == "YES":
        return clv._mid(row.get("yes_bid"), row.get("yes_ask"))
    return clv._mid(row.get("no_bid"), row.get("no_ask"))


def entries() -> pd.DataFrame:
    """First sighting per (source, leg1) from the flag logs, with the sharp line and pull it used."""
    parts = []
    for src, p in FLAG_FILES.items():
        try:
            f = pd.read_csv(p)
        except Exception:
            continue
        if f.empty:
            continue
        if "leg1" not in f:
            f["leg1"] = f["side"].astype(str) + " " + f["ticker"].astype(str)
        f["_ts"] = pd.to_datetime(f["snapshot_ts"], utc=True, errors="coerce")
        f = f.dropna(subset=["_ts"]).sort_values("_ts").drop_duplicates("leg1")
        f["source"] = src
        parts.append(f[["source", "leg1", "league", "snapshot_ts", "sharp_line", "odds_fetched_ts"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def line_move_fav(side: str, entry_line, close_line):
    """Every SHARP rung is 'greater' (YES = margin/total above the strike), so a higher line helps YES."""
    try:
        d = float(close_line) - float(entry_line)
    except (TypeError, ValueError):
        return None
    if math.isnan(d):
        return None
    return round(d if side == "YES" else -d, 2)


def close_line(row: dict, lines: pd.DataFrame, league: str, as_of: pd.Timestamp, cache: dict):
    """(line, pull ts) for this rung from the last usable sharp pull at or before as_of."""
    key = (league, as_of)
    if key not in cache:
        f = odds._fair_lines(lines, as_of)                 # uncached: as_of differs per game
        cache[key] = f[f["league"] == league] if len(f) else f
    fair = cache[key]
    if fair is None or fair.empty:
        return None, None
    sides, fe = sharp.match_event(league, row, fair)
    if fe.empty:
        return None, None
    rp = sharp.rung_prob(_DUMMY, fe, row, sides or {})
    if rp is None:
        return None, None
    return rp[1], str(fe["fetched_ts"].max())


def analyse(bets: pd.DataFrame, flags: pd.DataFrame, snaps: pd.DataFrame, lines: pd.DataFrame,
            clv_rows: pd.DataFrame, now) -> pd.DataFrame:
    now = pd.Timestamp(now)
    if bets is None or bets.empty:
        return pd.DataFrame(columns=COLS)
    s = snaps.copy() if snaps is not None and len(snaps) else pd.DataFrame(columns=clv.SNAP_COLS + SNAP_EXTRA)
    s["_ts"] = pd.to_datetime(s["snapshot_ts"], utc=True, errors="coerce")
    s = s[s["_ts"].notna() & (s["_ts"] <= now)]
    by_t = {t: g.sort_values("_ts") for t, g in s.groupby("ticker")}
    fl = {(r["source"], r["leg1"]): r for r in flags.to_dict("records")} if flags is not None and len(flags) else {}
    cv = ({(r["bet_id"], r["source"]): r for r in clv_rows.to_dict("records")}
          if clv_rows is not None and len(clv_rows) else {})
    lines = lines if lines is not None else pd.DataFrame()
    cache: dict = {}
    rows = []
    for b in bets.to_dict("records"):
        side, ticker = str(b["leg1"]).split(" ", 1)
        entry = pd.to_datetime(b["snapshot_ts"], utc=True)
        f = fl.get((b["source"], b["leg1"]), {})
        c = cv.get((b["bet_id"], b["source"]), {})
        g = by_t.get(ticker)
        ko = None
        if g is not None and len(g):
            occ = pd.to_datetime(g["occurrence_datetime"].dropna().iloc[-1], utc=True, errors="coerce") \
                if g["occurrence_datetime"].notna().any() else None
            ko = occ - clv.OCC_OFFSET if occ is not None and not pd.isna(occ) else None
        r = {"bet_id": b["bet_id"], "source": b["source"], "league": b["league"], "leg1": b["leg1"],
             "entry_ts": entry.strftime("%Y-%m-%dT%H:%M:%SZ"),
             "kickoff_utc": ko.strftime("%Y-%m-%dT%H:%M:%SZ") if ko is not None else None,
             "odds_fetched_ts": f.get("odds_fetched_ts"), "entry_line": f.get("sharp_line"),
             "clv_c": c.get("clv_c"), "close_status": c.get("close_status")}
        pulled = pd.to_datetime(f.get("odds_fetched_ts"), utc=True, errors="coerce")
        r["pull_age_min"] = None if pd.isna(pulled) else round((entry - pulled).total_seconds() / 60, 1)
        r["age_bucket"] = age_bucket(r["pull_age_min"])
        # (b) Kalshi pre-move: side mid at entry vs the last quote at/before entry - PRE_WINDOW
        r["pre_move_c"] = r["kalshi_first"] = None
        if g is not None and len(g):
            at = g[g["_ts"] <= entry]
            before = g[g["_ts"] <= entry - PRE_WINDOW]
            if len(at) and len(before) and (entry - before["_ts"].iloc[-1]) <= PRE_WINDOW * 2:
                m1, m0 = side_mid(at.iloc[-1].to_dict(), side), side_mid(before.iloc[-1].to_dict(), side)
                if m1 is not None and m0 is not None:
                    r["pre_move_c"] = round(100 * (m1 - m0), 2)
                    r["kalshi_first"] = bool(r["pre_move_c"] <= -PRE_MOVE_C)
        # (a) sharp line at the last usable pull before kickoff (never after now)
        r["close_line"] = r["close_pull_ts"] = r["line_move_fav"] = r["adverse"] = None
        if ko is not None and ko <= now and g is not None and len(lines):
            snap_row = g.iloc[-1].to_dict()
            snap_row["kickoff_utc"] = r["kickoff_utc"]
            snap_row["snapshot_ts"] = r["entry_ts"]
            cl, cts = close_line(snap_row, lines, b["league"], ko - pd.Timedelta(seconds=1), cache)
            later = cts is not None and not pd.isna(pulled) and pd.Timestamp(cts) >= pulled
            if cl is not None and later:
                r["close_line"], r["close_pull_ts"] = cl, cts
                r["line_move_fav"] = line_move_fav(side, r["entry_line"], cl)
                r["adverse"] = None if r["line_move_fav"] is None else bool(r["line_move_fav"] < 0)
        rows.append(r)
    return pd.DataFrame(rows, columns=COLS)


def _cell(g: pd.DataFrame) -> dict:
    ok = g[g["close_status"] == "ok"]
    v = pd.to_numeric(ok["clv_c"], errors="coerce").dropna()
    ev = ok.loc[v.index, "leg1"].map(clv._event)
    return {"plays": int(len(g)), "with_close": int(len(v)),
            "avg_clv_c": round(float(v.mean()), 2) if len(v) else None,
            "games": int(ev.nunique()) if len(v) else 0,
            "t_by_game": clv._t(v.groupby(ev).mean()) if len(v) else None}


def summary(p: pd.DataFrame) -> dict:
    out = {"proposals": [], "in_effect": []}
    if p is None or p.empty:
        return out
    p = p.copy()
    p["_grp"] = p["league"].map(lambda x: "cbb" if x == "CBB" else "football")
    for grp, g in p.groupby("_grp"):
        moved = g[g["adverse"].notna()]
        d = {"plays": int(len(g)),
             "line_checked": int(len(moved)),
             "adverse_pct": round(100 * float(moved["adverse"].astype(bool).mean()), 1) if len(moved) else None,
             "avg_line_move_fav": round(float(pd.to_numeric(moved["line_move_fav"]).mean()), 2) if len(moved) else None,
             "median_pull_age_min": (round(float(pd.to_numeric(g["pull_age_min"]).median()), 1)
                                     if g["pull_age_min"].notna().any() else None),
             "kalshi_first": {str(k): _cell(x) for k, x in g[g["kalshi_first"].notna()].groupby(g["kalshi_first"].astype(bool))},
             "by_pull_age": {b: _cell(g[g["age_bucket"] == b]) for _, _, b in AGE_BUCKETS if (g["age_bucket"] == b).any()}}
        out[grp] = d
        checks = [("plays entered after Kalshi already moved toward us", d["kalshi_first"].get("True"),
                   "skip a play when Kalshi's price already fell 1c+ toward ours in the 30 min before"),
                  *[(f"plays priced on a {b} old sharp line", d["by_pull_age"].get(b),
                     f"alert only within {AGE_BUCKETS[i][0]} min of a fresh sharp pull")
                    for i, (_, _, b) in enumerate(AGE_BUCKETS) if i > 0]]
        guarded = grp == "cbb" and "CBB" in premove.GUARD_LEAGUES          # #50: this guard is live for CBB
        if guarded:
            cell = d["kalshi_first"].get("True")
            out["in_effect"].append(f"{grp}: kalshi-moved-first guard is ON (Emerson OK'd 11-12)"
                                    + (f"; those plays {cell['avg_clv_c']}c vs Kalshi's close "
                                       f"(t {cell['t_by_game']} by game, {cell['with_close']} plays)"
                                       if cell and cell["with_close"] else ""))
            if cell and cell["with_close"] >= MIN_PLAYS and cell["t_by_game"] is not None \
                    and cell["t_by_game"] >= -T_BAD and (cell["avg_clv_c"] or 0) > 0:
                out["proposals"].append(f"PROPOSE {grp}: turn OFF the kalshi-moved-first guard: the plays it holds "
                                        f"back beat Kalshi's close by {cell['avg_clv_c']}c "
                                        f"(t {cell['t_by_game']} by game, {cell['with_close']} plays)")
            checks = checks[1:]                                              # never re-propose a live guard
        for what, cell, fix in checks:
            if cell and cell["with_close"] >= MIN_PLAYS and cell["t_by_game"] is not None \
                    and cell["t_by_game"] <= T_BAD and (cell["avg_clv_c"] or 0) < 0:
                out["proposals"].append(f"PROPOSE {grp}: {what} lose {cell['avg_clv_c']}c to Kalshi's close "
                                        f"(t {cell['t_by_game']} by game, {cell['with_close']} plays): {fix}")
    return out


def run(now=None) -> dict:
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    frames = []
    for src, p in clv.BET_FILES.items():
        try:
            b = pd.read_csv(p)
        except Exception:
            continue
        if len(b):
            b["source"] = src
            frames.append(b)
    if not frames:
        pd.DataFrame(columns=COLS).to_csv(OUT, index=False)
        return {"proposals": []}
    bets = pd.concat(frames, ignore_index=True)
    bets = bets[pd.to_datetime(bets["snapshot_ts"], utc=True, errors="coerce").notna()]
    bets = bets[pd.to_datetime(bets["snapshot_ts"], utc=True) <= now]
    try:
        clv_rows = pd.read_csv(clv.OUT)
    except Exception:
        clv_rows = pd.DataFrame()
    try:
        lines = odds.load_lines()
        lines = lines[pd.to_datetime(lines["fetched_ts"], utc=True, errors="coerce") <= now]
    except Exception:
        lines = pd.DataFrame()
    tickers = set(bets["leg1"].astype(str).str.split(" ", n=1).str[1])
    start = pd.to_datetime(bets["snapshot_ts"], utc=True).min() - PRE_WINDOW * 2
    snaps = load_snapshots(tickers, start, now)
    p = analyse(bets, entries(), snaps, lines, clv_rows, now)
    p.to_csv(OUT, index=False)
    s = summary(p)
    print("pickoff: " + " | ".join(f"{k} adverse {v['adverse_pct']}% of {v['line_checked']} "
                                   f"median pull age {v['median_pull_age_min']}m"
                                   for k, v in s.items() if k != "proposals")
          + (f" | {len(s['proposals'])} proposal(s)" if s["proposals"] else ""))
    return s


RERUN_EVERY = pd.Timedelta(minutes=60)
STATE = DATA / "pickoff_summary.json"


def run_if_due(now=None) -> dict:
    """The live loop calls weekly every 15 min; this takes ~35 s, so re-run at most hourly and
    serve the last summary in between (stamped when it was made, never ahead of now)."""
    import json
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    try:
        last = json.loads(STATE.read_text())
        made = pd.Timestamp(last["made_utc"])
        if made <= now and now - made < RERUN_EVERY:
            return last["summary"]
    except Exception:
        pass
    s = run(now)
    STATE.write_text(json.dumps({"made_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "summary": s}, default=str))
    return s


def load_snapshots(tickers: set[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """clv.load_snapshots plus the columns match_event/rung_prob need."""
    cols = clv.SNAP_COLS + SNAP_EXTRA
    parts = []
    if not tickers or not clv.SNAP_DIR.exists():
        return pd.DataFrame(columns=cols)
    for day in sorted(x for x in clv.SNAP_DIR.iterdir() if x.is_dir()):
        try:
            d = pd.Timestamp(day.name).date()
        except Exception:
            continue
        if d < start.date() or d > end.date():
            continue
        for f in sorted(day.glob("*.parquet")):
            try:
                df = pd.read_parquet(f, columns=cols)
            except Exception:
                continue
            df = df[df["ticker"].isin(tickers)]
            if len(df):
                parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


if __name__ == "__main__":
    run()
