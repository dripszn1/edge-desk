"""Line-move lag (backlog #12): when the sharp line moves >= 1 pt, how long does Kalshi take to follow?

For every NFL event + market (spread, total):
  sharp line series   - the sharp fair point at each odds pull (odds.fair_lines as of that pull).
  Kalshi implied line - at each Kalshi snapshot, the strike where the rung mid-price crosses 50%,
                        linearly interpolated across the ladder (spreads: home-team rungs plus the
                        away-team rungs mirrored, P(home margin > -k) = 1 - P(away margin > k)).
A MOVE is a change of >= MOVE_PTS between consecutive pulls (prev_pull_ts -> detected_ts). The true
move time is somewhere in that window, so lag is measured from detection (what we could act on):
  already_moved - at the first Kalshi snapshot at/after detection, Kalshi is already within
                  FOLLOW_FRAC of the way to the new line (lag <= 0: no timing edge on this one).
  followed      - Kalshi got there later; lag_minutes = first such snapshot - detected_ts.
  not_followed  - never got there in the snapshots so far (or kickoff came first).
Look-ahead: a move is only evaluated with Kalshi snapshots taken at/after its detection and
before kickoff; the Kalshi line "before" uses only snapshots at/before prev_pull_ts.
Money check (fees in): at the first snapshot after detection, the best rung edge vs the NEW sharp
fair = fair - ask - taker fee, the same rule as SHARP (fees at a 1-lot, worst case).
Outputs data/odds/lag.csv (one row per move) and data/odds/lag_summary.json (verdict).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import odds, sharp
from .fees import fee_per_contract

ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = ROOT / "data" / "snapshots"
OUT = ROOT / "data" / "odds" / "lag.csv"
SUMMARY = ROOT / "data" / "odds" / "lag_summary.json"

MOVE_PTS = 1.0
FOLLOW_FRAC = 0.5
MIN_MOVES = 10
ACT_MINUTES = 30            # a lag shorter than this is too fast to act on by hand
COLS = ["league", "event_ticker", "market", "prev_pull_ts", "detected_ts", "sharp_from", "sharp_to",
        "kalshi_before", "kalshi_at_detect", "status", "followed_ts", "lag_minutes",
        "best_edge_after_fee", "kickoff_utc"]


def _ts(x):
    return sharp._ts(x)


def implied_line(rows: pd.DataFrame, market: str, home_abbr: str | None = None) -> float | None:
    """Strike where P(YES) crosses 0.5 for one Kalshi event's ladder at one snapshot.
    Spread: returns the expected HOME margin. Total: the expected total."""
    r = rows.dropna(subset=["yes_bid", "yes_ask", "floor_strike"])
    r = r[(r["yes_ask"] > 0) & (r["strike_type"] == "greater")]
    if r.empty:
        return None
    mid = ((r["yes_bid"] + r["yes_ask"]) / 2).to_numpy(float)
    k = r["floor_strike"].to_numpy(float)
    if market == "SPREAD":
        team = r["ticker"].str.extract(r"-([A-Z&]+)\d+$")[0].to_numpy()
        home = team == home_abbr
        k = np.where(home, k, -k)                    # away "by over k" -> home margin < -k
        mid = np.where(home, mid, 1 - mid)           # P(home margin > -k) ~ 1 - P(away by > k)
    o = np.argsort(k)
    k, p = k[o], np.minimum.accumulate(mid[o])        # enforce a non-increasing ladder
    if p[0] < 0.5 or p[-1] > 0.5:
        return None                                   # 50% not bracketed by the listed rungs
    i = int(np.argmax(p <= 0.5))
    if i == 0 or p[i] == 0.5:
        return float(k[i])
    k0, k1, p0, p1 = k[i - 1], k[i], p[i - 1], p[i]
    return float(k0 + (p0 - 0.5) * (k1 - k0) / (p0 - p1))


def load_snapshots(snap_dir: Path = SNAP_DIR, league: str = "NFL", days: int = 14) -> pd.DataFrame:
    parts = []
    dirs = sorted(d for d in snap_dir.glob("*") if d.is_dir())[-days:]
    for f in (f for d in dirs for f in sorted(d.glob("*.parquet"))):
        try:
            d = pd.read_parquet(f, columns=["snapshot_ts", "league", "series", "event_ticker", "event_sub",
                                            "ticker", "strike_type", "floor_strike", "yes_bid", "yes_ask",
                                            "yes_bid_size", "yes_ask_size", "fee_multiplier",
                                            "occurrence_datetime", "status"])
        except Exception:
            continue
        parts.append(d[(d["league"] == league) & d["series"].isin(["KXNFLSPREAD", "KXNFLTOTAL"])])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def sharp_series(lines: pd.DataFrame, league: str = "NFL") -> pd.DataFrame:
    """Sharp point per (event, market) at each odds pull: home spread point -> home margin line."""
    rows = []
    if lines is None or lines.empty:
        return pd.DataFrame(columns=["fetched_ts", "event_id", "home_team", "away_team", "commence_utc",
                                     "market", "line"])
    lines = lines[lines["league"] == league]
    for ts in sorted(lines["fetched_ts"].unique()):
        f = odds.fair_lines(lines[lines["fetched_ts"] <= ts], ts)
        f = f[f["fetched_ts"] == ts]                 # only events re-quoted in this pull
        for _, r in f.iterrows():
            if r["market"] == "spreads" and r["outcome"] == r["home_team"] and pd.notna(r["point"]):
                rows.append((ts, r["event_id"], r["home_team"], r["away_team"], r["commence_utc"], "SPREAD", -float(r["point"])))
            elif r["market"] == "totals" and r["outcome"] == "Over" and pd.notna(r["point"]):
                rows.append((ts, r["event_id"], r["home_team"], r["away_team"], r["commence_utc"], "TOTAL", float(r["point"])))
    return pd.DataFrame(rows, columns=["fetched_ts", "event_id", "home_team", "away_team", "commence_utc",
                                       "market", "line"])


def _kalshi_event(snaps: pd.DataFrame, home_full: str, away_full: str, commence, market: str):
    """Kalshi event ticker + home abbr for an odds event (names via sharp.NFL_NAMES)."""
    series = "KXNFLSPREAD" if market == "SPREAD" else "KXNFLTOTAL"
    s = snaps[snaps["series"] == series].drop_duplicates("event_ticker")
    for r in s.to_dict("records"):
        sides = sharp.nfl_sides(r)
        if not sides or set(sides.values()) != {home_full, away_full}:
            continue
        ko = r.get("occurrence_datetime")
        if ko is not None and pd.notna(ko) and abs(_ts(ko) - pd.Timedelta(hours=3) - _ts(commence)) > sharp.KICKOFF_TOL:
            continue
        home_abbr = next(a for a, n in sides.items() if n == home_full)
        return r["event_ticker"], home_abbr
    return None, None


def _best_edge(rows: pd.DataFrame, market: str, home_abbr: str, line: float, dists: dict) -> float | None:
    best = None
    for r in rows.dropna(subset=["yes_bid", "yes_ask", "floor_strike"]).to_dict("records"):
        if market == "SPREAD":
            team = r["ticker"].rsplit("-", 1)[1].rstrip("0123456789")
            L = line if team == home_abbr else -line
            p = dists["spread"].p_greater(L, float(r["floor_strike"]))
        else:
            p = dists["total"].p_greater(line, float(r["floor_strike"]))
        mult = float(r.get("fee_multiplier") or 1.0)
        for price, q in ((r["yes_ask"], p), (round(1 - r["yes_bid"], 4), 1 - p)):
            if 0 < price < 1:
                e = q - price - fee_per_contract(price, 1, mult)
                best = e if best is None or e > best else best
    return None if best is None else round(best, 4)


def measure(lines: pd.DataFrame, snaps: pd.DataFrame, games: pd.DataFrame | None = None) -> pd.DataFrame:
    ss = sharp_series(lines)
    if ss.empty or snaps is None or snaps.empty:
        return pd.DataFrame(columns=COLS)
    snaps = snaps.copy()
    snaps["_ts"] = pd.to_datetime(snaps["snapshot_ts"], utc=True)
    out = []
    for (eid, mkt), g in ss.groupby(["event_id", "market"]):
        g = g.sort_values("fetched_ts")
        meta = g.iloc[0]
        ko = _ts(meta["commence_utc"])
        ev, home_abbr = _kalshi_event(snaps, meta["home_team"], meta["away_team"], ko, mkt)
        if ev is None:
            continue
        ks = snaps[snaps["event_ticker"] == ev]
        kl = {t: implied_line(r, mkt, home_abbr) for t, r in ks.groupby("_ts")}
        times = sorted(t for t, v in kl.items() if v is not None)
        prev = None
        for _, r in g.iterrows():
            if prev is not None and abs(r["line"] - prev["line"]) >= MOVE_PTS:
                t0, t1 = _ts(prev["fetched_ts"]), _ts(r["fetched_ts"])
                before = [kl[t] for t in times if t <= t0]
                after = [t for t in times if t1 <= t < ko]
                row = dict(league="NFL", event_ticker=ev, market=mkt, prev_pull_ts=prev["fetched_ts"],
                           detected_ts=r["fetched_ts"], sharp_from=prev["line"], sharp_to=r["line"],
                           kalshi_before=round(before[-1], 2) if before else None, kalshi_at_detect=None,
                           status="no_kalshi", followed_ts=None, lag_minutes=None,
                           best_edge_after_fee=None, kickoff_utc=meta["commence_utc"])
                if after:
                    target = prev["line"] + FOLLOW_FRAC * (r["line"] - prev["line"])
                    up = r["line"] > prev["line"]
                    there = lambda v: v >= target if up else v <= target
                    row["kalshi_at_detect"] = round(kl[after[0]], 2)
                    if games is not None and len(games):
                        dists = sharp.build_dists(games, after[0])
                        row["best_edge_after_fee"] = _best_edge(ks[ks["_ts"] == after[0]], mkt, home_abbr,
                                                                r["line"], dists)
                    hit = next((t for t in after if there(kl[t])), None)
                    if hit == after[0]:
                        row.update(status="already_moved", followed_ts=hit.strftime("%Y-%m-%dT%H:%M:%SZ"), lag_minutes=0.0)
                    elif hit is not None:
                        row.update(status="followed", followed_ts=hit.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                   lag_minutes=round((hit - t1).total_seconds() / 60, 1))
                    else:
                        row["status"] = "not_followed"
                out.append(row)
            prev = r
    return pd.DataFrame(out, columns=COLS)


def summarize(m: pd.DataFrame) -> dict:
    ev = m[m["status"].isin(["already_moved", "followed", "not_followed"])] if len(m) else m
    s = {"moves": int(len(m)), "evaluated": int(len(ev)),
         "status": m["status"].value_counts().to_dict() if len(m) else {},
         "rules": {"move_pts": MOVE_PTS, "follow_frac": FOLLOW_FRAC, "act_minutes": ACT_MINUTES}}
    if len(ev) < MIN_MOVES:
        s["verdict"] = f"not enough data ({len(ev)} evaluated moves < {MIN_MOVES})"
        return s
    lagged = ev[(ev["status"] == "not_followed") | (ev["lag_minutes"].fillna(0) >= ACT_MINUTES)]
    edges = ev["best_edge_after_fee"].dropna()
    s.update(share_already_moved=round(float((ev["status"] == "already_moved").mean()), 3),
             median_lag_minutes_followed=None if not (ev["status"] == "followed").any() else
             float(ev.loc[ev["status"] == "followed", "lag_minutes"].median()),
             share_lagged_actionable=round(len(lagged) / len(ev), 3),
             mean_best_edge_after_fee_c=None if edges.empty else round(float(edges.mean() * 100), 2))
    actionable = s["share_lagged_actionable"] >= 0.25 and (not edges.empty and edges.mean() > 0)
    s["verdict"] = ("possible timing edge: Kalshi often lags >= %d min and the rung edge after fees is positive"
                    % ACT_MINUTES) if actionable else "no timing edge seen: Kalshi mostly keeps up, or the lag isn't worth fees"
    return s


def run() -> dict:
    from . import nfl_results
    m = measure(odds.load_lines(), load_snapshots(), nfl_results.load())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT, index=False)
    s = summarize(m)
    SUMMARY.write_text(json.dumps(s, indent=2, default=str))
    print(f"lag: {s['moves']} sharp moves, {s['evaluated']} evaluated | {s['verdict']}")
    return s


if __name__ == "__main__":
    run()
