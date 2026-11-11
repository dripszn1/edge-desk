"""Historical CBB (men's D1) closing lines + final scores from CollegeBasketballData (#22). Read-only.

Secret CBBD_API_KEY (falls back to CFBD_API_KEY: the two sites share accounts). Runs inside
GitHub Actions only; if no key is set it prints a note and exits 0 (workflow stays green).
UNVERIFIED until the first Actions run with the key: field names below follow the public CBBD
schema (`/lines`: gameId, season, startDate, homeTeam, awayTeam, homeScore, awayScore,
lines[{provider, spread, overUnder}]), with snake_case fallbacks.

Calls: the API caps results per call, so history is fetched in half-month date chunks
(Nov 1..Apr 15 of each season since FIRST_SEASON: ~11 calls/season, once, only while
data/cbb/lines.csv is missing); after that only the last REFRESH_DAYS, at most every REFRESH_HOURS.

Closing line per game = `consensus` provider if present, else the provider median. Stored as
home-margin `spread_line` (= -spread, same sign as nflverse) and `total_line`, plus scores,
with `fetched_ts`. Used ONLY to fit the CBB rung distribution (sharp.py), never as a signal.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .cfb_lines import _num

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "cbb" / "lines.csv"
API = "https://api.collegebasketballdata.com"
FIRST_SEASON = 2021          # season label = calendar year it ends (2021 = 2020-21)
REFRESH_HOURS = 12
REFRESH_DAYS = 10
MIN_TEAM_SCORE = 20         # #29: 0-0 = unplayed/cancelled/not final yet, 1-0/2-0 = forfeit
FINAL_AFTER_H = 3           # scores fetched sooner than this after the tip may be mid-game
COLS = ["game_id", "season", "kickoff_utc", "home_team", "away_team", "home_score", "away_score",
        "spread_line", "total_line", "providers", "source", "fetched_ts"]


def _g(d: dict, *keys):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def parse(payload: list[dict], fetched_ts: str) -> pd.DataFrame:
    rows = []
    for g in payload or []:
        ls = [l for l in (g.get("lines") or []) if l]
        sp = {l.get("provider"): _num(l.get("spread")) for l in ls}
        ou = {l.get("provider"): _num(_g(l, "overUnder", "over_under")) for l in ls}
        sp = {k: v for k, v in sp.items() if v == v}
        ou = {k: v for k, v in ou.items() if v == v}
        if not sp and not ou:
            continue

        def pick(d):
            if not d:
                return float("nan"), ""
            if "consensus" in d:
                return d["consensus"], "consensus"
            return float(pd.Series(list(d.values())).median()), f"median:{len(d)}"
        s, src = pick(sp)
        t, _ = pick(ou)
        ko = _g(g, "startDate", "start_date")
        rows.append(dict(
            game_id=_g(g, "gameId", "game_id", "id"), season=g.get("season"),
            kickoff_utc=pd.Timestamp(ko).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ") if ko else None,
            home_team=_g(g, "homeTeam", "home_team"), away_team=_g(g, "awayTeam", "away_team"),
            home_score=_num(_g(g, "homeScore", "home_score", "homePoints")),
            away_score=_num(_g(g, "awayScore", "away_score", "awayPoints")),
            spread_line=-s if s == s else s, total_line=t,
            providers=",".join(sorted(k for k in sp if k)), source=src, fetched_ts=fetched_ts))
    return pd.DataFrame(rows, columns=COLS)


def load(path: Path | None = None) -> pd.DataFrame:
    path = path or OUT
    if not path.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(path)


def final_games(g: pd.DataFrame) -> pd.DataFrame:
    """Only games whose stored score is a real final: both teams >= MIN_TEAM_SCORE and the row was
    fetched >= FINAL_AFTER_H after the tip (when those columns exist)."""
    if g is None or g.empty:
        return g
    g = g.dropna(subset=["home_score", "away_score"])
    g = g[(g["home_score"] >= MIN_TEAM_SCORE) & (g["away_score"] >= MIN_TEAM_SCORE)]
    if "fetched_ts" in g and "kickoff_utc" in g:
        lag = pd.to_datetime(g["fetched_ts"], utc=True) - pd.to_datetime(g["kickoff_utc"], utc=True)
        g = g[lag >= pd.Timedelta(hours=FINAL_AFTER_H)]
    return g


def history(as_of=None, path: Path | None = None) -> pd.DataFrame:
    """Settled games usable at `as_of`: fetched at/before it (look-ahead guard) and tipped before it.
    Same columns as the football history frames, so sharp.build_dists / backtest work unchanged."""
    g = load(path)
    if g.empty:
        return g
    g = final_games(g)
    if as_of is not None:
        t = pd.Timestamp(as_of)
        t = t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")
        g = g[(pd.to_datetime(g["fetched_ts"], utc=True) <= t) & (pd.to_datetime(g["kickoff_utc"], utc=True) < t)]
    return g.reset_index(drop=True)


def chunks(start: datetime, end: datetime, days: int = 15) -> list[tuple[str, str]]:
    out, a = [], start
    while a < end:
        b = min(a + timedelta(days=days), end)
        out.append((a.strftime("%Y-%m-%dT%H:%M:%SZ"), b.strftime("%Y-%m-%dT%H:%M:%SZ")))
        a = b
    return out


def due(now: datetime, old: pd.DataFrame | None = None) -> list[tuple[str, str]]:
    old = load() if old is None else old
    if old.empty:
        out = []
        last = now.year + (1 if now.month >= 8 else 0)
        for s in range(FIRST_SEASON, last + 1):
            a = datetime(s - 1, 11, 1, tzinfo=timezone.utc)
            b = min(datetime(s, 4, 15, tzinfo=timezone.utc), now)
            if a < b:
                out += chunks(a, b)
        return out
    age = (pd.Timestamp(now) - pd.to_datetime(old["fetched_ts"], utc=True).max()).total_seconds() / 3600
    return chunks(now - timedelta(days=REFRESH_DAYS), now) if age >= REFRESH_HOURS else []


def run(now: datetime | None = None) -> pd.DataFrame | None:
    key = (os.environ.get("CBBD_API_KEY") or os.environ.get("CFBD_API_KEY") or "").strip()
    if not key:
        print("cbb lines: no CBBD_API_KEY/CFBD_API_KEY, skipping (CBB uses assumed SDs)")
        return None
    now = now or datetime.now(timezone.utc)
    todo = due(now)
    if not todo:
        print("cbb lines fresh, skip fetch")
        return None
    import requests
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    parts, fails = [], 0
    for a, b in todo:
        try:
            r = requests.get(f"{API}/lines", params={"startDateRange": a, "endDateRange": b},
                             headers={"Authorization": f"Bearer {key}"}, timeout=60)
            r.raise_for_status()
            p = parse(r.json(), ts)
            parts.append(p)
            if len(r.json() or []) >= 3000:
                print(f"cbb lines {a[:10]}..{b[:10]}: hit 3000 rows, chunk may be truncated")
        except Exception as e:                     # noqa: BLE001 - never break the workflow
            fails += 1
            print(f"cbb lines {a[:10]}..{b[:10]} failed: {type(e).__name__}")
            if fails >= 3 and not parts:
                break                              # key/endpoint wrong: stop burning calls
    if not parts:
        return None
    new = pd.concat(parts, ignore_index=True)
    allp = pd.concat([load(), new], ignore_index=True).drop_duplicates("game_id", keep="last")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    allp.sort_values("kickoff_utc").to_csv(OUT, index=False)
    settled = allp.dropna(subset=["home_score", "away_score", "spread_line"])
    print(f"cbb lines: {len(new)} games fetched in {len(todo)} calls, {len(allp)} stored, {len(settled)} settled with a line")
    return allp


if __name__ == "__main__":
    run()
