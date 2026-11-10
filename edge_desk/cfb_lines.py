"""Historical CFB closing lines from CollegeFootballData (backlog #13). Read-only.

Secret CFBD_API_KEY (free tier: 1,000 calls/month). Runs inside GitHub Actions only; if the key
is missing it prints a note and exits 0 (workflow stays green).
Calls: the full history (FIRST_SEASON..now, regular + postseason = 2 calls/season) only when
data/cfb/lines.csv is missing; after that only the current season, at most every REFRESH_HOURS
(~2 calls x 2/day ~ 120/month).

CFBD `/lines` gives, per game, each provider's `spread` (HOME perspective: -7 = home favoured by 7)
and `overUnder`. Closing line per game = the `consensus` provider if present, else the median of
the providers. Stored as home-margin `spread_line` (= -spread, same sign as nflverse) and
`total_line`, keyed by CFBD game id (== ESPN id == cfbfastR `game_id`), with `fetched_ts`.
These are used ONLY as a market benchmark (model backtest proxy, SHARP rung distribution),
never as a model input.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "cfb" / "lines.csv"
API = "https://api.collegefootballdata.com"
FIRST_SEASON = 2014
REFRESH_HOURS = 12
COLS = ["game_id", "season", "week", "season_type", "kickoff_utc", "home_team", "away_team",
        "spread_line", "total_line", "providers", "source", "fetched_ts"]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def parse(payload: list[dict], fetched_ts: str) -> pd.DataFrame:
    rows = []
    for g in payload or []:
        ls = [l for l in (g.get("lines") or []) if l]
        sp = {l.get("provider"): _num(l.get("spread")) for l in ls}
        ou = {l.get("provider"): _num(l.get("overUnder", l.get("over_under"))) for l in ls}
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
        ko = g.get("startDate", g.get("start_date"))
        rows.append(dict(game_id=g.get("id"), season=g.get("season"), week=g.get("week"),
                         season_type=g.get("seasonType", g.get("season_type")),
                         kickoff_utc=pd.Timestamp(ko).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ") if ko else None,
                         home_team=g.get("homeTeam", g.get("home_team")), away_team=g.get("awayTeam", g.get("away_team")),
                         spread_line=-s if s == s else s, total_line=t,
                         providers=",".join(sorted(k for k in sp if k)), source=src, fetched_ts=fetched_ts))
    return pd.DataFrame(rows, columns=COLS)


def load(path: Path | None = None) -> pd.DataFrame:
    path = path or OUT
    if not path.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(path)


def attach(games: pd.DataFrame, lines: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fill CFB games' spread_line / total_line from CFBD closing lines (by game id)."""
    lines = load() if lines is None else lines
    if games is None or games.empty or lines.empty:
        return games
    l = lines.drop_duplicates("game_id", keep="last").set_index("game_id")
    g = games.copy()
    ids = pd.to_numeric(g["game_id"], errors="coerce")
    g["spread_line"] = ids.map(l["spread_line"]).astype(float)
    g["total_line"] = ids.map(l["total_line"]).astype(float)
    return g


def _due(now: datetime) -> list[int] | None:
    last = now.year if now.month >= 8 else now.year - 1
    old = load()
    if old.empty:
        return list(range(FIRST_SEASON, last + 1))
    age = (pd.Timestamp(now) - pd.to_datetime(old["fetched_ts"], utc=True).max()).total_seconds() / 3600
    return [last] if age >= REFRESH_HOURS else None


def run(now: datetime | None = None) -> pd.DataFrame | None:
    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        print("cfb lines: CFBD_API_KEY not set, skipping (no CFB market benchmark this run)")
        return None
    now = now or datetime.now(timezone.utc)
    seasons = _due(now)
    if not seasons:
        print("cfb lines fresh, skip fetch")
        return None
    import requests
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    parts = []
    for yr in seasons:
        for st in ("regular", "postseason"):
            try:
                r = requests.get(f"{API}/lines", params={"year": yr, "seasonType": st},
                                 headers={"Authorization": f"Bearer {key}"}, timeout=60)
                r.raise_for_status()
                parts.append(parse(r.json(), ts))
            except Exception as e:                     # noqa: BLE001 - never break the workflow
                print(f"cfb lines {yr} {st} failed: {type(e).__name__}")
    if not parts:
        return None
    new = pd.concat(parts, ignore_index=True)
    allp = pd.concat([load(), new], ignore_index=True).drop_duplicates("game_id", keep="last")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    allp.sort_values(["season", "kickoff_utc"]).to_csv(OUT, index=False)
    print(f"cfb lines: {len(new)} games fetched for {seasons[0]}-{seasons[-1]}, {len(allp)} stored")
    return allp


if __name__ == "__main__":
    run()
