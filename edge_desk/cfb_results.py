"""Public CFB results + schedule for the rating model (backlog #7).

Source: sportsdataverse/cfbfastR-data `schedules/csv/cfb_schedules_<season>.csv` (public, CFBD-
derived, updated during the season; start times already UTC). Fetched inside GitHub Actions.
Output: data/cfb/games.csv in the same columns as data/nfl/games.csv, plus home/away division.
No closing lines in this source; `load()` attaches CFBD closing lines from data/cfb/lines.csv
(cfb_lines.py, backlog #13) when present, so the backtest can benchmark against the market.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd

from . import nfl_results

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "cfb" / "games.csv"
SOURCE = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/schedules/csv/cfb_schedules_{season}.csv"
FIRST_SEASON = 2014
# Kickoff "TBD" rows carry a midnight-ET placeholder. Push them to 8pm ET so a game is never
# treated as final (kickoff + FINAL_LAG) before it could actually have been played.
TBD_SHIFT = pd.Timedelta(hours=20)

COLS = ["game_id", "season", "game_type", "week", "kickoff_utc", "away_team", "home_team",
        "away_score", "home_score", "neutral", "spread_line", "total_line",
        "home_division", "away_division", "fetched_ts"]


def normalize(raw: pd.DataFrame, fetched_ts: str) -> pd.DataFrame:
    g = raw.copy()
    ko = pd.to_datetime(g["start_date"], utc=True, errors="coerce")
    tbd = g.get("start_time_tbd", False)
    tbd = tbd.astype(str).str.upper().eq("TRUE") if hasattr(tbd, "astype") else False
    ko = ko.where(~tbd, ko + TBD_SHIFT)
    g["kickoff_utc"] = ko.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    g["game_type"] = g["season_type"].astype(str).str.upper().replace({"REGULAR": "REG", "POSTSEASON": "POST"})
    g["neutral"] = g["neutral_site"].astype(str).str.upper().eq("TRUE").astype(int)
    g = g.rename(columns={"home_points": "home_score", "away_points": "away_score"})
    done = g["completed"].astype(str).str.upper().eq("TRUE")
    g.loc[~done, ["home_score", "away_score"]] = None      # never trust a score on an open game
    g["spread_line"], g["total_line"] = float("nan"), float("nan")
    g["fetched_ts"] = fetched_ts
    return g.dropna(subset=["kickoff_utc"])[COLS].sort_values("kickoff_utc").reset_index(drop=True)


def load(path: Path = OUT, with_lines: bool | None = None) -> pd.DataFrame:
    """CFB games; the default file also gets CFBD closing lines attached (backlog #13), used only
    as a market benchmark (backtest proxy, SHARP distribution), never as a model input."""
    g = nfl_results.load(path, COLS)
    if with_lines if with_lines is not None else path == OUT:
        from . import cfb_lines
        g = cfb_lines.attach(g)
    return g


def age_hours(path: Path = OUT, now: datetime | None = None) -> float:
    g = load(path)
    if g.empty:
        return float("inf")
    now = now or datetime.now(timezone.utc)
    return (pd.Timestamp(now) - pd.Timestamp(g["fetched_ts"].max())).total_seconds() / 3600


def run(min_age_hours: float = 0) -> pd.DataFrame | None:
    if age_hours() < min_age_hours:
        print(f"cfb games fresh ({age_hours():.1f}h old), skip fetch")
        return None
    import requests
    now_dt = datetime.now(timezone.utc)
    now = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    last = now_dt.year if now_dt.month >= 8 else now_dt.year - 1
    parts = []
    for season in range(FIRST_SEASON, last + 1):
        r = requests.get(SOURCE.format(season=season), timeout=60)
        r.raise_for_status()
        parts.append(normalize(pd.read_csv(StringIO(r.text)), now))
    g = pd.concat(parts, ignore_index=True).drop_duplicates("game_id", keep="last")
    changed = nfl_results.save(g, OUT)
    print(f"cfb games {len(g)} ({g['home_score'].notna().sum()} final) fetched {now} changed={changed}")
    return g


if __name__ == "__main__":
    import sys
    run(float(sys.argv[1]) if len(sys.argv) > 1 else 0)
