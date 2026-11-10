"""Public NFL results + schedule for the team-rating model (backlog #6).

Source: nflverse `games.csv` (public, free, scores since 1999, updated during the season).
Runs inside GitHub Actions only — Claude's cloud workspace is not the data path.
Output: data/nfl/games.csv, one row per game, times in UTC, plus `fetched_ts` so the model
can refuse rows fetched after a Kalshi snapshot (look-ahead) or too long before it (stale).

`spread_line` / `total_line` are nflverse's closing lines (home margin, positive = home
favoured). They are used ONLY by the backtest as a market benchmark, never as a model input.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "nfl" / "games.csv"
SOURCE = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
FIRST_SEASON = 2009

# Relocated franchises -> current nflverse code, so ratings carry across moves
RELOCATED = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
# Kalshi ticker codes -> nflverse codes
KALSHI_TO_NFLVERSE = {"JAC": "JAX", "LAR": "LA"}

COLS = ["game_id", "season", "game_type", "week", "kickoff_utc", "away_team", "home_team",
        "away_score", "home_score", "neutral", "spread_line", "total_line", "fetched_ts"]


def normalize(raw: pd.DataFrame, fetched_ts: str) -> pd.DataFrame:
    g = raw[raw["season"] >= FIRST_SEASON].copy()
    t = g["gametime"].fillna("13:00")              # nflverse times are US Eastern
    local = pd.to_datetime(g["gameday"] + " " + t, errors="coerce")
    g["kickoff_utc"] = (local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
                        .dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    for c in ("away_team", "home_team"):
        g[c] = g[c].replace(RELOCATED)
    g["neutral"] = (g["location"].astype(str).str.lower() == "neutral").astype(int)
    g["fetched_ts"] = fetched_ts
    return g.dropna(subset=["kickoff_utc"])[COLS].sort_values("kickoff_utc").reset_index(drop=True)


def _meta(path: Path) -> Path:
    return path.with_name(path.stem + "_fetched.json")


def save(g: pd.DataFrame, path: Path) -> bool:
    """Write the games CSV only if the data changed (keeps git history small); always record
    the fetch time in a sidecar JSON so freshness checks still see today's fetch."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    body = g.drop(columns=["fetched_ts"])
    changed = not path.exists() or not pd.read_csv(path).equals(pd.read_csv(StringIO(body.to_csv(index=False))))
    if changed:
        body.to_csv(path, index=False)
    _meta(path).write_text(json.dumps({"fetched_ts": str(g["fetched_ts"].max()), "rows": int(len(g))}))
    return changed


def load(path: Path = OUT, cols=COLS) -> pd.DataFrame:
    import json
    if not path.exists():
        return pd.DataFrame(columns=cols)
    g = pd.read_csv(path)
    if "fetched_ts" not in g:
        try:
            g["fetched_ts"] = json.loads(_meta(path).read_text())["fetched_ts"]
        except Exception:
            return pd.DataFrame(columns=cols)   # unknown fetch time -> treated as no data
    return g


def age_hours(path: Path = OUT, now: datetime | None = None) -> float:
    g = load(path)
    if g.empty:
        return float("inf")
    now = now or datetime.now(timezone.utc)
    return (pd.Timestamp(now) - pd.Timestamp(g["fetched_ts"].max())).total_seconds() / 3600


def run(min_age_hours: float = 0) -> pd.DataFrame | None:
    """Refetch unless the stored copy is younger than `min_age_hours` (keeps git churn down)."""
    if age_hours() < min_age_hours:
        print(f"nfl games fresh ({age_hours():.1f}h old), skip fetch")
        return None
    import requests
    r = requests.get(SOURCE, timeout=60)
    r.raise_for_status()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    g = normalize(pd.read_csv(StringIO(r.text)), now)
    changed = save(g, OUT)
    done = g["home_score"].notna().sum()
    print(f"nfl games {len(g)} ({done} final) fetched {now} changed={changed}")
    return g


if __name__ == "__main__":
    import sys
    run(float(sys.argv[1]) if len(sys.argv) > 1 else 0)
