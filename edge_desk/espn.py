"""ESPN public scoreboards -> kickoff times, game state and scores.

Runs inside GitHub Actions (Claude's cloud workspace can't reach ESPN). Read-only.
Output: data/espn/<YYYY-MM-DD>/<HHMMSS>Z.parquet (UTC), one row per game, stamped with
`fetched_ts` so later code can refuse anything fetched after the Kalshi snapshot it's joined to.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
ESPN_DIR = ROOT / "data" / "espn"

BASE = "https://site.api.espn.com/apis/site/v2/sports"
# league -> list of (path, extra params). CFB pulls FBS (80) and FCS (81): Kalshi lists both.
FEEDS = {
    "NFL": [("football/nfl", {})],
    "CFB": [("football/college-football", {"groups": "80"}),
            ("football/college-football", {"groups": "81"})],
    "CBB": [("basketball/mens-college-basketball", {"groups": "50"})],
}
DAYS_BACK, DAYS_AHEAD = 1, 12     # Kalshi lists games ~11 days out
ET = ZoneInfo("America/New_York")  # ESPN's `dates` param is the US-Eastern calendar day

session = requests.Session()
session.headers["User-Agent"] = "edge-desk/0.1 (read-only research)"

COLUMNS = ["fetched_ts", "league", "espn_id", "start_utc", "state", "completed", "status_detail",
           "home_abbr", "home_location", "home_short", "home_display", "home_score",
           "away_abbr", "away_location", "away_short", "away_display", "away_score"]


def _get(path: str, params: dict, tries: int = 3) -> dict:
    for i in range(tries):
        try:
            r = session.get(f"{BASE}/{path}/scoreboard", params=params, timeout=20)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(str(r.status_code))
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    return {}


def _score(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_scoreboard(league: str, payload: dict, fetched_ts: str) -> list[dict]:
    rows = []
    for ev in payload.get("events", []) or []:
        comp = (ev.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in comp.get("competitors", []) or []}
        if "home" not in sides or "away" not in sides:
            continue
        st = (ev.get("status") or comp.get("status") or {}).get("type", {}) or {}
        row = {
            "fetched_ts": fetched_ts, "league": league, "espn_id": str(ev.get("id")),
            "start_utc": _iso(ev.get("date") or comp.get("date")),
            "state": st.get("state"), "completed": bool(st.get("completed")),
            "status_detail": st.get("shortDetail") or st.get("detail"),
        }
        for side in ("home", "away"):
            c = sides[side]
            t = c.get("team", {}) or {}
            row[f"{side}_abbr"] = t.get("abbreviation")
            row[f"{side}_location"] = t.get("location")
            row[f"{side}_short"] = t.get("shortDisplayName")
            row[f"{side}_display"] = t.get("displayName")
            row[f"{side}_score"] = _score(c.get("score"))
        rows.append(row)
    return rows


def _iso(s: str | None) -> str | None:
    """ESPN gives '2026-11-15T18:00Z'; normalise to '2026-11-15T18:00:00Z'."""
    if not s:
        return None
    t = pd.Timestamp(s)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch(now: datetime | None = None) -> pd.DataFrame:
    now = now or datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    today = now.astimezone(ET).date()
    days = [today + timedelta(days=d) for d in range(-DAYS_BACK, DAYS_AHEAD + 1)]
    rows, errors = [], []
    for league, feeds in FEEDS.items():
        for path, extra in feeds:
            for d in days:
                params = {"dates": d.strftime("%Y%m%d"), "limit": "1000", **extra}
                try:
                    rows += parse_scoreboard(league, _get(path, params), ts)
                except Exception as e:
                    errors.append(f"{league} {d}: {e}")
                time.sleep(0.05)
    df = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates(["league", "espn_id"])
    out = ESPN_DIR / now.strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_parquet(out / f"{now.strftime('%H%M%S')}Z.parquet", index=False)
    log = ROOT / "data" / "espn_log.csv"
    new = not log.exists()
    with log.open("a") as fh:
        if new:
            fh.write("fetched_ts,rows,errors\n")
        fh.write(f"{ts},{len(df)},\"{' ; '.join(errors[:20])}\"\n")
    print(f"{ts}: {len(df)} ESPN games ({df.groupby('league').size().to_dict() if len(df) else {}}), "
          f"{len(errors)} errors")
    for e in errors[:20]:
        print("  ERROR", e, file=sys.stderr)
    return df


def load_asof(snapshot_ts: str | None, espn_dir: Path = ESPN_DIR) -> pd.DataFrame:
    """Latest ESPN pull fetched AT OR BEFORE `snapshot_ts` (no look-ahead). Empty if none."""
    if not snapshot_ts:
        return pd.DataFrame(columns=COLUMNS)
    cutoff = pd.Timestamp(snapshot_ts)
    best = None
    for f in sorted(espn_dir.glob("*/*.parquet")):
        t = pd.Timestamp(f"{f.parent.name}T{f.stem[:2]}:{f.stem[2:4]}:{f.stem[4:6]}Z")
        if t <= cutoff:
            best = f
    if best is None:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_parquet(best)
    return df[pd.to_datetime(df["fetched_ts"], utc=True) <= cutoff]


if __name__ == "__main__":
    fetch()
