"""Join Kalshi events to ESPN games and tag every market pregame / live / unknown.

Phase rules (conservative: anything uncertain is `unknown`, and the scanner skips it):
  pregame - matched to an ESPN game, ESPN pull is fresh (<= ESPN_STALE_MINUTES before the
            snapshot), ESPN said `pre` at that pull, and kickoff is >= PREGAME_MIN_LEAD_MINUTES
            after the snapshot (time for Emerson to place by hand).
  live    - matched, and kickoff has passed or ESPN already showed in-progress/final.
  unknown - no confident match, no ESPN data, ESPN data too old, or kickoff inside the lead window.
Only ESPN rows fetched at or before the snapshot are ever used (look-ahead guard).
"""
from __future__ import annotations

import re
from datetime import date, datetime
from difflib import SequenceMatcher
from zoneinfo import ZoneInfo

import pandas as pd

from . import config

CT = ZoneInfo(config.TIMEZONE)
ET = ZoneInfo("America/New_York")
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
MATCH_MIN = 0.85

# Kalshi short names that don't normalise onto an ESPN name on their own
ALIASES = {
    "uconn": "connecticut", "umass": "massachusetts", "ole miss": "mississippi",
    "pitt": "pittsburgh", "usc": "southern california", "lsu": "louisiana state",
    "ucf": "central florida", "smu": "southern methodist", "byu": "brigham young",
    "tcu": "texas christian", "utep": "texas el paso", "utsa": "texas san antonio",
    "unlv": "nevada las vegas", "miami fl": "miami", "miami oh": "miami oh",
    "app state": "appalachian state", "iu indy": "iu indianapolis", "at albany": "ualbany",
}


def norm(name: str | None) -> str:
    if not name:
        return ""
    s = name.lower().replace("&", " and ")
    s = re.sub(r"[.'()’]", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = ["state" if t in ("st", "saint") else t for t in s.split()]
    toks = [t for t in toks if t not in ("university", "the", "of")]
    s = " ".join(toks)
    return ALIASES.get(s, s)


def _game_key(event_ticker: str) -> tuple[str | None, date | None]:
    m = re.match(r"^[A-Z0-9]+-(\d{2})([A-Z]{3})(\d{2})([A-Z0-9]+)$", event_ticker or "")
    if not m or m.group(2) not in MONTHS:
        return None, None
    yy, mon, dd, teams = m.groups()
    return f"{yy}{mon}{dd}{teams}", date(2000 + int(yy), MONTHS[mon], int(dd))


def _split(s: str | None) -> list[str]:
    s = re.sub(r"\s*\(.*?\)\s*$", "", s or "")       # "BUF vs MIN (Nov 9)" -> "BUF vs MIN"
    s = re.sub(r":[^:]*$", "", s)                     # "A vs B: Spread" -> "A vs B"
    parts = re.split(r"\s+vs\.?\s+", s.strip())       # "vs" first: "University at Albany"
    if len(parts) != 2:
        parts = re.split(r"\s+(?:at|@)\s+", s.strip())
    return [p.strip() for p in parts] if len(parts) == 2 else []


def kalshi_games(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (league, game_key) with both teams' abbreviation + name."""
    rows = {}
    for r in df.drop_duplicates("event_ticker").to_dict("records"):
        key, d = _game_key(r["event_ticker"])
        if not key or (r["league"], key) in rows:
            continue
        names, abbrs = _split(r.get("event_title")), _split(r.get("event_sub"))
        if len(names) != 2:
            continue
        abbrs = abbrs if len(abbrs) == 2 else [None, None]
        rows[(r["league"], key)] = {"league": r["league"], "game_key": key, "game_date": d,
                                    "t1_name": names[0], "t1_abbr": abbrs[0],
                                    "t2_name": names[1], "t2_abbr": abbrs[1]}
    return pd.DataFrame(rows.values())


def team_score(name: str, abbr: str | None, g: dict, side: str) -> float:
    if abbr and g.get(f"{side}_abbr") and abbr.upper() == str(g[f"{side}_abbr"]).upper():
        return 1.0
    k = norm(name)
    if not k:
        return 0.0
    cands = {norm(g.get(f"{side}_{f}")) for f in ("location", "short", "display")} - {""}
    if k in cands:
        return 1.0
    best = 0.0
    kt = set(k.split())
    for c in cands:
        ct = set(c.split())
        if kt and kt <= ct and c.startswith(k):
            best = max(best, 0.9)       # "kent state" vs "kent state golden flashes"
        best = max(best, SequenceMatcher(None, k, c).ratio())
    return best


def match(kg: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Kalshi game -> ESPN game. Requires both teams to match and the ET date within a day.
    Ambiguous (two ESPN games score the same) -> no match."""
    out = []
    if kg.empty or games.empty:
        return pd.DataFrame(columns=["league", "game_key", "espn_id", "match_score"])
    g = games.copy()
    g["et_date"] = pd.to_datetime(g["start_utc"], utc=True).dt.tz_convert(ET).dt.date
    by_league = {lg: x.to_dict("records") for lg, x in g.groupby("league")}
    for k in kg.to_dict("records"):
        scored = []
        for e in by_league.get(k["league"], []):
            dd = abs((e["et_date"] - k["game_date"]).days)
            if dd > 1:
                continue
            s = max(min(team_score(k["t1_name"], k["t1_abbr"], e, "home"),
                        team_score(k["t2_name"], k["t2_abbr"], e, "away")),
                    min(team_score(k["t1_name"], k["t1_abbr"], e, "away"),
                        team_score(k["t2_name"], k["t2_abbr"], e, "home")))
            if s >= MATCH_MIN:
                scored.append((s, -dd, e["espn_id"]))
        if not scored:
            continue
        scored.sort(reverse=True)
        if len(scored) > 1 and scored[0][:2] == scored[1][:2]:
            continue
        out.append({"league": k["league"], "game_key": k["game_key"],
                    "espn_id": scored[0][2], "match_score": round(scored[0][0], 3)})
    return pd.DataFrame(out, columns=["league", "game_key", "espn_id", "match_score"])


def to_ct(ts_utc: str | None) -> str | None:
    """'2026-11-15T18:00:00Z' -> 'Sun Nov 15, 12:00 PM CT'."""
    if not ts_utc or pd.isna(ts_utc):
        return None
    t = pd.Timestamp(ts_utc).tz_convert(CT)
    return t.strftime("%a %b %-d, %-I:%M %p CT")


def tag(df: pd.DataFrame, games: pd.DataFrame, snapshot_ts: str | None) -> pd.DataFrame:
    """Add espn_id, kickoff_utc, kickoff_ct, phase to every Kalshi market row."""
    df = df.copy()
    for c in ("espn_id", "kickoff_utc", "kickoff_ct"):
        df[c] = None
    df["phase"] = "unknown"
    if df.empty or not snapshot_ts:
        return df
    snap = pd.Timestamp(snapshot_ts)
    if not games.empty:
        fetched = pd.to_datetime(games["fetched_ts"], utc=True)
        if (fetched > snap).any():
            raise ValueError("look-ahead: ESPN rows fetched after the Kalshi snapshot")
        if (snap - fetched.max()).total_seconds() > config.ESPN_STALE_MINUTES * 60:
            games = games.iloc[0:0]
    m = match(kalshi_games(df), games)
    if m.empty:
        return df
    g = games.set_index(["league", "espn_id"])
    keys = df["event_ticker"].map(lambda t: _game_key(t)[0])
    lookup = {(r["league"], r["game_key"]): r["espn_id"] for r in m.to_dict("records")}
    espn_ids = [lookup.get((lg, k)) for lg, k in zip(df["league"], keys)]
    df["espn_id"] = espn_ids
    lead = pd.Timedelta(minutes=config.PREGAME_MIN_LEAD_MINUTES)
    phases, kos = [], []
    for lg, eid in zip(df["league"], espn_ids):
        if eid is None:
            phases.append("unknown"), kos.append(None)
            continue
        e = g.loc[(lg, eid)]
        ko = e["start_utc"]
        kos.append(ko)
        kt = pd.Timestamp(ko) if ko else None
        if e["state"] in ("in", "post") or (kt is not None and snap >= kt):
            phases.append("live")
        elif e["state"] == "pre" and kt is not None and kt - snap >= lead:
            phases.append("pregame")
        else:
            phases.append("unknown")
    df["kickoff_utc"] = kos
    df["kickoff_ct"] = [to_ct(k) for k in kos]
    df["phase"] = phases
    return df


# --- Kalshi-only kickoffs (default since 2026-11-10) ---------------------------------------
# Kalshi's `occurrence_datetime` on every NFL/CFB game market is kickoff + 3h. Verified on the
# 2026-11-10 14:07Z snapshot against ESPN: NFL 54/54 events exactly +3.0h (std 0). Using the
# venue's own clock removes the ESPN name-matching step and its coverage gaps.
KALSHI_OCCURRENCE_OFFSET = pd.Timedelta(hours=3)


def tag_kalshi(df: pd.DataFrame, snapshot_ts: str | None) -> pd.DataFrame:
    """Add kickoff_utc, kickoff_ct, phase from Kalshi's own occurrence_datetime.

    pregame - kickoff >= PREGAME_MIN_LEAD_MINUTES after the snapshot
    live    - snapshot at/after kickoff
    unknown - no occurrence time, or kickoff inside the lead window
    """
    df = df.copy()
    df["kickoff_utc"], df["kickoff_ct"], df["phase"] = None, None, "unknown"
    if df.empty or not snapshot_ts or "occurrence_datetime" not in df:
        return df
    snap = pd.Timestamp(snapshot_ts)
    occ = pd.to_datetime(df["occurrence_datetime"], utc=True, errors="coerce")
    ko = occ - KALSHI_OCCURRENCE_OFFSET
    lead = pd.Timedelta(minutes=config.PREGAME_MIN_LEAD_MINUTES)
    phase = pd.Series("unknown", index=df.index)
    phase[ko.notna() & (snap >= ko)] = "live"
    phase[ko.notna() & (ko - snap >= lead)] = "pregame"
    df["phase"] = phase
    df["kickoff_utc"] = ko.dt.strftime("%Y-%m-%dT%H:%M:%SZ").where(ko.notna(), None)
    df["kickoff_ct"] = [to_ct(k) for k in df["kickoff_utc"]]
    return df
