"""Sharp-line fair value from sportsbook odds (backlog #10). Read-only.

Source: The Odds API v4 (secret ODDS_API_KEY), NFL + CFB, markets h2h/spreads/totals.
Runs inside GitHub Actions only. If the key is missing the step prints a note and exits 0.

Budget (plan-aware since #30: the quota is read from the x-requests-remaining + x-requests-used
headers, default 500 = free tier; see "plan-aware pacing" below for bigger plans): one /odds call costs len(markets) credits when all
requested bookmakers fit in one "region" (<= 10 books), so a pull of one sport = 3 credits.
`plan_pulls` decides which sports to pull this run:
  * never let the remaining credits drop below ODDS_RESERVE;
  * spend at most a daily allowance = (remaining - reserve) / days left in the month, times
    GAMEDAY_MULT when that sport has a game in the next 36h (self-correcting: overspending
    early shrinks later allowances);
  * at least GAMEDAY_INTERVAL_H between pulls of a sport with games inside 36h, else IDLE_INTERVAL_H.
Game-day detection uses the free /events endpoint (costs no credits).

Storage (everything cached; nothing deleted):
  data/odds/raw/<YYYY-MM-DD>/<league>_<HHMMZ>.json   raw API response
  data/odds/lines.csv     one row per book x market x outcome, with per-book de-vigged prob
  data/odds/state.json    last pull per league, credits remaining/used as last reported

Fair value (`fair_lines`): per event + market, Pinnacle's de-vigged line if present, else the
mean de-vigged probability of the other sharp books quoting the modal point. Guards:
  look-ahead - only rows fetched at/before `as_of` are used.
  stale      - rows older than ODDS_STALE_HOURS before `as_of` are ignored.
  live       - quotes fetched at/after the game's commence time are dropped (in-play odds).
"""
from __future__ import annotations

import calendar
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ODDS_DIR = ROOT / "data" / "odds"
LINES = ODDS_DIR / "lines.csv"
STATE = ODDS_DIR / "state.json"

API = "https://api.the-odds-api.com/v4"
SPORTS = {"NFL": "americanfootball_nfl", "CFB": "americanfootball_ncaaf", "CBB": "basketball_ncaab"}
MARKETS = ("h2h", "spreads", "totals")
SHARP_BOOKS = ("pinnacle", "lowvig", "betonlineag")          # in preference order
OTHER_BOOKS = ("draftkings", "fanduel")                      # stored, never used for fair value
BOOKMAKERS = SHARP_BOOKS + OTHER_BOOKS                      # <= 10 books -> 1 region of cost
COST_PER_PULL = len(MARKETS)

MONTHLY_CREDITS = 500
ODDS_RESERVE = 25            # never spend below this (manual checks, mid-month surprises)
GAMEDAY_MULT = 2.0          # allowance multiplier when a kickoff is inside GAMEDAY_HOT_H
GAMEDAY_HOT_H = 12
GAMEDAY_WINDOW_H = 36       # kickoff inside this -> short pull interval
GAMEDAY_INTERVAL_H = 4
IDLE_INTERVAL_H = 24
ODDS_STALE_HOURS = 6         # fair value older than this -> unusable

# CBB (#22): football always comes first. CBB may only spend credits above a football floor
# (reserve + FOOTBALL_DAILY_CREDITS for every day left in the month), at most CBB_PULLS_PER_DAY
# pulls per Central day (one before the evening slate, one near the main tips), only when a tip
# is within CBB_PULL_LEAD_H, and at least CBB_MIN_GAP_H apart. CBB spend never counts against
# football's daily allowance.
FOOTBALL = ("NFL", "CFB")
FOOTBALL_DAILY_CREDITS = 15  # ~470/month: what the football plan needs per day
CBB_PULLS_PER_DAY = 2
CBB_PULL_LEAD_H = 4
CBB_MIN_GAP_H = 2

# Confirm pulls (#28): SHARP holds back any play whose sharp line is >90 min old (60 near kickoff),
# but game-day pulls are 4h apart, so most candidates were never checked against a fresh line.
# When the latest scan (<= CONFIRM_MAX_SCAN_AGE_MIN old) has football plays held back ONLY for a
# stale line, that league may be re-pulled after CONFIRM_MIN_GAP_H instead of the usual interval.
# The reserve and the daily allowance still apply, so the monthly plan can't be overspent; the
# credits just move to the moments when a fresh line could turn into a play. Never for CBB.
CONFIRM_MIN_GAP_H = 1.0
CONFIRM_MAX_SCAN_AGE_MIN = 45
LATEST_SCAN = ROOT / "data" / "flags" / "latest.json"

LINE_COLS = ["fetched_ts", "league", "event_id", "commence_utc", "home_team", "away_team",
             "book", "book_update_utc", "market", "outcome", "point", "price", "implied", "fair"]


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")


def _iso(t) -> str:
    return _ts(t).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------- de-vig ----------
def implied(price_american: float) -> float:
    p = float(price_american)
    return 100 / (p + 100) if p > 0 else -p / (-p + 100)


def devig(probs: list[float]) -> list[float]:
    """Multiplicative (proportional) vig removal: scale implied probs to sum to 1."""
    s = sum(probs)
    if s <= 0 or any(p <= 0 for p in probs):
        return [float("nan")] * len(probs)
    return [p / s for p in probs]


def parse(payload: list[dict], league: str, fetched_ts: str) -> pd.DataFrame:
    """Odds API /odds response -> tidy rows with per-book de-vigged `fair` probability.
    Quotes for games that already started at fetch time are dropped (in-play)."""
    rows, fetched = [], _ts(fetched_ts)
    for ev in payload or []:
        commence = ev.get("commence_time")
        if not commence or _ts(commence) <= fetched:
            continue
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                outs = [o for o in mk.get("outcomes", []) if o.get("price") is not None]
                if len(outs) != 2:                      # 2-way only (no 3-way soccer-style h2h)
                    continue
                imp = [implied(o["price"]) for o in outs]
                for o, i, f in zip(outs, imp, devig(imp)):
                    rows.append(dict(
                        fetched_ts=fetched_ts, league=league, event_id=ev.get("id"),
                        commence_utc=_iso(commence), home_team=ev.get("home_team"),
                        away_team=ev.get("away_team"), book=bk.get("key"),
                        book_update_utc=mk.get("last_update") or bk.get("last_update"),
                        market=mk.get("key"), outcome=o.get("name"), point=o.get("point"),
                        price=o["price"], implied=round(i, 6), fair=round(f, 6)))
    return pd.DataFrame(rows, columns=LINE_COLS)


# ---------- fair value ----------
def usable(lines: pd.DataFrame, as_of) -> pd.DataFrame:
    """Rows known at `as_of` (no look-ahead), not stale, pregame at fetch."""
    if lines is None or lines.empty:
        return pd.DataFrame(columns=LINE_COLS)
    as_of = _ts(as_of)
    f = pd.to_datetime(lines["fetched_ts"], utc=True)
    c = pd.to_datetime(lines["commence_utc"], utc=True)
    ok = (f <= as_of) & (as_of - f <= pd.Timedelta(hours=ODDS_STALE_HOURS)) & (f < c)
    return lines[ok]


_FAIR_CACHE: dict = {}


def fair_lines(lines: pd.DataFrame, as_of) -> pd.DataFrame:
    """One row per event x market x outcome: fair prob + point + source, from the latest usable
    pull of each sharp book. Pinnacle wins; else mean of sharp books at the modal point.
    #44: one scan asks this 6 times (SHARP + ML x 3 leagues) with the same lines and as_of, ~2.5 s
    each, so the last answer is kept for that exact frame object (and length) and as_of."""
    c = _FAIR_CACHE
    if c.get("lines") is lines and c.get("n") == len(lines) and c.get("as_of") == str(as_of):
        return c["out"].copy()
    out = _fair_lines(lines, as_of)
    _FAIR_CACHE.clear()
    _FAIR_CACHE.update(lines=lines, n=len(lines), as_of=str(as_of), out=out.copy())
    return out


def _fair_lines(lines: pd.DataFrame, as_of) -> pd.DataFrame:
    u = usable(lines, as_of)
    u = u[u["book"].isin(SHARP_BOOKS)]
    cols = ["league", "event_id", "commence_utc", "home_team", "away_team", "market", "outcome",
            "point", "fair", "source", "n_books", "fetched_ts"]
    if u.empty:
        return pd.DataFrame(columns=cols)
    # latest pull per book/event/market
    last = u.groupby(["event_id", "market", "book"])["fetched_ts"].transform("max")
    u = u[u["fetched_ts"] == last]
    out = []
    for (eid, mkt), g in u.groupby(["event_id", "market"], sort=False):
        books = [b for b in SHARP_BOOKS if b in set(g["book"])]
        if "pinnacle" in books:
            pick, src = g[g["book"] == "pinnacle"], "pinnacle"
        else:
            # modal point of the first outcome (home/over); h2h has no point
            key = g.assign(pk=g["point"].fillna(0.0).abs())
            modal = key.groupby("pk")["book"].nunique().sort_values(ascending=False).index[0]
            pick = key[key["pk"] == modal]
            src = "consensus"
        meta = pick.iloc[0]
        agg = pick.groupby(["outcome"], sort=False).agg(
            point=("point", "first"), fair=("fair", "mean"), n_books=("book", "nunique"),
            fetched_ts=("fetched_ts", "max")).reset_index()
        tot = agg["fair"].sum()
        for _, r in agg.iterrows():
            out.append(dict(league=meta["league"], event_id=eid, commence_utc=meta["commence_utc"],
                            home_team=meta["home_team"], away_team=meta["away_team"], market=mkt,
                            outcome=r["outcome"], point=r["point"],
                            fair=round(r["fair"] / tot, 6) if tot > 0 else float("nan"),
                            source=src if src == "pinnacle" else f"consensus:{int(r['n_books'])}",
                            n_books=int(r["n_books"]), fetched_ts=r["fetched_ts"]))
    return pd.DataFrame(out, columns=cols)


# ---------- budget ----------
def days_left_in_month(now) -> int:
    now = _ts(now)
    return calendar.monthrange(now.year, now.month)[1] - now.day + 1


def spent_today(state: dict, now, leagues=None) -> int:
    """Credits spent today (UTC day). With `leagues`, only those leagues' spend (needs the
    per-league log; older days without it count everything as football)."""
    day = _ts(now).strftime("%Y-%m-%d")
    total = int(state.get("spent", {}).get(day, 0))
    if leagues is None:
        return total
    by = state.get("spent_by_league", {}).get(day)
    if by is None:
        return total if set(leagues) & set(FOOTBALL) else 0
    return int(sum(v for k, v in by.items() if k in leagues))


def _ct_day(now) -> str:
    return _ts(now).tz_convert("America/Chicago").strftime("%Y-%m-%d")


# ---------- plan-aware pacing (#30) ----------
# The real plan size comes from the API's own headers: quota = x-requests-remaining +
# x-requests-used (stored in state.json as "quota" at every pull). Unknown -> MONTHLY_CREDITS (free).
# Everything scales from it. Free tier (<= 500) is exactly the old behaviour. Bigger plans keep
# RESERVE_SHARE of the quota untouched (so a worst-case month spends <= 90%) and give football a
# FOOTBALL_SHARE floor of what is left. With >= HIGH_PLAN_CREDITS the cadence becomes:
#   kickoff within HIGH_NEAR_H        -> every HIGH_NEAR_MIN (5) min
#   in a live game window, kickoff <= GAMEDAY_HOT_H -> every HIGH_WINDOW_MIN (10) min
#   kickoff within GAMEDAY_WINDOW_H   -> hourly
#   otherwise                         -> every HIGH_IDLE_H
# CBB uses the same cadence on a big plan (no per-day cap) but still only above the football floor.
RESERVE_SHARE = 0.10
FOOTBALL_SHARE = 0.60
HIGH_PLAN_CREDITS = 20000
HIGH_NEAR_H = 1.5
HIGH_NEAR_MIN = 5
HIGH_WINDOW_MIN = 10
HIGH_GAMEDAY_H = 1.0
HIGH_IDLE_H = 6


def quota(state: dict) -> int:
    """Monthly plan size as last reported by the API; MONTHLY_CREDITS when unknown."""
    q = state.get("quota")
    if q is None and state.get("remaining") is not None and state.get("used") is not None:
        q = int(state["remaining"]) + int(state["used"])
    try:
        q = int(q)
    except (TypeError, ValueError):
        return MONTHLY_CREDITS
    return q if q > 0 else MONTHLY_CREDITS


def budget(state: dict) -> dict:
    q = quota(state)
    if q <= MONTHLY_CREDITS:
        return dict(quota=q, reserve=ODDS_RESERVE, football_daily=FOOTBALL_DAILY_CREDITS, high=False)
    reserve = max(ODDS_RESERVE, int(round(q * RESERVE_SHARE)))
    return dict(quota=q, reserve=reserve, football_daily=int(round((q - reserve) * FOOTBALL_SHARE / 30)),
                high=q >= HIGH_PLAN_CREDITS)


def check_minutes(state: dict, default: float) -> float:
    """How often the live loop should ask plan_pulls (5 min on a big plan)."""
    return min(default, HIGH_NEAR_MIN) if budget(state)["high"] else default


def _in_window(now) -> bool:
    from . import live                              # lazy: live imports odds
    return live.current_window(_ts(now).to_pydatetime()) is not None


def high_interval_h(now, kickoff, in_window: bool) -> float:
    secs = (_ts(kickoff) - _ts(now)).total_seconds()
    if secs <= HIGH_NEAR_H * 3600:
        return HIGH_NEAR_MIN / 60
    if in_window and secs <= GAMEDAY_HOT_H * 3600:
        return HIGH_WINDOW_MIN / 60
    if secs <= GAMEDAY_WINDOW_H * 3600:
        return HIGH_GAMEDAY_H
    return HIGH_IDLE_H


def football_floor(now, state: dict | None = None) -> int:
    """Credits CBB must leave untouched: the reserve plus football's need for the rest of the month."""
    b = budget(state or {})
    return b["reserve"] + b["football_daily"] * days_left_in_month(now)


def cbb_ok(state: dict, now, kickoff, left_after_football: int, in_window: bool | None = None) -> bool:
    now = _ts(now)
    if kickoff is None or _ts(kickoff) <= now:
        return False
    floor = football_floor(now, state)
    if budget(state)["high"]:
        if in_window is None:
            in_window = _in_window(now)
        last = state.get("last_pull", {}).get("CBB")
        if last and (now - _ts(last)).total_seconds() < high_interval_h(now, kickoff, in_window) * 3600:
            return False
        spare = max(left_after_football - floor, 0)
        cap = spare / days_left_in_month(now) * GAMEDAY_MULT
        if spent_today(state, now, ("CBB",)) + COST_PER_PULL > cap:
            return False
        return left_after_football - COST_PER_PULL >= floor
    if (_ts(kickoff) - now).total_seconds() > CBB_PULL_LEAD_H * 3600:
        return False
    if state.get("pulls", {}).get("CBB", {}).get(_ct_day(now), 0) >= CBB_PULLS_PER_DAY:
        return False
    last = state.get("last_pull", {}).get("CBB")
    if last and (now - _ts(last)).total_seconds() < CBB_MIN_GAP_H * 3600:
        return False
    return left_after_football - COST_PER_PULL >= floor


def remaining(state: dict, now) -> int:
    """Credits left as last reported by the API; a new month resets to the plan's quota."""
    now = _ts(now)
    seen = state.get("remaining_seen_utc")
    if state.get("remaining") is None or not seen or _ts(seen).strftime("%Y-%m") != now.strftime("%Y-%m"):
        return quota(state)
    return int(state["remaining"])


def confirm_leagues(now, path: Path | None = None) -> set[str]:
    """Football leagues whose latest scan held back plays only for a stale sharp line (#28)."""
    now = _ts(now)
    try:
        d = json.loads((path or LATEST_SCAN).read_text())
        snap = _ts(d["snapshot_ts"])
    except Exception:  # noqa: BLE001 - no scan yet -> nothing to confirm
        return set()
    age = (now - snap).total_seconds() / 60
    if age < 0 or age > CONFIRM_MAX_SCAN_AGE_MIN:
        return set()
    out = set()
    for k, v in (d.get("sharp") or {}).items():
        lg = k.split()[0]
        if lg in FOOTBALL and isinstance(v, dict) and int(v.get("stale_only") or 0) > 0:
            out.add(lg)
    return out


def plan_pulls(state: dict, now, next_kickoff: dict, near_h: float | None = None,
               near_interval_h: float | None = None, confirm=(), in_window: bool | None = None) -> list[str]:
    """Which leagues to pull now. `next_kickoff[league]` = soonest upcoming commence (or None).
    Live mode (#16) passes near_h/near_interval_h: a league whose kickoff is within near_h may be
    pulled every near_interval_h instead of GAMEDAY_INTERVAL_H. On a big plan (#30) the cadence
    comes from high_interval_h. The reserve and the daily allowance still apply, so the monthly
    plan can't be overspent."""
    now = _ts(now)
    b = budget(state)
    if b["high"] and in_window is None:
        in_window = _in_window(now)
    left = remaining(state, now) - b["reserve"]
    base = max(left, 0) / days_left_in_month(now)
    spent = spent_today(state, now, FOOTBALL)
    plan = []
    # game-day leagues first, soonest kickoff first
    def soon(lg):
        k = next_kickoff.get(lg)
        return k is not None and _ts(k) > now and (_ts(k) - now).total_seconds() <= GAMEDAY_WINDOW_H * 3600
    order = sorted((lg for lg in SPORTS if lg in FOOTBALL),
                   key=lambda lg: (not soon(lg), _ts(next_kickoff[lg]) if next_kickoff.get(lg) else now))
    for lg in order:
        k = next_kickoff.get(lg)
        if k is None or _ts(k) <= now:
            continue                                       # nothing upcoming -> nothing to price
        gd = soon(lg)
        if b["high"]:
            interval = high_interval_h(now, k, bool(in_window))
        else:
            interval = GAMEDAY_INTERVAL_H if gd else IDLE_INTERVAL_H
            if near_h is not None and near_interval_h is not None and (_ts(k) - now).total_seconds() <= near_h * 3600:
                interval = min(interval, near_interval_h)
        if lg in confirm:
            interval = min(interval, CONFIRM_MIN_GAP_H)
        last = state.get("last_pull", {}).get(lg)
        if last and (now - _ts(last)).total_seconds() < interval * 3600:
            continue
        hot = (_ts(k) - now).total_seconds() <= GAMEDAY_HOT_H * 3600
        allowance = base * (GAMEDAY_MULT if hot else 1.0)
        if left - COST_PER_PULL < 0 or spent + COST_PER_PULL > max(allowance, 0):
            continue
        plan.append(lg)
        spent += COST_PER_PULL
        left -= COST_PER_PULL
    # CBB last, only from what football doesn't need (skipped first when credits run short)
    if "CBB" in SPORTS and cbb_ok(state, now, next_kickoff.get("CBB"), left + b["reserve"], in_window):
        plan.append("CBB")
    return plan


def record_pull(state: dict, league: str, now, headers: dict) -> dict:
    now = _ts(now)
    s = dict(state)
    s.setdefault("last_pull", {})[league] = _iso(now)
    rem, used = headers.get("x-requests-remaining"), headers.get("x-requests-used")
    last_cost = headers.get("x-requests-last")
    cost = int(float(last_cost)) if last_cost is not None else COST_PER_PULL
    day = now.strftime("%Y-%m-%d")
    s.setdefault("spent", {})[day] = int(s.get("spent", {}).get(day, 0)) + cost
    s["spent"] = dict(sorted(s["spent"].items())[-40:])    # keep ~a month of history
    by = dict(s.get("spent_by_league", {}))
    d = dict(by.get(day, {}))
    d[league] = int(d.get(league, 0)) + cost
    by[day] = d
    s["spent_by_league"] = dict(sorted(by.items())[-40:])
    pulls = {k: dict(v) for k, v in s.get("pulls", {}).items()}
    lg = pulls.setdefault(league, {})
    cday = _ct_day(now)
    lg[cday] = int(lg.get(cday, 0)) + 1
    pulls[league] = dict(sorted(lg.items())[-40:])
    s["pulls"] = pulls
    if rem is not None:
        s["remaining"] = int(float(rem))
        s["remaining_seen_utc"] = _iso(now)
    if used is not None:
        s["used"] = int(float(used))
    if rem is not None and used is not None:
        s["quota"] = int(float(rem)) + int(float(used))
    return s


def load_state(path: Path | None = None) -> dict:
    try:
        return json.loads((path or STATE).read_text())
    except Exception:
        return {}


def load_lines(path: Path | None = None) -> pd.DataFrame:
    path = path or LINES
    if not path.exists():
        return pd.DataFrame(columns=LINE_COLS)
    return pd.read_csv(path)


def append_lines(df: pd.DataFrame, path: Path | None = None) -> None:
    path = path or LINES
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


# ---------- network (Actions only) ----------
def _get(url: str, params: dict):
    import requests
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json(), {k.lower(): v for k, v in r.headers.items()}


def next_kickoffs(key: str, now) -> dict:
    """Soonest upcoming commence per league via the free /events endpoint."""
    out = {}
    for lg, sport in SPORTS.items():
        try:
            evs, _ = _get(f"{API}/sports/{sport}/events", {"apiKey": key, "dateFormat": "iso"})
        except Exception as e:                       # noqa: BLE001 - never break the workflow
            print(f"odds: events {lg} failed: {type(e).__name__}")
            out[lg] = None
            continue
        ks = sorted(_ts(e["commence_time"]) for e in evs if e.get("commence_time") and _ts(e["commence_time"]) > _ts(now))
        out[lg] = _iso(ks[0]) if ks else None
    return out


def run(now=None, near_h: float | None = None, near_interval_h: float | None = None,
        confirm=None) -> list[str]:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        print("odds: ODDS_API_KEY not set, skipping (no sharp-line data this run)")
        return []
    now = _ts(now or datetime.now(timezone.utc))
    state = load_state()
    confirm = confirm_leagues(now) if confirm is None else set(confirm)
    plan = plan_pulls(state, now, next_kickoffs(key, now), near_h, near_interval_h, confirm)
    b = budget(state)
    print(f"odds: plan {b['quota']} credits/month (reserve {b['reserve']}{', fast cadence' if b['high'] else ''}), "
          f"credits left ~{remaining(state, now)}, spent today {spent_today(state, now)} (CBB {spent_today(state, now, ('CBB',))}), "
          f"confirm {sorted(confirm) or 'none'}, plan {plan or 'none'}")
    for lg in plan:
        try:
            payload, headers = _get(f"{API}/sports/{SPORTS[lg]}/odds", {
                "apiKey": key, "markets": ",".join(MARKETS), "oddsFormat": "american",
                "dateFormat": "iso", "bookmakers": ",".join(BOOKMAKERS)})
        except Exception as e:                       # noqa: BLE001
            print(f"odds: {lg} pull failed: {type(e).__name__}")
            continue
        fetched = _iso(now)
        raw = ODDS_DIR / "raw" / now.strftime("%Y-%m-%d") / f"{lg}_{now.strftime('%H%M')}Z.json"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(json.dumps(payload, separators=(",", ":")))
        lines = parse(payload, lg, fetched)
        append_lines(lines)
        state = record_pull(state, lg, now, headers)
        print(f"odds: {lg} {len(payload)} events, {len(lines)} rows, credits left {state.get('remaining')}")
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1, sort_keys=True))
    fair = fair_lines(load_lines(), now)
    if not fair.empty:
        fair.to_csv(ODDS_DIR / "fair_latest.csv", index=False)
    return plan


if __name__ == "__main__":
    run()
