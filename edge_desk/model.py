"""Team-rating model, NFL (backlog #6) + CFB (#7): margin + total projections -> fair prob per Kalshi rung.

Ratings: each team gets an offence and a defence rating in points; a game's expected score is
    home = mu + hfa/2 + off[home] - def[away]      away = mu - hfa/2 + off[away] - def[home]
(hfa = 0 at neutral sites). Fitted by ridge-weighted least squares on public final scores,
older games decayed with a half-life in days. Margin and total ~ Normal(proj, SD).

Guards (every one is tested):
  look-ahead - only games final (kickoff + FINAL_LAG) AND fetched at/before the snapshot are
               used; a results file fetched after the snapshot is refused outright.
  stale      - no flags if the Kalshi snapshot is stale, the results file is older than
               RESULTS_STALE_HOURS, or a game that ended > SCORE_LAG_DAYS ago still has no score.
  fees       - edge = fair - price - Kalshi taker fee at the actual paper order size.
  risk       - quarter Kelly, $10 cap, one flag per Kalshi event (best edge), and the day's
               combined stakes never exceed what is left of DAILY_LOSS_CAP.
Paper only. Never places orders.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .fees import fee_per_contract
from difflib import SequenceMatcher

from .kickoffs import _split, norm, to_ct
from .nfl_results import KALSHI_TO_NFLVERSE
from .sizing import bet_size

# Tuned by walk-forward on 2012-2017 only (see backtest.py / data/model/nfl_backtest.json)
HALF_LIFE_DAYS = 240.0
RIDGE = 10.0
SD_MARGIN = 13.5
SD_TOTAL = 13.6

CFB_HALF_LIFE_DAYS = 240.0              # tuned offline on 2016-2021 only (see backtest.LEAGUE_RUNS)
CFB_RIDGE = 0.1
CFB_SD_MARGIN = 16.9
CFB_SD_TOTAL = 17.7


@dataclass(frozen=True)
class League:
    name: str
    spread: str          # Kalshi series tickers
    total: str
    half_life_days: float
    ridge: float
    sd_margin: float
    sd_total: float
    backtest_file: str


LEAGUES = {
    "NFL": League("NFL", "KXNFLSPREAD", "KXNFLTOTAL", HALF_LIFE_DAYS, RIDGE, SD_MARGIN, SD_TOTAL,
                  "nfl_backtest.json"),
    "CFB": League("CFB", "KXNCAAFSPREAD", "KXNCAAFTOTAL", CFB_HALF_LIFE_DAYS, CFB_RIDGE,
                  CFB_SD_MARGIN, CFB_SD_TOTAL, "cfb_backtest.json"),
}

FINAL_LAG = pd.Timedelta(hours=4)       # a game counts as final this long after kickoff
SCORE_LAG_DAYS = 2                      # a final game still unscored after this -> feed is stale
RESULTS_STALE_HOURS = 36                # results file older than this -> no model flags
KALSHI_KICKOFF_TOLERANCE = pd.Timedelta(hours=36)
CFB_MATCH_MIN = 0.85                    # same bar as the old ESPN name matcher


class LookAheadError(ValueError):
    pass


def _ts(x) -> pd.Timestamp:
    return pd.Timestamp(x).tz_convert("UTC") if pd.Timestamp(x).tzinfo else pd.Timestamp(x, tz="UTC")


def training_games(games: pd.DataFrame, as_of) -> pd.DataFrame:
    """Games known and final at `as_of`: fetched at/before it and kicked off FINAL_LAG before it."""
    if games.empty:
        return games
    as_of = _ts(as_of)
    ko = pd.to_datetime(games["kickoff_utc"], utc=True)
    fetched = pd.to_datetime(games["fetched_ts"], utc=True)
    ok = (ko + FINAL_LAG <= as_of) & (fetched <= as_of) & games["home_score"].notna() \
        & games["away_score"].notna()
    return games[ok]


def results_ok(games: pd.DataFrame, as_of) -> tuple[bool, str]:
    """Is the results file usable at `as_of`? (look-ahead + freshness)."""
    if games is None or games.empty:
        return False, "no results"
    as_of = _ts(as_of)
    fetched = pd.to_datetime(games["fetched_ts"], utc=True)
    if (fetched > as_of).any():
        return False, "look-ahead: results fetched after the snapshot"
    if (as_of - fetched.max()).total_seconds() > RESULTS_STALE_HOURS * 3600:
        return False, f"results older than {RESULTS_STALE_HOURS}h"
    ko = pd.to_datetime(games["kickoff_utc"], utc=True)
    window = (ko + FINAL_LAG + pd.Timedelta(days=SCORE_LAG_DAYS) <= as_of) & \
             (ko >= as_of - pd.Timedelta(days=14))
    if (window & games["home_score"].isna()).any():
        return False, "results feed lagging: recent final game has no score"
    return True, "ok"


@dataclass
class Ratings:
    mu: float
    hfa: float
    off: dict = field(default_factory=dict)
    dfn: dict = field(default_factory=dict)
    as_of: str = ""
    n_games: int = 0
    last_game_utc: str | None = None

    def project(self, home: str, away: str, neutral: bool = False) -> tuple[float, float] | None:
        if home not in self.off or away not in self.off:
            return None
        h = 0.0 if neutral else self.hfa / 2
        hs = self.mu + h + self.off[home] - self.dfn[away]
        as_ = self.mu - h + self.off[away] - self.dfn[home]
        return hs - as_, hs + as_


def fit(games: pd.DataFrame, as_of, half_life_days: float = HALF_LIFE_DAYS,
        ridge: float = RIDGE) -> Ratings | None:
    tr = training_games(games, as_of)
    as_of = _ts(as_of)
    if len(tr) < 32:
        return None
    ko = pd.to_datetime(tr["kickoff_utc"], utc=True)
    age = (as_of - ko).dt.total_seconds().to_numpy() / 86400
    w = 0.5 ** (age / half_life_days)
    keep = w > 1e-3
    tr, w = tr[keep], w[keep]
    teams = sorted(set(tr["home_team"]) | set(tr["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(tr), len(teams)
    hi = tr["home_team"].map(idx).to_numpy()
    ai = tr["away_team"].map(idx).to_numpy()
    hf = 0.5 * (1 - tr["neutral"].fillna(0).to_numpy())
    # Two observations per game, each with 4 non-zeros [mu, hfa, off, -def]. Build the weighted
    # normal equations directly (O(games) + O(teams^3)) instead of a dense design matrix.
    one = np.ones(n)
    cols = np.concatenate([np.stack([0 * one, one, 2 + hi, 2 + k + ai], 1),
                           np.stack([0 * one, one, 2 + ai, 2 + k + hi], 1)]).astype(int)
    vals = np.concatenate([np.stack([one, hf, one, -one], 1), np.stack([one, -hf, one, -one], 1)])
    y = np.concatenate([tr["home_score"].to_numpy(float), tr["away_score"].to_numpy(float)])
    W = np.concatenate([w, w])
    m = 2 + 2 * k
    A = np.zeros((m, m))
    for i in range(4):
        for j in range(4):
            np.add.at(A, (cols[:, i], cols[:, j]), W * vals[:, i] * vals[:, j])
    rhs = np.zeros(m)
    for i in range(4):
        np.add.at(rhs, cols[:, i], W * vals[:, i] * y)
    A[2:, 2:] += ridge * np.eye(2 * k)
    b = np.linalg.solve(A, rhs)
    return Ratings(mu=float(b[0]), hfa=float(b[1]),
                   off={t: float(b[2 + i]) for t, i in idx.items()},
                   dfn={t: float(b[2 + k + i]) for t, i in idx.items()},
                   as_of=as_of.strftime("%Y-%m-%dT%H:%M:%SZ"), n_games=int(n),
                   last_game_utc=str(tr["kickoff_utc"].max()))


def p_greater(mean: float, sd: float, strike: float) -> float:
    """P(X > strike) for X ~ Normal(mean, sd)."""
    return 0.5 * math.erfc((strike - mean) / (sd * math.sqrt(2)))


# --- Kalshi join ---------------------------------------------------------------------------
def _pair(event_sub: str | None) -> tuple[str, str] | None:
    s = re.sub(r"\s*\(.*?\)\s*$", "", event_sub or "")
    parts = [p.strip() for p in re.split(r"\s+vs\.?\s+", s)]
    if len(parts) != 2 or not all(parts):
        return None
    return tuple(KALSHI_TO_NFLVERSE.get(p, p) for p in parts)


def _rung_team(ticker: str) -> str | None:
    m = re.search(r"-([A-Z]+)\d+$", ticker or "")
    return KALSHI_TO_NFLVERSE.get(m.group(1), m.group(1)) if m else None


def match_game(games: pd.DataFrame, pair: tuple[str, str], kickoff_utc) -> dict | None:
    if not pair or kickoff_utc is None or games.empty:
        return None
    ko = pd.to_datetime(games["kickoff_utc"], utc=True)
    same = ((games["home_team"] == pair[0]) & (games["away_team"] == pair[1])) | \
           ((games["home_team"] == pair[1]) & (games["away_team"] == pair[0]))
    near = (ko - _ts(kickoff_utc)).abs() <= KALSHI_KICKOFF_TOLERANCE
    m = games[same & near]
    return m.iloc[0].to_dict() if len(m) == 1 else None


def _abbr(ticker: str) -> str | None:
    m = re.search(r"-([A-Z&]+)\d+$", ticker or "")
    return m.group(1) if m else None


# Kalshi CFB names that don't normalise onto the CFBD schedule name (applied after norm)
CFB_ALIASES = {"louisiana monroe": "ul monroe"}


def _cfb_norm(name: str) -> str:
    n = norm(name)
    return CFB_ALIASES.get(n, n)


def _sim(a: str, b: str) -> float:
    a, b = _cfb_norm(a), _cfb_norm(b)
    return 1.0 if a == b else SequenceMatcher(None, a, b).ratio()


def match_cfb(games: pd.DataFrame, names: list[str], abbrs: list[str], kickoff_utc) -> tuple[dict, dict] | None:
    """Kalshi CFB event -> one FBS-vs-FBS schedule row, by school name within the kickoff window.
    Returns (game, {kalshi_abbr: schedule team name}). Ambiguous or weak matches -> None."""
    if len(names) != 2 or kickoff_utc is None or games.empty:
        return None
    ko = pd.to_datetime(games["kickoff_utc"], utc=True)
    cand = games[((ko - _ts(kickoff_utc)).abs() <= KALSHI_KICKOFF_TOLERANCE)
                 & (games["home_division"] == "fbs") & (games["away_division"] == "fbs")]
    scored = []
    for g in cand.to_dict("records"):
        for order in ((g["away_team"], g["home_team"]), (g["home_team"], g["away_team"])):
            sc = min(_sim(names[0], order[0]), _sim(names[1], order[1]))
            if sc >= CFB_MATCH_MIN:
                scored.append((sc, g, order))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    if len(scored) > 1 and scored[1][0] == scored[0][0] and scored[1][1]["game_id"] != scored[0][1]["game_id"]:
        return None
    _, g, order = scored[0]
    sides = dict(zip(abbrs, order)) if len(abbrs) == 2 else {}
    return g, sides


def resolve(league: str, games: pd.DataFrame, row: dict) -> tuple[dict, dict] | None:
    """(schedule game, {kalshi rung abbr: schedule team}) for a Kalshi market row."""
    if league == "NFL":
        pair = _pair(row.get("event_sub"))
        g = match_game(games, pair, row.get("kickoff_utc"))
        if not g:
            return None
        raw = re.sub(r"\s*\(.*?\)\s*$", "", row.get("event_sub") or "").split(" vs ")
        return g, {a.strip(): KALSHI_TO_NFLVERSE.get(a.strip(), a.strip()) for a in raw}
    return match_cfb(games, _split(row.get("event_title")), _split(row.get("event_sub")),
                     row.get("kickoff_utc"))


def _candidates(row: dict, p_yes: float):
    """(side, price, available size, fair prob for that side)."""
    yield "YES", row["yes_ask"], row.get("yes_ask_size") or 0, p_yes
    yield "NO", round(1 - row["yes_bid"], 4), row.get("yes_bid_size") or 0, 1 - p_yes


def price_side(p: float, price: float, size: float, mult: float, room: float) -> dict | None:
    """Fee-inclusive edge and paper size for buying one side at `price`. None if no flag."""
    if not 0 < price <= config.MAX_FLAG_PRICE or size < config.MIN_SIZE or room <= 0:
        return None
    fee1 = fee_per_contract(price, 1, mult)           # worst-case fee (1-lot rounding)
    stake = bet_size(p, price, fee1, lost_today=config.DAILY_LOSS_CAP - room)
    n = int(min(size, stake // (price + fee1))) if stake > 0 else 0
    if n < 1:
        return None
    fee = fee_per_contract(price, n, mult)
    edge = p - price - fee
    if edge < config.MIN_MODEL_EDGE:
        return None
    return {"contracts": n, "fee_per_contract": round(fee, 4), "edge": round(edge, 4),
            "stake": round(n * (price + fee), 2)}


def model_flags(df: pd.DataFrame, games: pd.DataFrame, snapshot_ts: str | None,
                lost_today: float = 0.0, now=None, ratings: Ratings | None = None,
                league: str = "NFL") -> tuple[list[dict], dict]:
    """MODEL flags for pregame spread/total rungs of one league. Returns (flags, status)."""
    L = LEAGUES[league]
    from .scan import is_stale
    status = {"ok": False, "reason": "", "ratings_as_of": None, "games_priced": 0}
    if df is None or df.empty or is_stale(snapshot_ts, now):
        status["reason"] = "stale or empty snapshot"
        return [], status
    ok, why = results_ok(games, snapshot_ts)
    if not ok:
        status["reason"] = why
        return [], status
    r = ratings or fit(games, snapshot_ts, L.half_life_days, L.ridge)
    if r is None:
        status["reason"] = "not enough games to fit"
        return [], status
    status.update(ok=True, reason="ok", ratings_as_of=r.as_of, n_games=r.n_games)

    q = df[(df["league"] == league) & df["series"].isin([L.spread, L.total])
           & (df["status"] == "active") & (df.get("phase", "unknown") == "pregame")
           & (df["strike_type"] == "greater")].dropna(subset=["floor_strike", "yes_bid", "yes_ask"])
    best: dict[str, dict] = {}
    proj_cache: dict[str, tuple] = {}
    for row in q.to_dict("records"):
        ev = row["event_ticker"]
        if ev not in proj_cache:
            m = resolve(league, games, row)
            g, sides = m if m else (None, {})
            pr = r.project(g["home_team"], g["away_team"], bool(g["neutral"])) if g else None
            proj_cache[ev] = (g, sides, pr)
        g, sides, pr = proj_cache[ev]
        if pr is None:
            continue
        margin_home, total = pr
        if row["series"] == L.spread:
            team = sides.get(_abbr(row["ticker"]))
            if team not in (g["home_team"], g["away_team"]):
                continue
            mean, sd, market = (margin_home if team == g["home_team"] else -margin_home), L.sd_margin, "SPREAD"
        else:
            mean, sd, market = total, L.sd_total, "TOTAL"
        p_yes = p_greater(mean, sd, float(row["floor_strike"]))
        mult = float(row.get("fee_multiplier") or 1.0)
        for side, price, size, p in _candidates(row, p_yes):
            s = price_side(p, price, size, mult, config.DAILY_LOSS_CAP - lost_today)
            if not s:
                continue
            f = {"snapshot_ts": row["snapshot_ts"], "league": league, "kind": "MODEL", "market": market,
                 "event": row["event_title"], "event_ticker": ev, "ticker": row["ticker"],
                 "leg1": f"{side} {row['ticker']}", "leg1_desc": row["yes_sub_title"],
                 "side": side, "price": price, "fair": round(p, 4), **s,
                 "size_available": size, "fee_multiplier": mult,
                 "home": g["home_team"], "away": g["away_team"],
                 "proj_margin_home": round(margin_home, 2), "proj_total": round(total, 2),
                 "ratings_as_of": r.as_of, "kickoff_utc": row.get("kickoff_utc"),
                 "kickoff_ct": row.get("kickoff_ct") or to_ct(row.get("kickoff_utc"))}
            if ev not in best or f["edge"] > best[ev]["edge"]:
                best[ev] = f
        status["games_priced"] = sum(1 for v in proj_cache.values() if v[2] is not None)
    status["events_unmatched"] = sum(1 for v in proj_cache.values() if v[0] is None)

    # Combined stakes never exceed what's left of today's loss cap (each bet can lose its stake)
    room = config.DAILY_LOSS_CAP - lost_today
    out = []
    for f in sorted(best.values(), key=lambda x: -x["edge"]):
        if f["stake"] > room:
            s = price_side(f["fair"], f["price"], f["size_available"], f["fee_multiplier"], room)
            if not s:
                continue
            f.update(s)
        room -= f["stake"]
        out.append(f)
    return out, status


# --- Backtest gate -------------------------------------------------------------------------
# MODEL flags only go live (alerts, paper ledger) once the walk-forward backtest shows the model
# beats closing-line prices after fees. Until then they are logged as SHADOW flags, so the
# model's calls against real Kalshi prices can still be graded. Missing/broken backtest = closed.
GATE_MIN_T = 2.0


def gate(backtest_path=None, league: str = "NFL") -> tuple[bool, str]:
    import json
    from pathlib import Path
    path = Path(backtest_path) if backtest_path else \
        Path(__file__).resolve().parents[1] / "data" / "model" / LEAGUES[league].backtest_file
    try:
        allp = json.loads(path.read_text())["test"]["proxy"]["ALL"]
        pnl, t = float(allp["pnl_per_contract"]), float(allp["t_stat"])
    except Exception:
        return False, f"gated: no {league} market benchmark in the backtest yet"
    if pnl > 0 and t >= GATE_MIN_T:
        return True, f"open: backtest {pnl*100:+.1f}c/contract, t={t:.2f}"
    return False, f"gated: backtest {pnl*100:+.1f}c/contract after fees, t={t:.2f} (needs >0 and t>={GATE_MIN_T})"
