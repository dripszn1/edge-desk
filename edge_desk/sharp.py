"""Kalshi vs sharp (backlog #11): price Kalshi spread/total rungs off the sharp main line.

Rung pricing: P(outcome > strike | market line) from an EMPIRICAL distribution of NFL outcomes
around closing lines (nflverse 2009+), not a normal curve, so key numbers (3, 7, 10) carry their
real weight:
  * spreads: home margin of games whose closing spread is within BW of the target line, mirrored
    (line, margin) -> (-line, -margin) to double the sample. Outcomes are NOT shifted, so the
    spikes at 3/7/10 stay where they are. If fewer than MIN_N games are that close (big
    favourites), fall back to the MIN_N nearest games with outcomes shifted by the line gap.
  * totals: Normal(line, SD) with SD = std of (total - closing total) in the training games.
    The empirical method lost to the normal curve on totals in BOTH the 2012-17 selection
    period and the 2018-25 test (NFL totals have no strong key numbers), so it is not used there.
  * a light prior (PRIOR_N pseudo-games from Normal(line, SD)) keeps thin buckets sane.
Only games with kickoff before the snapshot are used (look-ahead), and the backtest builds each
season's distribution from earlier seasons only.

SHARP signal: sharp fair (odds.fair_lines: Pinnacle, else sharp consensus) -> rung prob ->
flag when fair - Kalshi ask - taker fee at the paper size >= MIN_MODEL_EDGE (model.price_side:
same 90c cap, quarter Kelly, $10 cap). One flag per Kalshi event; combined stakes <= what is left
of DAILY_LOSS_CAP. Stale Kalshi snapshot or stale/after-the-snapshot odds -> no flags.
Gated like MODEL: flags go to data/flags/sharp_shadow.csv until the graded shadow record shows
>0 P&L with t >= 2 over >= 30 settled bets (`gate`, per league). CFB uses the same method on
CFBD closing lines (backlog #13) and is skipped until those lines are in. CFB teams are matched
to sportsbook names by school-name prefix ("Kent St." -> "Kent State Golden Flashes"), shortest
mascot remainder wins, ties -> no match. Paper only; never places orders.
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, odds
from .kickoffs import _split, norm, to_ct
from .model import _abbr, price_side
from .nfl_results import KALSHI_TO_NFLVERSE

ROOT = Path(__file__).resolve().parents[1]
BACKTEST_OUT = ROOT / "data" / "model" / "sharp_backtest.json"
SHADOW = ROOT / "data" / "flags" / "sharp_shadow.csv"
LIVE = ROOT / "data" / "flags" / "sharp_flags.csv"
SHADOW_BETS = ROOT / "data" / "sharp_shadow_bets.csv"

BW = {"spread": 1.0, "total": 1.5}
MIN_N = 300
PRIOR_N = 20
SD = {"spread": 13.3, "total": 13.6}      # only for the prior and the normal benchmark
KICKOFF_TOL = pd.Timedelta(hours=6)       # odds commence vs Kalshi kickoff
# #41: Kalshi stamps some TBD-time night games at a placeholder (11-14: TTU/OKST Kalshi 20:00Z vs books
# 02:30Z; TEX/LSU 20:00Z vs 00:30Z). When nothing matches inside KICKOFF_TOL, a unique name match up to
# this far away is accepted (a team never plays twice in 12h). The earlier, Kalshi kickoff still drives
# phase / stale guards, so the fallback can only make us stricter, never price an in-play market.
KICKOFF_TOL_WIDE = pd.Timedelta(hours=12)
MAX_RUNG_OFFSET = 10.5   # only price rungs inside the backtested range (OFFSETS = line +-10); the
# CFB: NFL-validated cap doesn't transfer; stay near the main line until a CFB backtest says otherwise
MAX_RUNG_OFFSET_BY_LEAGUE = {"NFL": 10.5, "CFB": 6.0, "CBB": 3.0}
# CBB has no historical closing-line feed yet, so its rungs use a normal around the sharp line with
# typical NCAAB residual SDs (assumption, not fitted). Only rungs within +-3 pts of the line alert,
# where the SD matters least; everything else is tracked (#24). Replace with fitted SDs when #22 lands.
CBB_SD = {"spread": 11.4, "total": 16.5}   # fallback only, until data/cbb/lines.csv has history.
# (#29) Was 11.0/11.5; CBBD 2021-26 finals (32k games, 0-0/forfeit rows dropped) fit 11.35 / 16.5.
CBB_MIN_FIT_GAMES = 500     # settled CBB games with a closing line needed to fit the SDs
SHADOW_ONLY = ("CBB",)      # Emerson: CBB is a shadow/PAPER test; never live, whatever its gate says
ODDS_MAX_AGE_MIN = 90          # no SHARP flag from a sharp line older than this...
ODDS_MAX_AGE_NEAR_KO_MIN = 60  # ...or older than this within NEAR_KO_H of kickoff
NEAR_KO_H = 3
EDGE_SANITY_CAP = 0.12         # an "edge" this big vs a sharp line is almost always a data/matching error
# Low-confidence plays are NOT thrown away: they are logged here with the reason, graded like
# everything else, and only promoted to alerts once their own record proves them (Emerson, 11-10:
# "if the data is pointing that way we don't want to skip").
REJECTED = Path(__file__).resolve().parents[1] / "data" / "flags" / "sharp_tracked.csv"


def odds_too_old(snapshot_ts, odds_fetched_ts, kickoff_utc) -> bool:
    """True when the sharp line is too stale to trust for this snapshot."""
    try:
        snap = pd.Timestamp(snapshot_ts); fet = pd.Timestamp(odds_fetched_ts)
    except Exception:  # noqa: BLE001
        return True
    if snap.tzinfo is None: snap = snap.tz_localize("UTC")
    if fet.tzinfo is None: fet = fet.tz_localize("UTC")
    age = (snap - fet).total_seconds() / 60
    limit = ODDS_MAX_AGE_MIN
    if kickoff_utc:
        ko = pd.Timestamp(kickoff_utc)
        if ko.tzinfo is None: ko = ko.tz_localize("UTC")
        if ko - snap <= pd.Timedelta(hours=NEAR_KO_H):
            limit = ODDS_MAX_AGE_NEAR_KO_MIN
    return age > limit


# (#27) path -> {(ticker, side): signature of the last row logged}. The live loop re-scans every
# 30 s; a held-back play is re-logged only when it is new or something about it changed. Grading
# uses the first sighting, so dropping identical repeats changes no result.
_TRACK_SEEN: dict[str, dict] = {}
_TRACK_COLS = ("snapshot_ts", "league", "event", "ticker", "side", "price",
               "fair", "edge", "sharp_line", "odds_fetched_ts")


def _track_sig(price, fair, line, odds_ts, reason) -> tuple:
    def r(x, n):
        try:
            return round(float(x), n)
        except (TypeError, ValueError):
            return None
    return (r(price, 4), r(fair, 3), r(line, 2), str(odds_ts), str(reason))


def _track_seen() -> dict:
    key = str(REJECTED)
    if key not in _TRACK_SEEN:
        seen = {}
        if REJECTED.exists():
            try:
                old = pd.read_csv(REJECTED, dtype=str)
                for r in old.itertuples(index=False):
                    seen[(r.ticker, r.side)] = _track_sig(r.price, r.fair, r.sharp_line, r.odds_fetched_ts, r.reason)
            except Exception:  # noqa: BLE001 - an unreadable log just means "log everything"
                seen = {}
        _TRACK_SEEN[key] = seen
    return _TRACK_SEEN[key]


def log_rejected(f: dict, reason: str) -> None:
    seen = _track_seen()
    k = (str(f.get("ticker")), str(f.get("side")))
    sig = _track_sig(f.get("price"), f.get("fair"), f.get("sharp_line"), f.get("odds_fetched_ts"), reason)
    if seen.get(k) == sig:
        return
    REJECTED.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{**{c: f.get(c) for c in _TRACK_COLS}, "reason": reason}])
    row.to_csv(REJECTED, mode="a", header=not REJECTED.exists(), index=False)
    seen[k] = sig
                         # far tails of the totals normal run fat (NFL 2012-25: P(under line-20)
                         # normal 6.6% vs actual 5.1%), which made deep NO-on-low-overs look +EV
GATE_MIN_BETS = 30
GATE_MIN_T = 2.0

NFL_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")


# ---------- empirical rung distribution ----------
class LineDist:
    def __init__(self, lines, outcomes, kind: str):
        lines, outcomes = np.asarray(lines, float), np.asarray(outcomes, float)
        ok = ~(np.isnan(lines) | np.isnan(outcomes))
        lines, outcomes = lines[ok], outcomes[ok]
        if kind == "spread":
            lines, outcomes = np.concatenate([lines, -lines]), np.concatenate([outcomes, -outcomes])
        self.kind, self.bw, self.sd = kind, BW[kind], SD[kind]
        self.normal = kind == "total"
        if len(lines) >= 100:                       # prior / normal SD fitted on the training games
            self.sd = float(np.std(outcomes - lines, ddof=1))
        o = np.lexsort((outcomes, lines))           # (#36) canonical order: ties by value, never by
        self.lines, self.out = lines[o], outcomes[o]  # input position or the CPU's sort algorithm
        self.n = len(self.lines)

    def p_greater(self, line: float, strike: float) -> float:
        """P(outcome > strike | market line = `line`)."""
        prior = 0.5 * math.erfc((strike - line) / (self.sd * math.sqrt(2)))
        if self.n == 0 or self.normal:
            return prior
        d = self.lines - line
        near = np.abs(d) <= self.bw + 1e-9
        if near.sum() >= MIN_N:
            w = 1.0 - np.abs(d[near]) / (self.bw + 0.5)       # triangular, never zero inside BW
            hit = (self.out[near] > strike).astype(float)
        else:
            k = min(MIN_N, self.n)
            a = np.abs(d)
            cut = np.partition(a, k - 1)[k - 1]           # (#36) the k-th nearest gap; every game tied
            idx = np.flatnonzero(a <= cut + 1e-9)        # with it is in, so no tie is broken by order
            w = np.ones(len(idx))
            hit = (self.out[idx] - d[idx] > strike).astype(float)   # shift by the line gap
        return float((w @ hit + PRIOR_N * prior) / (w.sum() + PRIOR_N))


def build_dists(games: pd.DataFrame, before) -> dict[str, LineDist]:
    """Distributions from games that kicked off before `before` (look-ahead guard)."""
    g = games.dropna(subset=["home_score", "away_score"])
    g = g[pd.to_datetime(g["kickoff_utc"], utc=True) < _ts(before)]
    return {"spread": LineDist(g["spread_line"], g["home_score"] - g["away_score"], "spread"),
            "total": LineDist(g["total_line"], g["home_score"] + g["away_score"], "total")}


def cbb_dists(games: pd.DataFrame | None, before) -> dict[str, LineDist]:
    """CBB rung distributions: normal around the sharp line, SD fitted walk-forward from settled
    games tipped before `before` (build_dists' look-ahead guard); the assumed CBB_SD until there
    are CBB_MIN_FIT_GAMES of them."""
    from .cbb_lines import final_games
    games = final_games(games) if games is not None else None
    fitted = build_dists(games, before) if games is not None and not games.empty else {}
    out = {}
    for k in ("spread", "total"):
        d = fitted.get(k)
        n = (d.n // 2 if k == "spread" else d.n) if d is not None else 0
        if d is None or n < CBB_MIN_FIT_GAMES:
            d = LineDist([], [], k)
            d.sd, d.source = CBB_SD[k], f"assumed ({n} games < {CBB_MIN_FIT_GAMES})"
        else:
            d.source = f"fitted on {n} games"
        d.normal = True
        out[k] = d
    return out


# ---------- backtest: closing line -> fair rung prices -> realised outcomes ----------
OFFSETS = np.arange(-10, 11)                 # rungs at floor(line)+0.5+off, x.5 like Kalshi


def _normal_p(line, strike, sd):
    return 0.5 * math.erfc((strike - line) / (sd * math.sqrt(2)))


OFFSET_BANDS = ((0.0, 3.0), (3.0, 6.0), (6.0, 10.5))   # |rung - line|, matching the per-league caps


def offset_bands(P_e, P_n, Y, OFF) -> list[dict]:
    """(#33) Calibration by distance from the line. Each rung is folded to its longshot side
    (pred = min(p, 1-p), hit = that side won), so over/under errors don't cancel: a band whose
    longshot_gap is clearly negative is one where we overprice longshots (fair too high -> fake
    edges on far rungs). Calibration only; widening a cap still needs a tracked record (#24)."""
    P_e, P_n, Y, A = map(np.asarray, (P_e, P_n, Y, np.abs(np.asarray(OFF))))
    out = []
    for i, (lo, hi) in enumerate(OFFSET_BANDS):
        m = ((A > lo) if i else (A >= lo)) & (A <= hi)
        if not m.any():
            continue
        p, pn, y = P_e[m], P_n[m], Y[m]
        flip = p > 0.5
        q, hit = np.where(flip, 1 - p, p), np.where(flip, 1 - y, y)
        out.append({"band": f"{lo:g}-{hi:g}", "rungs": int(m.sum()),
                    "brier": round(float(np.mean((p - y) ** 2)), 5),
                    "brier_normal": round(float(np.mean((pn - y) ** 2)), 5),
                    "longshot_pred": round(float(q.mean()), 4), "longshot_actual": round(float(hit.mean()), 4),
                    "longshot_gap": round(float(hit.mean() - q.mean()), 4)})
    return out


def backtest(games: pd.DataFrame, seasons=range(2018, 2026)) -> dict:
    g = games.dropna(subset=["home_score", "away_score", "spread_line", "total_line"]).copy()
    g["margin"] = g["home_score"] - g["away_score"]
    g["total"] = g["home_score"] + g["away_score"]
    res = {}
    for kind, line_col, out_col in (("spread", "spread_line", "margin"), ("total", "total_line", "total")):
        P_e, P_n, Y, OFF = [], [], [], []
        for s in seasons:
            test = g[g["season"] == s]
            if test.empty:
                continue
            first = pd.to_datetime(test["kickoff_utc"], utc=True).min()
            d = build_dists(games, first)[kind]
            sd_n = d.sd
            for L, y in zip(test[line_col].to_numpy(float), test[out_col].to_numpy(float)):
                for off in OFFSETS:
                    k = math.floor(L) + 0.5 + float(off)
                    P_e.append(d.p_greater(L, k))
                    P_n.append(_normal_p(L, k, sd_n))
                    Y.append(float(y > k))
                    OFF.append(k - L)
        P_e, P_n, Y, OFF = map(np.asarray, (P_e, P_n, Y, OFF))
        eps = 1e-6
        ll = lambda p: float(-np.mean(Y * np.log(np.clip(p, eps, 1)) + (1 - Y) * np.log(np.clip(1 - p, eps, 1))))
        cal = []
        for lo in np.arange(0, 1, 0.1):
            m = (P_e >= lo) & (P_e < lo + 0.1)
            if m.sum():
                cal.append({"bucket": f"{lo:.1f}-{lo + 0.1:.1f}", "n": int(m.sum()),
                            "pred": round(float(P_e[m].mean()), 4), "actual": round(float(Y[m].mean()), 4)})
        near = np.abs(OFF) <= 4                      # rungs within 4 pts of the line: where Kalshi trades
        by_off = offset_bands(P_e, P_n, Y, OFF)
        res[kind] = {
            "rungs": int(len(Y)), "games": int(len(Y) // len(OFFSETS)),
            "brier_empirical": round(float(np.mean((P_e - Y) ** 2)), 5),
            "brier_normal": round(float(np.mean((P_n - Y) ** 2)), 5),
            "logloss_empirical": round(ll(P_e), 5), "logloss_normal": round(ll(P_n), 5),
            "near_line_brier_empirical": round(float(np.mean((P_e[near] - Y[near]) ** 2)), 5),
            "near_line_brier_normal": round(float(np.mean((P_n[near] - Y[near]) ** 2)), 5),
            "max_calibration_gap": round(max(abs(c["pred"] - c["actual"]) for c in cal), 4),
            "calibration": cal,
            "by_offset": by_off,
        }
    d = build_dists(games, pd.Timestamp("2100-01-01", tz="UTC"))["spread"]
    res["key_numbers_at_minus3"] = {
        "p_cover_2.5_minus_3.5_empirical": round(d.p_greater(3, 2.5) - d.p_greater(3, 3.5), 4),
        "p_cover_2.5_minus_3.5_normal": round(_normal_p(3, 2.5, SD["spread"]) - _normal_p(3, 3.5, SD["spread"]), 4)}
    res["seasons"] = [int(s) for s in seasons]
    res["method"] = {"spread": "empirical (key numbers)", "total": "normal, SD fitted on training seasons"}
    res["method_better_than_normal"] = {k: bool(res[k]["logloss_empirical"] < res[k]["logloss_normal"])
                                        for k in ("spread", "total")}
    return res


# ---------- Kalshi <-> odds matching ----------
def _odds_event(fair: pd.DataFrame, home_full: str, away_full: str, kickoff) -> pd.DataFrame:
    if fair.empty or kickoff is None or (isinstance(kickoff, float) and math.isnan(kickoff)):
        return fair.iloc[0:0]
    c = pd.to_datetime(fair["commence_utc"], utc=True)
    teams = {home_full, away_full}
    same = fair.apply(lambda r: {r["home_team"], r["away_team"]} == teams, axis=1)
    for tol in (KICKOFF_TOL, KICKOFF_TOL_WIDE):      # #41: placeholder Kalshi times
        m = fair[((c - _ts(kickoff)).abs() <= tol) & same]
        if m["event_id"].nunique() == 1:
            return m
        if m["event_id"].nunique() > 1:
            break                                    # ambiguous -> no match
    return fair.iloc[0:0]


SERIES = {"NFL": ("KXNFLSPREAD", "KXNFLTOTAL"), "CFB": ("KXNCAAFSPREAD", "KXNCAAFTOTAL"),
          "CBB": ("KXNCAAMBSPREAD", "KXNCAAMBTOTAL")}

# Kalshi CFB short names -> how sportsbooks spell the school (cleaned, before the mascot)
CFB_BOOK_ALIASES = {
    "miami fl": ["miami"], "miami oh": ["miami oh", "miami ohio"], "ole miss": ["ole miss", "mississippi"],
    "nc state": ["nc state", "north carolina state"], "uconn": ["uconn", "connecticut"],
    "umass": ["umass", "massachusetts"], "louisiana monroe": ["ul monroe", "louisiana monroe"],
    "louisiana": ["louisiana", "louisiana lafayette", "ul lafayette"], "hawaii": ["hawaii"],
    "southern miss": ["southern miss", "southern mississippi"], "sam houston": ["sam houston state", "sam houston"],
    "florida international": ["fiu", "florida international"], "usc": ["usc", "southern california"],
    "lsu": ["lsu"], "byu": ["byu"], "tcu": ["tcu"], "smu": ["smu"], "ucf": ["ucf"], "utsa": ["utsa"],
    "utep": ["utep"], "unlv": ["unlv"], "app state": ["appalachian state"],
    "appalachian state": ["appalachian state"], "san jose state": ["san jose state"],
    "texas a and m": ["texas a and m"], "university at albany": ["albany", "ualbany"],
}


def _clean(name: str | None) -> str:
    return _clean_cached(name or "")


@lru_cache(maxsize=65536)
def _clean_cached(name: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[.'()’ʻ`]", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = ["state" if t == "st" else t for t in s.split()]
    return " ".join(toks)


def _prefix_score(kalshi_name: str, book_name: str) -> int | None:
    """Words left after the school name (the mascot); None if the book name isn't this school."""
    b = _clean(book_name).split()
    k = _clean(kalshi_name)
    best = None
    for alt in CFB_BOOK_ALIASES.get(k, [k]):
        a = alt.split()
        if a and b[:len(a)] == a:
            rem = len(b) - len(a)
            best = rem if best is None or rem < best else best
    return best


_ID_CACHE: dict = {}


def _per_frame(name: str, frame: pd.DataFrame, build):
    """#44: memo of `build(frame)` for this exact frame object (kept alive, so its id can't be reused)."""
    hit = _ID_CACHE.get(name)
    if hit is not None and hit[0] is frame and hit[1] == len(frame):
        return hit[2]
    val = build(frame)
    _ID_CACHE[name] = (frame, len(frame), val)
    return val


def _commence(fair: pd.DataFrame) -> pd.Series:
    return _per_frame("commence", fair, lambda f: pd.to_datetime(f["commence_utc"], utc=True))


def _event_lines(fe: pd.DataFrame) -> dict:
    """{('spreads', team) | ('totals', 'Over'): (point, source)} from the FIRST matching row, as before."""
    out = {}
    for r in fe[fe["market"].isin(["spreads", "totals"])].to_dict("records"):
        out.setdefault((r["market"], r["outcome"]), (r["point"], r["source"]))
    out["_has_spreads"] = bool((fe["market"] == "spreads").any())
    return out


def match_event(league: str, row: dict, fair: pd.DataFrame) -> tuple[dict | None, pd.DataFrame]:
    """({kalshi abbr: sportsbook team name}, that event's fair rows) or (None, empty)."""
    empty = fair.iloc[0:0]
    if league == "NFL":
        sides = nfl_sides(row)
        names = list(sides.values()) if sides else []
        return (sides, _odds_event(fair, *names, row.get("kickoff_utc"))) if len(names) == 2 else (None, empty)
    names, abbrs = _split(row.get("event_title")), _split(row.get("event_sub"))
    ko = row.get("kickoff_utc")
    if len(names) != 2 or len(abbrs) != 2 or ko is None or fair.empty or (isinstance(ko, float) and math.isnan(ko)):
        return None, empty
    c = _commence(fair)
    for tol in (KICKOFF_TOL, KICKOFF_TOL_WIDE):      # #41: wide window only when the near one is empty
        cand = fair[(c - _ts(ko)).abs() <= tol].drop_duplicates("event_id")
        scored = []
        for r in cand.to_dict("records"):
            for bk in ((r["home_team"], r["away_team"]), (r["away_team"], r["home_team"])):
                s0, s1 = _prefix_score(names[0], bk[0]), _prefix_score(names[1], bk[1])
                if s0 is not None and s1 is not None:
                    scored.append((s0 + s1, r["event_id"], {abbrs[0]: bk[0], abbrs[1]: bk[1]}))
        if scored:
            break
    if not scored:
        return None, empty
    scored.sort(key=lambda x: x[0])
    if len(scored) > 1 and scored[1][0] == scored[0][0] and scored[1][1] != scored[0][1]:
        return None, empty                         # ambiguous -> no match (fails safe)
    _, eid, sides = scored[0]
    return sides, fair[fair["event_id"] == eid]


def nfl_sides(row: dict) -> dict | None:
    """{kalshi abbr: odds-api full name} for an NFL Kalshi row."""
    raw = re.sub(r"\s*\(.*?\)\s*$", "", row.get("event_sub") or "").split(" vs ")
    if len(raw) != 2:
        return None
    out = {}
    for a in raw:
        a = a.strip()
        full = NFL_NAMES.get(KALSHI_TO_NFLVERSE.get(a, a))
        if not full:
            return None
        out[a] = full
    return out


def rung_prob(dists: dict, fair_ev: pd.DataFrame, row: dict, sides: dict) -> tuple[float, float, str] | None:
    """(P(YES), sharp line used, source) for one Kalshi rung, or None if no sharp line."""
    strike = float(row["floor_strike"])
    ev = _per_frame("event_lines", fair_ev, _event_lines)
    if row["series"].endswith("SPREAD"):
        team = sides.get(_abbr(row["ticker"]))
        if team is None or not ev["_has_spreads"]:
            return None
        r = ev.get(("spreads", team))
        if r is None or pd.isna(r[0]):
            return None
        line = -float(r[0])                          # team -3.5 -> expected margin +3.5
        return dists["spread"].p_greater(line, strike), line, r[1]
    tt = ev.get(("totals", "Over"))
    if tt is None or pd.isna(tt[0]):
        return None
    line = float(tt[0])
    return dists["total"].p_greater(line, strike), line, tt[1]


def fair_entry(row: dict, fair: float, price, fe: pd.DataFrame, line, stale: bool) -> dict:
    """#35: this scan's fair value for one side of one contract, so an alerted play can be called
    'gone' when a fresh sharp pull moves the line against it (even when it is no longer a flag)."""
    return {"fair": round(float(fair), 4), "price": None if price is None else float(price),
            "odds_fetched_ts": str(fe["fetched_ts"].max()), "snapshot_ts": str(row.get("snapshot_ts")),
            "sharp_line": None if line is None else float(line), "stale": bool(stale),
            "kickoff_utc": row.get("kickoff_utc")}


def sharp_flags(df: pd.DataFrame, lines: pd.DataFrame, games: pd.DataFrame, snapshot_ts: str | None,
                lost_today: float = 0.0, now=None, league: str = "NFL") -> tuple[list[dict], dict]:
    from .scan import is_stale
    status = {"ok": False, "reason": "", "events_priced": 0, "events_no_sharp_line": 0}
    if df is None or df.empty or is_stale(snapshot_ts, now):
        status["reason"] = "stale or empty snapshot"
        return [], status
    fair = odds.fair_lines(lines, snapshot_ts)          # look-ahead + stale + in-play guards
    fair = fair[fair["league"] == league] if len(fair) else fair
    if fair.empty:
        status["reason"] = "no usable sharp odds (missing key, none fetched, or older than %dh)" % odds.ODDS_STALE_HOURS
        return [], status
    if league == "CBB":
        dists = cbb_dists(games, snapshot_ts)
        status["cbb_sd"] = {k: {"sd": round(d.sd, 2), "source": d.source} for k, d in dists.items()}
    elif games is None or games.empty or games["spread_line"].notna().sum() < 500:
        status["reason"] = "no historical closing lines for the rung distribution"
        return [], status
    else:
        dists = build_dists(games, snapshot_ts)
    status.update(ok=True, reason="ok", odds_fetched=str(fair["fetched_ts"].max()))
    q = df[(df["league"] == league) & df["series"].isin(SERIES[league])
           & (df["status"] == "active") & (df.get("phase", "unknown") == "pregame")
           & (df["strike_type"] == "greater")].dropna(subset=["floor_strike", "yes_bid", "yes_ask"])
    best, seen, fairs = {}, {}, {}
    status["_fairs"] = fairs                            # #35: popped by scan.run_sharp, never in latest.json
    ev_vol = (df[df["league"] == league].groupby("event_ticker")["volume_24h"].sum()
              if "volume_24h" in df else pd.Series(dtype=float))
    for row in q.to_dict("records"):
        if league == "CBB" and float(ev_vol.get(row["event_ticker"], 0) or 0) < config.CBB_MIN_EVENT_VOL_24H:
            status["low_volume"] = status.get("low_volume", 0) + 1
            continue                                   # Emerson: no volume -> don't touch
        ev = row["event_ticker"]
        if ev not in seen:
            seen[ev] = match_event(league, row, fair)
        sides, fe = seen[ev]
        if fe.empty:
            continue
        rp = rung_prob(dists, fe, row, sides)
        if rp is None:
            continue
        p_yes, line, src = rp
        why = []
        if abs(float(row["floor_strike"]) - line) > MAX_RUNG_OFFSET_BY_LEAGUE.get(league, MAX_RUNG_OFFSET):
            status["rungs_out_of_range"] = status.get("rungs_out_of_range", 0) + 1
            why.append("rung far from sharp line (unvalidated range)")
        if odds_too_old(row["snapshot_ts"], fe["fetched_ts"].max(), row.get("kickoff_utc")):
            status["odds_too_old"] = status.get("odds_too_old", 0) + 1
            why.append("sharp line stale")
        mult = float(row.get("fee_multiplier") or 1.0)
        stale_line = "sharp line stale" in why
        for side, price, size, p in (("YES", row["yes_ask"], row.get("yes_ask_size") or 0, p_yes),
                                     ("NO", round(1 - row["yes_bid"], 4), row.get("yes_bid_size") or 0, 1 - p_yes)):
            fairs[f"{row['ticker']}|{side}"] = fair_entry(row, p, price, fe, line, stale_line)   # #35
            if league == "CBB" and (size or 0) < config.CBB_MIN_SIZE:
                continue
            s = price_side(p, price, size, mult, config.DAILY_LOSS_CAP - lost_today)
            if not s:
                continue
            f = {"snapshot_ts": row["snapshot_ts"], "league": league, "kind": "SHARP",
                 "market": "SPREAD" if row["series"].endswith("SPREAD") else "TOTAL",
                 "event": row["event_title"], "event_ticker": ev, "ticker": row["ticker"],
                 "leg1": f"{side} {row['ticker']}", "leg1_desc": row["yes_sub_title"],
                 "side": side, "price": price, "fair": round(p, 4), **s, "size_available": size,
                 "fee_multiplier": mult, "sharp_line": line, "sharp_source": src,
                 "odds_fetched_ts": str(fe["fetched_ts"].max()), "kickoff_utc": row.get("kickoff_utc"),
                 "kickoff_ct": row.get("kickoff_ct") or to_ct(row.get("kickoff_utc"))}
            reasons = list(why)
            if f["edge"] > EDGE_SANITY_CAP:
                status["rejected_edge_cap"] = status.get("rejected_edge_cap", 0) + 1
                reasons.append("edge above sanity cap")
            if reasons == ["sharp line stale"]:              # #28: a fresh line could make it a play
                status["stale_only"] = status.get("stale_only", 0) + 1
            if reasons:
                log_rejected(f, "; ".join(reasons))       # tracked + graded, never alerted
                continue
            if ev not in best or f["edge"] > best[ev]["edge"]:
                best[ev] = f
    status["events_priced"] = sum(1 for v in seen.values() if not v[1].empty)
    status["events_no_sharp_line"] = sum(1 for v in seen.values() if v[1].empty)
    room, out = config.DAILY_LOSS_CAP - lost_today, []
    for f in sorted(best.values(), key=lambda x: -x["edge"]):
        if f["stake"] > room:
            s = price_side(f["fair"], f["price"], f["size_available"], f["fee_multiplier"], room)
            if not s:
                continue
            f.update(s)
        room -= f["stake"]
        out.append(f)
    return out, status


# ---------- gate ----------
def gate(bets_path: Path | None = None, league: str | None = None, market: str | None = None) -> tuple[bool, str]:
    """Open only when graded SHARP shadow bets (of `league`, if given) show >0 P&L with t >= 2
    over >= 30 settled. market="ML" -> moneylines only; otherwise spreads/totals only (#23)."""
    from .weekly import t_stat
    try:
        b = pd.read_csv(bets_path or SHADOW_BETS)
        s = b[b["status"] == "settled"]
        if league is not None and "league" in s:
            s = s[s["league"] == league]
        if "market" in s:
            s = s[s["market"] == "ML"] if market == "ML" else s[s["market"] != "ML"]
    except Exception:
        return False, "gated: no graded SHARP shadow bets yet"
    if len(s) < GATE_MIN_BETS:
        return False, f"gated: {len(s)} settled SHARP shadow bets (< {GATE_MIN_BETS})"
    t = t_stat(s["pnl"] / s["stake"])
    pnl = float(s["pnl"].sum())
    if pnl > 0 and t is not None and t >= GATE_MIN_T:
        return True, f"open: {len(s)} bets, P&L ${pnl:+.2f}, t={t:.2f}"
    return False, f"gated: {len(s)} bets, P&L ${pnl:+.2f}, t={t if t is None else round(t, 2)} (needs >0 and t>={GATE_MIN_T})"


CFB_BACKTEST_OUT = ROOT / "data" / "model" / "sharp_cfb_backtest.json"
CBB_BACKTEST_OUT = ROOT / "data" / "model" / "sharp_cbb_backtest.json"


def cbb_sd_check(games: pd.DataFrame, max_off: float = 3.0) -> dict:
    """Walk-forward per season: SD fitted on earlier seasons vs the assumed CBB_SD, scored by log
    loss on rungs within +-max_off of the closing line (the only CBB rungs that alert)."""
    g = games.dropna(subset=["home_score", "away_score", "spread_line", "total_line"]).copy()
    g["margin"] = g["home_score"] - g["away_score"]
    g["total"] = g["home_score"] + g["away_score"]
    eps, out = 1e-6, {}
    seasons = sorted(int(x) for x in g["season"].dropna().unique())
    for kind, lc, oc in (("spread", "spread_line", "margin"), ("total", "total_line", "total")):
        Pf, Pa, Y, per = [], [], [], []
        for s in seasons:
            test = g[g["season"] == s]
            first = pd.to_datetime(test["kickoff_utc"], utc=True).min()
            d = cbb_dists(games, first)[kind]
            if not d.source.startswith("fitted"):
                continue
            n0 = len(Y)
            for L, y in zip(test[lc].to_numpy(float), test[oc].to_numpy(float)):
                for off in np.arange(-max_off, max_off + 1):
                    k = math.floor(L) + 0.5 + float(off)
                    if abs(k - L) > max_off:
                        continue
                    Pf.append(_normal_p(L, k, d.sd)); Pa.append(_normal_p(L, k, CBB_SD[kind])); Y.append(float(y > k))
            per.append({"season": s, "sd_fitted": round(d.sd, 2), "games": int(len(test)), "rungs": len(Y) - n0})
        if not Y:
            out[kind] = {"note": "not enough earlier seasons to fit"}
            continue
        Pf, Pa, Y = map(np.asarray, (Pf, Pa, Y))
        ll = lambda p: float(-np.mean(Y * np.log(np.clip(p, eps, 1)) + (1 - Y) * np.log(np.clip(1 - p, eps, 1))))
        out[kind] = {"rungs": int(len(Y)), "logloss_fitted": round(ll(Pf), 5),
                     "logloss_assumed": round(ll(Pa), 5), "sd_assumed": CBB_SD[kind],
                     "fitted_better": bool(ll(Pf) < ll(Pa)), "by_season": per}
    return out


def history(league: str) -> pd.DataFrame:
    """Past games with closing lines for the rung distribution."""
    from . import cbb_lines, cfb_results, nfl_results
    if league == "CBB":
        return cbb_lines.history()
    return nfl_results.load() if league == "NFL" else cfb_results.load()


def inputs_hash(g: pd.DataFrame) -> str:
    """(#37) Fingerprint of a backtest's inputs: the history rows plus the pricing code. Same
    fingerprint -> same result (#36 made it deterministic), so collect can skip the ~80 s re-run."""
    import hashlib
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(g.reset_index(drop=True), index=True).to_numpy().tobytes())
    for f in ("sharp.py", "cbb_lines.py"):
        p = Path(__file__).with_name(f)
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()[:16]


def run_backtest(force: bool = False) -> dict:
    reps = {}
    for lg, out in (("NFL", BACKTEST_OUT), ("CFB", CFB_BACKTEST_OUT), ("CBB", CBB_BACKTEST_OUT)):
        g = history(lg)
        if g.empty or g["spread_line"].notna().sum() < 500:
            print(f"sharp backtest {lg}: no closing lines yet, skip")
            continue
        ih = inputs_hash(g)
        if not force and out.exists():
            try:
                if json.loads(out.read_text()).get("inputs_hash") == ih:
                    print(f"sharp backtest {lg}: inputs unchanged ({ih}), kept")
                    continue
            except (ValueError, OSError):
                pass
        if lg == "CBB":
            ss = sorted(int(x) for x in g["season"].dropna().unique())[1:]
            rep = backtest(g, seasons=ss)
            rep["method"] = {"spread": "normal, SD fitted walk-forward", "total": "normal, SD fitted walk-forward"}
            rep["sd_check"] = cbb_sd_check(g)
        else:
            rep = backtest(g)
        rep["league"] = lg
        rep["inputs_hash"] = ih
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2))
        reps[lg] = rep
        for k in ("spread", "total"):
            r = rep[k]
            print(f"sharp backtest {lg} {k}: {r['games']} games, logloss emp {r['logloss_empirical']} vs normal "
                  f"{r['logloss_normal']}, near-line brier {r['near_line_brier_empirical']} vs {r['near_line_brier_normal']}")
    return reps


if __name__ == "__main__":
    import sys
    run_backtest(force="--force" in sys.argv)
