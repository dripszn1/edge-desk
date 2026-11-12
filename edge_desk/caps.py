"""Distance caps that learn (#46). Report only: nothing here changes pricing or alerts.

SHARP prices a Kalshi rung only within a per-league distance of the sharp line
(sharp.MAX_RUNG_OFFSET_BY_LEAGUE: NFL 10.5 / CFB 6 / CBB 3 pts). CFB and CBB were set tight out of
caution, before their pricing was validated, so this module collects the evidence for each cap:

(a) caps_in_sd: each cap in points AND in units of that league x market's outcome SD around the
    closing line (std of outcome - line over settled history games played before `now`).
(b) calibration-adjusted fair: fair minus the walk-forward longshot bias of its distance band
    (sharp backtest `by_offset_season`, #33). Only seasons whose LAST game kicked off before the
    bet's entry are pooled, so a bet is never adjusted with outcomes from after it was made.
    The side's fair is folded like the backtest: a longshot side (fair < 50%) moves by +gap, a
    favourite side by -gap. adjusted edge = logged edge (fair - price - fee) + (fair_adj - fair),
    so fees stay counted; a play "qualifies adjusted" when that edge is still >= MIN_MODEL_EDGE.
    Raw and adjusted are graded side by side; alerts stay on raw.
(c) cap review: per league x market x distance band, SHARP shadow plays plus held-back plays
    whose ONLY reason is distance (stale-line and sanity-cap plays are left out: a stale line or a
    data error would muddy what distance does). CLV vs Kalshi's close (clv_bets.csv, after the
    entry fee, t by game since rungs of one game move together) and settled P&L.
    Beyond the cap: >= MIN_PLAYS with a close, CLV > 0 and t >= 2  -> "PROPOSE widening".
    Inside the cap: >= MIN_PLAYS, CLV < 0 and t <= -2              -> "PROPOSE tightening".
Emerson decides; the caps never change here. Rows entered after `now` are ignored.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "cap_review_bets.csv"
BET_FILES = {"shadow": DATA / "sharp_shadow_bets.csv", "tracked": DATA / "sharp_tracked_bets.csv"}
FLAG_FILES = {"shadow": DATA / "flags" / "sharp_shadow.csv", "tracked": DATA / "flags" / "sharp_tracked.csv"}
CLV_FILE = DATA / "clv_bets.csv"
BACKTESTS = {"NFL": DATA / "model" / "sharp_backtest.json", "CFB": DATA / "model" / "sharp_cfb_backtest.json",
             "CBB": DATA / "model" / "sharp_cbb_backtest.json"}
PT_BANDS = ((0.0, 3.0), (3.0, 6.0), (6.0, 10.5), (10.5, math.inf))
SD_BANDS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, math.inf))
MIN_PLAYS = 30
PROPOSE_T = 2.0
COLS = ["bet_id", "source", "league", "market", "event", "entry_ts", "side", "price", "fair", "strike",
        "sharp_line", "dist_pts", "sd", "dist_sd", "band", "inside_cap", "bias", "fair_adj", "edge_raw",
        "edge_adj", "adj_qualifies", "status", "pnl", "clv_c", "close_status"]


def _caps() -> dict:
    from .sharp import MAX_RUNG_OFFSET_BY_LEAGUE
    return dict(MAX_RUNG_OFFSET_BY_LEAGUE)


def _band(x: float, bands) -> str | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    for i, (lo, hi) in enumerate(bands):
        if ((x > lo) if i else (x >= lo)) and x <= hi + 1e-9:
            return f"{lo:g}-{hi:g}" if hi != math.inf else f"{lo:g}+"
    return None


def strike_of(ticker: str) -> float | None:
    """Kalshi rung strike from its ticker: '...-24' -> 23.5 (Over 23.5), '...-AUB16' -> 15.5
    (Auburn wins by over 15.5). Kalshi suffixes are the strike rounded up to a whole point."""
    m = re.search(r"-[A-Z]*(\d+)$", str(ticker))
    return float(m.group(1)) - 0.5 if m else None


def market_of(ticker: str) -> str | None:
    t = str(ticker).split("-")[0]
    return "SPREAD" if t.endswith("SPREAD") else "TOTAL" if t.endswith("TOTAL") else None


# ---------- (a) SDs ----------
def outcome_sds(histories: dict[str, pd.DataFrame], now) -> dict:
    """{league: {market: (sd, games)}}: std of outcome - closing line over settled games played
    before `now` (CBB: final games only, see cbb_lines.final_games)."""
    now = pd.Timestamp(now)
    out = {}
    for lg, g in histories.items():
        if g is None or g.empty:
            continue
        g = g.dropna(subset=["home_score", "away_score"])
        g = g[pd.to_datetime(g["kickoff_utc"], utc=True, errors="coerce") < now]
        out[lg] = {}
        for mkt, line, y in (("SPREAD", g["spread_line"], g["home_score"] - g["away_score"]),
                             ("TOTAL", g["total_line"], g["home_score"] + g["away_score"])):
            r = (y - line).dropna()
            if len(r) >= 100:
                out[lg][mkt] = (float(r.std(ddof=1)), int(len(r)))
    return out


def caps_in_sd(sds: dict, caps: dict | None = None) -> dict:
    caps = caps or _caps()
    out = {}
    for lg, cap in caps.items():
        out[lg] = {}
        for mkt in ("SPREAD", "TOTAL"):
            sd, n = sds.get(lg, {}).get(mkt, (None, 0))
            out[lg][mkt] = {"cap_pts": cap, "sd": None if sd is None else round(sd, 2), "games": n,
                            "cap_sd": None if not sd else round(cap / sd, 3)}
    return out


# ---------- (b) calibration bias, walk-forward ----------
def band_bias(backtest: dict | None, market: str, entry_ts) -> dict:
    """{band: longshot gap (actual - predicted)} pooled over backtest seasons that ended before
    `entry_ts`. Empty when no season qualifies or the backtest predates #46."""
    kind = "spread" if market == "SPREAD" else "total"
    seasons = ((backtest or {}).get(kind) or {}).get("by_offset_season") or []
    t = pd.Timestamp(entry_ts)
    t = t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")
    acc: dict[str, list[float]] = {}
    for s in seasons:
        if pd.Timestamp(s["last_kickoff_utc"]) >= t:
            continue
        for b in s["bands"]:
            a = acc.setdefault(b["band"], [0.0, 0.0, 0.0])
            a[0] += b["rungs"]; a[1] += b["pred_sum"]; a[2] += b["hit_sum"]
    return {k: (v[2] - v[1]) / v[0] for k, v in acc.items() if v[0] > 0}


def adjust(fair: float, bias: float) -> float:
    """Fold like the backtest: longshot side moves by +bias, favourite side by -bias."""
    f = fair + bias if fair < 0.5 else fair - bias
    return float(min(max(f, 0.001), 0.999))


# ---------- per-bet table ----------
def _first_flags(flags: pd.DataFrame) -> pd.DataFrame:
    f = flags.copy()
    f["_ts"] = pd.to_datetime(f["snapshot_ts"], utc=True, errors="coerce")
    f["leg1"] = f["side"].astype(str) + " " + f["ticker"].astype(str)
    return f[f["_ts"].notna()].sort_values("_ts").drop_duplicates(["leg1", "snapshot_ts"])


def bet_table(bets: dict[str, pd.DataFrame], flags: dict[str, pd.DataFrame], clv: pd.DataFrame | None,
              sds: dict, backtests: dict, now, caps: dict | None = None) -> pd.DataFrame:
    caps = caps or _caps()
    now = pd.Timestamp(now)
    clv_map = {}
    if clv is not None and len(clv) and "bet_id" in clv:
        for r in clv.to_dict("records"):
            clv_map[(r["bet_id"], r["source"])] = (r.get("clv_c"), r.get("close_status"))
    rows = []
    for src, b in bets.items():
        if b is None or b.empty:
            continue
        b = b.copy()
        b["_ts"] = pd.to_datetime(b["snapshot_ts"], utc=True, errors="coerce")
        b = b[b["_ts"].notna() & (b["_ts"] <= now)]
        if src == "tracked" and "reason_bucket" in b:
            b = b[b["reason_bucket"] == "far_rung"]      # distance must be the only reason
        fl = flags.get(src)
        fmap = {}
        if fl is not None and len(fl):
            for r in _first_flags(fl).to_dict("records"):
                fmap[(r["leg1"], r["snapshot_ts"])] = r
        for r in b.to_dict("records"):
            leg1 = str(r["leg1"])
            side, ticker = leg1.split(" ", 1) if " " in leg1 else ("", leg1)
            mkt = market_of(ticker)
            if mkt is None:
                continue                                   # moneylines have no distance
            fr = fmap.get((leg1, r["snapshot_ts"]))
            line = None if fr is None else pd.to_numeric(fr.get("sharp_line"), errors="coerce")
            strike = strike_of(ticker)
            if line is None or pd.isna(line) or strike is None:
                continue
            lg = r["league"]
            dist = abs(strike - float(line))
            sd = sds.get(lg, {}).get(mkt, (None, 0))[0]
            band = _band(dist, PT_BANDS)
            bias = band_bias(backtests.get(lg), mkt, r["_ts"]).get(band)
            fair, price = float(r["fair"]), float(r["price"])
            edge = float(r["claimed_edge"])
            fair_adj = None if bias is None else adjust(fair, bias)
            edge_adj = None if fair_adj is None else edge + (fair_adj - fair)
            cv, cs = clv_map.get((r["bet_id"], src), (None, None))
            rows.append({"bet_id": r["bet_id"], "source": src, "league": lg, "market": mkt,
                         "event": _event(leg1), "entry_ts": r["snapshot_ts"], "side": side, "price": price,
                         "fair": fair, "strike": strike, "sharp_line": float(line), "dist_pts": round(dist, 2),
                         "sd": None if sd is None else round(sd, 3),
                         "dist_sd": None if not sd else round(dist / sd, 3), "band": band,
                         "inside_cap": dist <= caps.get(lg, 10.5) + 1e-9,
                         "bias": None if bias is None else round(bias, 5),
                         "fair_adj": None if fair_adj is None else round(fair_adj, 4),
                         "edge_raw": round(edge, 4), "edge_adj": None if edge_adj is None else round(edge_adj, 4),
                         "adj_qualifies": None if edge_adj is None else bool(edge_adj >= config.MIN_MODEL_EDGE - 1e-9),
                         "status": r.get("status"), "pnl": r.get("pnl"), "clv_c": cv, "close_status": cs})
    out = pd.DataFrame(rows, columns=COLS)
    # a contract graded both as shadow and as held back counts once (shadow wins)
    return out.sort_values("source", key=lambda s: s.map({"shadow": 0, "tracked": 1})).drop_duplicates("bet_id")


def _event(leg1: str) -> str:
    parts = str(leg1).split(" ", 1)[-1].split("-")
    return parts[1] if len(parts) > 1 else parts[0]


def _t(x: pd.Series):
    x = pd.Series(x).dropna().astype(float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return None
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2)


def _cell(g: pd.DataFrame) -> dict:
    ok = g[g["close_status"] == "ok"]
    v = pd.to_numeric(ok["clv_c"], errors="coerce")
    st = g[g["status"] == "settled"]
    pnl = pd.to_numeric(st["pnl"], errors="coerce")
    return {"plays": int(len(g)), "games": int(g["event"].nunique()),
            "clv_n": int(v.notna().sum()), "clv_games": int(ok["event"].nunique()),
            "avg_clv_c": None if v.dropna().empty else round(float(v.mean()), 2),
            "clv_t_by_game": _t(v.groupby(ok["event"]).mean()) if len(ok) else None,
            "settled": int(len(st)), "pnl": round(float(pnl.sum()), 2) if len(st) else 0.0,
            "pnl_t_by_game": _t(pnl.groupby(st["event"]).sum()) if len(st) else None}


def review(t: pd.DataFrame, sds: dict, caps: dict | None = None) -> dict:
    caps = caps or _caps()
    out = {"caps": caps_in_sd(sds, caps), "cells": {}, "by_sd": {}, "proposals": [],
           "rules": {"min_plays": MIN_PLAYS, "t": PROPOSE_T, "plays": "SHARP shadow + held back for distance only"}}
    if t is None or t.empty:
        return out
    for (lg, mkt), g in t.groupby(["league", "market"]):
        key = f"{lg} {mkt}"
        cap = caps.get(lg, 10.5)
        sd = sds.get(lg, {}).get(mkt, (None, 0))[0]
        cells = {}
        for lo, hi in PT_BANDS:
            name = f"{lo:g}-{hi:g}" if hi != math.inf else f"{lo:g}+"
            gb = g[g["band"] == name]
            if gb.empty:
                continue
            c = _cell(gb)
            c["sd_range"] = None if not sd else f"{lo / sd:.2f}-{(hi / sd) if hi != math.inf else math.inf:.2f}"
            c["inside_cap"] = hi <= cap + 1e-9
            adj = gb[gb["adj_qualifies"] == True]  # noqa: E712 - None means no bias available
            c["adjusted"] = {**_cell(adj), "dropped": int((gb["adj_qualifies"] == False).sum()),  # noqa: E712
                             "no_bias": int(gb["adj_qualifies"].isna().sum())} if len(adj) or len(gb) else {}
            cells[name] = c
            n, avg, tt = c["clv_n"], c["avg_clv_c"], c["clv_t_by_game"]
            if n >= MIN_PLAYS and avg is not None and tt is not None:
                if not c["inside_cap"] and lo >= cap - 1e-9 and avg > 0 and tt >= PROPOSE_T:
                    to = hi if hi != math.inf else math.ceil(float(gb["dist_pts"].max()) * 2) / 2
                    out["proposals"].append(f"PROPOSE widening {lg} {mkt.lower()} to {to:g} pts "
                                            f"(band {name}: {n} plays, CLV {avg:+.1f}c, t/game {tt:+.1f})")
                if c["inside_cap"] and avg < 0 and tt <= -PROPOSE_T:
                    out["proposals"].append(f"PROPOSE tightening {lg} {mkt.lower()} to {lo:g} pts "
                                            f"(band {name}: {n} plays, CLV {avg:+.1f}c, t/game {tt:+.1f})")
        out["cells"][key] = cells
        out["by_sd"][key] = {b: _cell(gs) for b, gs in g.groupby(g["dist_sd"].map(lambda x: _band(x, SD_BANDS)))}
        out["cells"][key]["_all"] = {"raw": _cell(g), "adjusted": _cell(g[g["adj_qualifies"] == True])}  # noqa: E712
    return out


def _load(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def histories() -> dict:
    from . import cbb_lines, sharp
    out = {}
    for lg in ("NFL", "CFB", "CBB"):
        try:
            g = sharp.history(lg)
            out[lg] = cbb_lines.final_games(g) if lg == "CBB" else g
        except Exception:  # noqa: BLE001
            out[lg] = pd.DataFrame()
    return out


def run(now=None) -> dict:
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    sds = outcome_sds(histories(), now)
    bts = {}
    for lg, p in BACKTESTS.items():
        try:
            bts[lg] = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            bts[lg] = {}
    t = bet_table({k: _load(p) for k, p in BET_FILES.items()}, {k: _load(p) for k, p in FLAG_FILES.items()},
                  _load(CLV_FILE), sds, bts, now)
    t.to_csv(OUT, index=False)
    rep = review(t, sds)
    print("caps: " + " | ".join(f"{lg} {m.lower()} {v['cap_pts']:g}pt={v['cap_sd']}SD"
                                for lg, d in rep["caps"].items() for m, v in d.items())
          + (" | " + "; ".join(rep["proposals"]) if rep["proposals"] else " | no proposals"))
    return rep


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str)[:4000])
