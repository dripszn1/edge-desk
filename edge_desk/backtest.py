"""Walk-forward backtest of the rating model: NFL (backlog #6), CFB (#7).

For every week of every test season: fit ratings on games final before that week's first
kickoff (point-in-time: each result is treated as fetched at kickoff + FINAL_LAG), then predict
that week. Hyperparameters are chosen on TUNE seasons only and reported on TEST seasons.

Benchmarks
  - accuracy: model MAE vs the closing line's MAE (nflverse spread_line / total_line).
  - Kalshi proxy: there is no Kalshi price history yet, so each game's rungs are priced at the
    closing line's Normal probability + HALF_SPREAD (what a taker pays). The model "buys" the
    best rung per game per market when fair - price - fee >= MIN_MODEL_EDGE, exactly as live.
    Settled on the real final score. P&L is per contract, fees included.
Output: data/model/nfl_backtest.json
"""
from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from . import cfb_results, config, model, nfl_results
from .fees import fee_per_contract

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "model" / "nfl_backtest.json"
TUNE = range(2012, 2018)
TEST = range(2018, 2026)
HALF_SPREAD = 0.01
GRID_HL = (120.0, 240.0, 365.0)
GRID_RIDGE = (5.0, 10.0, 20.0)
PROXY_CONTRACTS = 10


def point_in_time(games: pd.DataFrame) -> pd.DataFrame:
    """Pretend each result was fetched FINAL_LAG after its kickoff."""
    g = games.copy()
    ko = pd.to_datetime(g["kickoff_utc"], utc=True)
    g["fetched_ts"] = (ko + model.FINAL_LAG).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return g


def walk_forward(games: pd.DataFrame, seasons, half_life_days=model.HALF_LIFE_DAYS,
                 ridge=model.RIDGE) -> pd.DataFrame:
    g = point_in_time(games.dropna(subset=["home_score", "away_score"]))
    rows = []
    for s in seasons:
        sg = g[g["season"] == s]
        for (_, wk), w in sg.groupby(["game_type", "week"]):
            as_of = pd.to_datetime(w["kickoff_utc"], utc=True).min() - pd.Timedelta(hours=1)
            r = model.fit(g, as_of, half_life_days, ridge)
            if r is None:
                continue
            train_max = model.training_games(g, as_of)["kickoff_utc"].max()
            for x in w.to_dict("records"):
                pr = r.project(x["home_team"], x["away_team"], bool(x["neutral"]))
                if pr is None:
                    continue
                rows.append({"game_id": x["game_id"], "season": s, "week": wk,
                             "kickoff_utc": x["kickoff_utc"], "train_max_kickoff": train_max,
                             "pred_margin": pr[0], "pred_total": pr[1],
                             "margin": x["home_score"] - x["away_score"],
                             "total": x["home_score"] + x["away_score"],
                             "spread_line": x["spread_line"], "total_line": x["total_line"]})
    return pd.DataFrame(rows)


def _proxy_bets(p: pd.DataFrame, sd_m: float, sd_t: float) -> pd.DataFrame:
    bets = []
    for x in p.dropna(subset=["spread_line", "total_line"]).to_dict("records"):
        for market, pred, line, actual, sd in (("SPREAD", x["pred_margin"], x["spread_line"], x["margin"], sd_m),
                                               ("TOTAL", x["pred_total"], x["total_line"], x["total"], sd_t)):
            best = None
            base = math.floor(line)
            for k in range(-10, 11):
                strike = base + k + 0.5
                fair_yes = model.p_greater(pred, sd, strike)
                mkt_yes = model.p_greater(line, sd, strike)
                for side, fair, price in (("YES", fair_yes, mkt_yes + HALF_SPREAD),
                                          ("NO", 1 - fair_yes, 1 - mkt_yes + HALF_SPREAD)):
                    price = round(price, 2)
                    if not 0.02 <= price <= 0.98:
                        continue
                    fee = fee_per_contract(price, PROXY_CONTRACTS)
                    edge = fair - price - fee
                    if edge >= config.MIN_MODEL_EDGE and (best is None or edge > best["edge"]):
                        won = (actual > strike) == (side == "YES")
                        best = {"season": x["season"], "week": x["week"], "market": market,
                                "edge": edge, "price": price, "fee": fee,
                                "pnl": (1.0 if won else 0.0) - price - fee, "won": won}
            if best:
                bets.append(best)
    return pd.DataFrame(bets)


def _t(x: pd.Series) -> float:
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 1 and x.std() > 0 else 0.0


def summarize(p: pd.DataFrame, sd_m: float, sd_t: float) -> dict:
    out = {"games": int(len(p)),
           "mae_margin_model": round(float((p["pred_margin"] - p["margin"]).abs().mean()), 3),
           "mae_total_model": round(float((p["pred_total"] - p["total"]).abs().mean()), 3)}
    c = p.dropna(subset=["spread_line", "total_line"])
    has = len(c) > 0     # CFB source has no closing lines -> no market benchmark
    out["mae_margin_closing_line"] = round(float((c["spread_line"] - c["margin"]).abs().mean()), 3) if has else None
    out["mae_total_closing_line"] = round(float((c["total_line"] - c["total"]).abs().mean()), 3) if has else None
    b = _proxy_bets(p, sd_m, sd_t)
    out["proxy"] = {}
    for m in ("SPREAD", "TOTAL", "ALL"):
        s = b if m == "ALL" else (b[b["market"] == m] if len(b) else b)
        out["proxy"][m] = {} if s.empty else {
            "bets": int(len(s)), "win_rate": round(float(s["won"].mean()), 4),
            "avg_price": round(float(s["price"].mean()), 4),
            "avg_claimed_edge": round(float(s["edge"].mean()), 4),
            "pnl_per_contract": round(float(s["pnl"].mean()), 4), "t_stat": round(_t(s["pnl"]), 2)}
    if len(b):
        out["proxy_by_season"] = {int(k): round(float(v), 4) for k, v in b.groupby("season")["pnl"].mean().items()}
    return out


def tune(games: pd.DataFrame, seasons=TUNE) -> dict:
    best = None
    for hl, rg in product(GRID_HL, GRID_RIDGE):
        p = walk_forward(games, seasons, hl, rg)
        score = (p["pred_margin"] - p["margin"]).abs().mean() + (p["pred_total"] - p["total"]).abs().mean()
        if best is None or score < best["score"]:
            sd_m = float((p["pred_margin"] - p["margin"]).std())
            sd_t = float((p["pred_total"] - p["total"]).std())
            best = {"half_life_days": hl, "ridge": rg, "score": float(score), "sd_margin": sd_m, "sd_total": sd_t}
    return best


# Per league: results loader, tune/test seasons, game types, output file, tune in Actions?
LEAGUE_RUNS = {
    "NFL": dict(load=nfl_results.load, tune=TUNE, test=TEST, types=["REG", "WC", "DIV", "CON", "SB"],
                do_tune=True),
    # CFB params were tuned offline on 2016-2021 only (HL 120-540 days x ridge 0.03-10; best
    # 240 / 0.1 on margin MAE + total MAE) and are fixed in model.py, so weekly runs stay fast.
    "CFB": dict(load=cfb_results.load, tune=range(2016, 2022), test=range(2022, 2026),
                types=["REG", "POST"], do_tune=False),
}


def run(games: pd.DataFrame | None = None, do_tune: bool | None = None, league: str = "NFL") -> dict:
    cfg, L = LEAGUE_RUNS[league], model.LEAGUES[league]
    games = cfg["load"]() if games is None else games
    games = games[games["game_type"].isin(cfg["types"])]
    do_tune = cfg["do_tune"] if do_tune is None else do_tune
    live = {"half_life_days": L.half_life_days, "ridge": L.ridge, "sd_margin": L.sd_margin, "sd_total": L.sd_total}
    params = tune(games, cfg["tune"]) if do_tune else dict(live)
    p = walk_forward(games, cfg["test"], params["half_life_days"], params["ridge"])
    from datetime import datetime, timezone
    res = {"league": league, "run_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "tuned_on": [cfg["tune"].start, cfg["tune"].stop - 1],
           "tested_on": [cfg["test"].start, cfg["test"].stop - 1],
           "params": {k: round(v, 3) for k, v in params.items()}, "live_params": live,
           "min_model_edge": config.MIN_MODEL_EDGE, "half_spread": HALF_SPREAD,
           "test": summarize(p, L.sd_margin, L.sd_total)}
    out = OUT.parent / L.backtest_file
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


def run_if_stale(league: str = "NFL", max_age_days: float = 7) -> dict | None:
    """Re-run at most once a week per league."""
    from datetime import datetime, timezone
    out = OUT.parent / model.LEAGUES[league].backtest_file
    try:   # file mtime is useless after a fresh checkout, so read the run time from the JSON
        prev = json.loads(out.read_text())
        age = datetime.now(timezone.utc) - datetime.fromisoformat(prev["run_ts"].replace("Z", "+00:00"))
        no_bench = prev.get("test", {}).get("mae_margin_closing_line") is None
        has_lines = LEAGUE_RUNS[league]["load"]()["spread_line"].notna().sum() >= 500
        if no_bench and has_lines:
            print(f"{league}: closing lines now available, re-running backtest")
        elif "test" in prev and age.total_seconds() < max_age_days * 86400:
            print(f"{league} backtest fresh, skip")
            return None
    except Exception:
        pass
    return run(league=league)


if __name__ == "__main__":
    import sys
    leagues = [a for a in sys.argv[1:] if a in LEAGUE_RUNS] or ["NFL"]
    for lg in leagues:
        try:
            run_if_stale(lg) if "--if-stale" in sys.argv else run(league=lg)
        except Exception as e:   # one league's failure must not block the other
            print(f"{lg} backtest failed: {e}")
