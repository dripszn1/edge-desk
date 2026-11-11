"""SHARP moneylines (#23): fees, ties, stale lines, look-ahead, liquidity, own gate."""
import pandas as pd
import pytest

from edge_desk import alerts, config, moneyline, odds, sharp
from edge_desk.fees import fee_per_contract

SNAP = "2026-11-15T15:00:00Z"
KO = "2026-11-15T18:00:00Z"


def _odds(home_ml=-150, away_ml=130, ts="2026-11-15T14:30:00Z", home="Detroit Lions", away="New England Patriots",
          league="NFL"):
    pay = [{"id": "e1", "commence_time": KO, "home_team": home, "away_team": away, "bookmakers": [{"key": "pinnacle", "markets": [
        {"key": "h2h", "outcomes": [{"name": home, "price": home_ml}, {"name": away, "price": away_ml}]}]}]}]
    return odds.parse(pay, league, ts)


def _row(abbr="DET", bid=0.50, ask=0.52, size=500, league="NFL", series="KXNFLGAME", vol=10000.0,
         ev="KXNFLGAME-26NOV15NEDET", title="New England vs Detroit", sub="NE vs DET (Nov 15)"):
    return dict(snapshot_ts=SNAP, league=league, series=series, event_ticker=ev, event_title=title, event_sub=sub,
                ticker=f"{ev}-{abbr}", yes_sub_title=abbr, strike_type="structured", floor_strike=None,
                yes_bid=bid, yes_ask=ask, yes_bid_size=size, yes_ask_size=size, status="active",
                fee_multiplier=1.0, phase="pregame", kickoff_utc=KO, kickoff_ct="Sun 12:00 PM CT", volume_24h=vol)


def _p_home():
    f = odds.fair_lines(_odds(), SNAP)
    return float(f[(f.market == "h2h") & (f.outcome == "Detroit Lions")]["fair"].iloc[0])


def test_yes_flag_fee_inclusive_and_tie_adjusted():
    flags, st = moneyline.ml_flags(pd.DataFrame([_row()]), _odds(), SNAP, now=pd.Timestamp(SNAP))
    assert st["ok"] and len(flags) == 1
    f = flags[0]
    assert f["market"] == "ML" and f["side"] == "YES" and f["kind"] == "SHARP"
    assert f["fair"] == pytest.approx(_p_home() * (1 - moneyline.NFL_TIE_P), abs=1e-4)
    assert f["edge"] == pytest.approx(f["fair"] - 0.52 - fee_per_contract(0.52, f["contracts"]), abs=1e-4)


def test_tie_helps_no_hurts_yes():
    assert moneyline.fair_yes(0.6, "NFL") < 0.6 and moneyline.fair_yes(0.6, "CFB") == 0.6


def test_no_flag_when_edge_eaten_by_fee():
    p = _p_home() * (1 - moneyline.NFL_TIE_P)
    ask = round(p - 0.035, 2)                         # ~3c raw edge, fee pushes it under MIN_MODEL_EDGE
    flags, _ = moneyline.ml_flags(pd.DataFrame([_row(ask=ask, bid=ask - 0.02)]), _odds(), SNAP, now=pd.Timestamp(SNAP))
    assert all(f["edge"] >= config.MIN_MODEL_EDGE for f in flags)
    assert not any(f["side"] == "YES" for f in flags)


def test_odds_after_snapshot_refused():
    flags, st = moneyline.ml_flags(pd.DataFrame([_row()]), _odds(ts="2026-11-15T15:30:00Z"), SNAP, now=pd.Timestamp(SNAP))
    assert flags == [] and not st["ok"]


def test_stale_line_tracked_not_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp, "REJECTED", tmp_path / "t.csv")
    flags, st = moneyline.ml_flags(pd.DataFrame([_row()]), _odds(ts="2026-11-15T12:30:00Z"), SNAP, now=pd.Timestamp(SNAP))
    assert flags == [] and st.get("odds_too_old")
    assert "stale" in pd.read_csv(tmp_path / "t.csv")["reason"].iloc[0]


def test_stale_snapshot_no_flags():
    flags, st = moneyline.ml_flags(pd.DataFrame([_row()]), _odds(), SNAP, now=pd.Timestamp(SNAP) + pd.Timedelta(hours=2))
    assert flags == [] and "stale" in st["reason"]


def test_price_cap_and_one_per_game():
    rows = [_row(abbr="DET", bid=0.50, ask=0.52), _row(abbr="NE", bid=0.30, ask=0.32)]
    flags, _ = moneyline.ml_flags(pd.DataFrame(rows), _odds(), SNAP, now=pd.Timestamp(SNAP))
    assert len(flags) == 1
    big = _odds(home_ml=-2000, away_ml=1200)
    flags, _ = moneyline.ml_flags(pd.DataFrame([_row(bid=0.90, ask=0.91)]), big, SNAP, now=pd.Timestamp(SNAP))
    assert all(f["price"] <= config.MAX_FLAG_PRICE for f in flags)


def test_cbb_liquidity_rule():
    cbb = dict(league="CBB", series="KXNCAAMBGAME", ev="KXNCAAMBGAME-26NOV15DUKEKU", title="Duke vs Kansas",
               sub="DUKE vs KU (Nov 15)")
    o = _odds(home="Duke Blue Devils", away="Kansas Jayhawks", league="CBB")
    ok, _ = moneyline.ml_flags(pd.DataFrame([_row(abbr="DUKE", **cbb)]), o, SNAP, now=pd.Timestamp(SNAP), league="CBB")
    assert len(ok) == 1
    lo, st = moneyline.ml_flags(pd.DataFrame([_row(abbr="DUKE", vol=100.0, **cbb)]), o, SNAP, now=pd.Timestamp(SNAP), league="CBB")
    assert lo == [] and st["low_volume"] == 1
    thin, _ = moneyline.ml_flags(pd.DataFrame([_row(abbr="DUKE", size=10, **cbb)]), o, SNAP, now=pd.Timestamp(SNAP), league="CBB")
    assert thin == []


def test_ml_gate_separate_from_spreads(tmp_path):
    import numpy as np
    p = tmp_path / "b.csv"
    n = 40
    rng = np.random.default_rng(0)
    pnl = rng.normal(1.0, 0.5, n)
    pd.DataFrame({"status": "settled", "league": "NFL", "market": ["SPREAD"] * n, "pnl": pnl, "stake": 5.0}).to_csv(p, index=False)
    assert sharp.gate(p, league="NFL")[0]
    assert not sharp.gate(p, league="NFL", market="ML")[0]


def test_alert_title_ml():
    flags, _ = moneyline.ml_flags(pd.DataFrame([_row()]), _odds(), SNAP, now=pd.Timestamp(SNAP))
    assert alerts.build_new(flags[0], False, SNAP)["title"] == "PAPER SHARP NFL ML"
