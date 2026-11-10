"""NFL rating model (backlog #6): fees, stale data, look-ahead, risk caps, walk-forward."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from edge_desk import backtest, config, model, nfl_results
from edge_desk.fees import fee_per_contract

SNAP = "2026-11-12T18:00:00Z"
NOW = pd.Timestamp(SNAP)
TEAMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
STRENGTH = {t: s for t, s in zip(TEAMS, [7, 5, 3, 1, -1, -3, -5, -7])}


def _games(n_weeks=12, fetched=SNAP, end="2026-11-09T18:00:00Z", seed=0):
    """Round-robin-ish season ending `end`, scores driven by STRENGTH, plus upcoming week."""
    rng = np.random.default_rng(seed)
    rows, end = [], pd.Timestamp(end)
    for w in range(n_weeks):
        ko = end - pd.Timedelta(days=7 * (n_weeks - 1 - w))
        order = list(rng.permutation(TEAMS))
        for i in range(0, 8, 2):
            h, a = order[i], order[i + 1]
            m = STRENGTH[h] - STRENGTH[a] + 2 + rng.normal(0, 3)
            rows.append(dict(game_id=f"{w}_{h}_{a}", season=2026, game_type="REG", week=w + 1,
                             kickoff_utc=ko.strftime("%Y-%m-%dT%H:%M:%SZ"), away_team=a, home_team=h,
                             away_score=round(22 - m / 2), home_score=round(22 + m / 2), neutral=0,
                             spread_line=np.nan, total_line=np.nan, fetched_ts=fetched))
    rows.append(dict(game_id="next_AAA_HHH", season=2026, game_type="REG", week=n_weeks + 1,
                     kickoff_utc="2026-11-15T18:00:00Z", away_team="HHH", home_team="AAA",
                     away_score=np.nan, home_score=np.nan, neutral=0, spread_line=np.nan,
                     total_line=np.nan, fetched_ts=fetched))
    return pd.DataFrame(rows)


def _row(ticker, series, strike, bid, ask, size=100, sub="Over", phase="pregame"):
    return dict(snapshot_ts=SNAP, league="NFL", series=series, event_ticker=ticker.rsplit("-", 1)[0],
                event_title="H vs A", event_sub="HHH vs AAA (Nov 15)", ticker=ticker,
                yes_sub_title=sub, strike_type="greater", floor_strike=strike, team_key="x",
                yes_bid=bid, yes_ask=ask, yes_bid_size=size, yes_ask_size=size, status="active",
                fee_multiplier=1.0, phase=phase, kickoff_utc="2026-11-15T18:00:00Z",
                kickoff_ct="Sun Nov 15, 12:00 PM CT")


def _market(**kw):
    # AAA (home, +7) vs HHH (-7): projected AAA margin ~ 16. Kalshi lists "away vs home".
    return pd.DataFrame([
        _row("KXNFLSPREAD-26NOV15HHHAAA-AAA3", "KXNFLSPREAD", 2.5, 0.40, 0.42, **kw),   # cheap: big edge
        _row("KXNFLSPREAD-26NOV15HHHAAA-AAA10", "KXNFLSPREAD", 9.5, 0.55, 0.57, **kw),
        _row("KXNFLTOTAL-26NOV15HHHAAA-44", "KXNFLTOTAL", 43.5, 0.50, 0.52, **kw),
    ])


def _fit():
    return model.fit(_games(), SNAP)


# --- the model learns something real ---
def test_ratings_recover_strength_order():
    r = _fit()
    net = {t: r.off[t] + r.dfn[t] for t in TEAMS}   # dfn > 0 = good defence
    assert sorted(net, key=net.get, reverse=True)[:2] == ["AAA", "BBB"]
    margin, total = r.project("AAA", "HHH")
    assert 8 < margin < 22 and 38 < total < 50


def test_p_greater_symmetry():
    assert model.p_greater(3, 13.5, 3) == pytest.approx(0.5)
    assert model.p_greater(10, 13.5, 2.5) + model.p_greater(-10, 13.5, -2.5) == pytest.approx(1)


# --- look-ahead ---
def test_future_and_unfinished_games_are_ignored():
    g = _games()
    base = model.fit(g, SNAP)
    # a blowout AFTER the snapshot, and one kicked off 1h before it (not final yet)
    extra = pd.DataFrame([
        dict(g.iloc[0]) | dict(game_id="fut", kickoff_utc="2026-11-13T00:00:00Z", home_team="HHH",
                               away_team="AAA", home_score=70, away_score=0),
        dict(g.iloc[0]) | dict(game_id="live", kickoff_utc="2026-11-12T17:00:00Z", home_team="HHH",
                               away_team="AAA", home_score=60, away_score=0)])
    after = model.fit(pd.concat([g, extra]), SNAP)
    assert after.off["HHH"] == pytest.approx(base.off["HHH"])


def test_results_fetched_after_snapshot_refused():
    g = _games(fetched="2026-11-12T18:30:00Z")
    flags, st = model.model_flags(_market(), g, SNAP, now=NOW)
    assert flags == [] and "look-ahead" in st["reason"]


def test_rows_fetched_after_as_of_not_trained_on():
    g = _games()
    g.loc[g.index[-5:-1], "fetched_ts"] = "2026-11-13T00:00:00Z"
    assert len(model.training_games(g, SNAP)) == len(g) - 1 - 4


def test_walk_forward_trains_only_on_the_past():
    g = pd.concat([_games(end="2025-12-28T18:00:00Z", seed=1).assign(season=2025),
                   _games(seed=2)]).dropna(subset=["home_score"])
    p = backtest.walk_forward(g, [2026])
    assert len(p) > 0
    assert (pd.to_datetime(p["train_max_kickoff"], utc=True) < pd.to_datetime(p["kickoff_utc"], utc=True)).all()


# --- stale data ---
def test_stale_results_no_flags():
    g = _games(fetched="2026-11-10T00:00:00Z")    # 66h old
    flags, st = model.model_flags(_market(), g, SNAP, now=NOW)
    assert flags == [] and "older" in st["reason"]


def test_lagging_scores_no_flags():
    g = _games()
    g.loc[g.index[-2], ["home_score", "away_score"]] = np.nan   # last week's game never scored
    flags, st = model.model_flags(_market(), g, SNAP, now=NOW)
    assert flags == [] and "lagging" in st["reason"]


def test_stale_snapshot_no_flags():
    flags, st = model.model_flags(_market(), _games(), SNAP, now=NOW + pd.Timedelta(hours=2))
    assert flags == []


def test_fresh_data_flags():
    flags, st = model.model_flags(_market(), _games(), SNAP, now=NOW)
    assert st["ok"] and flags
    f = next(x for x in flags if x["market"] == "SPREAD")
    assert f["ticker"].endswith("AAA3") and f["side"] == "YES" and f["kind"] == "MODEL"
    assert f["home"] == "AAA" and f["proj_margin_home"] > 8


# --- fees ---
def test_edge_is_after_fees_at_order_size():
    flags, _ = model.model_flags(_market(), _games(), SNAP, now=NOW)
    for f in flags:
        assert f["fee_per_contract"] == pytest.approx(fee_per_contract(f["price"], f["contracts"]), abs=1e-4)
        assert f["edge"] == pytest.approx(f["fair"] - f["price"] - f["fee_per_contract"], abs=2e-4)
        assert f["edge"] >= config.MIN_MODEL_EDGE


def test_fee_kills_thin_model_edge():
    # fair - price = 3.5c raw; the ~1.7c fee pushes it under MIN_MODEL_EDGE
    assert model.price_side(0.535, 0.50, 100, 1.0, config.DAILY_LOSS_CAP) is None
    assert model.price_side(0.60, 0.50, 100, 1.0, config.DAILY_LOSS_CAP) is not None


# --- risk ---
def test_one_flag_per_event_and_caps():
    flags, _ = model.model_flags(_market(), _games(), SNAP, now=NOW)
    evs = [f["event_ticker"] for f in flags]
    assert len(evs) == len(set(evs))
    assert all(f["stake"] <= config.MAX_BET for f in flags)
    assert sum(f["stake"] for f in flags) <= config.DAILY_LOSS_CAP


def test_no_flags_after_daily_cap():
    flags, _ = model.model_flags(_market(), _games(), SNAP, lost_today=config.DAILY_LOSS_CAP, now=NOW)
    assert flags == []


def test_live_and_thin_markets_skipped():
    assert model.model_flags(_market(phase="live"), _games(), SNAP, now=NOW)[0] == []
    assert model.model_flags(_market(size=2), _games(), SNAP, now=NOW)[0] == []


def test_gate_closed_without_winning_backtest(tmp_path):
    assert not model.gate(tmp_path / "missing.json")[0]
    p = tmp_path / "bt.json"
    p.write_text(json.dumps({"test": {"proxy": {"ALL": {"pnl_per_contract": -0.04, "t_stat": -4.0}}}}))
    assert not model.gate(p)[0]
    p.write_text(json.dumps({"test": {"proxy": {"ALL": {"pnl_per_contract": 0.01, "t_stat": 1.2}}}}))
    assert not model.gate(p)[0]
    p.write_text(json.dumps({"test": {"proxy": {"ALL": {"pnl_per_contract": 0.02, "t_stat": 2.5}}}}))
    assert model.gate(p)[0]


# --- results feed ---
def test_normalize_eastern_to_utc_and_codes():
    raw = pd.DataFrame([dict(game_id="x", season=2026, game_type="REG", week=1, gameday="2026-11-15",
                             gametime="13:00", away_team="OAK", home_team="JAX", away_score=1,
                             home_score=2, location="Neutral", spread_line=3.0, total_line=44.0)])
    g = nfl_results.normalize(raw, SNAP)
    assert g.kickoff_utc[0] == "2026-11-15T18:00:00Z"
    assert g.away_team[0] == "LV" and g.neutral[0] == 1
    assert model._pair("JAC vs LAR (Nov 15)") == ("JAX", "LA")
    assert model._rung_team("KXNFLSPREAD-26NOV15JACTEN-JAC7") == "JAX"


def test_closed_gate_routes_flags_to_shadow(tmp_path, monkeypatch):
    from edge_desk import scan
    monkeypatch.setattr(scan, "FLAG_DIR", tmp_path)
    monkeypatch.setitem(scan.RESULT_LOADERS, "NFL", lambda: _games())
    monkeypatch.setattr(scan, "lost_today", lambda: 0.0)
    monkeypatch.setattr(scan.model, "gate", lambda league="NFL": (False, "gated: test"))
    real = model.model_flags
    monkeypatch.setattr(scan.model, "model_flags", lambda df, g, ts, lost_today, league: real(df, g, ts, lost_today, now=NOW, league=league))
    m = scan.run_model(_market(), SNAP, leagues=("NFL",))
    assert m["live"] == [] and m["shadow"]
    assert (tmp_path / "model_shadow.csv").exists() and not (tmp_path / "model_flags.csv").exists()


def test_never_flag_above_90c():
    # Kalshi audit rule (CLAUDE.md #9): 90c+ buys were a top leak -> never flagged, even with edge
    assert model.price_side(0.99, 0.91, 100, 1.0, config.DAILY_LOSS_CAP) is None
    assert model.price_side(0.99, 0.90, 100, 1.0, config.DAILY_LOSS_CAP) is not None
    from edge_desk.scan import scan
    from tests.test_core import row
    # broken ladder whose NO leg costs 92c: would be an arb, but a leg is above 90c
    df = pd.DataFrame([row("A3", "A", 2.5, 0.02, 0.03), row("A7", "A", 6.5, 0.08, 0.09)])
    assert scan(df) == []
