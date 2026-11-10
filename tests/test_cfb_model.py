"""CFB extension of the rating model (backlog #7): feed, name matching, guards, gate."""
import json

import numpy as np
import pandas as pd
import pytest

from edge_desk import backtest, cfb_results, config, model
from edge_desk.fees import fee_per_contract

SNAP = "2026-11-12T18:00:00Z"
NOW = pd.Timestamp(SNAP)
SCHOOLS = ["Ohio State", "Miami", "Miami (OH)", "San José State", "UL Monroe", "Rice",
           "Tulane", "Air Force"]
STRENGTH = dict(zip(SCHOOLS, [20, 12, -4, 0, -10, -8, 6, 2]))


def _games(fetched=SNAP, weeks=12, end="2026-11-08T20:00:00Z", seed=0, fcs_home=False):
    rng = np.random.default_rng(seed)
    rows, end = [], pd.Timestamp(end)
    for w in range(weeks):
        ko = end - pd.Timedelta(days=7 * (weeks - 1 - w))
        order = list(rng.permutation(SCHOOLS))
        for i in range(0, 8, 2):
            h, a = order[i], order[i + 1]
            m = STRENGTH[h] - STRENGTH[a] + 3 + rng.normal(0, 4)
            rows.append(dict(game_id=f"{w}{i}", season=2026, game_type="REG", week=w + 1,
                             kickoff_utc=ko.strftime("%Y-%m-%dT%H:%M:%SZ"), away_team=a, home_team=h,
                             away_score=round(28 - m / 2), home_score=round(28 + m / 2), neutral=0,
                             spread_line=np.nan, total_line=np.nan, home_division="fbs",
                             away_division="fbs", fetched_ts=fetched))
    rows.append(dict(game_id="next", season=2026, game_type="REG", week=weeks + 1,
                     kickoff_utc="2026-11-14T20:00:00Z", away_team="Rice", home_team="Ohio State",
                     away_score=np.nan, home_score=np.nan, neutral=0, spread_line=np.nan,
                     total_line=np.nan, home_division="fcs" if fcs_home else "fbs",
                     away_division="fbs", fetched_ts=fetched))
    return pd.DataFrame(rows)


def _row(ticker, series, strike, bid, ask, size=100, phase="pregame"):
    return dict(snapshot_ts=SNAP, league="CFB", series=series, event_ticker=ticker.rsplit("-", 1)[0],
                event_title="Rice vs Ohio St.: Spread", event_sub="RICE vs OSU (Nov 14)", ticker=ticker,
                yes_sub_title=ticker, strike_type="greater", floor_strike=strike, team_key="x",
                yes_bid=bid, yes_ask=ask, yes_bid_size=size, yes_ask_size=size, status="active",
                fee_multiplier=1.0, phase=phase, kickoff_utc="2026-11-14T20:00:00Z",
                kickoff_ct="Sat Nov 14, 2:00 PM CT")


def _market(**kw):
    # Ohio State (home, +20) vs Rice (-8): projected OSU margin ~ 30.
    return pd.DataFrame([
        _row("KXNCAAFSPREAD-26NOV14RICEOSU-OSU14", "KXNCAAFSPREAD", 13.5, 0.40, 0.42, **kw),
        _row("KXNCAAFSPREAD-26NOV14RICEOSU-RICE7", "KXNCAAFSPREAD", 6.5, 0.30, 0.32, **kw),
        _row("KXNCAAFTOTAL-26NOV14RICEOSU-50", "KXNCAAFTOTAL", 49.5, 0.50, 0.52, **kw),
    ])


def _flags(games=None, **kw):
    return model.model_flags(_market(**kw.pop("mkt", {})), _games() if games is None else games,
                             SNAP, now=kw.pop("now", NOW), league="CFB", **kw)


# --- feed ---
def test_normalize_cfbd_schedule():
    raw = pd.DataFrame([
        dict(game_id=1, season=2026, week=12, season_type="regular", start_date="2026-11-14T20:00:00.000Z",
             start_time_tbd=False, completed=True, neutral_site=False, home_team="Rice", home_division="fbs",
             home_points=10, away_team="Tulane", away_division="fbs", away_points=31),
        dict(game_id=2, season=2026, week=12, season_type="regular", start_date="2026-11-14T05:00:00.000Z",
             start_time_tbd=True, completed=False, neutral_site=True, home_team="Army", home_division="fbs",
             home_points=0, away_team="Navy", away_division="fbs", away_points=0)])
    g = cfb_results.normalize(raw, SNAP)
    a, b = g.set_index("game_id").loc[1], g.set_index("game_id").loc[2]
    assert a.kickoff_utc == "2026-11-14T20:00:00Z" and a.home_score == 10 and a.neutral == 0
    assert b.neutral == 1 and pd.isna(b.home_score)            # open game: score not trusted
    assert b.kickoff_utc == "2026-11-15T01:00:00Z"              # TBD pushed to 8pm ET


# --- name matching ---
def test_match_kalshi_short_names_and_orientation():
    g = _games()
    m = model.resolve("CFB", g, _row("KXNCAAFSPREAD-26NOV14RICEOSU-OSU14", "KXNCAAFSPREAD", 13.5, .4, .42))
    assert m is not None
    game, sides = m
    assert game["game_id"] == "next" and sides == {"RICE": "Rice", "OSU": "Ohio State"}
    assert model._sim("San Jose St.", "San José State") >= model.CFB_MATCH_MIN
    assert model._sim("Louisiana-Monroe", "UL Monroe") == 1.0
    assert model._sim("Miami (FL)", "Miami") == 1.0 and model._sim("Miami (OH)", "Miami") < model.CFB_MATCH_MIN


def test_non_fbs_games_not_priced():
    flags, st = _flags(games=_games(fcs_home=True))
    assert flags == [] and st["events_unmatched"] >= 1


def test_spread_rung_uses_the_right_team():
    flags, st = _flags()
    assert st["ok"]
    sp = [f for f in flags if f["market"] == "SPREAD"]
    assert sp and sp[0]["home"] == "Ohio State" and sp[0]["proj_margin_home"] > 15
    # OSU by 13.5+ is cheap at .42 for a ~30-point favourite -> buy YES
    assert sp[0]["ticker"].endswith("OSU14") and sp[0]["side"] == "YES"


# --- look-ahead / stale / fees / risk (same guards as NFL) ---
def test_cfb_look_ahead_refused():
    flags, st = _flags(games=_games(fetched="2026-11-12T19:00:00Z"))
    assert flags == [] and "look-ahead" in st["reason"]


def test_cfb_stale_results_refused():
    flags, st = _flags(games=_games(fetched="2026-11-10T00:00:00Z"))
    assert flags == [] and "older" in st["reason"]


def test_cfb_fees_and_caps():
    flags, _ = _flags()
    assert flags
    for f in flags:
        assert f["fee_per_contract"] == pytest.approx(fee_per_contract(f["price"], f["contracts"]), abs=1e-4)
        assert f["edge"] >= config.MIN_MODEL_EDGE and f["stake"] <= config.MAX_BET
    assert sum(f["stake"] for f in flags) <= config.DAILY_LOSS_CAP
    assert _flags(lost_today=config.DAILY_LOSS_CAP)[0] == []
    assert _flags(mkt={"phase": "live"})[0] == []


def test_cfb_walk_forward_no_look_ahead():
    g = pd.concat([_games(end="2025-11-30T20:00:00Z", seed=1).assign(season=2025),
                   _games(seed=2)]).dropna(subset=["home_score"])
    p = backtest.walk_forward(g, [2026], model.CFB_HALF_LIFE_DAYS, model.CFB_RIDGE)
    assert len(p) > 0
    assert (pd.to_datetime(p["train_max_kickoff"], utc=True) < pd.to_datetime(p["kickoff_utc"], utc=True)).all()


def test_cfb_gate_closed_without_market_benchmark(tmp_path):
    # the CFB feed has no closing lines -> backtest proxy is empty -> gate stays closed
    p = tmp_path / "cfb.json"
    p.write_text(json.dumps({"test": {"proxy": {"SPREAD": {}, "TOTAL": {}, "ALL": {}}}}))
    ok, why = model.gate(p, league="CFB")
    assert not ok and "CFB" in why


def test_save_skips_unchanged_csv_but_records_fetch(tmp_path):
    from edge_desk import nfl_results
    path = tmp_path / "games.csv"
    g = _games(fetched="2026-11-11T00:00:00Z")
    assert nfl_results.save(g, path)
    assert not nfl_results.save(g.assign(fetched_ts="2026-11-12T00:00:00Z"), path)   # data same
    back = cfb_results.load(path)
    assert (back["fetched_ts"] == "2026-11-12T00:00:00Z").all() and len(back) == len(g)
    (tmp_path / "games_fetched.json").unlink()
    assert cfb_results.load(path).empty          # unknown fetch time -> no data, fails safe
