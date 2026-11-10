"""Line-move lag (backlog #12): implied line, move detection, look-ahead, fees in the money check."""
import numpy as np
import pandas as pd
import pytest

from edge_desk import lag, odds
from tests.test_sharp import _hist

KO = "2026-11-16T18:00:00Z"
HOME, AWAY = "New York Giants", "Washington Commanders"


def _pull(ts, home_pt, total=44.5):
    pay = [{"id": "ev1", "commence_time": KO, "home_team": HOME, "away_team": AWAY,
            "bookmakers": [{"key": "pinnacle", "markets": [
                {"key": "spreads", "outcomes": [{"name": HOME, "price": -110, "point": home_pt},
                                                 {"name": AWAY, "price": -110, "point": -home_pt}]},
                {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": total},
                                                {"name": "Under", "price": -110, "point": total}]}]}]}]
    return odds.parse(pay, "NFL", ts)


def _ladder(ts, home_line, sd=13.0):
    """Kalshi spread ladder for NYG (home) and WAS centred on `home_line` (expected NYG margin)."""
    from math import erfc, sqrt
    rows = []
    for team, sign in (("NYG", 1), ("WAS", -1)):
        for k in np.arange(0.5, 15, 1.0):
            p = 0.5 * erfc((k - sign * home_line) / (sd * sqrt(2)))
            mid = round(p, 2)
            rows.append(dict(snapshot_ts=ts, league="NFL", series="KXNFLSPREAD",
                             event_ticker="KXNFLSPREAD-26NOV16WASNYG", event_sub="WAS vs NYG (Nov 16)",
                             ticker=f"KXNFLSPREAD-26NOV16WASNYG-{team}{int(k + 0.5)}", strike_type="greater",
                             floor_strike=k, yes_bid=max(mid - 0.01, 0.01), yes_ask=min(mid + 0.01, 0.99),
                             yes_bid_size=100, yes_ask_size=100, fee_multiplier=1.0,
                             occurrence_datetime="2026-11-16T21:00:00Z", status="active"))
    return rows


def _snaps(seq):
    return pd.DataFrame([r for ts, L in seq for r in _ladder(ts, L)])


def test_implied_line_recovers_centre():
    df = pd.DataFrame(_ladder("2026-11-14T12:00:00Z", 3.0))
    assert lag.implied_line(df, "SPREAD", "NYG") == pytest.approx(3.0, abs=0.3)
    df = pd.DataFrame(_ladder("2026-11-14T12:00:00Z", -2.0))
    assert lag.implied_line(df, "SPREAD", "NYG") == pytest.approx(-2.0, abs=0.3)


def test_followed_with_lag():
    lines = pd.concat([_pull("2026-11-14T12:00:00Z", -3.0), _pull("2026-11-14T16:00:00Z", -5.0)])
    snaps = _snaps([("2026-11-14T11:30:00Z", 3.0), ("2026-11-14T16:00:00Z", 3.0),
                    ("2026-11-14T16:30:00Z", 3.2), ("2026-11-14T17:30:00Z", 4.8)])
    m = lag.measure(lines, snaps)
    sp = m[m.market == "SPREAD"].iloc[0]
    assert sp.status == "followed" and sp.lag_minutes == 90.0
    assert sp.sharp_from == 3.0 and sp.sharp_to == 5.0


def test_already_moved_is_zero_lag():
    lines = pd.concat([_pull("2026-11-14T12:00:00Z", -3.0), _pull("2026-11-14T16:00:00Z", -5.0)])
    snaps = _snaps([("2026-11-14T11:30:00Z", 3.0), ("2026-11-14T16:00:00Z", 4.9)])
    assert lag.measure(lines, snaps).iloc[0].status == "already_moved"


def test_small_move_ignored():
    lines = pd.concat([_pull("2026-11-14T12:00:00Z", -3.0), _pull("2026-11-14T16:00:00Z", -3.5)])
    assert lag.measure(lines, _snaps([("2026-11-14T16:00:00Z", 3.0)])).empty


def test_no_look_ahead_snapshots_before_detection_not_used_for_follow():
    """Kalshi moved at 14:00, before the sharp move was detected at 16:00 -> already_moved, not
    'followed' at a time we could not have known about; and 'before' uses only <= prev pull."""
    lines = pd.concat([_pull("2026-11-14T12:00:00Z", -3.0), _pull("2026-11-14T16:00:00Z", -5.0)])
    snaps = _snaps([("2026-11-14T11:00:00Z", 3.0), ("2026-11-14T14:00:00Z", 5.0), ("2026-11-14T16:10:00Z", 5.0)])
    r = lag.measure(lines, snaps).iloc[0]
    assert r.status == "already_moved" and r.kalshi_before == pytest.approx(3.0, abs=0.3)


def test_snapshots_after_kickoff_ignored():
    lines = pd.concat([_pull("2026-11-14T12:00:00Z", -3.0), _pull("2026-11-14T16:00:00Z", -5.0)])
    snaps = _snaps([("2026-11-14T16:00:00Z", 3.0), ("2026-11-16T19:00:00Z", 6.0)])   # in-game
    assert lag.measure(lines, snaps).iloc[0].status == "not_followed"


def test_best_edge_includes_fees():
    lines = pd.concat([_pull("2026-11-14T12:00:00Z", -3.0), _pull("2026-11-14T16:00:00Z", -6.0)])
    snaps = _snaps([("2026-11-14T16:00:00Z", 3.0)])
    r = lag.measure(lines, snaps, _hist()).iloc[0]
    assert r.best_edge_after_fee is not None
    # an un-moved ladder vs a 3-point sharp move must show a real edge, and less than the raw gap
    assert 0 < r.best_edge_after_fee < 0.25


def test_summary_needs_min_moves():
    assert "not enough data" in lag.summarize(pd.DataFrame(columns=lag.COLS))["verdict"]
