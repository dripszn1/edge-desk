"""SPEC.md 'Tests that must pass before anything goes live'."""
from datetime import datetime, timedelta, timezone

import pandas as pd

from edge_desk import config
from edge_desk.fees import taker_fee, fee_per_contract
from edge_desk.sizing import bet_size, kelly_fraction, structural_size
from edge_desk.scan import scan, is_stale


def row(ticker, team, strike, bid, ask, bsize=100, asize=100, series="KXNFLSPREAD",
        league="NFL", stype="greater", event="EV1"):
    return dict(snapshot_ts="2026-11-09T20:00:00Z", league=league, series=series,
                event_ticker=event, event_title="A vs B", event_sub="", ticker=ticker,
                title=ticker, yes_sub_title=ticker, strike_type=stype, floor_strike=strike,
                cap_strike=None, team_key=team, yes_bid=bid, yes_ask=ask, yes_bid_size=bsize,
                yes_ask_size=asize, no_bid=1 - ask, no_ask=1 - bid, last_price=bid,
                volume=1, volume_24h=1, open_interest=1, status="active",
                occurrence_datetime=None, close_time=None, fee_multiplier=1.0,
                phase="pregame")  # tagged by kickoffs.tag in production


# --- fees on every leg ---
def test_fee_matches_kalshi_formula():
    assert taker_fee(0.50, 100) == 1.75          # 0.07*100*.25
    assert taker_fee(0.50, 1) == 0.02            # rounds UP to the cent
    assert fee_per_contract(0.99, 1) == 0.01


# --- sizing caps ---
def test_bet_never_exceeds_cap():
    assert bet_size(0.95, 0.10, 0.01) <= config.MAX_BET


def test_no_bets_after_daily_cap():
    assert bet_size(0.70, 0.50, 0.01, lost_today=config.DAILY_LOSS_CAP) == 0.0


def test_fees_shrink_kelly():
    assert kelly_fraction(0.60, 0.50, 0.01) < kelly_fraction(0.60, 0.50, 0.0)


def test_structural_size_capped():
    assert structural_size(0.95, 10_000)["stake"] <= config.MAX_BET


# --- scanner ---
def test_clean_ladder_no_flags():
    df = pd.DataFrame([row("A3", "A", 2.5, 0.60, 0.61), row("A7", "A", 6.5, 0.40, 0.41)])
    assert scan(df) == []


def test_broken_ladder_flags():
    # higher strike bid 0.70 > lower strike ask 0.55: buy YES lo @.55 + NO hi @.30 = .85 + fees
    df = pd.DataFrame([row("A3", "A", 2.5, 0.54, 0.55), row("A7", "A", 6.5, 0.70, 0.71)])
    f = scan(df)
    assert len(f) == 1 and f[0]["kind"] == "LADDER" and f[0]["edge_per_set"] > 0.10


def test_tiny_gap_eaten_by_fees():
    # 1c raw gap: fees on two legs kill it
    df = pd.DataFrame([row("A3", "A", 2.5, 0.49, 0.50), row("A7", "A", 6.5, 0.51, 0.52)])
    assert scan(df) == []


def test_exclusive_spread_flags():
    df = pd.DataFrame([row("A3", "A", 2.5, 0.60, 0.61), row("B3", "B", 2.5, 0.55, 0.56)])
    f = scan(df)
    assert any(x["kind"] == "EXCLUSIVE" for x in f)


def test_thin_size_ignored():
    df = pd.DataFrame([row("A3", "A", 2.5, 0.54, 0.55, asize=2), row("A7", "A", 6.5, 0.70, 0.71)])
    assert scan(df) == []


def test_cover_only_without_ties():
    nfl = pd.DataFrame([row("A", "A", None, 0.40, 0.42, series="KXNFLGAME", stype="", league="NFL"),
                        row("B", "B", None, 0.50, 0.52, series="KXNFLGAME", stype="", league="NFL")])
    assert not any(x["kind"] == "COVER" for x in scan(nfl))
    cbb = nfl.assign(series="KXNCAAMBGAME", league="CBB")
    assert any(x["kind"] == "COVER" for x in scan(cbb))


# --- stale data ---
def test_stale_snapshot_blocks_flags():
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert is_stale(old) and is_stale(None)
    fresh = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert not is_stale(fresh)
