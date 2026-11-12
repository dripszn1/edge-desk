"""#38 trade-out grading: fees on both legs, exits at the bid, no look-ahead, candles catch spikes."""
import pandas as pd
import pytest

from edge_desk import fees, trades

ENTRY = "2026-11-14T15:00:00Z"
OCC = "2026-11-14T21:00:00Z"          # kickoff = occurrence - 3h = 18:00Z
NOW = pd.Timestamp("2026-11-15T03:00:00Z")


def bet(side="NO", price=0.33, fair=0.38, n=10, status="settled", payout=0.0, **kw):
    fee = fees.taker_fee(price, n)
    return {"bet_id": "b1", "source": "shadow", "league": "CFB", "market": "SPREAD",
            "leg1": f"{side} KXNCAAFSPREAD-26NOV14MICHORE-ORE6", "snapshot_ts": ENTRY, "price": price,
            "fair": fair, "contracts": n, "fees": fee, "status": status, "payout": payout, **kw}


def snaps(*rows):
    """rows: (ts, yes_bid, yes_ask)."""
    return pd.DataFrame([{"snapshot_ts": t, "ticker": "KXNCAAFSPREAD-26NOV14MICHORE-ORE6", "yes_bid": yb,
                          "yes_ask": ya, "occurrence_datetime": OCC} for t, yb, ya in rows])


def by_variant(df):
    return {r["variant"]: r for r in df.to_dict("records")}


def test_no_side_bid_is_one_minus_yes_ask():
    assert trades.side_bid("NO", 0.60, 0.65) == 0.35
    assert trades.side_bid("YES", 0.60, 0.65) == 0.60
    assert trades.side_bid("NO", 0.60, None) is None


def test_exit_at_tip_sells_at_bid_and_pays_both_fees():
    s = snaps(("2026-11-14T16:00:00Z", 0.60, 0.65), ("2026-11-14T17:45:00Z", 0.58, 0.62))
    v = by_variant(trades.grade_all(pd.DataFrame([bet()]), s, NOW, candles_fn=lambda t: None))
    tip = v["EXIT_AT_TIP"]
    assert tip["exit_price"] == 0.38                         # NO bid = 1 - 0.62, the bid not the mid
    expect = 10 * (0.38 - 0.33) - fees.taker_fee(0.33, 10) - fees.taker_fee(0.38, 10)
    assert tip["pnl_usd"] == pytest.approx(expect)


def test_hold_uses_settlement_and_entry_fee_only():
    v = by_variant(trades.grade_all(pd.DataFrame([bet(payout=10.0)]), snaps(), NOW, candles_fn=lambda t: None))
    assert v["HOLD"]["pnl_usd"] == pytest.approx(10 * (1 - 0.33) - fees.taker_fee(0.33, 10))


def test_resting_fills_in_game_at_target_with_maker_fee():
    # pregame never reaches +10c, an in-game spike does (NO bid 0.45 = 1 - 0.55)
    s = snaps(("2026-11-14T17:00:00Z", 0.62, 0.66), ("2026-11-14T19:00:00Z", 0.50, 0.55))
    v = by_variant(trades.grade_all(pd.DataFrame([bet()]), s, NOW, candles_fn=lambda t: None))
    r = v["RESTING_10"]
    assert r["exit_kind"] == "auto_sell" and r["exit_price"] == pytest.approx(0.43)
    expect = 10 * 0.10 - fees.taker_fee(0.33, 10) - trades.maker_fee(0.43, 10)
    assert r["pnl_usd"] == pytest.approx(expect)
    assert v["RESTING_15"]["exit_kind"] == "settle"            # never reached -> held to the end
    # TRADE is pregame only: no pregame touch of fair -> sold at the tip
    assert v["TRADE"]["exit_kind"] == "tip"


def test_candle_spike_between_snapshots_counts_as_fill():
    s = snaps(("2026-11-14T16:00:00Z", 0.62, 0.66), ("2026-11-14T16:30:00Z", 0.62, 0.66))
    candles = pd.DataFrame([{"end_ts": "2026-11-14T16:12:00Z", "yes_bid_hi": 0.63, "yes_bid_lo": 0.50,
                             "yes_ask_hi": 0.66, "yes_ask_lo": 0.52}])   # NO bid high = 0.48
    t = trades.grade_all(pd.DataFrame([bet()]), s, NOW, candles_fn=lambda tk: candles)
    v = by_variant(t)
    assert v["RESTING_15"]["exit_kind"] == "auto_sell" and v["RESTING_15"]["used_candles"]
    no_c = by_variant(trades.grade_all(pd.DataFrame([bet()]), s, NOW, candles_fn=lambda tk: None))
    assert no_c["RESTING_15"]["exit_kind"] == "settle"


def test_no_lookahead_future_snapshots_and_candles_ignored():
    now = pd.Timestamp("2026-11-14T16:00:00Z")
    s = snaps(("2026-11-14T15:30:00Z", 0.62, 0.66), ("2026-11-14T16:30:00Z", 0.40, 0.45))  # 2nd is future
    candles = pd.DataFrame([{"end_ts": "2026-11-14T16:20:00Z", "yes_bid_hi": 0.4, "yes_bid_lo": 0.3,
                             "yes_ask_hi": 0.45, "yes_ask_lo": 0.40}])
    v = by_variant(trades.grade_all(pd.DataFrame([bet(status="pending", payout=None)]), s, now,
                                    candles_fn=lambda tk: candles))
    assert all(r["status"] == "pending" for k, r in v.items()), v
    # a quote at or before the entry never counts either
    s2 = snaps((ENTRY, 0.40, 0.45))
    v2 = by_variant(trades.grade_all(pd.DataFrame([bet(status="pending", payout=None)]), s2, NOW,
                                     candles_fn=lambda tk: None))
    assert v2["RESTING_5"]["status"] == "pending"


def test_bracket_stop_first_when_both_hit_in_one_candle_and_labels():
    candles = pd.DataFrame([{"end_ts": "2026-11-14T16:05:00Z", "yes_bid_hi": 0.75, "yes_bid_lo": 0.50,
                             "yes_ask_hi": 0.78, "yes_ask_lo": 0.52}])   # NO bid lo 0.22 (stop), hi 0.48 (goal)
    v = by_variant(trades.grade_all(pd.DataFrame([bet()]), snaps(), NOW, candles_fn=lambda tk: candles))
    assert v["BRACKET_10_10"]["exit_kind"] == "stop"
    assert v["BRACKET_10_10"]["label"] == "API-ONLY" and v["RESTING_10"]["label"] == "MANUAL-OK"


def test_summary_gate_only_for_manual_variants():
    rows = []
    for i in range(32):
        rows += [{"bet_id": f"b{i}", "league": "NFL", "leg1": f"NO KXNFLSPREAD-26NOV{i:02d}AAABBB-AAA3",
                  "variant": v, "label": trades.label(v), "status": "closed",
                  "pnl_usd": 1.0 + (i % 3) * 0.1, "pnl_c": 10} for v in ("RESTING_10", "BRACKET_10_5")]
    s = trades.summary(pd.DataFrame(rows))
    assert s["football"]["variants"]["RESTING_10"]["gate_open"] is True
    assert s["football"]["variants"]["BRACKET_10_5"]["gate_open"] is False
    assert s["football"]["best_manual"] == "RESTING_10"


def test_summary_same_denominator_ignores_early_fills_of_unsettled_bets():
    rows = [{"bet_id": "a", "league": "CBB", "leg1": "NO KXNCAAMBTOTAL-26NOV11AAABBB-140", "variant": "RESTING_10",
             "label": "MANUAL-OK", "status": "closed", "pnl_usd": 1.0, "pnl_c": 10},
            {"bet_id": "a", "league": "CBB", "leg1": "NO KXNCAAMBTOTAL-26NOV11AAABBB-140", "variant": "HOLD",
             "label": "MANUAL-OK", "status": "pending", "pnl_usd": None, "pnl_c": None}]
    s = trades.summary(pd.DataFrame(rows))
    r = s["cbb"]["variants"]["RESTING_10"]
    assert r["closed"] == 0 and r["pnl_usd"] == 0.0 and r["closed_early_not_counted"] == 1
