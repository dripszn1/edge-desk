import pandas as pd

from edge_desk import fees, maker

E = "2026-11-14T15:00:00Z"
KO = "2026-11-14T18:00:00Z"
NOW = "2026-11-15T03:00:00Z"


def snaps(rows, ticker="KXNCAAFSPREAD-26NOV14AAABBB-AAA3"):
    return pd.DataFrame([{"snapshot_ts": t, "ticker": ticker, "yes_bid": b, "yes_ask": a,
                          "occurrence_datetime": "2026-11-14T21:00:00Z"} for t, b, a in rows])


def bet(side="YES", price=0.50, status="pending", payout=None):
    return {"bet_id": "SHARP|x", "source": "shadow", "league": "CFB", "leg1": f"{side} KXNCAAFSPREAD-26NOV14AAABBB-AAA3",
            "snapshot_ts": E, "price": price, "contracts": 10, "fair": 0.56, "status": status, "payout": payout}


def test_no_fill_on_flag_quote_only_later():
    s = snaps([(E, 0.48, 0.50)])                           # the flag's own quote: ask 50 > limit 48
    rows = maker.grade_play(bet(), s, None, None, KO, 0.50, -1.0, NOW)
    assert {r["variant"]: r["status"] for r in rows} == {"MAKER_BID": "unfilled", "MAKER_1": "unfilled"}
    s2 = snaps([(E, 0.48, 0.48 + 0.02), ("2026-11-14T16:00:00Z", 0.47, 0.48)])
    r = {x["variant"]: x for x in maker.grade_play(bet(), s2, None, None, KO, 0.50, -1.0, NOW)}
    assert r["MAKER_BID"]["status"] == "filled" and r["MAKER_BID"]["limit"] == 0.48
    assert r["MAKER_BID"]["fee_c"] == round(100 * fees._round_up_cent(fees.MAKER_RATE * 10 * 0.48 * 0.52) / 10, 4)
    assert r["MAKER_BID"]["clv_c"] == round(100 * (0.50 - 0.48) - r["MAKER_BID"]["fee_c"], 4)
    assert r["MAKER_1"]["status"] == "filled" and r["MAKER_1"]["limit"] == 0.49   # ask 48 <= limit 49

def test_maker1_fills_when_ask_reaches_it():
    s = snaps([(E, 0.48, 0.50), ("2026-11-14T16:00:00Z", 0.48, 0.49)])
    r = {x["variant"]: x for x in maker.grade_play(bet(), s, None, None, KO, 0.50, -1.0, NOW)}
    assert r["MAKER_1"]["status"] == "filled" and r["MAKER_1"]["limit"] == 0.49
    assert r["MAKER_BID"]["status"] == "unfilled"


def test_no_fill_after_kickoff_or_after_now():
    s = snaps([(E, 0.48, 0.50), ("2026-11-14T18:30:00Z", 0.40, 0.45)])      # after kickoff
    assert all(x["status"] == "unfilled" for x in maker.grade_play(bet(), s, None, None, KO, 0.5, 0, NOW))
    s = snaps([(E, 0.48, 0.50), ("2026-11-14T17:00:00Z", 0.40, 0.45)])      # after "now"
    rows = maker.grade_play(bet(), s, None, None, KO, 0.5, 0, "2026-11-14T16:00:00Z")
    assert all(x["status"] == "open" for x in rows)


def test_maker1_skipped_when_it_would_take():
    assert maker.limits(0.49, 0.50) == {"MAKER_BID": 0.49}
    assert maker.limits(0.50, 0.50) == {}
    assert maker.limits(None, 0.50) == {}


def test_no_side_uses_one_minus_yes_bid():
    # NO side: bid = 1 - yes_ask, ask = 1 - yes_bid
    s = snaps([(E, 0.50, 0.52), ("2026-11-14T16:00:00Z", 0.53, 0.55)])       # NO ask 0.47 <= NO bid 0.48
    r = {x["variant"]: x for x in maker.grade_play(bet("NO", price=0.50), s, None, None, KO, 0.5, 0, NOW)}
    assert r["MAKER_BID"]["limit"] == 0.48 and r["MAKER_BID"]["status"] == "filled"


def test_filled_hold_uses_limit_and_maker_fee():
    s = snaps([(E, 0.48, 0.50), ("2026-11-14T16:00:00Z", 0.47, 0.48)])
    b = bet(status="settled", payout=10.0)
    r = {x["variant"]: x for x in maker.grade_play(b, s, None, None, KO, 0.5, 0, NOW)}["MAKER_BID"]
    fee = fees._round_up_cent(fees.MAKER_RATE * 10 * 0.48 * 0.52)
    assert r["HOLD_status"] == "closed"
    assert abs(r["HOLD_pnl_usd"] - round(10 * (1 - 0.48) - fee, 4)) < 1e-9


def test_summary_adverse_selection_split():
    m = pd.DataFrame([
        {"league": "CFB", "variant": "MAKER_BID", "leg1": "YES KXNCAAFSPREAD-26NOV14AAABBB-AAA3", "status": "filled",
         "clv_c": -1.0, "taker_clv_c": -3.0, **{f"{e}_status": "pending" for e in maker.EXITS}, **{f"{e}_pnl_usd": None for e in maker.EXITS}},
        {"league": "CFB", "variant": "MAKER_BID", "leg1": "YES KXNCAAFSPREAD-26NOV14CCCDDD-CCC3", "status": "unfilled",
         "clv_c": None, "taker_clv_c": 1.0, **{f"{e}_status": None for e in maker.EXITS}, **{f"{e}_pnl_usd": None for e in maker.EXITS}},
    ])
    b = maker.summary(m)["CFB"]["MAKER_BID"]
    assert b["fill_rate_pct"] == 50.0
    assert b["adverse_selection"]["taker_clv_if_filled"]["avg_c"] == -3.0
    assert b["adverse_selection"]["taker_clv_if_unfilled"]["avg_c"] == 1.0
