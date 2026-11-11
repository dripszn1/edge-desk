"""#31 closing-price value: did the shadow/tracked entry beat Kalshi's own pregame close?"""
import pandas as pd

from edge_desk import clv

KO = "2026-11-14T18:00:00Z"            # kickoff
OCC = "2026-11-14T21:00:00Z"           # Kalshi occurrence_datetime = kickoff + 3h
T = "KXNFLSPREAD-26NOV14AAABBB-AAA3"


def snap(ts, yes_bid, yes_ask, ticker=T, occ=OCC):
    return {"snapshot_ts": ts, "ticker": ticker, "yes_bid": yes_bid, "yes_ask": yes_ask,
            "no_bid": round(1 - yes_ask, 2), "no_ask": round(1 - yes_bid, 2), "occurrence_datetime": occ}


def bet(side="YES", price=0.40, contracts=10, fees=0.17, ts="2026-11-12T12:00:00Z", source="shadow"):
    return {"bet_id": f"SHARP|{side} {T}", "snapshot_ts": ts, "league": "NFL", "leg1": f"{side} {T}",
            "contracts": contracts, "price": price, "fees": fees, "source": source}


NOW = pd.Timestamp("2026-11-15T00:00:00Z")


def run(bets, snaps, now=NOW):
    return clv.closing_value(pd.DataFrame(bets), pd.DataFrame(snaps), now=now)


def test_fee_counts_against_clv():
    # close mid 0.41 > entry 0.40, but the 1.7c/contract entry fee eats it: net CLV must be negative
    out = run([bet()], [snap("2026-11-12T12:00:00Z", 0.39, 0.40), snap("2026-11-14T17:50:00Z", 0.40, 0.42)])
    r = out.iloc[0]
    assert r["close_status"] == "ok"
    assert abs(r["close_mid"] - 0.41) < 1e-9
    assert abs(r["clv_raw_c"] - 1.0) < 1e-6
    assert abs(r["clv_c"] - (1.0 - 1.7)) < 1e-6
    assert not r["beat_close"]


def test_no_side_uses_no_quotes():
    out = run([bet(side="NO", price=0.55, fees=0.0)],
              [snap("2026-11-12T12:00:00Z", 0.44, 0.45), snap("2026-11-14T17:50:00Z", 0.30, 0.32)])
    # NO mid at close = (0.68 + 0.70) / 2 = 0.69
    assert abs(out.iloc[0]["close_mid"] - 0.69) < 1e-9
    assert out.iloc[0]["beat_close"]


def test_snapshot_after_kickoff_is_look_ahead():
    out = run([bet()], [snap("2026-11-14T17:50:00Z", 0.40, 0.42), snap("2026-11-14T18:05:00Z", 0.90, 0.92)])
    assert abs(out.iloc[0]["close_mid"] - 0.41) < 1e-9


def test_snapshot_from_the_future_is_ignored():
    now = pd.Timestamp("2026-11-14T17:00:00Z")      # before the 17:50 snapshot "exists"
    out = run([bet()], [snap("2026-11-14T16:30:00Z", 0.40, 0.42), snap("2026-11-14T17:50:00Z", 0.90, 0.92)], now=now)
    # kickoff hasn't happened yet at `now` -> no close yet, never the future quote
    assert out.iloc[0]["close_status"] == "pending"
    assert pd.isna(out.iloc[0]["close_mid"])


def test_stale_close_is_no_close():
    # last pregame quote 3h before kickoff: too old to call it the close
    out = run([bet()], [snap("2026-11-12T12:00:00Z", 0.39, 0.40), snap("2026-11-14T15:00:00Z", 0.40, 0.42)])
    assert out.iloc[0]["close_status"] == "stale"
    assert pd.isna(out.iloc[0]["clv_c"])


def test_close_must_be_after_entry():
    out = run([bet(ts="2026-11-14T17:50:00Z")], [snap("2026-11-14T17:50:00Z", 0.40, 0.42)])
    assert out.iloc[0]["close_status"] == "no_close"


def test_one_sided_or_crossed_quote_skipped():
    out = run([bet()], [snap("2026-11-14T17:40:00Z", 0.40, 0.42), snap("2026-11-14T17:55:00Z", 0.0, 0.42)])
    assert abs(out.iloc[0]["close_mid"] - 0.41) < 1e-9


def test_summary_groups_and_t():
    rows = []
    for i, p in enumerate([0.30, 0.31, 0.32, 0.33]):
        t = f"KXNFLSPREAD-26NOV14AAABBB-X{i}"
        rows.append({**bet(price=p, fees=0.0), "leg1": f"YES {t}", "bet_id": t})
    snaps = [snap("2026-11-14T17:50:00Z", 0.36, 0.38, ticker=f"KXNFLSPREAD-26NOV14AAABBB-X{i}") for i in range(4)]
    out = run(rows, snaps)
    s = clv.summary(out)
    g = s["shadow"]["football"]
    assert g["n"] == 4 and g["beat_close_pct"] == 100.0
    assert abs(g["avg_clv_c"] - 5.5) < 1e-6
    assert g["t_stat"] is not None and g["t_stat"] > 2
    # all four rungs are one game: the per-game t can't claim significance from one game
    assert g["events"] == 1 and g["t_by_event"] is None


def test_future_stamped_snapshot_cannot_move_kickoff_or_close():
    now = pd.Timestamp("2026-11-14T18:30:00Z")
    out = run([bet()], [snap("2026-11-14T17:50:00Z", 0.40, 0.42),
                        snap("2026-11-14T19:00:00Z", 0.89, 0.91, occ="2026-11-14T23:00:00Z")], now=now)
    r = out.iloc[0]
    assert r["close_status"] == "ok" and abs(r["close_mid"] - 0.41) < 1e-9
    assert r["kickoff_utc"] == KO
