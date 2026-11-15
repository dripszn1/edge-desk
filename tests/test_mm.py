"""#67 market-making paper test on the Kalshi stream: quotes both sides around the sharp fair,
fills only when a LATER update trades through our price, maker fee on every fill, cancel at kickoff,
inventory cap, adverse selection, t by game. No look-ahead: a quote uses only fairs logged before it."""
from pathlib import Path

import pandas as pd
import pytest

from edge_desk import fees, mm, stream

T0 = 1_794_700_000_000
KO = "2026-11-15T18:00:00Z"
KO_MS = stream._utc_ms(KO)


def fair(f=0.50, pulled_ms=T0 - 60_000, ko=KO, stale=False, far=False):
    return {"fair": f, "stale": stale, "far": far, "kickoff_utc": ko, "odds_fetched_ts": stream._iso(pulled_ms)}


# ---------- engine logging ----------
def test_engine_logs_each_quote_once_and_fair_changes_only(tmp_path):
    now = [T0]
    e = stream.Engine(now_ms=lambda: now[0], directory=tmp_path)
    e.set_fairs({"X-1|YES": fair(0.50), "X-1|NO": fair(0.50)})
    e.set_fairs({"X-1|YES": fair(0.50), "X-1|NO": fair(0.50)})          # same pull: not logged again
    e.set_fairs({"X-1|YES": fair(0.52, pulled_ms=T0)})                    # new pull: logged
    e.set_fairs({"X-2|YES": fair(0.50, far=True)})                        # far rung: never logged
    e.on_quote("X-1", 0.45, 0.55, 10, 10, T0)
    e.flush()
    day = sorted(tmp_path.glob("quotes_*.csv"))[0]
    assert len(pd.read_csv(day)) == 1                                    # was 2 (once per side)
    f = pd.read_csv(sorted(tmp_path.glob("fairs_*.csv"))[0])
    assert list(f["fair"]) == [0.50, 0.52] and set(f["ticker"]) == {"X-1"}


# ---------- replay ----------
def Q(ms, yb, ya, t="KXNFLSPREAD-26NOV15NEDET-NE4"):
    return {"recv_ms": ms, "ts_ms": ms, "ticker": t, "yes_bid": yb, "yes_ask": ya, "bid_size": 100, "ask_size": 100}


def F(ms, f, t="KXNFLSPREAD-26NOV15NEDET-NE4", pulled=None, ko=KO, stale=False):
    return {"logged_ms": ms, "ticker": t, "fair": f, "odds_fetched_ts": stream._iso(pulled or ms),
            "stale": stale, "kickoff_utc": ko}


def run(quotes, fairs, k=2, **kw):
    return mm.simulate(pd.DataFrame(quotes), pd.DataFrame(fairs), k_c=k, **kw)


def test_quotes_both_sides_and_fills_only_when_traded_through_later():
    # fair 50 -> our YES bid 48, YES offer 52 (= buy NO at 48)
    fills = run([Q(T0 + 1, 0.47, 0.53), Q(T0 + 2, 0.46, 0.48),         # ask 48 = our bid: touch, not through
                 Q(T0 + 3, 0.45, 0.47),                                  # ask 47 < 48: our YES bid fills
                 Q(T0 + 4, 0.53, 0.55)],                                 # bid 53 > 52: our YES offer fills
                [F(T0, 0.50)])
    assert list(fills["side"]) == ["YES", "NO"]
    assert list(fills["price"]) == [0.48, 0.48]
    y = fills.iloc[0]
    assert y["fee"] == fees._round_up_cent(fees.MAKER_RATE * y["contracts"] * 0.48 * 0.52) and y["fee"] > 0


def test_no_look_ahead_fair_logged_after_the_quote_is_not_used():
    assert run([Q(T0 + 1, 0.40, 0.45)], [F(T0 + 5, 0.50)]).empty


def test_no_fill_on_the_quote_that_placed_it():
    # the first usable update already trades through the new price: it only places the quote
    fills = run([Q(T0 + 1, 0.40, 0.45), Q(T0 + 2, 0.40, 0.45)], [F(T0, 0.50)])
    assert len(fills) == 0
    fills = run([Q(T0 + 1, 0.40, 0.45), Q(T0 + 2, 0.40, 0.50), Q(T0 + 3, 0.40, 0.44)], [F(T0, 0.50)])
    assert len(fills) == 1                                               # placed once the book is clear, then traded through


def test_cancel_at_kickoff_and_on_stale_fair():
    ko = stream._iso(T0 + 3)
    assert run([Q(T0 + 1, 0.47, 0.53), Q(T0 + 5, 0.40, 0.45)], [F(T0, 0.50, ko=ko)]).empty
    old = T0 - 2 * 3600_000                                             # 2h-old pull
    assert run([Q(T0 + 1, 0.47, 0.53), Q(T0 + 2, 0.40, 0.45)], [F(T0, 0.50, pulled=old)]).empty


def test_inventory_cap_stops_buying_more_of_one_side():
    qs = [Q(T0 + 1, 0.47, 0.53)]
    for i in range(10):                                                  # ask dips through our bid 10 times
        qs += [Q(T0 + 10 + 2 * i, 0.46, 0.49), Q(T0 + 11 + 2 * i, 0.44, 0.46)]
    fills = run(qs, [F(T0, 0.50)], contracts=10, max_inventory=30)
    assert (fills["side"] == "YES").sum() == 3


def test_only_liquid_near_coin_flip_rungs():
    assert run([Q(T0 + 1, 0.04, 0.06), Q(T0 + 2, 0.01, 0.02)], [F(T0, 0.05)]).empty


def test_summary_pnl_pairs_and_marks_at_kickoff_with_t_by_game():
    fills = run([Q(T0 + 1, 0.47, 0.53), Q(T0 + 2, 0.45, 0.47), Q(T0 + 3, 0.53, 0.55),
                 Q(T0 + 4, 0.49, 0.51)], [F(T0, 0.50)], contracts=10)
    s = mm.summarize(fills, pd.DataFrame([Q(T0 + 1, 0.47, 0.53), Q(T0 + 4, 0.49, 0.51)]), results=None)
    nfl = s["by_k"]["2"]["NFL"]
    # bought YES 48 + NO 48 = 96c for a sure $1: +4c x 10 minus two maker fees
    exp = round(10 * 0.04 - 2 * fees._round_up_cent(fees.MAKER_RATE * 10 * 0.48 * 0.52), 2)
    assert nfl["fills"] == 2 and nfl["games"] == 1 and nfl["pnl_usd"] == pytest.approx(exp)
    assert "adverse_c" in nfl and "max_exposure_usd" in nfl


def test_adverse_selection_measures_mid_after_fill():
    fills = run([Q(T0 + 1, 0.47, 0.53), Q(T0 + 2, 0.45, 0.47), Q(T0 + 400_000, 0.30, 0.32)], [F(T0, 0.50)])
    a = mm.adverse(fills, pd.DataFrame([Q(T0 + 2, 0.45, 0.47), Q(T0 + 400_000, 0.30, 0.32)]))
    assert a.iloc[0] == pytest.approx(100 * (0.31 - 0.48))               # bought YES at 48, mid 5 min later 31


def test_open_inventory_settles_only_on_a_result_fetched_after_kickoff():
    fills = run([Q(T0 + 1, 0.47, 0.53), Q(T0 + 2, 0.45, 0.47)], [F(T0, 0.50)], contracts=10)
    t = fills.iloc[0]["ticker"]
    quotes = pd.DataFrame([Q(T0 + 2, 0.45, 0.47)])
    early = pd.DataFrame([{"ticker": t, "status": "finalized", "result": "yes", "fetched_ts": "2026-11-15T17:00:00Z"}])
    late = pd.DataFrame([{"ticker": t, "status": "finalized", "result": "yes", "fetched_ts": "2026-11-15T22:00:00Z"}])
    assert mm.summarize(fills, quotes, early)["by_k"]["2"]["NFL"]["settled_pnl_usd"] is None
    s = mm.summarize(fills, quotes, late)["by_k"]["2"]["NFL"]
    fee = fees._round_up_cent(fees.MAKER_RATE * 10 * 0.48 * 0.52)
    assert s["settled_pnl_usd"] == pytest.approx(round(10 * 0.52 - fee, 2))


def test_mm_never_touches_orders():
    src = (Path(mm.__file__)).read_text().lower()
    for bad in ("/portfolio", "/orders", "create_order", "balance"):
        assert bad not in src


def test_weekly_and_digest_carry_the_report():
    from datetime import datetime, timezone
    from edge_desk import digest
    s = {"by_k": {"2": {"NFL": {"fills": 4, "games": 2, "pnl_usd": 1.2, "t_by_game": 0.5, "adverse_c": -1.0}}}}
    w = {"generated_utc": "2026-11-15T00:00:00Z", "market_making": s}
    line = digest.stream_line(w, datetime(2026, 11, 15, 1, tzinfo=timezone.utc))
    assert line and "market-making paper" in line[-1] and "NFL ±2c 4 fills" in line[-1]
    assert "market_making" in (Path(mm.__file__).parent / "weekly.py").read_text()
