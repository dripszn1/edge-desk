"""Backlog #4: grader settles paper flags from Kalshi results, P&L after fees, daily ledger."""
import pandas as pd

from edge_desk import config
from edge_desk.fees import taker_fee
from edge_desk.grade import paper_bets, ledger
from edge_desk.settle import flag_tickers

NOW = "2026-11-16T12:00:00Z"


def flag(ts="2026-11-15T16:00:00Z", lo="T-NE3", hi="T-NE7", p1=0.55, p2=0.30, sets=11, kind="LADDER"):
    return dict(snapshot_ts=ts, league="NFL", kind=kind, event="NE vs DET", leg1=f"YES {lo}",
                leg2=f"NO {hi}", leg1_price=p1, leg2_price=p2, paper_sets=sets, fee_multiplier=1.0,
                kickoff_ct="Sun Nov 15, 12:00 PM CT")


def res(ticker, result, ts="2026-11-16T03:00:00Z", status="finalized"):
    return dict(fetched_ts=ts, ticker=ticker, status=status, result=result)


def grade(flags, results):
    return paper_bets(pd.DataFrame(flags), pd.DataFrame(results, columns=["fetched_ts", "ticker", "status", "result"]), NOW)


def test_settled_pnl_after_fees_on_both_legs():
    # NE wins by 5: YES >2.5 pays, NO >6.5 pays -> $2 per set
    b = grade([flag()], [res("T-NE3", "yes"), res("T-NE7", "no")]).iloc[0]
    fees = taker_fee(0.55, 11) + taker_fee(0.30, 11)
    assert b.status == "settled" and b.fees == round(fees, 2)
    assert b.pnl == round(2 * 11 - (0.85 * 11 + fees), 2)


def test_fees_recomputed_at_paper_size_not_per_contract_estimate():
    b = grade([flag(sets=3)], [res("T-NE3", "yes"), res("T-NE7", "yes")]).iloc[0]
    assert b.fees == round(taker_fee(0.55, 3) + taker_fee(0.30, 3), 2)  # rounded up per order
    assert b.pnl == round(3 * 1 - 0.85 * 3 - b.fees, 2)


def test_result_fetched_before_flag_is_ignored():
    early = [res("T-NE3", "yes", ts="2026-11-15T15:00:00Z"), res("T-NE7", "no", ts="2026-11-15T15:00:00Z")]
    assert grade([flag()], early).iloc[0].status == "pending"


def test_non_final_status_stays_pending():
    b = grade([flag()], [res("T-NE3", "yes", status="closed"), res("T-NE7", "no")]).iloc[0]
    assert b.status == "pending" and pd.isna(b.pnl)


def test_old_unsettled_flag_marked_stale():
    b = grade([flag(ts="2026-11-01T16:00:00Z")], []).iloc[0]
    assert b.status == "stale"


def test_repeat_sightings_are_one_bet_at_first_price():
    fl = [flag(ts="2026-11-15T16:30:00Z", p1=0.50), flag(ts="2026-11-15T16:00:00Z")]
    b = grade(fl, [])
    assert len(b) == 1 and b.iloc[0].snapshot_ts == "2026-11-15T16:00:00Z"


def test_void_refunds_principal_but_not_fees():
    b = grade([flag()], [res("T-NE3", "void"), res("T-NE7", "void")]).iloc[0]
    assert b.pnl == -b.fees


def test_daily_cap_skips_later_flags_same_day():
    # first bet loses ~$9.70 x 3 settled before the 4th flag -> 4th is skipped
    fl, rs = [], []
    for i in range(4):
        fl.append(flag(ts=f"2026-11-15T{14 + 2 * i}:00:00Z", lo=f"L{i}", hi=f"H{i}"))
        rs += [res(f"L{i}", "no", ts=f"2026-11-15T{15 + 2 * i}:00:00Z"),
               res(f"H{i}", "yes", ts=f"2026-11-15T{15 + 2 * i}:00:00Z")]
    b = grade(fl, rs).sort_values("snapshot_ts")
    assert list(b.status) == ["settled", "settled", "settled", "skipped_daily_cap"]
    assert b.iloc[3].pnl == 0.0


def test_drawdown_stop_pauses_everything_after():
    fl, rs = [], []
    for i in range(7):   # one losing bet per day, each ~-$9.7 -> drawdown passes $50 on day 6
        d = f"2026-11-{10 + i:02d}"
        fl.append(flag(ts=f"{d}T16:00:00Z", lo=f"L{i}", hi=f"H{i}"))
        rs += [res(f"L{i}", "no", ts=f"{d}T23:00:00Z"), res(f"H{i}", "yes", ts=f"{d}T23:00:00Z")]
    b = grade(fl, rs).sort_values("snapshot_ts")
    assert b.iloc[-1].status == "skipped_drawdown"
    assert (config.BANKROLL + b[b.status == "settled"].pnl.sum()) > config.BANKROLL - config.DRAWDOWN_STOP - config.MAX_BET


def test_ledger_daily_and_cumulative():
    fl = [flag(), flag(ts="2026-11-16T16:00:00Z", lo="X", hi="Y")]
    rs = [res("T-NE3", "yes"), res("T-NE7", "no"), res("X", "yes", ts=NOW), res("Y", "yes", ts=NOW)]
    led = ledger(grade(fl, rs))
    assert list(led.date_ct) == ["2026-11-15", "2026-11-16"]
    assert led.iloc[-1].cum_pnl == round(led.pnl.sum(), 2)
    assert led.iloc[-1].bankroll == round(config.BANKROLL + led.pnl.sum(), 2)


def test_settle_collects_both_leg_tickers():
    assert flag_tickers(pd.DataFrame([flag()])) == {"T-NE3", "T-NE7"}
