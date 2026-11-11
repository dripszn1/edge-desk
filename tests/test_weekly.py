"""Shadow MODEL grading + weekly skill-or-luck report (backlog #8)."""
import numpy as np
import pandas as pd
import pytest

from edge_desk import weekly
from edge_desk.fees import taker_fee
from edge_desk.grade import model_bets

SNAP = "2026-11-14T15:00:00Z"


def _flag(ticker="T1", side="YES", price=0.40, contracts=20, snap=SNAP):
    return dict(snapshot_ts=snap, league="NFL", kind="MODEL", market="SPREAD", event="E",
                leg1=f"{side} {ticker}", price=price, fair=0.5, edge=0.08, contracts=contracts,
                fee_multiplier=1.0, kickoff_ct="Sun")


def _res(ticker, result, ts, status="finalized"):
    return dict(fetched_ts=ts, ticker=ticker, status=status, result=result)


# --- grading shadow MODEL flags ---
def test_model_bet_settles_with_fee_at_size():
    b = model_bets(pd.DataFrame([_flag()]), pd.DataFrame([_res("T1", "yes", "2026-11-15T03:00:00Z")]),
                   now=pd.Timestamp("2026-11-15T04:00:00Z")).iloc[0]
    fee = taker_fee(0.40, 20)
    assert b.status == "settled" and b.fees == pytest.approx(fee)
    assert b.pnl == pytest.approx(20 - (0.40 * 20 + fee), abs=0.01)


def test_model_bet_ignores_results_from_before_the_flag():
    # a "final" row fetched BEFORE the flag's snapshot can't settle it (look-ahead)
    b = model_bets(pd.DataFrame([_flag()]), pd.DataFrame([_res("T1", "no", "2026-11-14T14:00:00Z")]),
                   now=pd.Timestamp("2026-11-15T04:00:00Z")).iloc[0]
    assert b.status == "pending" and pd.isna(b.pnl)


def test_model_bet_goes_stale_and_first_sighting_wins():
    f = pd.DataFrame([_flag(), _flag(price=0.30, snap="2026-11-14T16:00:00Z")])
    b = model_bets(f, pd.DataFrame(columns=["fetched_ts", "ticker", "status", "result"]),
                   now=pd.Timestamp("2026-11-30T00:00:00Z"))
    assert len(b) == 1 and b.iloc[0].price == 0.40 and b.iloc[0].status == "stale"


def test_void_refunds_principal_not_fee():
    b = model_bets(pd.DataFrame([_flag()]), pd.DataFrame([_res("T1", "void", "2026-11-15T03:00:00Z")]),
                   now=pd.Timestamp("2026-11-15T04:00:00Z")).iloc[0]
    assert b.pnl == pytest.approx(-taker_fee(0.40, 20), abs=0.01)


# --- skill or luck ---
def _settled(n, wins, price=0.5, contracts=10, kind="MODEL", weeks=1):
    rows = []
    for i in range(n):
        won = i < wins
        fee = taker_fee(price, contracts)
        stake = price * contracts + fee
        rows.append(dict(kind=kind, status="settled", price=price, contracts=contracts, stake=stake,
                         fees=fee, pnl=(contracts if won else 0) - stake, claimed_edge=0.05,
                         date_ct=f"2026-11-{2 + 7 * (i % weeks):02d}"))
    return pd.DataFrame(rows)


def test_not_enough_data_under_min_bets():
    assert weekly.assess(_settled(10, 9))["verdict"] == "not enough data"
    assert weekly.assess(pd.DataFrame())["verdict"] == "not enough data"


def test_fair_coin_record_is_luck():
    r = weekly.assess(_settled(60, 30), n_sims=2000)
    assert r["verdict"] == "luck / no edge"
    assert r["coin_flip"]["beat_pct"] < 0.95


def test_big_winning_record_is_skill():
    r = weekly.assess(_settled(60, 45), n_sims=2000)
    assert r["verdict"] == "skill" and r["t_stat"] >= 2 and r["coin_flip"]["beat_pct"] >= 0.95


def test_coin_flip_null_loses_the_fees():
    d = _settled(40, 20)
    sims = weekly.coin_flips(d.price.to_numpy(), d.contracts.to_numpy(float), d.stake.to_numpy(), 20000)
    assert np.mean(sims) == pytest.approx(-d.fees.sum(), abs=1.0)


def test_unsettled_and_structural_excluded_from_coin_flip():
    d = pd.concat([_settled(5, 5), _settled(3, 3, kind="LADDER"),
                   _settled(4, 0).assign(status="pending")])
    r = weekly.assess(d, n_sims=500)
    assert r["settled"] == 8 and r["coin_flip"]["bets"] == 5


def test_edge_by_week():
    r = weekly.assess(_settled(40, 20, weeks=2), n_sims=500)
    assert [w["week_of"] for w in r["by_week"]] == ["2026-11-02", "2026-11-09"]
    assert all("realised_edge_c" in w and "claimed_edge_c" in w for w in r["by_week"])


def test_sharp_scorecards_never_pool_cbb():
    rows = [dict(status="settled", pnl=1.0, stake=2.0, fees=0.1, league=lg, kind="SHARP", price=0.5, contracts=4,
                 date_ct="2026-11-14")
            for lg in ("NFL", "CFB", "CBB")]
    rep = weekly.sharp_split(pd.DataFrame(rows))
    assert rep["sharp_shadow"]["settled"] == 2 and rep["sharp_shadow_cbb"]["settled"] == 1
    assert weekly.sharp_split(pd.DataFrame())["sharp_shadow_cbb"]["settled"] == 0


def test_sharp_moneyline_breakdown():
    rows = [dict(status="settled", pnl=p, stake=2.0, fees=0.1, league="NFL", kind="SHARP", price=0.5, contracts=4,
                 date_ct="2026-11-14", market=m) for p, m in ((1.0, "ML"), (-1.0, "SPREAD"), (0.5, "ML"))]
    rep = weekly.sharp_split(pd.DataFrame(rows))["sharp_shadow"]["by_market"]
    assert rep["MONEYLINE"]["settled"] == 2 and rep["MONEYLINE"]["pnl"] == 1.5
    assert rep["SPREAD/TOTAL"]["settled"] == 1
