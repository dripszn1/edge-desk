"""#24: held-back SHARP plays are graded like shadow bets, broken down by reason."""
from datetime import datetime, timezone

import pandas as pd

from edge_desk import digest, grade, weekly

NOW = datetime(2026, 11, 12, 12, tzinfo=timezone.utc)
COLS = ["snapshot_ts", "league", "event", "ticker", "side", "price", "fair", "edge", "sharp_line",
        "odds_fetched_ts", "reason"]


def tracked(*rows):
    return pd.DataFrame([dict(zip(COLS, r)) for r in rows])


def results(*rows):
    return pd.DataFrame(rows, columns=["fetched_ts", "ticker", "status", "result"])


FAR = "rung far from sharp line (unvalidated range)"
STALE = "sharp line stale"


def test_first_sighting_only_sized_and_graded_with_fees():
    t = tracked(
        ("2026-11-11T01:00:00Z", "CBB", "A vs B: Spread", "KXNCAAMBSPREAD-X-A5", "YES", 0.40, 0.50, 0.08, 1, "", FAR),
        ("2026-11-11T01:00:30Z", "CBB", "A vs B: Spread", "KXNCAAMBSPREAD-X-A5", "YES", 0.30, 0.50, 0.18, 1, "", FAR),
    )
    r = results(("2026-11-11T05:00:00Z", "KXNCAAMBSPREAD-X-A5", "finalized", "yes"))
    b = grade.tracked_bets(t, r, NOW)
    assert len(b) == 1
    row = b.iloc[0]
    assert row["price"] == 0.40 and row["contracts"] >= 1 and row["fees"] > 0
    assert row["status"] == "settled" and row["pnl"] == round(row["contracts"] - row["stake"], 2)
    assert row["reason_bucket"] == "far_rung" and row["market"] == "SPREAD" and row["source"] == "tracked"


def test_no_lookahead_and_future_rows_ignored():
    t = tracked(
        ("2026-11-11T01:00:00Z", "NFL", "e", "KXNFLTOTAL-Y-45", "NO", 0.20, 0.30, 0.08, 40, "", STALE),
        ("2026-11-14T15:00:00Z", "CFB", "e", "KXNCAAFSPREAD-Z-A16", "YES", 0.12, 0.27, 0.14, 7, "", FAR),
    )
    r = results(("2026-11-11T00:59:00Z", "KXNFLTOTAL-Y-45", "finalized", "no"))   # before entry
    b = grade.tracked_bets(t, r, NOW)
    assert list(b["leg1"]) == ["NO KXNFLTOTAL-Y-45"]
    assert b.iloc[0]["status"] == "pending"


def test_reason_buckets():
    assert grade.reason_bucket(FAR + "; " + STALE) == "far_rung+stale_line"
    assert grade.reason_bucket("edge above sanity cap") == "sanity_cap"


def _settled(bucket, pnls):
    return pd.DataFrame([{"kind": "SHARP", "status": "settled", "stake": 5.0, "fees": 0.1, "pnl": p, "price": 0.4,
                          "contracts": 12, "claimed_edge": 0.05, "date_ct": "2026-11-11",
                          "settled_ts": "2026-11-11T05:00:00Z", "reason_bucket": bucket} for p in pnls])


def test_weekly_proposes_widening_only_for_a_proven_bucket():
    good = _settled("stale_line", [2.0, 1.5, 2.5] * 12)            # 36 settled, all wins
    bad = _settled("far_rung", [2.0, -5.0] * 20)
    rep = weekly.tracked_report(pd.concat([good, bad]), n_sims=200)
    assert rep["propose_widen"] == ["stale_line"]
    assert set(rep["by_reason"]) == {"stale_line", "far_rung"}


def test_digest_line_by_reason():
    lines = digest.held_back_by_reason(_settled("far_rung", [2.0, -5.0]), NOW)
    assert lines == ["  held back by reason: far_rung 1-1 $-3.00"]
