import pandas as pd

from edge_desk import grade, trades


def test_contract_in_both_files_counts_once_and_shadow_wins():
    df = pd.DataFrame([
        {"bet_id": "t1", "source": "tracked", "leg1": "NO KXNFLTOTAL-26NOV12WASNYG-33", "snapshot_ts": "2026-11-12T14:00:00Z"},
        {"bet_id": "s1", "source": "shadow", "leg1": "NO KXNFLTOTAL-26NOV12WASNYG-33", "snapshot_ts": "2026-11-12T14:46:00Z"},
        {"bet_id": "t2", "source": "tracked", "leg1": "NO KXNCAAFTOTAL-26NOV11SACCMU-27", "snapshot_ts": "2026-11-10T23:58:00Z"},
    ])
    out = grade.one_per_contract(df)
    assert sorted(out["bet_id"]) == ["s1", "t2"]


def test_trades_summary_counts_a_double_logged_contract_once():
    rows = []
    for bid, src in (("SHARP|NO KXNFLTOTAL-26NOV12WASNYG-33", "tracked"), ("SHARP|NO KXNFLTOTAL-26NOV12WASNYG-33", "shadow")):
        for v in ("HOLD", "EXIT_AT_TIP"):
            rows.append({"bet_id": bid, "source": src, "league": "NFL", "leg1": "NO KXNFLTOTAL-26NOV12WASNYG-33",
                         "variant": v, "label": "MANUAL-OK", "status": "closed", "pnl_usd": -2.0, "pnl_c": -22.0, "entry_ts": "2026-11-12T14:00:00Z"})
    s = trades.summary(pd.DataFrame(rows))
    assert s["football"]["variants"]["HOLD"]["closed"] == 1


def test_same_bet_id_in_both_sources_keeps_only_shadow():
    df = pd.DataFrame([
        {"bet_id": "SHARP|NO X-1", "source": "tracked", "leg1": "NO X-1", "snapshot_ts": "2026-11-12T14:00:00Z", "v": "HOLD"},
        {"bet_id": "SHARP|NO X-1", "source": "tracked", "leg1": "NO X-1", "snapshot_ts": "2026-11-12T14:00:00Z", "v": "TIP"},
        {"bet_id": "SHARP|NO X-1", "source": "shadow", "leg1": "NO X-1", "snapshot_ts": "2026-11-12T14:46:00Z", "v": "HOLD"},
        {"bet_id": "SHARP|NO X-1", "source": "shadow", "leg1": "NO X-1", "snapshot_ts": "2026-11-12T14:46:00Z", "v": "TIP"},
    ])
    out = grade.one_per_contract(df)
    assert list(out["source"]) == ["shadow", "shadow"]
