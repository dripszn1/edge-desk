import pandas as pd

from edge_desk import stream_grade

NOW = "2026-11-16T12:00:00Z"


def fills():
    return pd.DataFrame([
        {"kind": "taker", "decided_utc": "2026-11-15T03:33:33.770Z", "ticker": "KXNFLSPREAD-26NOV19INDHOU-HOU15",
         "side": "NO", "price": 0.61, "fair": 0.68, "edge_c": 5.5, "contracts": 16, "latency_ms": 384},
        {"kind": "maker_quote", "decided_utc": "2026-11-15T03:33:33.770Z", "ticker": "KXNFLSPREAD-26NOV19INDHOU-HOU15",
         "side": "NO", "price": 0.60, "fair": 0.68, "edge_c": 5.5, "contracts": 16, "latency_ms": 384},
        {"kind": "taker", "decided_utc": "2026-11-17T00:00:00Z", "ticker": "KXNCAAFTOTAL-26NOV16AAABBB-50",
         "side": "YES", "price": 0.40, "fair": 0.45, "edge_c": 3.5, "contracts": 25, "latency_ms": 100},
    ])


def test_to_flags_only_takers_and_never_future():
    f = stream_grade.to_flags(fills(), NOW)
    assert list(f["leg1"]) == ["NO KXNFLSPREAD-26NOV19INDHOU-HOU15"]          # maker quote and future row dropped
    assert f.loc[0, "league"] == "NFL" and f.loc[0, "contracts"] == 16 and f.loc[0, "snapshot_ts"] == "2026-11-15T03:33:33Z"


def test_graded_like_sharp_bets_with_fee_and_settlement():
    from edge_desk import grade
    f = stream_grade.to_flags(fills(), NOW)
    res = pd.DataFrame([{"ticker": "KXNFLSPREAD-26NOV19INDHOU-HOU15", "status": "finalized", "result": "no",
                         "fetched_ts": "2026-11-20T06:00:00Z"}])
    b = grade.model_bets(f, res, "2026-11-21T00:00:00Z", source="stream", kind="STREAM")
    r = b.iloc[0]
    assert r["status"] == "settled" and r["source"] == "stream" and r["contracts"] == 16
    assert r["fees"] > 0 and abs(r["pnl"] - (16 - 16 * 0.61 - r["fees"])) < 0.02
    early = pd.DataFrame([{"ticker": "KXNFLSPREAD-26NOV19INDHOU-HOU15", "status": "finalized", "result": "no",
                           "fetched_ts": "2026-11-15T03:00:00Z"}])            # fetched before entry: not used
    assert grade.model_bets(f, early, "2026-11-21T00:00:00Z", source="stream", kind="STREAM").iloc[0]["status"] != "settled"


def test_league_of_and_digest_line_empty():
    assert stream_grade.league_of("KXNCAAFSPREAD-X") == "CFB" and stream_grade.league_of("FOO") is None
    assert stream_grade.digest_line({}) == []
