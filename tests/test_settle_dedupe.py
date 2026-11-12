"""#45: results.csv gets a row only when a ticker's (status, result) changes, so the log
doesn't grow by every pending ticker on every 15-min pass; grades stay the same."""
from datetime import datetime, timezone

import pandas as pd

from edge_desk import grade, settle


def _setup(tmp_path, monkeypatch, markets):
    flags = tmp_path / "sharp_tracked.csv"
    pd.DataFrame([{"snapshot_ts": "2026-11-12T01:00:00Z", "ticker": t, "side": "YES"}
                  for t in ("KXA-1", "KXB-2", "KXC-3")]).to_csv(flags, index=False)
    monkeypatch.setattr(settle, "FLAGS", tmp_path / "none.csv")
    monkeypatch.setattr(settle, "MODEL_FLAGS", (flags,))
    monkeypatch.setattr(settle, "RESULTS", tmp_path / "results.csv")
    state = {"markets": markets}
    monkeypatch.setattr(settle, "_get", lambda path, params: {"markets": state["markets"]})
    return state


def _t(h):
    return datetime(2026, 11, 12, h, 0, tzinfo=timezone.utc)


def test_unchanged_status_is_not_appended_again(tmp_path, monkeypatch):
    st = _setup(tmp_path, monkeypatch, [{"ticker": "KXA-1", "status": "active", "result": ""},
                                        {"ticker": "KXB-2", "status": "active", "result": ""}])
    settle.run(now=_t(2))
    settle.run(now=_t(3))
    settle.run(now=_t(4))
    r = pd.read_csv(settle.RESULTS)
    assert len(r) == 2                                # one row per ticker, not one per pass
    st["markets"] = [{"ticker": "KXA-1", "status": "finalized", "result": "yes"},
                     {"ticker": "KXB-2", "status": "active", "result": ""}]
    settle.run(now=_t(5))
    r = pd.read_csv(settle.RESULTS)
    assert len(r) == 3
    final = r[r["status"] == "finalized"].iloc[0]
    assert final["ticker"] == "KXA-1" and final["fetched_ts"] == "2026-11-12T05:00:00Z"
    settle.run(now=_t(6))                             # final tickers are no longer asked
    assert len(pd.read_csv(settle.RESULTS)) == 3


def test_grade_unchanged_by_dedupe(tmp_path, monkeypatch):
    """Same settlement time as the full log: the first final row is always written."""
    _setup(tmp_path, monkeypatch, [{"ticker": "KXA-1", "status": "active", "result": ""}])
    for h in (2, 3, 4):
        settle.run(now=_t(h))
    monkeypatch.setattr(settle, "_get", lambda p, q: {"markets": [
        {"ticker": "KXA-1", "status": "finalized", "result": "no"}]})
    settle.run(now=_t(5))
    r = pd.read_csv(settle.RESULTS)
    r["_ts"] = pd.to_datetime(r["fetched_ts"], utc=True)
    res, ts = grade._settlement(r, "KXA-1", pd.Timestamp("2026-11-12T01:00:00Z"))
    assert res == "no" and ts == pd.Timestamp("2026-11-12T05:00:00Z")


def test_missing_ticker_is_reported(tmp_path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch, [{"ticker": "KXA-1", "status": "active", "result": ""}])
    settle.run(now=_t(2))
    out = capsys.readouterr().out
    assert "KXB-2" in out and "KXC-3" in out and "not returned" in out
