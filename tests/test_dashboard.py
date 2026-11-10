"""Backlog #5: data/dashboard.json."""
from datetime import datetime, timezone

import pandas as pd

from edge_desk import config
from edge_desk.dashboard import build
from edge_desk.grade import paper_bets, ledger
from tests.test_grade import flag, res

NOW = datetime(2026, 11, 16, 18, 0, tzinfo=timezone.utc)     # Mon Nov 16, 12:00 PM CT


def bets_and_ledger():
    fl = [flag(), flag(ts="2026-11-16T16:00:00Z", lo="X", hi="Y", kind="EXCLUSIVE"),
          flag(ts="2026-11-16T19:00:00Z", lo="F1", hi="F2")]          # after NOW: must not show
    fl[1]["league"] = "CBB"
    rs = [res("T-NE3", "yes"), res("T-NE7", "no"),
          res("X", "yes", ts="2026-11-16T17:00:00Z"), res("Y", "yes", ts="2026-11-16T17:00:00Z")]
    b = paper_bets(pd.DataFrame(fl), pd.DataFrame(rs), NOW)
    return b, ledger(b)


def latest(stale=False, phase="pregame"):
    return {"snapshot_ts": "2026-11-16T17:50:00Z", "stale": stale, "markets_scanned": 10,
            "flags": [{"kind": "LADDER", "phase": phase, "edge_per_set": 0.05}]}


def test_record_is_after_fees_and_splits_add_up():
    b, led = bets_and_ledger()
    d = build(latest(), b, led, NOW)
    settled = b[(b.status == "settled") & (pd.to_datetime(b.snapshot_ts, utc=True) <= NOW)]
    assert d["record"]["pnl"] == round(settled.pnl.sum(), 2)
    assert d["record"]["fees"] > 0 and d["record"]["settled"] == 2
    assert round(sum(r["pnl"] for r in d["by_league"].values()), 2) == d["record"]["pnl"]
    assert set(d["by_signal"]) == {"LADDER", "EXCLUSIVE"}
    assert d["risk"]["bankroll"] == round(config.BANKROLL + d["record"]["pnl"], 2)


def test_no_future_bets_on_dashboard():
    b, led = bets_and_ledger()
    d = build(latest(), b, led, NOW)
    assert d["record"]["bets"] == 2
    assert all("F1" not in x["leg1"] for x in d["today"]["bets"])


def test_stale_snapshot_shows_no_live_flags():
    b, led = bets_and_ledger()
    assert build(latest(stale=True), b, led, NOW)["flags_now"] == []
    assert len(build(latest(), b, led, NOW)["flags_now"]) == 1
    assert build(latest(phase="live"), b, led, NOW)["flags_now"] == []


def test_times_shown_in_central():
    b, led = bets_and_ledger()
    d = build(latest(), b, led, NOW)
    assert d["generated_ct"] == "Mon Nov 16, 12:00 PM CT" and d["today_ct"] == "2026-11-16"
    assert d["snapshot"]["ct"] == "Mon Nov 16, 11:50 AM CT"


def test_empty_history_builds():
    d = build({}, pd.DataFrame(), pd.DataFrame(), NOW)
    assert d["record"]["bets"] == 0 and d["snapshot"]["stale"] and d["risk"]["bankroll"] == config.BANKROLL


def test_snapshot_goes_stale_by_build_time():
    b, led = bets_and_ledger()
    later = datetime(2026, 11, 16, 20, 0, tzinfo=timezone.utc)   # 2h10m after the snapshot
    d = build(latest(), b, led, later)
    assert d["snapshot"]["stale"] and d["flags_now"] == []
