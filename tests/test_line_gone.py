"""#35: a "gone, don't chase" push when the SHARP LINE moves against an alerted play."""
import csv
from datetime import timedelta

from edge_desk import alerts, sharp
from tests.test_alerts import NOW, Post, sharp_flag

T = "KXNFLSPREAD-26NOV15LARARI-ARI3"


def flag():
    return dict(sharp_flag(), odds_fetched_ts="2026-11-14T15:50:00Z", sharp_line=-3.5)


def fe(fair, fetched="2026-11-14T16:20:00Z", stale=False, snap="2026-11-14T16:20:30Z", price=0.45, line=-2.0):
    return {f"{T}|YES": {"fair": fair, "price": price, "odds_fetched_ts": fetched, "snapshot_ts": snap,
                         "sharp_line": line, "stale": stale}}


def q(snap="2026-11-14T16:20:30Z"):
    return {T: {"yes_bid": 0.44, "yes_ask": 0.45, "snapshot_ts": snap}}


def _new(tmp_path):
    st, lg = tmp_path / "s.json", tmp_path / "l.csv"
    out = alerts.process([([flag()], False)], {}, NOW, post=Post(), state_path=st, log_path=lg, clock=lambda: NOW)
    assert [a["type"] for a in out] == ["new"]
    return st, lg


def test_moved_line_fires_once(tmp_path):
    st, lg = _new(tmp_path)
    later = NOW + timedelta(minutes=21)
    out = alerts.process([], q(), later, post=Post(), state_path=st, log_path=lg, fairs=fe(0.47), clock=lambda: later)
    assert [a["type"] for a in out] == ["gone"]
    b = out[0]["body"]
    assert "Line moved: fair now 47¢ (was 52¢)" in b and "-3.5 → -2" in b and "Skip it" in b
    assert "line moved" in out[0]["title"]
    row = [r for r in csv.DictReader(lg.open()) if r["type"] == "gone"][0]
    assert row["quote_utc"] == "2026-11-14T16:20:30Z"
    again = alerts.process([], q(), later + timedelta(minutes=5), post=Post(), state_path=st, log_path=lg,
                           fairs=fe(0.40, fetched="2026-11-14T16:25:00Z"))
    assert again == []                                     # max one gone per contract


def test_stale_line_never_fires(tmp_path):
    st, lg = _new(tmp_path)
    out = alerts.process([], q(), NOW + timedelta(minutes=21), post=Post(), state_path=st, log_path=lg,
                         fairs=fe(0.40, stale=True))
    assert out == []


def test_same_pull_never_fires(tmp_path):
    """The pull the alert was raised on (or an older one) can't say the line moved since."""
    st, lg = _new(tmp_path)
    for fetched in ("2026-11-14T15:50:00Z", "2026-11-14T15:00:00Z"):
        out = alerts.process([], q(), NOW + timedelta(minutes=1), post=Post(), state_path=st, log_path=lg,
                             fairs=fe(0.40, fetched=fetched))
        assert out == []


def test_small_move_still_clears_no_push(tmp_path):
    st, lg = _new(tmp_path)
    out = alerts.process([], q(), NOW + timedelta(minutes=21), post=Post(), state_path=st, log_path=lg,
                         fairs=fe(0.515))
    assert out == []


def test_fee_counts_in_line_moved(tmp_path):
    """fair - price = 3.5c clears MIN_MODEL_EDGE before the fee but not after it: gone."""
    st, lg = _new(tmp_path)
    out = alerts.process([], q(), NOW + timedelta(minutes=21), post=Post(), state_path=st, log_path=lg,
                         fairs=fe(0.485))
    assert [a["type"] for a in out] == ["gone"]


def test_no_line_gone_after_kickoff(tmp_path):
    st, lg = _new(tmp_path)
    out = alerts.process([], q("2026-11-15T21:30:00Z"), NOW + timedelta(days=2), post=Post(), state_path=st,
                         log_path=lg, fairs=fe(0.30, fetched="2026-11-15T21:20:00Z", snap="2026-11-15T21:30:00Z"))
    assert out == []


def test_fair_entry_marks_stale():
    import pandas as pd
    row = {"snapshot_ts": "2026-11-14T16:00:00Z", "kickoff_utc": "2026-11-15T21:05:00Z"}
    e = sharp.fair_entry(row, 0.512345, 0.45, pd.DataFrame({"fetched_ts": ["2026-11-14T15:00:00Z"]}), -3.0, True)
    assert e["stale"] is True and e["fair"] == 0.5123 and e["odds_fetched_ts"].startswith("2026-11-14T15:00")
