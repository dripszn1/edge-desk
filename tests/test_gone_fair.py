"""#50: price-gone pushes use the CURRENT sharp fair (while that pull is fresh), not the alert-time max."""
import csv
from datetime import timedelta

from edge_desk import alerts
from tests.test_alerts import NOW, Post, sharp_flag
from tests.test_line_gone import T, fe, flag


def _new(tmp_path):
    st, lg = tmp_path / "s.json", tmp_path / "l.csv"
    out = alerts.process([([flag()], False)], {}, NOW, post=Post(), state_path=st, log_path=lg, clock=lambda: NOW)
    assert [a["type"] for a in out] == ["new"]
    return st, lg, out[0]["max_price"]


def quote(ask, snap):
    return {T: {"yes_bid": round(ask - 0.01, 2), "yes_ask": ask, "snapshot_ts": snap}}


def test_line_moved_our_way_no_false_gone(tmp_path):
    st, lg, mp = _new(tmp_path)                           # fair 52c -> max ~46c
    later = NOW + timedelta(minutes=21)
    ask = round(mp + 0.02, 2)
    # fresh pull: fair rose to 58c, so the ask above the old max still clears fee + edge
    out = alerts.process([], quote(ask, "2026-11-14T16:20:30Z"), later, post=Post(), state_path=st, log_path=lg,
                         fairs=fe(0.58, price=ask), clock=lambda: later)
    assert out == []
    # fast lane (no fairs passed) a minute later: still uses the stored fresh fair
    t2 = later + timedelta(minutes=1)
    out = alerts.process([], quote(ask, "2026-11-14T16:21:30Z"), t2, post=Post(), state_path=st, log_path=lg,
                         clock=lambda: t2)
    assert out == []


def test_stored_fair_expires_when_stale(tmp_path):
    st, lg, mp = _new(tmp_path)
    later = NOW + timedelta(minutes=21)
    ask = round(mp + 0.02, 2)
    alerts.process([], quote(mp, "2026-11-14T16:20:30Z"), later, post=Post(), state_path=st, log_path=lg,
                   fairs=fe(0.58, price=mp), clock=lambda: later)
    # 2h after that pull (>90 min): the stored fair is stale -> back to the alert's own max -> gone
    t2 = NOW + timedelta(hours=2, minutes=30)
    out = alerts.process([], quote(ask, "2026-11-14T18:30:00Z"), t2, post=Post(), state_path=st, log_path=lg,
                         clock=lambda: t2)
    assert [a["type"] for a in out] == ["gone"]
    assert "don't chase" in out[0]["title"] and "line moved" not in out[0]["title"]


def test_stale_or_same_pull_fair_ignored(tmp_path):
    later = NOW + timedelta(minutes=21)
    for i, kw in enumerate(({"stale": True}, {"fetched": "2026-11-14T15:50:00Z"})):
        d = tmp_path / str(i)
        d.mkdir()
        st, lg, mp = _new(d)
        ask = round(mp + 0.02, 2)
        out = alerts.process([], quote(ask, "2026-11-14T16:20:30Z"), later, post=Post(), state_path=st,
                             log_path=lg, fairs=fe(0.58, price=ask, **kw), clock=lambda: later)
        assert [a["type"] for a in out] == ["gone"]          # old max applies


def test_over_both_maxes_is_price_gone_with_current_max(tmp_path):
    st, lg, mp = _new(tmp_path)
    later = NOW + timedelta(minutes=21)
    out = alerts.process([], quote(0.60, "2026-11-14T16:20:30Z"), later, post=Post(), state_path=st, log_path=lg,
                         fairs=fe(0.58, price=0.60), clock=lambda: later)
    out = [a for a in out if a["type"] != "sell"]           # bid 59c also hits the 52c Auto Sell: fine
    assert [a["type"] for a in out] == ["gone"]
    new_mp = alerts.max_price(0.58, flag()["contracts"])
    assert new_mp > mp
    assert f"past your {new_mp * 100:.0f}¢ max" in out[0]["body"]
    row = [r for r in csv.DictReader(lg.open()) if r["type"] == "gone"][0]
    assert abs(float(row["max_price"]) - new_mp) < 1e-9
    # never a second gone
    t2 = later + timedelta(minutes=1)
    assert [a for a in alerts.process([], quote(0.70, "2026-11-14T16:21:30Z"), t2, post=Post(), state_path=st,
                                      log_path=lg, clock=lambda: t2) if a["type"] == "gone"] == []


def test_fast_lane_uses_lower_current_fair(tmp_path):
    """Line moved against us but the price still cleared at that pass; a later fast-lane tick above the
    NEW max (but under the old one) is a line-moved gone, once."""
    st, lg, mp = _new(tmp_path)
    later = NOW + timedelta(minutes=21)
    new_mp = alerts.max_price(0.50, flag()["contracts"])
    assert new_mp < mp
    out = alerts.process([], quote(new_mp, "2026-11-14T16:20:30Z"), later, post=Post(), state_path=st, log_path=lg,
                         fairs=fe(0.50, price=new_mp), clock=lambda: later)
    assert out == []
    t2 = later + timedelta(minutes=1)
    out = alerts.process([], quote(round(new_mp + 0.01, 2), "2026-11-14T16:21:30Z"), t2, post=Post(),
                         state_path=st, log_path=lg, clock=lambda: t2)
    assert [a["type"] for a in out] == ["gone"] and "line moved" in out[0]["title"]


def test_future_pull_never_stored(tmp_path):
    """A fair entry stamped after the quote it is checked against is look-ahead: ignored."""
    st, lg, mp = _new(tmp_path)
    later = NOW + timedelta(minutes=21)
    ask = round(mp + 0.02, 2)
    out = alerts.process([], quote(ask, "2026-11-14T16:20:30Z"), later, post=Post(), state_path=st, log_path=lg,
                         fairs=fe(0.58, fetched="2026-11-14T16:40:00Z", price=ask), clock=lambda: later)
    assert [a["type"] for a in out] == ["gone"]
