import pandas as pd

from edge_desk import alerts


def _log():
    return pd.DataFrame([
        {"key": "SHARP|YES KXA-1", "type": "new", "price": 0.50, "max_price": 0.50, "sent_utc": "2026-11-13T01:00:00Z"},
        {"key": "SHARP|YES KXA-1", "type": "gone", "price": 0.51, "max_price": 0.50, "sent_utc": "2026-11-13T01:00:30Z"},
        {"key": "SHARP|NO KXB-2", "type": "new", "price": 0.40, "max_price": 0.43, "sent_utc": "2026-11-13T01:00:00Z"},
        {"key": "SHARP|NO KXB-2", "type": "gone", "price": 0.44, "max_price": 0.43, "sent_utc": "2026-11-13T09:00:00Z"},
        {"key": "SHARP|YES KXC-3", "type": "new", "price": 0.30, "max_price": 0.30, "sent_utc": "2026-11-13T09:00:00Z"},
    ])


def test_room_buckets_gone_and_no_lookahead():
    clv = pd.DataFrame([{"leg1": "YES KXA-1", "source": "shadow", "close_status": "ok", "clv_c": -2.0},
                        {"leg1": "NO KXB-2", "source": "tracked", "close_status": "ok", "clv_c": 5.0}])
    r = alerts.room_report(_log(), clv, "2026-11-13T02:00:00Z")
    assert r["0c"]["alerts"] == 1                     # KXC-3 sent after now: ignored
    assert r["0c"]["gone_pct"] == 100.0 and r["0c"]["median_min_to_gone"] == 0.5
    assert r["0c"]["avg_clv_c"] == -2.0
    assert r["1c+"]["alerts"] == 1 and r["1c+"]["gone_pct"] == 0.0   # its gone is after now
    assert r["1c+"]["with_close"] == 0                # tracked CLV row is not an alert's


def test_room_report_empty():
    assert alerts.room_report(None, None, "2026-11-13T02:00:00Z") == {}

