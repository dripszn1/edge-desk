"""Kalshi links must open a real event page (series-only pages error on Kalshi)."""
import json

from edge_desk.alerts import kalshi_url, load_url_format


def test_link_is_event_page():
    u = kalshi_url("KXNFLSPREAD-26NOV12WASNYG-NYG2", "KXNFLSPREAD-26NOV12WASNYG")
    assert u == "https://kalshi.com/markets/kxnflspread/event/kxnflspread-26nov12wasnyg"


def test_stored_series_only_template_is_ignored(tmp_path):
    p = tmp_path / "url_format.json"
    p.write_text(json.dumps({"template": "https://kalshi.com/markets/{series}", "verified": False}))
    assert "{event}" in load_url_format(p)


def test_gone_alert_has_working_link_and_plain_words():
    from edge_desk.alerts import build_gone
    old = {"key": "SHARP|YES KXNCAAFSPREAD-26NOV14BCND-ND36", "kind": "SHARP", "side": "YES",
           "ticker": "KXNCAAFSPREAD-26NOV14BCND-ND36", "max_price": 0.32,
           "click": "https://kalshi.com/markets/kxncaafspread", "desc": "Notre Dame wins by over 35.5 points"}
    g = build_gone(old, 0.33, "2026-11-10T23:39:21Z")
    assert g["click"] == "https://kalshi.com/markets/kxncaafspread/event/kxncaafspread-26nov14bcnd"
    assert "Notre Dame wins by over 35.5 points" in g["body"]


def test_gone_sent_once_even_with_stale_state(tmp_path, monkeypatch):
    from edge_desk import alerts
    log = tmp_path / "log.csv"
    log.write_text("key,type\nSHARP|YES KXNCAAFSPREAD-26NOV14BCND-ND36,gone\n")
    monkeypatch.setattr(alerts, "LOG", log)
    state = {"alerts": {"SHARP|YES KXNCAAFSPREAD-26NOV14BCND-ND36": {
        "key": "SHARP|YES KXNCAAFSPREAD-26NOV14BCND-ND36", "kind": "SHARP", "side": "YES",
        "ticker": "KXNCAAFSPREAD-26NOV14BCND-ND36", "max_price": 0.32, "gone": False,
        "sent_utc": "2026-11-10T22:44:59Z", "kickoff_utc": "2026-11-14T20:30:00Z"}}}
    quotes = {"KXNCAAFSPREAD-26NOV14BCND-ND36": {"yes_ask": 0.33, "yes_bid": 0.31}}
    out = alerts.decide([], quotes, state, "2026-11-10T23:40:00Z")
    assert [a for a in out if a["type"] == "gone"] == []
