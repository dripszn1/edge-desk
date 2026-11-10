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
