"""CBB removed 11-13: an alert sent before removal gets no fast-lane slot and no follow-up pushes."""
from datetime import datetime, timedelta, timezone

from edge_desk import alerts, live

NOW = datetime(2026, 11, 14, 1, 30, tzinfo=timezone.utc)


def _alert(ticker):
    return {"ticker": ticker, "key": ticker + "|YES", "side": "YES", "legs": None,
            "sent_utc": (NOW - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "quote_utc": (NOW - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "kickoff_utc": (NOW - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S"),
            "max_price": 0.40, "price": 0.40, "autosell": 0.45, "fair": 0.46, "contracts": 10,
            "gone": False, "kind": "SHARP", "game": "g", "desc": "d", "paper": True}


def test_in_scope():
    assert alerts.in_scope("KXNFLSPREAD-26NOV16X-Y3")
    assert alerts.in_scope("KXNCAAFTOTAL-26NOV14A-50")
    assert not alerts.in_scope("KXNCAAMBSPREAD-26NOV13DUKE-DUKE5")


def test_cbb_alert_not_fast_laned():
    st = {"alerts": {"a": _alert("KXNCAAMBSPREAD-26NOV13DUKE-DUKE5"),
                     "b": _alert("KXNCAAFSPREAD-26NOV13PSU-PSU6")}}
    assert live.fast_tickers(st, NOW) == ["KXNCAAFSPREAD-26NOV13PSU-PSU6"]


def test_cbb_alert_no_auto_sell_push(monkeypatch):
    monkeypatch.setattr(alerts, "_gone_keys_logged", lambda: set())
    cbb, cfb = "KXNCAAMBSPREAD-26NOV13DUKE-DUKE5", "KXNCAAFSPREAD-26NOV13PSU-PSU6"
    st = {"alerts": {cbb + "|YES": _alert(cbb), cfb + "|YES": _alert(cfb)}}
    q = {t: {"snapshot_ts": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"), "yes_bid": 0.60, "yes_ask": 0.62,
             "no_bid": 0.38, "no_ask": 0.40} for t in (cbb, cfb)}
    out = alerts.decide([], q, st, NOW)
    assert [a.get("ticker") for a in out] == [cfb]
    assert not st["alerts"][cbb + "|YES"].get("sell_hit")
