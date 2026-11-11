"""Phone alerts (#17), Kalshi links (#18), speed log (#19)."""
from datetime import datetime, timedelta, timezone

import pytest

from edge_desk import alerts, config
from edge_desk.fees import fee_per_contract

UTC = timezone.utc
NOW = datetime(2026, 11, 14, 16, 0, 0, tzinfo=UTC)


def sharp_flag(ticker="KXNFLSPREAD-26NOV15LARARI-ARI3", side="YES", price=0.45, fair=0.52, n=20, stake=9.0):
    fee = fee_per_contract(price, n)
    return {"snapshot_ts": "2026-11-14T15:59:50Z", "league": "NFL", "kind": "SHARP", "market": "SPREAD",
            "event": "Los Angeles R at Arizona: Spread", "event_ticker": "KXNFLSPREAD-26NOV15LARARI",
            "ticker": ticker, "leg1": f"{side} {ticker}", "leg1_desc": "Arizona wins by over 3.5 points",
            "side": side, "price": price, "fair": fair, "contracts": n, "fee_per_contract": fee,
            "edge": round(fair - price - fee, 4), "stake": stake, "fee_multiplier": 1.0,
            "kickoff_utc": "2026-11-15T21:05:00Z", "kickoff_ct": "Sun Nov 15, 3:05 PM CT"}


def ladder_flag():
    return {"snapshot_ts": "2026-11-14T15:59:50Z", "league": "CFB", "kind": "LADDER", "event": "A vs B",
            "event_ticker": "KXNCAAFSPREAD-26NOV14AB", "leg1": "YES KXNCAAFSPREAD-26NOV14AB-A3", "leg1_desc": "A by 3+",
            "leg1_price": 0.55, "leg2": "NO KXNCAAFSPREAD-26NOV14AB-A7", "leg2_desc": "A by 7+", "leg2_price": 0.30,
            "cost_per_set": 0.87, "edge_per_set": 0.13, "size_available": 100, "paper_sets": 11,
            "paper_stake": 9.57, "fee_multiplier": 1.0, "kickoff_utc": "2026-11-14T20:00:00Z",
            "kickoff_ct": "Sat Nov 14, 2:00 PM CT"}


class Post:
    def __init__(self):
        self.calls = []

    def __call__(self, url, **kw):
        self.calls.append((url, kw))
        return type("R", (), {"status_code": 200})()


# ---------- max price includes fees ----------
def test_max_price_clears_edge_after_fees():
    mp = alerts.max_price(0.52, 20)
    assert 0.52 - mp - fee_per_contract(mp, 20) >= config.MIN_MODEL_EDGE - 1e-9
    nxt = round(mp + 0.01, 2)
    assert 0.52 - nxt - fee_per_contract(nxt, 20) < config.MIN_MODEL_EDGE      # 1c higher fails
    assert mp < 0.52 - config.MIN_MODEL_EDGE                                   # fee is not ignored
    assert alerts.max_price(0.02, 5) is None


# ---------- message ----------
def test_new_alert_body_and_title():
    a = alerts.build_new(sharp_flag(), gate_open=False, now=NOW)
    assert a["title"].startswith("PAPER")
    b = a["body"]
    # mirrors the Kalshi ticket: event title, market title, then the side button
    assert b.startswith("Los Angeles R at Arizona: Spread\nArizona wins by over 3.5 points\nBuy Yes at 45¢")
    for s in ("pay up to", "Fair 52¢", "$9.00 = 20 contracts", "Sun Nov 15, 3:05 PM CT", "sent 10:00:00 CT"):
        assert s in b, s
    assert a["click"] == "https://kalshi.com/markets/kxnflspread/event/kxnflspread-26nov15larari"


def test_live_title_needs_open_gate_and_paper_mode_off(monkeypatch):
    assert alerts.build_new(sharp_flag(), True, NOW)["title"].startswith("PAPER")   # PAPER_MODE on
    monkeypatch.setattr(config, "PAPER_MODE", False)
    assert alerts.build_new(sharp_flag(), True, NOW)["title"].startswith("LIVE")
    assert alerts.build_new(sharp_flag(), False, NOW)["title"].startswith("PAPER")
    cbb = dict(sharp_flag(), league="CBB")
    assert alerts.build_new(cbb, True, NOW)["title"].startswith("PAPER SHARP CBB")   # CBB never LIVE


def test_priority_by_edge():
    small = sharp_flag(price=0.45, fair=0.495)
    big = sharp_flag(price=0.40, fair=0.52)
    assert alerts.build_new(small, False, NOW)["priority"] == 4
    assert alerts.build_new(big, False, NOW)["priority"] == 5


def test_no_side_reads_plainly():
    a = alerts.build_new(sharp_flag(side="NO"), False, NOW)
    assert "Arizona wins by over 3.5 points\nBuy No at " in a["body"]
    assert "KXNFL" not in a["body"]


def test_gone_without_stored_titles_uses_quote_row():
    old = {"key": "k", "kind": "SHARP", "ticker": "KXNCAAFSPREAD-26NOV14MICHORE-ORE2", "side": "NO",
           "max_price": 0.22, "event_ticker": "KXNCAAFSPREAD-26NOV14MICHORE"}
    row = {"event_title": "Michigan vs Oregon: Spread", "yes_sub_title": "Oregon wins by over 1.5 points"}
    g = alerts.build_gone(old, 0.23, NOW, row)
    assert g["body"].startswith("Michigan vs Oregon: Spread\nOregon wins by over 1.5 points\n"
                                "No is now 23¢, past your 22¢ max. Skip it.")
    assert "KXNCAAF" not in g["body"]


def test_structural_alert():
    a = alerts.build_new(ladder_flag(), False, NOW)
    assert "LADDER" in a["title"] and "Locked edge 13.0¢/set" in a["body"] and "11 sets" in a["body"]
    assert a["max_price"] < 1 - config.MIN_EDGE                               # fees on both legs


# ---------- dedupe / gone / cap ----------
def test_one_push_per_contract_then_gone_once(tmp_path):
    st, lg, post = tmp_path / "s.json", tmp_path / "log.csv", Post()
    f = sharp_flag()
    q = {f["ticker"]: {"yes_bid": 0.44, "yes_ask": 0.45}}
    a1 = alerts.process([([f], False)], q, NOW, post=post, state_path=st, log_path=lg, clock=lambda: NOW)
    a2 = alerts.process([([f], False)], q, NOW + timedelta(seconds=30), post=post, state_path=st, log_path=lg)
    assert [a["type"] for a in a1] == ["new"] and a2 == []
    moved = {f["ticker"]: {"yes_bid": 0.49, "yes_ask": 0.50}}                  # past max price
    a3 = alerts.process([([f], False)], moved, NOW + timedelta(minutes=1), post=post, state_path=st, log_path=lg)
    a4 = alerts.process([([f], False)], moved, NOW + timedelta(minutes=2), post=post, state_path=st, log_path=lg)
    assert [a["type"] for a in a3] == ["gone"] and "don't chase" in a3[0]["title"] and a4 == []


def test_gone_uses_no_side_price(tmp_path):
    st = tmp_path / "s.json"
    f = sharp_flag(side="NO", price=0.45)
    alerts.process([([f], False)], {}, NOW, post=Post(), state_path=st, log_path=tmp_path / "l.csv")
    # NO costs 1 - yes_bid: yes_bid .50 -> NO .50, past max
    out = alerts.process([([f], False)], {f["ticker"]: {"yes_bid": 0.50, "yes_ask": 0.51}}, NOW,
                         post=Post(), state_path=st, log_path=tmp_path / "l.csv")
    assert [a["type"] for a in out] == ["gone"]


def test_no_gone_after_kickoff(tmp_path):
    st = tmp_path / "s.json"
    f = sharp_flag()
    alerts.process([([f], False)], {}, NOW, post=Post(), state_path=st, log_path=tmp_path / "l.csv")
    later = datetime(2026, 11, 15, 22, 0, tzinfo=UTC)
    out = alerts.process([], {f["ticker"]: {"yes_bid": 0.9, "yes_ask": 0.91}}, later,
                         post=Post(), state_path=st, log_path=tmp_path / "l.csv")
    assert out == []


def test_daily_cap_across_pushes(tmp_path):
    st = tmp_path / "s.json"
    fs = [sharp_flag(ticker=f"KXNFLSPREAD-26NOV15G{i}-X3", stake=9.0) for i in range(3)]
    out = alerts.process([(fs, False)], {}, NOW, post=Post(), state_path=st, log_path=tmp_path / "l.csv")
    assert len(out) == 2                                     # 3 x $9 > $20 cap
    out = alerts.process([([sharp_flag(ticker="KXNFLSPREAD-26NOV15Z-X3", stake=9.0)], False)], {}, NOW,
                         post=Post(), state_path=st, log_path=tmp_path / "l.csv")
    assert out == []                                         # room used by earlier pushes today
    out = alerts.process([([sharp_flag(stake=1.0, ticker="KXNFLSPREAD-26NOV15Y-X3")], False)], {}, NOW,
                         lost_today=config.DAILY_LOSS_CAP, post=Post(), state_path=st, log_path=tmp_path / "l2.csv")
    assert out == []                                         # cap already lost


# ---------- sending ----------
def test_skip_cleanly_without_secrets(tmp_path, monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    monkeypatch.delenv("PUSHOVER_TOKEN", raising=False)
    post = Post()
    alerts.process([([sharp_flag()], False)], {}, NOW, post=post, state_path=tmp_path / "s.json",
                   log_path=tmp_path / "l.csv")
    assert post.calls == []
    assert "none" in (tmp_path / "l.csv").read_text()


def test_ntfy_headers_and_topic_never_logged(tmp_path, monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "secret-topic-xyz")
    monkeypatch.setenv("PUSHOVER_TOKEN", "tok")
    monkeypatch.setenv("PUSHOVER_USER", "usr")
    post = Post()
    alerts.process([([sharp_flag(price=0.40)], False)], {}, NOW, post=post, state_path=tmp_path / "s.json",
                   log_path=tmp_path / "l.csv")
    url, kw = post.calls[0]
    assert url == "https://ntfy.sh/secret-topic-xyz"
    assert kw["headers"]["Priority"] == "5" and kw["headers"]["Click"].startswith("https://kalshi.com/markets/")
    assert post.calls[1][0].startswith("https://api.pushover.net/")
    for p in (tmp_path / "s.json", tmp_path / "l.csv"):
        assert "secret-topic-xyz" not in p.read_text() and "tok" not in p.read_text().split("ticker")[0]


def test_speed_log_and_median(tmp_path):
    lg = tmp_path / "l.csv"
    t = iter([NOW + timedelta(seconds=2), NOW + timedelta(seconds=4)])
    fs = [sharp_flag(ticker="KXNFLSPREAD-26NOV15A-X3", stake=1), sharp_flag(ticker="KXNFLSPREAD-26NOV15B-X3", stake=1)]
    alerts.process([(fs, False)], {}, NOW, post=Post(), state_path=tmp_path / "s.json", log_path=lg,
                   clock=lambda: next(t))
    txt = lg.read_text()
    assert "2026-11-14T15:59:50Z" in txt                     # quote time kept
    assert alerts.median_latency(lg) == 3.0


# ---------- links ----------
def test_kalshi_url_and_verify():
    assert alerts.kalshi_url("KXNCAAFTOTAL-26NOV14OSUMICH-51", "KXNCAAFTOTAL-26NOV14OSUMICH",
                             alerts.URL_TEMPLATES[0]) == "https://kalshi.com/markets/kxncaaftotal/event/kxncaaftotal-26nov14osumich"

    def get_ok(u):
        ok = u == "https://kalshi.com/markets/kxnflgame/event/kxnflgame-26nov15larari"
        return type("R", (), {"status_code": 200 if ok else 404, "url": u, "text": "kxnflgame-26nov15larari" if ok else ""})()
    r = alerts.verify_url("KXNFLGAME-26NOV15LARARI-ARI", "KXNFLGAME-26NOV15LARARI", get_ok)
    assert r["verified"] and r["template"] == alerts.URL_TEMPLATES[0]

    def get_spa(u):     # 200 for everything but never mentions the event -> not trusted
        return type("R", (), {"status_code": 200, "url": u, "text": "<html>app</html>"})()
    r = alerts.verify_url("KXNFLGAME-26NOV15LARARI-ARI", "KXNFLGAME-26NOV15LARARI", get_spa)
    assert not r["verified"] and r["template"] == alerts.FALLBACK_TEMPLATE


# ---------- live loop pushes in the same iteration ----------
def test_live_loop_alerts_same_pass(monkeypatch, tmp_path):
    from tests import test_live
    calls = []
    s = test_live._run(monkeypatch, tmp_path, datetime(2026, 11, 14, 16, 0, tzinfo=UTC), 2, test_live._pass_frame)
    assert s.stats["full_passes"] > 0
    from edge_desk import live
    c = test_live.FakeClock(datetime(2026, 11, 14, 16, 0, tzinfo=UTC))
    monkeypatch.setattr(live, "deadline", lambda *a, **k: c.t + timedelta(minutes=1))
    live.run_loop(now_fn=c.now, clock=c.clock, sleep=c.sleep,
                  fetch_full=lambda m, ts: (test_live._pass_frame(ts), [], False), odds_fn=lambda: None,
                  scan_fn=lambda df, ts, seen, partial: None, persist_fn=lambda *a: None, post_fn=lambda: [],
                  commit_fn=lambda m: False, used_min=0, seen=set(),
                  alert_fn=lambda df, ts: calls.append((ts, set(df["snapshot_ts"]))) or [])
    assert calls and all(stamps == {ts} for ts, stamps in calls)


# ---------- #18: link everywhere ----------
def test_links_in_flags_digest_dashboard(monkeypatch, tmp_path):
    import pandas as pd
    from edge_desk import dashboard, digest, scan
    from tests.test_core import row
    monkeypatch.setattr(alerts, "URL_FORMAT", tmp_path / "none.json")
    df = pd.DataFrame([row("KXNFLSPREAD-26NOV15AB-A3", "A", 2.5, 0.54, 0.55, event="KXNFLSPREAD-26NOV15AB"),
                       row("KXNFLSPREAD-26NOV15AB-A7", "A", 6.5, 0.70, 0.71, event="KXNFLSPREAD-26NOV15AB")])
    f = scan.with_url(scan.scan(df))
    assert f[0]["kalshi_url"] == "https://kalshi.com/markets/kxnflspread/event/kxnflspread-26nov15ab"
    line = digest._play_line({**sharp_flag(), "_tag": "SHARP shadow"})
    assert "[Kalshi](https://kalshi.com/markets/kxnflspread/event/kxnflspread-26nov15larari)" in line
    bets = pd.DataFrame([{"date_ct": "2026-11-14", "league": "NFL", "kind": "SHARP", "event": "x",
                          "leg1": "YES KXNFLSPREAD-26NOV15LARARI-ARI3", "leg2": None, "sets": 1, "stake": 1,
                          "fees": 0.01, "pnl": 0, "status": "pending", "kickoff_ct": "", "snapshot_ts": "2026-11-14T15:00:00Z"}])
    d = dashboard.build({"snapshot_ts": None}, bets, pd.DataFrame(), NOW)
    assert d["today"]["bets"][0]["kalshi_url"].endswith("/kxnflspread-26nov15larari")


def test_cap_spent_on_credible_signals_first(tmp_path):
    model = {**sharp_flag(ticker="KXNFLTOTAL-26NOV15M-45", price=0.30, fair=0.60, stake=9.0), "kind": "MODEL"}
    sh = [sharp_flag(ticker=f"KXNFLSPREAD-26NOV15S{i}-X3", stake=9.0) for i in range(2)]
    out = alerts.process([([model], False), (sh, False)], {}, NOW, post=Post(),
                         state_path=tmp_path / "s.json", log_path=tmp_path / "l.csv")
    assert [a["kind"] for a in out] == ["SHARP", "SHARP"]     # bigger MODEL edge waits behind SHARP


def test_rate_limited_url_check_is_retried_soon():
    def get_429(u):
        return type("R", (), {"status_code": 429, "url": u, "text": ""})()
    r = alerts.verify_url("KXNFLGAME-26NOV15LARARI-ARI", "KXNFLGAME-26NOV15LARARI", get_429)
    assert not r["verified"] and r["inconclusive"]
    now = datetime(2026, 11, 14, 16, 0, tzinfo=UTC)
    r["checked_utc"] = "2026-11-14T08:00:00Z"                 # 8 h ago, inconclusive -> recheck
    assert alerts.url_check_due(r, now) is True
    r["checked_utc"] = "2026-11-14T13:00:00Z"                 # 3 h ago -> wait
    assert alerts.url_check_due(r, now) is False

    def get_404(u):
        return type("R", (), {"status_code": 404, "url": u, "text": ""})()
    r = alerts.verify_url("KXNFLGAME-26NOV15LARARI-ARI", "KXNFLGAME-26NOV15LARARI", get_404)
    assert not r["inconclusive"]
    r["checked_utc"] = "2026-11-12T16:00:00Z"                 # definitive answer, 2 days old -> keep
    assert alerts.url_check_due(r, now) is False
    assert alerts.url_check_due({}, now) is True


def test_old_url_file_without_inconclusive_field():
    prev = {"verified": False, "tried": [{"url": "x", "status": 429}], "checked_utc": "2026-11-14T08:00:00Z"}
    assert alerts.url_check_due(prev, datetime(2026, 11, 14, 16, 0, tzinfo=UTC)) is True


def test_model_flags_never_pushed():
    f = {"kind": "MODEL", "ticker": "KXNCAAFSPREAD-26NOV11MASSTOL-TOL22", "event_ticker": "KXNCAAFSPREAD-26NOV11MASSTOL",
         "side": "YES", "price": 0.31, "fair": 0.55, "edge": 0.217, "contracts": 1, "stake": 0.33,
         "kickoff_utc": "2026-11-12T00:30:00Z", "event": "UMass vs Toledo: Spread", "leg1_desc": "Toledo wins by over 21.5 points"}
    out = alerts.decide([([f], False)], {}, {}, "2026-11-10T23:30:00Z")
    assert out == []


# ---------- speed log for "gone" pushes (#34) ----------
def test_gone_logs_quote_time(tmp_path):
    import csv
    import pandas as pd
    st, lg = tmp_path / "s.json", tmp_path / "log.csv"
    f = sharp_flag()
    alerts.process([([f], False)], {}, NOW, post=Post(), state_path=st, log_path=lg, clock=lambda: NOW)
    df = pd.DataFrame([{"ticker": f["ticker"], "yes_bid": 0.49, "yes_ask": 0.50,
                        "snapshot_ts": "2026-11-14T16:00:40Z"}])
    later = NOW + timedelta(minutes=1)
    out = alerts.process([([f], False)], alerts.quotes_of(df), later, post=Post(), state_path=st,
                         log_path=lg, clock=lambda: later)
    assert [a["type"] for a in out] == ["gone"]
    rows = list(csv.DictReader(lg.open()))
    gone = [r for r in rows if r["type"] == "gone"][0]
    assert gone["quote_utc"] == "2026-11-14T16:00:40Z" and float(gone["quote_to_send_s"]) == 20.0


def test_gone_never_on_quote_older_than_the_alert(tmp_path):
    """A quote from before the 'new' push can't say the price moved since: no gone."""
    import pandas as pd
    st, lg = tmp_path / "s.json", tmp_path / "log.csv"
    f = sharp_flag()
    alerts.process([([f], False)], {}, NOW, post=Post(), state_path=st, log_path=lg, clock=lambda: NOW)
    df = pd.DataFrame([{"ticker": f["ticker"], "yes_bid": 0.49, "yes_ask": 0.50,
                        "snapshot_ts": "2026-11-14T15:30:00Z"}])
    out = alerts.process([([f], False)], alerts.quotes_of(df), NOW + timedelta(minutes=1), post=Post(),
                         state_path=st, log_path=lg)
    assert out == []
