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


# ---------- #42 signal scorecards + open alerts ----------
def _weekly(gen="2026-11-16T17:55:00Z"):
    card = {"settled": 3, "pnl": -1.5, "roi": -0.2, "t_stat": None, "verdict": "not enough data", "why": "3 < 30"}
    return {"generated_utc": gen, "model_shadow": card, "sharp_shadow": card, "sharp_shadow_cbb": card,
            "sharp_tracked": {**card, "by_reason": {"far_rung": {"settled": 2, "pnl": 1.0}}},
            "clv": {"shadow": {"football": {"n": 2, "avg_clv_c": 0.5}}},
            "trades": {"football": {"best_manual": "RESTING_FAIR", "variants": {
                "HOLD": {"label": "MANUAL-OK", "closed": 5, "pnl_usd": -8.0, "t_by_game": None, "gate_open": False},
                "RESTING_FAIR": {"label": "MANUAL-OK", "closed": 5, "pnl_usd": 3.0, "t_by_game": 0.4, "gate_open": False},
                "BRACKET_5_5": {"label": "API-ONLY", "closed": 5, "pnl_usd": 9.0, "t_by_game": 1.0, "gate_open": False}}}}}


def _alerts():
    a = {"kind": "SHARP", "side": "NO", "game": "JMU vs UConn: Spread", "desc": "JMU wins by over 3.5",
         "max_price": 0.48, "stake": 3.9, "gone": False, "click": "https://kalshi.com/markets/x"}
    return {"alerts": {
        "open": {**a, "ticker": "T-OPEN", "sent_utc": "2026-11-16T10:00:00Z", "kickoff_utc": "2026-11-16T21:00:00Z"},
        "over": {**a, "ticker": "T-OVER", "sent_utc": "2026-11-15T10:00:00Z", "kickoff_utc": "2026-11-16T11:00:00Z"},
        "future": {**a, "ticker": "T-FUT", "sent_utc": "2026-11-16T18:30:00Z", "kickoff_utc": "2026-11-16T21:00:00Z"}}}


def test_signals_show_scorecards_exits_and_open_alerts():
    b, led = bets_and_ledger()
    lat = {**latest(), "sharp": {"NFL": {"gate": "gated: 0 settled"}}, "model": {"NFL": {"gate": "open: t=2.5"}}}
    s = build(lat, b, led, NOW, weekly=_weekly(), alert_state=_alerts())["signals"]
    names = [c["name"] for c in s["scorecards"]]
    assert names == ["SHARP shadow football", "SHARP shadow CBB", "SHARP held back", "MODEL shadow"]
    assert s["scorecards"][0]["pnl"] == -1.5 and s["scorecards"][2]["by_reason"]["far_rung"]["settled"] == 2
    ex = s["exits"]["football"]
    assert ex["best_manual"] == "RESTING_FAIR" and set(ex["variants"]) == {"HOLD", "RESTING_FAIR"}  # API-ONLY hidden
    assert [a["ticker"] for a in s["open_alerts"]] == ["T-OPEN"]          # game over / sent after now dropped
    assert s["open_alerts"][0]["kickoff_ct"] == "Mon Nov 16, 3:00 PM CT" and s["open_alerts"][0]["kalshi_url"]
    assert s["gates"] == {"MODEL NFL": "open", "SHARP NFL": "closed"}


def test_signals_ignore_a_weekly_report_from_the_future():
    b, led = bets_and_ledger()
    s = build(latest(), b, led, NOW, weekly=_weekly(gen="2026-11-16T18:30:00Z"), alert_state={})["signals"]
    assert s["scorecards"] == [] and s["exits"] == {} and s["weekly_ct"] is None


def test_signals_missing_inputs_build():
    b, led = bets_and_ledger()
    s = build(latest(), b, led, NOW)["signals"]
    assert s["scorecards"] == [] and s["open_alerts"] == []
