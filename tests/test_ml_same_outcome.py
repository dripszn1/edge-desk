"""#61: a KX*GAME event has one market per team, so YES team A and NO team B are the same outcome.
One flag, one push, one graded bet per (event, outcome)."""
from datetime import timedelta

import pandas as pd

from edge_desk import alerts, grade
from tests.test_alerts import NOW, Post, sharp_flag


def test_outcome_key_maps_equivalent_legs_together():
    a = grade.outcome_key("NO KXNCAAFGAME-26NOV14CINISU-CIN")
    b = grade.outcome_key("YES KXNCAAFGAME-26NOV14CINISU-ISU")
    c = grade.outcome_key("YES KXNCAAFGAME-26NOV14CINISU-CIN")
    assert a == b == "KXNCAAFGAME-26NOV14CINISU>ISU"
    assert c == "KXNCAAFGAME-26NOV14CINISU>CIN" and c != a
    assert grade.outcome_key("NO KXNFLGAME-26NOV15BUFNYJ-NYJ") == "KXNFLGAME-26NOV15BUFNYJ>BUF"
    # spreads/totals and unparseable legs keep their own leg as key
    assert grade.outcome_key("NO KXNFLSPREAD-26NOV15BUFNYJ-BUF7") == "NO KXNFLSPREAD-26NOV15BUFNYJ-BUF7"
    assert grade.outcome_key("NO KXNCAAFGAME-26NOV14CINISU-XYZ") == "NO KXNCAAFGAME-26NOV14CINISU-XYZ"


def _ml(ticker, side, price, ts="2026-11-14T17:04:28Z"):
    f = sharp_flag(ticker=ticker, side=side, price=price, fair=0.52, n=10, stake=5.0)
    f.update(market="ML", league="CFB", event_ticker=ticker.rsplit("-", 1)[0], snapshot_ts=ts,
             kickoff_utc="2026-11-14T20:00:00Z")
    return f


def test_two_equivalent_legs_give_one_push_cheaper_ask(tmp_path):
    st, lg, post = tmp_path / "s.json", tmp_path / "log.csv", Post()
    no_cin = _ml("KXNCAAFGAME-26NOV14CINISU-CIN", "NO", 0.45)
    yes_isu = _ml("KXNCAAFGAME-26NOV14CINISU-ISU", "YES", 0.44)
    q = {no_cin["ticker"]: {"yes_bid": 0.55, "yes_ask": 0.57}, yes_isu["ticker"]: {"yes_bid": 0.43, "yes_ask": 0.44}}
    out = alerts.process([([no_cin, yes_isu], False)], q, NOW, post=post, state_path=st, log_path=lg, clock=lambda: NOW)
    assert [(a["type"], a["ticker"]) for a in out] == [("new", yes_isu["ticker"])]


def test_equivalent_leg_in_a_later_pass_is_not_pushed(tmp_path):
    st, lg, post = tmp_path / "s.json", tmp_path / "log.csv", Post()
    no_cin = _ml("KXNCAAFGAME-26NOV14CINISU-CIN", "NO", 0.45)
    yes_isu = _ml("KXNCAAFGAME-26NOV14CINISU-ISU", "YES", 0.44, ts="2026-11-14T17:04:58Z")
    q = {no_cin["ticker"]: {"yes_bid": 0.55, "yes_ask": 0.57}, yes_isu["ticker"]: {"yes_bid": 0.43, "yes_ask": 0.44}}
    a1 = alerts.process([([no_cin], False)], q, NOW, post=post, state_path=st, log_path=lg, clock=lambda: NOW)
    a2 = alerts.process([([yes_isu], False)], q, NOW + timedelta(seconds=30), post=post, state_path=st, log_path=lg)
    assert [a["type"] for a in a1] == ["new"] and a2 == []
    # the opposite outcome (CIN wins) is a different play and still alerts
    no_isu = _ml("KXNCAAFGAME-26NOV14CINISU-ISU", "NO", 0.40, ts="2026-11-14T17:19:05Z")
    a3 = alerts.process([([no_isu], False)], q, NOW + timedelta(minutes=1), post=post, state_path=st, log_path=lg)
    assert [(a["type"], a["ticker"]) for a in a3 if a["type"] == "new"] == [("new", no_isu["ticker"])]


def _flagrows():
    base = {"league": "CFB", "kind": "SHARP", "market": "ML", "event": "Cincinnati at Iowa St", "contracts": 10,
            "fee_multiplier": 1.0, "fair": 0.52, "edge": 0.05}
    return pd.DataFrame([
        {**base, "snapshot_ts": "2026-11-14T17:04:28Z", "leg1": "NO KXNCAAFGAME-26NOV14CINISU-CIN", "price": 0.45},
        {**base, "snapshot_ts": "2026-11-14T17:04:58Z", "leg1": "YES KXNCAAFGAME-26NOV14CINISU-ISU", "price": 0.44},
        {**base, "snapshot_ts": "2026-11-14T17:19:05Z", "leg1": "NO KXNCAAFGAME-26NOV14CINISU-ISU", "price": 0.40},
    ])


def test_grading_counts_one_bet_per_outcome_first_sighting():
    res = pd.DataFrame([
        {"fetched_ts": "2026-11-15T01:00:00Z", "ticker": "KXNCAAFGAME-26NOV14CINISU-CIN", "status": "finalized", "result": "no"},
        {"fetched_ts": "2026-11-15T01:00:00Z", "ticker": "KXNCAAFGAME-26NOV14CINISU-ISU", "status": "finalized", "result": "yes"},
    ])
    b = grade.model_bets(_flagrows(), res, now=pd.Timestamp("2026-11-15T02:00:00Z"), kind="SHARP")
    assert sorted(b["leg1"]) == ["NO KXNCAAFGAME-26NOV14CINISU-CIN", "NO KXNCAAFGAME-26NOV14CINISU-ISU"]


def test_same_snapshot_keeps_cheaper_ask():
    f = _flagrows().iloc[:2].copy()
    f["snapshot_ts"] = "2026-11-14T17:04:28Z"
    b = grade.model_bets(f, pd.DataFrame(columns=["fetched_ts", "ticker", "status", "result"]),
                         now=pd.Timestamp("2026-11-14T18:00:00Z"), kind="SHARP")
    assert list(b["leg1"]) == ["YES KXNCAAFGAME-26NOV14CINISU-ISU"]


def test_tracked_and_pooled_summaries_count_outcome_once():
    tr = _flagrows().rename(columns={"leg1": "_l"})
    tr["side"] = tr["_l"].str.split(" ").str[0]
    tr["ticker"] = tr["_l"].str.split(" ").str[1]
    tr["reason"] = "sharp line stale"
    tb = grade.tracked_bets(tr.drop(columns=["_l"]), pd.DataFrame(columns=["fetched_ts", "ticker", "status", "result"]),
                            now=pd.Timestamp("2026-11-14T18:00:00Z"))
    assert len(tb) == 2
    pooled = pd.DataFrame([
        {"source": "shadow", "leg1": "NO KXNCAAFGAME-26NOV14CINISU-CIN", "snapshot_ts": "2026-11-14T17:04:28Z"},
        {"source": "tracked", "leg1": "YES KXNCAAFGAME-26NOV14CINISU-ISU", "snapshot_ts": "2026-11-14T16:00:00Z"},
    ])
    assert list(grade.one_per_contract(pooled)["source"]) == ["shadow"]
