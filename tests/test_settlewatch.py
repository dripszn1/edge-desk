"""#55: held-back bets carry their kickoff, and a settlement watch names any play still pending
long after its game (every shadow AND tracked play must eventually settle)."""
from datetime import datetime, timezone

import pandas as pd

from edge_desk import digest, grade

NOW = datetime(2026, 11, 15, 12, tzinfo=timezone.utc)
COLS = ["snapshot_ts", "league", "event", "ticker", "side", "price", "fair", "edge", "sharp_line",
        "odds_fetched_ts", "reason"]
FAR = "rung far from sharp line (unvalidated range)"


def tracked(*rows):
    return pd.DataFrame([dict(zip(COLS, r)) for r in rows])


def no_results():
    return pd.DataFrame(columns=["fetched_ts", "ticker", "status", "result"])


KO = {"KXNCAAFSPREAD-26NOV14JMUCONN": pd.Timestamp("2026-11-14T17:00:00Z"),
      "KXNFLTOTAL-26NOV16DALLV": pd.Timestamp("2026-11-17T01:15:00Z")}


def test_tracked_bets_get_kickoff_from_event_map():
    t = tracked(("2026-11-13T01:00:00Z", "CFB", "e", "KXNCAAFSPREAD-26NOV14JMUCONN-JMU10", "NO",
                 0.40, 0.50, 0.08, 3.5, "", FAR),
                ("2026-11-13T01:00:00Z", "CFB", "e", "KXNCAAFSPREAD-26NOV14ZZZYYY-ZZZ3", "NO",
                 0.40, 0.50, 0.08, 3.5, "", FAR))
    b = grade.tracked_bets(t, no_results(), NOW, kickoffs=KO)
    ko = dict(zip(b["leg1"], b["kickoff_ct"]))
    assert ko["NO KXNCAAFSPREAD-26NOV14JMUCONN-JMU10"] == "Sat Nov 14, 11:00 AM CT"
    assert pd.isna(ko["NO KXNCAAFSPREAD-26NOV14ZZZYYY-ZZZ3"])          # unknown event -> blank, no guess


def test_event_of_ticker():
    assert grade.event_of("KXNFLSPREAD-26NOV15BUFNYJ-BUF7") == "KXNFLSPREAD-26NOV15BUFNYJ"
    assert grade.event_of("NO KXNFLSPREAD-26NOV15BUFNYJ-BUF7") == "KXNFLSPREAD-26NOV15BUFNYJ"


def test_event_kickoffs_reads_map(tmp_path):
    p = tmp_path / "event_map.csv"
    p.write_text("event_ticker,league,espn_id,kickoff_utc,matched_at\n"
                 "KXNFLTOTAL-26NOV16DALLV,NFL,,2026-11-17T01:15:00Z,2026-11-13T00:00:00Z\n"
                 "KXBAD-1,NFL,,,2026-11-13T00:00:00Z\n")
    m = grade.event_kickoffs(p)
    assert m == {"KXNFLTOTAL-26NOV16DALLV": pd.Timestamp("2026-11-17T01:15:00Z")}
    assert grade.event_kickoffs(tmp_path / "missing.csv") == {}


def _bets(*rows):
    return pd.DataFrame([{"snapshot_ts": s, "leg1": leg, "status": st} for s, leg, st in rows])


def test_watch_counts_overdue_only_after_grace():
    frames = {"shadow": _bets(("2026-11-13T01:00:00Z", "NO KXNCAAFSPREAD-26NOV14JMUCONN-JMU10", "pending"),
                              ("2026-11-13T01:00:00Z", "YES KXNFLTOTAL-26NOV16DALLV-45", "pending")),
              "tracked": _bets(("2026-11-13T01:00:00Z", "YES KXNCAAFSPREAD-26NOV14JMUCONN-JMU3", "settled"),
                               ("2026-11-13T01:00:00Z", "YES KXZZZ-1-2", "pending"))}
    w = grade.settlement_watch(frames, KO, NOW)
    # JMU kicked off 11-14 17:00Z -> 19h ago: overdue. DAL/LV is tomorrow: pending. Settled: ignored.
    assert w["pending"] == 3 and w["overdue"] == 1 and w["no_kickoff"] == 1
    assert w["oldest"][0]["leg1"] == "NO KXNCAAFSPREAD-26NOV14JMUCONN-JMU10"
    assert w["oldest"][0]["source"] == "shadow" and w["oldest"][0]["hours_past"] == 19.0
    # 6h after kickoff is inside the grace window -> not overdue yet
    w2 = grade.settlement_watch(frames, KO, datetime(2026, 11, 14, 23, tzinfo=timezone.utc))
    assert w2["overdue"] == 0


def test_watch_ignores_future_stamped_and_stale_counts_as_pending():
    frames = {"tracked": _bets(("2026-11-20T01:00:00Z", "NO KXNCAAFSPREAD-26NOV14JMUCONN-JMU10", "pending"),
                               ("2026-11-13T01:00:00Z", "NO KXNCAAFSPREAD-26NOV14JMUCONN-JMU7", "stale"))}
    w = grade.settlement_watch(frames, KO, NOW)
    assert w["pending"] == 1 and w["overdue"] == 1


def test_digest_status_shows_overdue_only_when_any():
    weekly = {"generated_utc": "2026-11-15T11:00:00Z",
              "settlement": {"pending": 3, "overdue": 2, "no_kickoff": 0, "oldest": []}}
    assert "2 plays unsettled 12h+ after kickoff" in digest.settlement_note(weekly, pd.Timestamp(NOW))
    assert digest.settlement_note({**weekly, "settlement": {"overdue": 0}}, pd.Timestamp(NOW)) == ""
    future = {**weekly, "generated_utc": "2026-11-16T00:00:00Z"}
    assert digest.settlement_note(future, pd.Timestamp(NOW)) == ""
