"""Sharp-line fair value (backlog #10): de-vig, look-ahead, stale, in-play, budget, missing key."""
import json

import pandas as pd
import pytest

from edge_desk import odds

FETCH = "2026-11-14T15:00:00Z"
KO = "2026-11-14T18:00:00Z"


def _ev(eid="e1", commence=KO, books=None):
    books = books or {
        "pinnacle": {"spreads": [("Home", -105, -3.0), ("Away", -105, 3.0)],
                     "totals": [("Over", -110, 44.5), ("Under", +100, 44.5)],
                     "h2h": [("Home", -160, None), ("Away", +145, None)]},
        "lowvig": {"spreads": [("Home", -110, -3.5), ("Away", -100, 3.5)]},
    }
    bks = []
    for b, mk in books.items():
        bks.append({"key": b, "last_update": FETCH, "markets": [
            {"key": m, "last_update": FETCH, "outcomes": [
                {"name": n, "price": p, **({"point": pt} if pt is not None else {})} for n, p, pt in outs]}
            for m, outs in mk.items()]})
    return {"id": eid, "commence_time": commence, "home_team": "Home", "away_team": "Away",
            "bookmakers": bks}


def test_implied_and_devig():
    assert odds.implied(-110) == pytest.approx(110 / 210)
    assert odds.implied(+150) == pytest.approx(0.4)
    a, b = odds.devig([odds.implied(-110), odds.implied(-110)])
    assert a == pytest.approx(0.5) and b == pytest.approx(0.5)
    # vig is removed: fair probs sum to 1 though implied sum > 1
    imp = [odds.implied(-160), odds.implied(+145)]
    assert sum(imp) > 1 and sum(odds.devig(imp)) == pytest.approx(1)


def test_parse_drops_inplay_and_devigs_per_book():
    live = _ev("live", commence="2026-11-14T14:00:00Z")      # started before fetch
    df = odds.parse([_ev(), live], "NFL", FETCH)
    assert set(df["event_id"]) == {"e1"}
    g = df.groupby(["book", "market"])["fair"].sum()
    assert all(g.round(6) == 1)


def test_fair_prefers_pinnacle():
    df = odds.parse([_ev()], "NFL", FETCH)
    f = odds.fair_lines(df, "2026-11-14T16:00:00Z")
    sp = f[(f.market == "spreads") & (f.outcome == "Home")].iloc[0]
    assert sp.source == "pinnacle" and sp.point == -3.0 and sp.fair == pytest.approx(0.5)


def test_fair_consensus_without_pinnacle_uses_modal_point():
    books = {"lowvig": {"totals": [("Over", -110, 44.5), ("Under", -110, 44.5)]},
             "betonlineag": {"totals": [("Over", -120, 44.5), ("Under", +100, 44.5)]},
             "draftkings": {"totals": [("Over", -200, 40.5), ("Under", +170, 40.5)]}}  # not sharp
    f = odds.fair_lines(odds.parse([_ev(books=books)], "NFL", FETCH), "2026-11-14T16:00:00Z")
    over = f[f.outcome == "Over"].iloc[0]
    assert over.source == "consensus:2" and over.point == 44.5
    exp = (0.5 + odds.devig([odds.implied(-120), odds.implied(100)])[0]) / 2
    assert over.fair == pytest.approx(exp, abs=1e-6)


def test_no_sharp_books_no_fair():
    books = {"draftkings": {"totals": [("Over", -110, 44.5), ("Under", -110, 44.5)]}}
    assert odds.fair_lines(odds.parse([_ev(books=books)], "NFL", FETCH), "2026-11-14T16:00:00Z").empty


def test_look_ahead_rows_fetched_after_as_of_ignored():
    df = odds.parse([_ev()], "NFL", FETCH)
    assert odds.fair_lines(df, "2026-11-14T14:59:00Z").empty


def test_stale_rows_ignored():
    df = odds.parse([_ev(commence="2026-11-15T18:00:00Z")], "NFL", FETCH)
    late = pd.Timestamp(FETCH) + pd.Timedelta(hours=odds.ODDS_STALE_HOURS, minutes=1)
    assert odds.fair_lines(df, late).empty
    assert not odds.fair_lines(df, pd.Timestamp(FETCH) + pd.Timedelta(hours=1)).empty


def test_latest_pull_wins():
    old = odds.parse([_ev()], "NFL", "2026-11-14T12:00:00Z")
    new_books = {"pinnacle": {"totals": [("Over", -110, 47.5), ("Under", -110, 47.5)]}}
    new = odds.parse([_ev(books=new_books)], "NFL", FETCH)
    f = odds.fair_lines(pd.concat([old, new]), "2026-11-14T16:00:00Z")
    assert f[(f.market == "totals")]["point"].unique().tolist() == [47.5]


# ---- budget ----
NOW = "2026-11-14T15:00:00Z"   # 17 days left in November


def test_budget_never_below_reserve():
    st = {"remaining": odds.ODDS_RESERVE + odds.COST_PER_PULL - 1, "remaining_seen_utc": NOW}
    assert odds.plan_pulls(st, NOW, {"NFL": KO, "CFB": KO}) == []


def test_budget_daily_allowance_and_interval():
    st = {"remaining": 200, "remaining_seen_utc": NOW}
    plan = odds.plan_pulls(st, NOW, {"NFL": KO, "CFB": KO})
    assert plan == ["NFL", "CFB"] or plan == ["CFB", "NFL"]
    # pulled 1h ago -> game-day interval blocks
    st2 = odds.record_pull(st, "NFL", "2026-11-14T14:00:00Z", {"x-requests-remaining": "197"})
    assert "NFL" not in odds.plan_pulls(st2, NOW, {"NFL": KO, "CFB": None})
    # nothing upcoming -> no pull
    assert odds.plan_pulls(st, NOW, {"NFL": None, "CFB": None}) == []


def test_budget_idle_day_is_slower():
    far = "2026-11-20T18:00:00Z"
    st = {"remaining": 200, "remaining_seen_utc": NOW,
          "last_pull": {"NFL": "2026-11-14T08:00:00Z"}}
    assert odds.plan_pulls(st, NOW, {"NFL": far, "CFB": None}) == []   # 7h < 24h idle interval


def test_budget_month_spend_fits_free_tier():
    """Simulate a month of 30-min workflow runs with games every Thu/Sat/Sun: total spend <= cap."""
    st, spent = {}, 0
    t = pd.Timestamp("2026-12-01T00:00:00Z")
    while t.month == 12:
        kick = {}
        for lg in odds.SPORTS:
            d = t.normalize()
            nxt = [d + pd.Timedelta(days=k, hours=18) for k in range(8)
                   if (d + pd.Timedelta(days=k)).dayofweek in (3, 5, 6)]
            kick[lg] = next((x for x in nxt if x > t), None)
        for lg in odds.plan_pulls(st, t, kick):
            rem = odds.remaining(st, t) - odds.COST_PER_PULL
            st = odds.record_pull(st, lg, t, {"x-requests-remaining": str(rem), "x-requests-last": "3"})
            spent += 3
        t += pd.Timedelta(minutes=30)
    assert spent <= odds.MONTHLY_CREDITS - odds.ODDS_RESERVE
    assert spent >= 300            # and it actually uses the budget


def test_new_month_resets_remaining():
    st = {"remaining": 0, "remaining_seen_utc": "2026-11-30T20:00:00Z"}
    assert odds.remaining(st, "2026-12-01T15:00:00Z") == odds.MONTHLY_CREDITS


def test_missing_key_skips_cleanly(monkeypatch, tmp_path):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setattr(odds, "_get", lambda *a, **k: pytest.fail("no network without a key"))
    assert odds.run(NOW) == []


def test_run_writes_cache_and_state(monkeypatch, tmp_path):
    monkeypatch.setenv("ODDS_API_KEY", "x")
    monkeypatch.setattr(odds, "ODDS_DIR", tmp_path)
    monkeypatch.setattr(odds, "LINES", tmp_path / "lines.csv")
    monkeypatch.setattr(odds, "STATE", tmp_path / "state.json")

    def fake(url, params):
        if url.endswith("/events"):
            return [{"commence_time": KO}], {}
        return [_ev()], {"x-requests-remaining": "480", "x-requests-last": "3"}
    monkeypatch.setattr(odds, "_get", fake)
    plan = odds.run(NOW)
    assert set(plan) == {"NFL", "CFB"}
    st = json.loads((tmp_path / "state.json").read_text())
    assert st["remaining"] == 480 and st["spent"]["2026-11-14"] == 6
    assert len(list((tmp_path / "raw" / "2026-11-14").glob("*.json"))) == 2
    assert (tmp_path / "fair_latest.csv").exists()
