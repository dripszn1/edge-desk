import pandas as pd

from edge_desk import pickoff


def test_line_move_sign():
    assert pickoff.line_move_fav("YES", 149, 145.5) == -3.5      # Over lost value
    assert pickoff.line_move_fav("NO", 149, 145.5) == 3.5
    assert pickoff.line_move_fav("YES", None, 1) is None


def test_age_bucket():
    assert pickoff.age_bucket(3) == "0-10 min"
    assert pickoff.age_bucket(45) == "30-90 min"
    assert pickoff.age_bucket(-1) is None and pickoff.age_bucket(None) is None


def _snap(ts, bid, ask, occ="2026-11-13T02:30:00Z"):
    return {"snapshot_ts": ts, "ticker": "KXT-1", "yes_bid": bid, "yes_ask": ask, "no_bid": 1 - ask,
            "no_ask": 1 - bid, "occurrence_datetime": occ, "series": "KXNCAAMBTOTAL", "event_title": "A vs B",
            "event_sub": "A vs B", "floor_strike": 149.5}


def test_pre_move_and_no_lookahead():
    bets = pd.DataFrame([{"bet_id": "b1", "source": "shadow", "league": "CBB", "leg1": "YES KXT-1",
                          "snapshot_ts": "2026-11-12T22:30:00Z"}])
    flags = pd.DataFrame([{"source": "shadow", "leg1": "YES KXT-1", "snapshot_ts": "2026-11-12T22:30:00Z",
                           "sharp_line": 149.0, "odds_fetched_ts": "2026-11-12T22:25:00Z", "league": "CBB"}])
    snaps = pd.DataFrame([_snap("2026-11-12T22:00:00Z", .46, .48), _snap("2026-11-12T22:30:00Z", .40, .42),
                          _snap("2026-11-13T05:00:00Z", .10, .12)])         # after now: must be ignored
    out = pickoff.analyse(bets, flags, snaps, pd.DataFrame(), pd.DataFrame(), "2026-11-12T23:00:00Z")
    r = out.iloc[0]
    assert r["pre_move_c"] == -6.0 and bool(r["kalshi_first"]) is True
    assert r["pull_age_min"] == 5.0 and r["age_bucket"] == "0-10 min"
    assert r["close_line"] is None or pd.isna(r["close_line"])        # kickoff not reached: no close line


def _rows(league, n=40, clv0=-3.0, step=-0.1):
    return [{"bet_id": f"b{i}", "source": "tracked", "league": league, "leg1": f"YES KXT-26NOV12G{i % 10}-1",
             "kalshi_first": True, "age_bucket": "30-90 min", "adverse": False, "line_move_fav": 0,
             "pull_age_min": 40, "clv_c": clv0 + (i % 10) * step, "close_status": "ok"} for i in range(n)]


def test_live_guard_is_not_reproposed_but_reported():
    """#52: CBB already runs the kalshi-moved-first guard (#50), so the digest must not keep
    proposing it; the pull-age check still proposes, and the guard's own record is shown."""
    s = pickoff.summary(pd.DataFrame(_rows("CBB")))
    assert not any("Kalshi already moved" in p for p in s["proposals"])
    assert any("30-90 min" in p for p in s["proposals"])
    assert any("guard is ON" in x and "40 plays" in x for x in s["in_effect"])


def test_live_guard_revert_proposed_only_on_positive_evidence():
    win = pickoff.summary(pd.DataFrame(_rows("CBB", clv0=3.0, step=0.1)))
    assert any("turn OFF" in p for p in win["proposals"])
    few = pickoff.summary(pd.DataFrame(_rows("CBB", n=20, clv0=3.0, step=0.1)))
    assert not any("turn OFF" in p for p in few["proposals"])
    lose = pickoff.summary(pd.DataFrame(_rows("CBB")))
    assert not any("turn OFF" in p for p in lose["proposals"])


def test_proposal_needs_30_plays_and_t():
    rows = []
    for i in range(40):
        rows.append({"bet_id": f"b{i}", "source": "tracked", "league": "NFL", "leg1": f"YES KXT-26NOV12G{i % 10}-1",
                     "kalshi_first": True, "age_bucket": "30-90 min", "adverse": False, "line_move_fav": 0,
                     "pull_age_min": 40, "clv_c": -3 - (i % 10) * 0.1, "close_status": "ok"})
    s = pickoff.summary(pd.DataFrame(rows))
    assert any("Kalshi already moved" in p for p in s["proposals"])
    assert any("30-90 min" in p for p in s["proposals"])
    s2 = pickoff.summary(pd.DataFrame(rows[:20]))
    assert s2["proposals"] == []
