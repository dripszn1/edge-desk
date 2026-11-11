"""#30: odds pacing scales from the real quota (x-requests-remaining + x-requests-used)."""
import pandas as pd

from edge_desk import odds

NOW = "2026-11-14T20:00:00Z"          # Sat 2:00 PM CT, inside the Saturday window
OUT = "2026-11-14T13:00:00Z"          # Sat 7:00 AM CT, before the window


def _hi(remaining=19000, used=1000, **kw):
    return dict({"remaining": remaining, "used": used, "remaining_seen_utc": NOW}, **kw)


def _ago(t, minutes):
    return (pd.Timestamp(t) - pd.Timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _in(t, minutes):
    return (pd.Timestamp(t) + pd.Timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------- quota ----------
def test_quota_from_headers_and_safe_default():
    assert odds.quota({}) == odds.MONTHLY_CREDITS                        # header never seen
    assert odds.quota({"remaining": 476}) == odds.MONTHLY_CREDITS        # 'used' missing -> default
    assert odds.quota({"remaining": 476, "used": 24}) == 500
    assert odds.quota(_hi()) == 20000
    st = odds.record_pull({}, "CFB", NOW, {"x-requests-remaining": "19997", "x-requests-used": "3",
                                          "x-requests-last": "3"})
    assert st["quota"] == 20000 and odds.quota(st) == 20000
    st = odds.record_pull({}, "CFB", NOW, {})                            # headers missing
    assert odds.quota(st) == odds.MONTHLY_CREDITS


def test_new_month_resets_to_the_known_quota_not_500():
    st = _hi(remaining=40, used=19960, quota=20000)
    assert odds.remaining(st, "2026-12-01T15:00:00Z") == 20000
    assert odds.remaining({"remaining": 40, "remaining_seen_utc": NOW}, "2026-12-01T15:00:00Z") == 500


def test_budget_scales_with_quota_and_is_identical_on_free_tier():
    free = odds.budget({"remaining": 476, "used": 24})
    assert (free["reserve"], free["football_daily"], free["high"]) == (odds.ODDS_RESERVE, odds.FOOTBALL_DAILY_CREDITS, False)
    hi = odds.budget(_hi())
    assert hi["high"] and hi["reserve"] >= 0.10 * 20000 and hi["football_daily"] > odds.FOOTBALL_DAILY_CREDITS
    assert odds.football_floor(NOW, _hi()) == hi["reserve"] + hi["football_daily"] * odds.days_left_in_month(NOW)


def test_free_tier_plans_identical_with_or_without_used_header():
    ko = {"NFL": _in(NOW, 60), "CFB": _in(NOW, 30), "CBB": _in(NOW, 120)}
    for last in (None, _ago(NOW, 30), _ago(NOW, 300)):
        a = {"remaining": 400, "remaining_seen_utc": NOW}
        if last:
            a["last_pull"] = {"NFL": last, "CFB": last, "CBB": last}
        b = dict(a, used=100)
        for kw in ({}, {"near_h": 1.5, "near_interval_h": 1.0}):
            assert odds.plan_pulls(a, NOW, ko, **kw) == odds.plan_pulls(b, NOW, ko, **kw)
    # free tier keeps the 4h game-day interval even though the plan header is now read
    st = {"remaining": 476, "used": 24, "remaining_seen_utc": NOW, "last_pull": {"CFB": _ago(NOW, 30)}}
    assert odds.plan_pulls(st, NOW, {"NFL": None, "CFB": _in(NOW, 30)}) == []


# ---------- 20k cadence ----------
def test_20k_ten_minute_cadence_in_window():
    ko = {"NFL": None, "CFB": _in(NOW, 180), "CBB": None}               # kickoff 3h out, in window
    st = _hi(last_pull={"CFB": _ago(NOW, 11)})
    assert odds.plan_pulls(st, NOW, ko) == ["CFB"]
    st = _hi(last_pull={"CFB": _ago(NOW, 6)})
    assert odds.plan_pulls(st, NOW, ko) == []


def test_20k_five_minutes_near_kickoff():
    ko = {"NFL": None, "CFB": _in(NOW, 60), "CBB": None}                # 1h out
    assert odds.plan_pulls(_hi(last_pull={"CFB": _ago(NOW, 6)}), NOW, ko) == ["CFB"]
    assert odds.plan_pulls(_hi(last_pull={"CFB": _ago(NOW, 3)}), NOW, ko) == []


def test_20k_hourly_outside_windows_on_game_days():
    ko = {"NFL": None, "CFB": _in(OUT, 300), "CBB": None}               # game day, 7 AM CT
    st = _hi(remaining_seen_utc=OUT, last_pull={"CFB": _ago(OUT, 30)})
    assert odds.plan_pulls(st, OUT, ko) == []
    st = _hi(remaining_seen_utc=OUT, last_pull={"CFB": _ago(OUT, 61)})
    assert odds.plan_pulls(st, OUT, ko) == ["CFB"]


def test_20k_cbb_pulls_every_ten_minutes_too_but_after_football():
    ko = {"NFL": None, "CFB": _in(NOW, 180), "CBB": _in(NOW, 180)}
    st = _hi(last_pull={"CFB": _ago(NOW, 11), "CBB": _ago(NOW, 11)},
             pulls={"CBB": {"2026-11-14": 9}})                          # no 2-per-day cap on 20k
    assert odds.plan_pulls(st, NOW, ko) == ["CFB", "CBB"]


def test_20k_never_below_reserve_and_cbb_never_eats_football_floor():
    b = odds.budget(_hi())
    ko = {"NFL": None, "CFB": _in(NOW, 60), "CBB": _in(NOW, 60)}
    low = _hi(remaining=b["reserve"] + odds.COST_PER_PULL - 1, used=20000 - b["reserve"] - 2)
    assert odds.plan_pulls(low, NOW, ko) == []
    floor = odds.football_floor(NOW, _hi())
    mid = _hi(remaining=floor + 1, used=20000 - floor - 1)
    assert "CBB" not in odds.plan_pulls(mid, NOW, ko)


def test_live_check_minutes_follow_plan():
    assert odds.check_minutes({"remaining": 476, "used": 24}, 10) == 10
    assert odds.check_minutes(_hi(), 10) == odds.HIGH_NEAR_MIN


def _simulate(st, start, days, step_min, ko_every_h=1):
    """Worst case: kickoffs for every league every hour, all month, polled every step_min."""
    t = pd.Timestamp(start)
    spent, lowest = 0, odds.remaining(st, t)
    q = odds.quota(st)
    while t < pd.Timestamp(start) + pd.Timedelta(days=days):
        nxt = (t.floor(f"{ko_every_h}h") + pd.Timedelta(hours=ko_every_h)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for lg in odds.plan_pulls(st, t, {"NFL": nxt, "CFB": nxt, "CBB": nxt}):
            rem = odds.remaining(st, t) - odds.COST_PER_PULL
            st = odds.record_pull(st, lg, t, {"x-requests-remaining": str(rem), "x-requests-used": str(q - rem),
                                              "x-requests-last": str(odds.COST_PER_PULL)})
            spent += odds.COST_PER_PULL
            lowest = min(lowest, rem)
        t += pd.Timedelta(minutes=step_min)
    return spent, lowest


def test_20k_worst_case_month_stays_under_90pct():
    st = {"remaining": 20000, "used": 0, "remaining_seen_utc": "2026-11-01T00:00:00Z"}
    spent, lowest = _simulate(st, "2026-11-01T00:00:00Z", 30, 5)
    assert spent <= 0.90 * 20000
    assert lowest >= odds.budget(st)["reserve"]
    assert spent > 0.5 * 20000                                         # and the plan is actually used
