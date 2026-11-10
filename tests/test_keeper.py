"""Cron-independent keeper (backlog #21)."""
from datetime import datetime, timedelta, timezone

from edge_desk import keeper

UTC = timezone.utc
# Tue 2026-11-10 23:45Z = 5:45 PM CST, inside the Tue 5:30-11:30 PM window
IN_WIN = datetime(2026, 11, 10, 23, 45, tzinfo=UTC)
# Tue 2026-11-10 16:00Z = 10 AM CST: collect hours, no game window
DAY = datetime(2026, 11, 10, 16, 0, tzinfo=UTC)
# Wed 2026-11-11 09:00Z = 3 AM CST: outside collect hours
NIGHT = datetime(2026, 11, 11, 9, 0, tzinfo=UTC)


def run(wf, mins_ago, status="completed", now=DAY, rid=1):
    return {"workflow": wf, "status": status, "created_at": now - timedelta(minutes=mins_ago), "id": rid}


def test_dispatch_collect_when_stale():
    acts, _ = keeper.plan(DAY, [run("collect.yml", 31)])
    assert acts == ["collect.yml"]


def test_no_collect_when_recent_or_running():
    assert keeper.plan(DAY, [run("collect.yml", 10)])[0] == []
    assert keeper.plan(DAY, [run("collect.yml", 45, "in_progress")])[0] == []
    assert keeper.plan(DAY, [run("collect.yml", 45, "queued")])[0] == []


def test_stale_data_counts_even_if_run_list_is_empty():
    assert keeper.plan(DAY, [])[0] == ["collect.yml"]


def test_future_created_at_is_not_fresh():
    # a clock-skewed / bogus timestamp from the future must not suppress collection forever
    acts, _ = keeper.plan(DAY, [run("collect.yml", -120)])
    assert acts == ["collect.yml"]


def test_no_collect_outside_collect_hours():
    assert keeper.plan(NIGHT, [run("collect.yml", 300, now=NIGHT)])[0] == []


def test_live_dispatched_in_window_and_collect_paused():
    acts, _ = keeper.plan(IN_WIN, [run("collect.yml", 60, now=IN_WIN)])
    assert "live.yml" in acts
    # while a live run is active it persists its own snapshots, so no collect duplicates
    acts, _ = keeper.plan(IN_WIN, [run("collect.yml", 60, now=IN_WIN),
                                   run("live.yml", 30, "in_progress", now=IN_WIN)])
    assert acts == []


def test_live_not_redispatched_too_often():
    state = {}
    acts, state = keeper.plan(IN_WIN, [run("collect.yml", 5, now=IN_WIN)], state)
    assert acts == ["live.yml"]
    later = IN_WIN + timedelta(minutes=5)
    # previous dispatch ran its guard and exited (completed) -> wait LIVE_RETRY_MINUTES
    acts, state = keeper.plan(later, [run("collect.yml", 10, now=later),
                                      run("live.yml", 4, now=later)], state)
    assert acts == []
    later2 = IN_WIN + timedelta(minutes=keeper.LIVE_RETRY_MINUTES + 1)
    acts, _ = keeper.plan(later2, [run("collect.yml", 1, now=later2)], state)
    assert acts == ["live.yml"]


def test_no_live_outside_window():
    assert "live.yml" not in keeper.plan(DAY, [run("collect.yml", 5)])[0]


def test_parse_runs_handles_z_and_bad_rows():
    rows = [{"databaseId": 7, "status": "completed", "createdAt": "2026-11-10T16:00:00Z"},
            {"databaseId": 8, "status": "queued", "createdAt": None}]
    out = keeper.parse_runs("collect.yml", rows)
    assert out == [{"workflow": "collect.yml", "status": "completed", "id": 7,
                    "created_at": datetime(2026, 11, 10, 16, 0, tzinfo=UTC)}]


def test_guard_single_keeper():
    assert keeper.guard(active_others=0, chained=False) is True
    assert keeper.guard(active_others=1, chained=False) is False
    assert keeper.guard(active_others=1, chained=True) is True


def test_loop_stops_before_job_limit_and_requests_redispatch():
    t = [DAY]
    sent = []

    def now():
        return t[0]

    def sleep(s):
        t[0] = t[0] + timedelta(seconds=s)

    def list_runs():
        return [run("collect.yml", 0, now=t[0])]          # always fresh -> nothing to do

    stats = keeper.loop(now_fn=now, sleep=sleep, list_runs=list_runs, dispatch=sent.append,
                        max_minutes=60, tick_seconds=300)
    assert sent == []
    assert stats["ticks"] == 12
    assert (t[0] - DAY) <= timedelta(minutes=60)
    assert stats["redispatch"] is True


def test_loop_survives_gh_errors():
    t = [DAY]

    def boom():
        raise RuntimeError("gh down")

    stats = keeper.loop(now_fn=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + timedelta(seconds=s)),
                        list_runs=boom, dispatch=lambda wf: None, max_minutes=10, tick_seconds=300)
    assert stats["errors"] == 2 and stats["redispatch"] is True
