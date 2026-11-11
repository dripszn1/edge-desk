"""Game-day live mode (backlog #16): windows, budget, pacing, ordering, staleness, fees, dedupe."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from edge_desk import config, live, odds, scan
from tests.test_core import row

UTC = timezone.utc


def ct(y, m, d, h, mi):
    return datetime(y, m, d, h, mi, tzinfo=live.CT).astimezone(UTC)


# ---------- windows ----------
@pytest.mark.parametrize("t,inside", [
    (ct(2026, 11, 14, 10, 0), True),     # Sat start
    (ct(2026, 11, 14, 9, 59), False),
    (ct(2026, 11, 14, 23, 29), True),
    (ct(2026, 11, 14, 23, 30), False),   # end is exclusive
    (ct(2026, 11, 15, 22, 31), False),   # Sun ends 10:30 PM
    (ct(2026, 11, 15, 11, 0), True),
    (ct(2026, 11, 16, 17, 29), False),   # Mon starts 5:30 PM (CBB tips)
    (ct(2026, 11, 16, 17, 30), True),
    (ct(2026, 11, 16, 18, 0), True),
    (ct(2026, 11, 10, 17, 30), True),    # Tue MACtion
    (ct(2026, 11, 11, 12, 0), False),    # Wed midday
    (ct(2026, 9, 12, 10, 0), True),      # CDT Saturday
])
def test_windows(t, inside):
    assert (live.current_window(t) is not None) is inside


def test_dst_window_in_utc():
    s, e = live.window_on(datetime(2026, 9, 12).date())     # CDT = UTC-5
    assert s == datetime(2026, 9, 12, 15, 0, tzinfo=UTC) and e == datetime(2026, 9, 13, 4, 30, tzinfo=UTC)
    s, _ = live.window_on(datetime(2026, 11, 14).date())    # CST = UTC-6
    assert s == datetime(2026, 11, 14, 16, 0, tzinfo=UTC)


def test_early_fire_waits_for_window():
    assert live.current_window(ct(2026, 11, 14, 9, 50), early_min=15) is not None
    assert live.current_window(ct(2026, 11, 14, 9, 40), early_min=15) is None


# ---------- guard / budget / deadline ----------
def test_guard():
    t = ct(2026, 11, 14, 12, 0)
    assert live.guard(t, 0, False, 0)[0] is True
    assert live.guard(t, 1, False, 0)[0] is False          # backup cron: one already running
    assert live.guard(t, 1, True, 0)[0] is True            # chained re-dispatch queues behind it
    assert live.guard(t, 0, False, config.LIVE_MINUTES_BUDGET_MONTH)[0] is False
    assert live.guard(ct(2026, 11, 11, 12, 0), 0, False, 0)[0] is False
    assert live.guard(ct(2026, 11, 14, 23, 25), 0, False, 0)[0] is False   # < 10 min left


def test_deadline_is_tightest_limit():
    s = ct(2026, 11, 14, 10, 0)
    _, e = live.window_on(s.astimezone(live.CT).date())
    assert live.deadline(s, e, 0) == s + timedelta(minutes=config.LIVE_MAX_RUN_MINUTES)
    late = ct(2026, 11, 14, 21, 0)
    assert live.deadline(late, e, 0) == e
    assert live.deadline(s, e, config.LIVE_MINUTES_BUDGET_MONTH - 30) == s + timedelta(minutes=30)
    assert live.deadline(s, e, config.LIVE_MINUTES_BUDGET_MONTH + 99) == s


def test_minutes_log_sums_this_month_only(tmp_path):
    p = tmp_path / "live_log.csv"
    live.append_log({"run_id": 1, "started_utc": "2026-10-31T20:00:00Z", "minutes": 300}, p)
    live.append_log({"run_id": 2, "started_utc": "2026-11-14T16:00:00Z", "minutes": 332}, p)
    live.append_log({"run_id": 3, "started_utc": "2026-11-14T21:30:00Z", "minutes": 120}, p)
    assert live.minutes_used_this_month(ct(2026, 11, 15, 12, 0), p) == 452
    assert live.billed_minutes(ct(2026, 11, 14, 10, 0), ct(2026, 11, 14, 10, 30)) == 32


def test_redispatch():
    _, e = live.window_on(datetime(2026, 11, 14).date())
    assert live.should_redispatch(e - timedelta(hours=2), e, 0) is True
    assert live.should_redispatch(e - timedelta(minutes=5), e, 0) is False
    assert live.should_redispatch(e - timedelta(hours=2), e, config.LIVE_MINUTES_BUDGET_MONTH) is False


# ---------- pacing ----------
def test_pacer_fast_full_every_pass():
    p = live.Pacer()
    assert p.next_kind(0, True) == "full"
    p.done("full", 0, 10, False)                 # 10 s full pass -> fast enough for 30 s
    assert p.next_kind(30, True) == "full"


def test_pacer_slow_full_uses_hot_passes():
    p = live.Pacer()
    p.done("full", 0, 50, False)                 # 50 s full pass -> too slow
    assert p.next_kind(60, True) == "hot"
    assert p.next_kind(60, False) == "full"      # nothing hot -> full
    assert p.next_kind(0 + config.LIVE_SLOW_SECONDS, True) == "full"


def test_pacer_backoff_on_429():
    p = live.Pacer()
    p.done("hot", 0, 1, True)
    assert p.interval == 2 * config.LIVE_SNAPSHOT_SECONDS
    for _ in range(10):
        p.done("hot", 0, 1, True)
    assert p.interval == config.LIVE_MAX_BACKOFF_SECONDS
    for _ in range(10):
        p.done("hot", 0, 1, False)
    assert p.interval == config.LIVE_SNAPSHOT_SECONDS


def _tagged(ts, kickoffs_utc):
    return pd.DataFrame([{"snapshot_ts": ts, "event_ticker": f"E{i}", "kickoff_utc": k}
                         for i, k in enumerate(kickoffs_utc)])


def test_hot_events_window_and_stale():
    now = datetime(2026, 11, 14, 16, 0, tzinfo=UTC)
    df = _tagged("2026-11-14T15:59:00Z", ["2026-11-14T17:00:00Z", "2026-11-14T20:30:00Z",
                                          "2026-11-14T15:00:00Z", None])
    assert live.hot_events(df, now) == ["E0"]                  # E1 > 3h, E2 started, E3 unknown
    stale = _tagged("2026-11-14T15:00:00Z", ["2026-11-14T17:00:00Z"])
    assert live.hot_events(stale, now) == []                   # stale full pass -> force a full


def test_odds_interval_by_plan():
    assert live.odds_interval_hours(500) == 1.0
    assert live.odds_interval_hours(20000) < 0.5


def test_odds_near_kickoff_interval_still_budgeted():
    now = pd.Timestamp("2026-11-14T16:00:00Z")
    ko = {"NFL": None, "CFB": "2026-11-14T17:00:00Z"}
    st = {"last_pull": {"CFB": "2026-11-14T14:30:00Z"}, "remaining": 400,
          "remaining_seen_utc": "2026-11-14T14:30:00Z", "spent": {"2026-11-14": 3}}
    assert odds.plan_pulls(st, now, ko) == []                          # 4h game-day interval
    assert odds.plan_pulls(st, now, ko, near_h=1.5, near_interval_h=1.0) == ["CFB"]
    broke = dict(st, remaining=odds.ODDS_RESERVE + 2)
    assert odds.plan_pulls(broke, now, ko, near_h=1.5, near_interval_h=1.0) == []   # reserve holds
    spent = dict(st, spent={"2026-11-14": 200})
    assert odds.plan_pulls(spent, now, ko, near_h=1.5, near_interval_h=1.0) == []   # daily allowance holds


# ---------- the loop: ordering, staleness, fees, dedupe ----------
class FakeClock:
    def __init__(self, start):
        self.t = start
        self.mono = 0.0

    def now(self):
        return self.t

    def clock(self):
        return self.mono

    def sleep(self, s):
        self.t += timedelta(seconds=s)
        self.mono += s


def _pass_frame(ts, bid_hi=0.70):
    # broken ladder: YES lo @.55 + NO hi @(1-bid_hi): a real arb only if bid_hi is high enough
    rs = [row("A3", "A", 2.5, 0.54, 0.55), row("A7", "A", 6.5, bid_hi, bid_hi + 0.01)]
    for r in rs:
        r["snapshot_ts"] = ts
        r["occurrence_datetime"] = "2026-11-14T22:00:00Z"    # kickoff 19:00Z
    return pd.DataFrame(rs)


def _run(monkeypatch, tmp_path, start, minutes, frame_fn, odds_log=None, scans=None, seen=None):
    monkeypatch.setattr(scan, "FLAG_DIR", tmp_path)
    monkeypatch.setattr(scan, "write_event_map", lambda df: None)
    monkeypatch.setattr(scan, "run_model", lambda *a, **k: {"status": {}, "live": [], "shadow": []})
    monkeypatch.setattr(scan, "run_sharp", lambda *a, **k: {"status": {}, "live": [], "shadow": []})
    monkeypatch.setattr(scan, "lost_today", lambda *a, **k: 0.0)
    c = FakeClock(start)
    stop = start + timedelta(minutes=minutes)
    monkeypatch.setattr(live, "deadline", lambda *a, **k: stop)

    def fetch(mults, ts):
        c.mono += 5
        return frame_fn(ts), [], False

    def odds_fn():
        (odds_log if odds_log is not None else []).append(c.now())

    def scan_fn(df, ts, seen, partial):
        if scans is not None:
            scans.append((ts, set(df["snapshot_ts"])))
        return scan.run(df, ts, seen=seen, partial=partial)

    return live.run_loop(now_fn=c.now, clock=c.clock, sleep=c.sleep, fetch_full=fetch,
                         fetch_hot=lambda *a: (pd.DataFrame(), [], False), odds_fn=odds_fn, scan_fn=scan_fn,
                         persist_fn=lambda *a: None, post_fn=lambda: [], commit_fn=lambda m: False,
                         used_min=0, seen=set() if seen is None else seen, alert_fn=lambda df, ts: [])


def test_loop_logs_each_contract_once(monkeypatch, tmp_path):
    s = _run(monkeypatch, tmp_path, datetime(2026, 11, 14, 16, 0, tzinfo=UTC), 5, _pass_frame)
    assert s.stats["full_passes"] >= 8                          # ~30 s cadence over 5 min
    f = pd.read_csv(tmp_path / "flags.csv")
    assert len(f) == 1 and f["kind"].iloc[0] == "LADDER"       # seen every pass, logged once
    assert s.stats["flags_logged"] == 1


def test_loop_odds_before_snapshot_and_one_ts_per_scan(monkeypatch, tmp_path):
    odds_log, scans = [], []
    _run(monkeypatch, tmp_path, datetime(2026, 11, 14, 16, 0, tzinfo=UTC), 25, _pass_frame, odds_log, scans)
    assert len(odds_log) >= 3                                   # every LIVE_ODDS_CHECK_MINUTES
    first_ts = pd.Timestamp(scans[0][0])
    assert odds_log[0] <= first_ts                              # odds fetched before the snapshot
    assert all(stamps == {ts} for ts, stamps in scans)          # each scan sees exactly one pass


def test_loop_fees_kill_thin_gap(monkeypatch, tmp_path):
    # 2c raw gap on two legs (.55 + .47 = 1.02 before fees? no: .55 + (1-.57)=.98) -> fees eat it
    _run(monkeypatch, tmp_path, datetime(2026, 11, 14, 16, 0, tzinfo=UTC), 2,
         lambda ts: _pass_frame(ts, bid_hi=0.57))
    assert not (tmp_path / "flags.csv").exists()


def test_loop_stale_pass_never_flags(monkeypatch, tmp_path):
    old = lambda ts: _pass_frame("2026-11-14T14:00:00Z")        # quotes 2 h old
    _run(monkeypatch, tmp_path, datetime(2026, 11, 14, 16, 0, tzinfo=UTC), 2,
         lambda ts: old(ts))
    assert not (tmp_path / "flags.csv").exists()


def test_scan_stale_ts_blocks_flags(monkeypatch, tmp_path):
    monkeypatch.setattr(scan, "FLAG_DIR", tmp_path)
    monkeypatch.setattr(scan, "run_model", lambda *a, **k: {"status": {}, "live": [], "shadow": []})
    monkeypatch.setattr(scan, "run_sharp", lambda *a, **k: {"status": {}, "live": [], "shadow": []})
    ts = (datetime.now(UTC) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert scan.run(_pass_frame(ts), ts, seen=set(), partial=True) == []


def test_fresh_rows_drops_old_quotes_and_old_passes():
    now = datetime(2026, 11, 14, 16, 0, 30, tzinfo=UTC)
    mixed = pd.concat([_pass_frame("2026-11-14T16:00:00Z"), _pass_frame("2026-11-14T15:58:00Z")])
    assert set(live.fresh_rows(mixed, "2026-11-14T16:00:00Z", now)["snapshot_ts"]) == {"2026-11-14T16:00:00Z"}
    old = _pass_frame("2026-11-14T15:00:00Z")
    assert live.fresh_rows(old, "2026-11-14T15:00:00Z", now).empty             # pass itself stale
    fut = _pass_frame("2026-11-14T16:05:00Z")
    assert live.fresh_rows(fut, "2026-11-14T16:05:00Z", now).empty             # from the future


def test_seed_seen_skips_relogging(monkeypatch, tmp_path):
    seen = {("LADDER", "YES A3", "NO A7")}
    _run(monkeypatch, tmp_path, datetime(2026, 11, 14, 16, 0, tzinfo=UTC), 2, _pass_frame, seen=seen)
    assert not (tmp_path / "flags.csv").exists()


def test_loop_outside_window_does_nothing(monkeypatch, tmp_path):
    c = FakeClock(ct(2026, 11, 11, 12, 0))
    s = live.run_loop(now_fn=c.now, clock=c.clock, sleep=c.sleep,
                      fetch_full=lambda *a: pytest.fail("no Kalshi calls outside a window"),
                      odds_fn=lambda: pytest.fail("no odds outside a window"), used_min=0, seen=set(),
                      alert_fn=lambda df, ts: pytest.fail("no alerts outside a window"))
    assert s.stats["full_passes"] == 0


def test_loop_stops_at_deadline_and_commits(monkeypatch, tmp_path):
    commits = []
    monkeypatch.setattr(scan, "FLAG_DIR", tmp_path)
    c = FakeClock(datetime(2026, 11, 14, 16, 0, tzinfo=UTC))
    monkeypatch.setattr(live, "deadline", lambda *a, **k: c.t + timedelta(minutes=40))
    s = live.run_loop(now_fn=c.now, clock=c.clock, sleep=c.sleep,
                      fetch_full=lambda m, ts: (pd.DataFrame(), [], False), odds_fn=lambda: None,
                      scan_fn=lambda *a: None, persist_fn=lambda *a: None, post_fn=lambda: [],
                      commit_fn=lambda m: commits.append(m) or True, used_min=0, seen=set(),
                      alert_fn=lambda df, ts: [])
    assert c.t <= s.stop_at + timedelta(seconds=1)
    assert len(commits) >= 3 and commits[-1].endswith("(end)")   # every 15 min + final
