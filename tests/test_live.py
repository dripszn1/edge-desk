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


def test_checkpoint_logs_minutes_even_if_run_is_killed(monkeypatch, tmp_path):
    """Each 15-min commit upserts this run's live_log row, so a cancelled run still counts."""
    monkeypatch.setattr(scan, "FLAG_DIR", tmp_path)
    log = tmp_path / "ll.csv"
    c = FakeClock(datetime(2026, 11, 14, 16, 0, tzinfo=UTC))
    monkeypatch.setattr(live, "deadline", lambda *a, **k: c.t + timedelta(minutes=40))
    live.run_loop(now_fn=c.now, clock=c.clock, sleep=c.sleep,
                  fetch_full=lambda m, ts: (pd.DataFrame(), [], False), odds_fn=lambda: None,
                  scan_fn=lambda *a: None, persist_fn=lambda *a: None, post_fn=lambda: [],
                  commit_fn=lambda m: True, used_min=0, seen=set(), alert_fn=lambda df, ts: [],
                  log_fn=lambda row: live.upsert_log(row, log))
    rows = pd.read_csv(log)
    assert len(rows) == 1 and rows["exit_reason"].iloc[0].startswith("running")   # one row per run, replaced
    assert rows["minutes"].iloc[0] >= 30
    assert live.minutes_used_this_month(datetime(2026, 11, 14, 17, tzinfo=UTC), log) >= 30


def test_upsert_replaces_same_run_only(tmp_path):
    p = tmp_path / "l.csv"
    live.upsert_log({"run_id": "a", "started_utc": "2026-11-14T16:00:00Z", "minutes": 10}, p)
    live.upsert_log({"run_id": "b", "started_utc": "2026-11-14T17:00:00Z", "minutes": 5}, p)
    live.upsert_log({"run_id": "a", "started_utc": "2026-11-14T16:00:00Z", "minutes": 40, "exit_reason": "window end"}, p)
    r = pd.read_csv(p)
    assert sorted(r["run_id"]) == ["a", "b"] and r.set_index("run_id").loc["a", "minutes"] == 40


def test_loop_odds_check_follows_plan_size(monkeypatch, tmp_path):
    """#30: a 20k plan checks odds every 5 min, the free tier every LIVE_ODDS_CHECK_MINUTES."""
    from edge_desk import odds
    start = datetime(2026, 11, 14, 16, 0, tzinfo=UTC)
    monkeypatch.setattr(odds, "load_state", lambda *a: {"remaining": 476, "used": 24})
    free = []
    _run(monkeypatch, tmp_path, start, 30, _pass_frame, free)
    monkeypatch.setattr(odds, "load_state", lambda *a: {"remaining": 19000, "used": 1000})
    big = []
    _run(monkeypatch, tmp_path / "b", start, 30, _pass_frame, big)
    assert len(free) <= 3 + 1 and len(big) >= 6


# ---------- #39b fast lane ----------
def test_fast_tickers_recent_alerts_only_and_capped():
    from datetime import datetime, timezone
    from edge_desk import live
    now = datetime(2026, 11, 14, 20, 0, tzinfo=timezone.utc)
    st = {"alerts": {
        "a": {"ticker": "T1", "sent_utc": "2026-11-14T19:00:00Z", "kickoff_utc": "2026-11-14T21:00:00Z"},
        "b": {"ticker": "T2", "sent_utc": "2026-11-13T10:00:00Z"},                                  # >24h
        "c": {"ticker": "T3", "sent_utc": "2026-11-14T10:00:00Z", "kickoff_utc": "2026-11-14T12:00:00Z"},  # game over
        "d": {"ticker": None, "legs": ["YES X", "NO Y"], "sent_utc": "2026-11-14T19:00:00Z"}}}
    assert live.fast_tickers(st, now) == ["T1"]
    many = {"alerts": {str(i): {"ticker": f"T{i}", "sent_utc": "2026-11-14T19:00:00Z"} for i in range(80)}}
    assert len(live.fast_tickers(many, now)) == live.config.LIVE_FAST_MAX_TICKERS


def test_fast_tickers_keep_open_alerts_until_game_over_soonest_first():
    """#40: an alert from days ago stays in the fast lane until kickoff + 6h; soonest kickoff wins the cap."""
    from datetime import datetime, timezone
    from edge_desk import live
    now = datetime(2026, 11, 14, 20, 0, tzinfo=timezone.utc)
    st = {"alerts": {
        "wed": {"ticker": "SAT", "sent_utc": "2026-11-11T06:00:00Z", "kickoff_utc": "2026-11-14T20:30:00Z"},
        "ingame": {"ticker": "LIVE", "sent_utc": "2026-11-10T22:00:00Z", "kickoff_utc": "2026-11-14T17:00:00Z"},
        "sun": {"ticker": "SUN", "sent_utc": "2026-11-14T19:00:00Z", "kickoff_utc": "2026-11-15T18:00:00Z"},
        "over": {"ticker": "OVER", "sent_utc": "2026-11-13T19:00:00Z", "kickoff_utc": "2026-11-14T13:00:00Z"},
        "ancient": {"ticker": "OLD", "sent_utc": "2026-11-01T19:00:00Z", "kickoff_utc": "2026-11-15T18:00:00Z"},
        "noko": {"ticker": "NOKO", "sent_utc": "2026-11-13T10:00:00Z"}}}
    assert live.fast_tickers(st, now) == ["LIVE", "SAT", "SUN"]
    # cap: in-game / near-tip contracts are kept, far-off ones dropped
    many = {"alerts": {str(i): {"ticker": f"F{i}", "sent_utc": "2026-11-14T19:00:00Z",
                                "kickoff_utc": "2026-11-16T01:00:00Z"} for i in range(60)}}
    many["alerts"]["x"] = {"ticker": "NOW", "sent_utc": "2026-11-12T19:00:00Z", "kickoff_utc": "2026-11-14T19:00:00Z"}
    got = live.fast_tickers(many, now)
    assert len(got) == live.config.LIVE_FAST_MAX_TICKERS and got[0] == "NOW"


def test_fast_lane_ticks_between_passes_and_backs_off(monkeypatch):
    from datetime import datetime, timedelta, timezone
    import pandas as pd
    from edge_desk import live
    t = {"now": datetime(2026, 11, 14, 19, 0, tzinfo=timezone.utc), "mono": 0.0}

    def now_fn():
        return t["now"]

    def clock():
        return t["mono"]

    def sleep(s):
        t["mono"] += s
        t["now"] += timedelta(seconds=s)

    calls = []

    def fast(nf, stats):
        calls.append(t["mono"])
        if len(calls) == 3:
            raise live.Throttled("x")

    def full(m, ts):
        t["mono"] += 1
        return pd.DataFrame(), [], False

    s = live.run_loop(now_fn=now_fn, clock=clock, sleep=sleep, fetch_full=full, fetch_hot=full,
                      odds_fn=lambda: None, scan_fn=lambda *a: None, persist_fn=lambda *a: None,
                      post_fn=lambda: [], commit_fn=lambda m: False, used_min=0, seen=set(),
                      alert_fn=lambda df, ts: [], log_fn=lambda r: None, fast_fn=fast)
    # ticks are >= LIVE_FAST_SECONDS apart, and a 429 stops the fast lane for the rest of that wait
    gaps = [b - a for a, b in zip(calls, calls[1:])]
    assert calls and all(g >= live.config.LIVE_FAST_SECONDS - 1e-9 for g in gaps)
    assert len(calls) > 3 and any(g > 2 * live.config.LIVE_FAST_SECONDS for g in gaps)


def test_log_fast_only_changes(tmp_path):
    import pandas as pd
    from edge_desk import live
    live._fast_last.clear()
    df = pd.DataFrame([{"snapshot_ts": "2026-11-14T19:00:00Z", "ticker": "T1", "yes_bid": 0.4, "yes_ask": 0.45}])
    assert live.log_fast(df, tmp_path) == 1
    assert live.log_fast(df, tmp_path) == 0
    df2 = df.assign(snapshot_ts="2026-11-14T19:00:03Z", yes_bid=0.41)
    assert live.log_fast(df2, tmp_path) == 1
    assert len(pd.read_csv(tmp_path / "2026-11-14.csv")) == 2


# ---------- #54: a merge mid-run restarts the loop on the new code ----------
def _code_run(monkeypatch, tmp_path, revs, minutes=60):
    monkeypatch.setattr(scan, "FLAG_DIR", tmp_path)
    commits = []
    c = FakeClock(datetime(2026, 11, 14, 16, 0, tzinfo=UTC))
    monkeypatch.setattr(live, "deadline", lambda *a, **k: c.t + timedelta(minutes=minutes))
    it = iter(revs)
    last = [None]

    def rev():
        last[0] = next(it, last[0])
        return last[0]
    s = live.run_loop(now_fn=c.now, clock=c.clock, sleep=c.sleep,
                      fetch_full=lambda m, ts: (pd.DataFrame(), [], False), odds_fn=lambda: None,
                      scan_fn=lambda *a: None, persist_fn=lambda *a: None, post_fn=lambda: [],
                      commit_fn=lambda m: commits.append(m) or True, used_min=0, seen=set(),
                      alert_fn=lambda df, ts: [], log_fn=lambda row: None, code_fn=rev)
    return s, c, commits


def test_loop_ends_after_a_commit_pulls_new_code(monkeypatch, tmp_path):
    # rev at start "a", after the first 15-min commit "b": stop right there, final commit still made
    s, c, commits = _code_run(monkeypatch, tmp_path, ["a", "b"])
    assert s.restart == "code updated"
    assert c.t < s.stop_at - timedelta(minutes=30)
    assert commits[-1].endswith("(end)")


def test_loop_same_code_runs_to_deadline(monkeypatch, tmp_path):
    s, c, _ = _code_run(monkeypatch, tmp_path, ["a"])
    assert s.restart is None and c.t >= s.stop_at - timedelta(seconds=31)


def test_no_restart_near_the_deadline(monkeypatch, tmp_path):
    # new code with < 10 min left: a restart would cost more than it buys
    s, c, _ = _code_run(monkeypatch, tmp_path, ["a", "b"], minutes=20)
    assert s.restart is None


def test_code_rev_tracks_python_files(tmp_path):
    (tmp_path / "edge_desk").mkdir()
    f = tmp_path / "edge_desk" / "x.py"
    f.write_text("a = 1\n")
    r1 = live.code_rev(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "y.csv").write_text("1\n")          # data never counts
    assert live.code_rev(tmp_path) == r1
    f.write_text("a = 2\n")
    assert live.code_rev(tmp_path) != r1


def test_main_redispatches_on_restart():
    end = datetime(2026, 11, 14, 18, 0, tzinfo=UTC)
    assert live.exit_reason(end, end + timedelta(hours=3), end + timedelta(hours=5), "code updated") == "code updated"
    assert live.exit_reason(end, end, end, None) == "window end"
