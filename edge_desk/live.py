"""Game-day live mode (backlog #16). Read-only: never places or prepares orders.

`.github/workflows/live.yml` runs two jobs:
  guard  `python3 -m edge_desk.live guard ...`  (stdlib only, no pip install)
         go only inside a game window (CT), under the monthly minutes budget, and when no other
         live run is already active (a chained re-dispatch skips that check: the concurrency
         group queues it behind the run that dispatched it).
  loop   `python -m edge_desk.live loop`
         one long job: sharp odds (budgeted, BEFORE the snapshot -> no look-ahead), Kalshi pass,
         scan + MODEL + SHARP on that pass only, persist/grade/commit every ~15 min, exit before
         the 6 h job limit and ask the workflow to re-dispatch itself if the window continues.

Windows (America/Chicago): Tue-Fri 5:30-11:30 PM (MACtion/TNF), Sat 10 AM-11:30 PM,
Sun 11 AM-10:30 PM, Mon 6-11:30 PM.

Passes: a FULL pass pulls every in-scope series (as collect.py does). If a full pass is fast
enough (<= 80% of LIVE_SNAPSHOT_SECONDS) every pass is full; otherwise HOT passes (only events
kicking off within LIVE_HOT_HOURS, one call per event) run every LIVE_SNAPSHOT_SECONDS and a full
pass every LIVE_SLOW_SECONDS. Every pass is scanned on its own rows only (one snapshot_ts), never
merged with older quotes, so the staleness check means what it says. 429s double the cadence
up to LIVE_MAX_BACKOFF_SECONDS; quiet passes halve it back.

Flags are deduped per contract for the whole session (seeded from today's logs), so a contract is
logged once however many passes see it. Minutes used per run go to data/live_log.csv.
"""
from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from . import config

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LIVE_LOG = DATA / "live_log.csv"
CT = ZoneInfo(config.TIMEZONE)
UTC = timezone.utc

# weekday (Mon=0) -> (start, end) local CT times as (h, m)
WINDOWS = {
    0: ((18, 0), (23, 30)),
    1: ((17, 30), (23, 30)),
    2: ((17, 30), (23, 30)),
    3: ((17, 30), (23, 30)),
    4: ((17, 30), (23, 30)),
    5: ((10, 0), (23, 30)),
    6: ((11, 0), (22, 30)),
}
LOG_COLS = ["run_id", "started_utc", "ended_utc", "minutes", "full_passes", "hot_passes",
            "snapshots_saved", "commits", "odds_checks", "flags_logged", "alerts", "errors", "exit_reason"]


# ---------- windows ----------
def _utc(t: datetime) -> datetime:
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def window_on(day: date) -> tuple[datetime, datetime]:
    (sh, sm), (eh, em) = WINDOWS[day.weekday()]
    s = datetime(day.year, day.month, day.day, sh, sm, tzinfo=CT)
    e = datetime(day.year, day.month, day.day, eh, em, tzinfo=CT)
    return s.astimezone(UTC), e.astimezone(UTC)


def current_window(now: datetime, early_min: float = 0) -> tuple[datetime, datetime] | None:
    """The window containing `now` (or starting within early_min minutes), in UTC."""
    now = _utc(now)
    s, e = window_on(now.astimezone(CT).date())
    if s - timedelta(minutes=early_min) <= now < e:
        return s, e
    return None


# ---------- minutes budget ----------
def minutes_used_this_month(now: datetime, path: Path | None = None) -> float:
    path = path or LIVE_LOG
    if not path.exists():
        return 0.0
    month = _utc(now).strftime("%Y-%m")
    tot = 0.0
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if str(r.get("started_utc", "")).startswith(month):
                try:
                    tot += float(r.get("minutes") or 0)
                except ValueError:
                    pass
    return tot


def deadline(start: datetime, window_end: datetime, used_min: float,
             budget_min: float = config.LIVE_MINUTES_BUDGET_MONTH,
             max_run_min: float = config.LIVE_MAX_RUN_MINUTES) -> datetime:
    """When the loop must stop: job limit, window end, or the rest of the monthly budget."""
    left = max(budget_min - used_min, 0)
    return min(_utc(start) + timedelta(minutes=max_run_min), _utc(window_end),
               _utc(start) + timedelta(minutes=left))


def guard(now: datetime, active_runs: int, chained: bool, used_min: float,
          budget_min: float = config.LIVE_MINUTES_BUDGET_MONTH) -> tuple[bool, str]:
    w = current_window(now, config.LIVE_EARLY_START_MINUTES)
    if w is None:
        return False, "outside game windows"
    if used_min + 10 > budget_min:
        return False, f"monthly live budget used ({used_min:.0f}/{budget_min:.0f} min)"
    if active_runs > 0 and not chained:
        return False, f"{active_runs} live run(s) already active"
    if w[1] - _utc(now) < timedelta(minutes=10):
        return False, "window ends in < 10 min"
    return True, "go"


def should_redispatch(now: datetime, window_end: datetime, used_min: float,
                      budget_min: float = config.LIVE_MINUTES_BUDGET_MONTH) -> bool:
    return _utc(window_end) - _utc(now) >= timedelta(minutes=10) and used_min + 10 <= budget_min


# ---------- pass planning ----------
@dataclass
class Pacer:
    """Decides full vs hot passes and the cadence (with 429 backoff)."""
    target: float = config.LIVE_SNAPSHOT_SECONDS
    slow: float = config.LIVE_SLOW_SECONDS
    ceiling: float = config.LIVE_MAX_BACKOFF_SECONDS
    interval: float = config.LIVE_SNAPSHOT_SECONDS
    full_seconds: float | None = None        # duration of the last full pass
    last_full: float | None = None           # monotonic time of the last full pass

    @property
    def full_is_fast(self) -> bool:
        return self.full_seconds is not None and self.full_seconds <= 0.8 * self.target

    def next_kind(self, t: float, have_hot: bool) -> str:
        if self.last_full is None or self.full_is_fast or not have_hot:
            return "full"
        return "full" if t - self.last_full >= self.slow else "hot"

    def done(self, kind: str, t_start: float, t_end: float, throttled: bool) -> None:
        if kind == "full":
            self.full_seconds, self.last_full = t_end - t_start, t_start
        if throttled:
            self.interval = min(self.interval * 2, self.ceiling)
        else:
            self.interval = max(self.interval / 2, self.target)


def hot_events(df, now: datetime, hours: float = config.LIVE_HOT_HOURS) -> list[str]:
    """Events from the last FULL pass whose kickoff is ahead and within `hours`. Uses that pass's
    own kickoff tags; if the pass is stale, nothing is hot (forces a full pass)."""
    from .scan import is_stale
    if df is None or df.empty or "kickoff_utc" not in df:
        return []
    ts = df["snapshot_ts"].iloc[0]
    if is_stale(ts, _utc(now)):
        return []
    import pandas as pd
    ko = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")
    n = pd.Timestamp(_utc(now))
    m = ko.notna() & (ko > n) & (ko <= n + pd.Timedelta(hours=hours))
    return sorted(df.loc[m, "event_ticker"].dropna().unique().tolist())


def odds_interval_hours(monthly_credits: int) -> float:
    """Near-kickoff odds interval sized to The Odds API plan (free tier -> hourly at most)."""
    if monthly_credits >= 20000:
        return 10 / 60
    if monthly_credits >= 5000:
        return 0.5
    return 1.0


# ---------- Kalshi passes ----------
class Throttled(Exception):
    pass


def _get(path: str, params: dict | None = None):
    """One Kalshi GET. 429 -> Throttled (the loop backs off; no blocking retries here)."""
    from . import collect
    r = collect.session.get(config.KALSHI_BASE + path, params=params, timeout=30)
    if r.status_code == 429:
        raise Throttled(path)
    r.raise_for_status()
    return r.json()


def full_pass(mults: dict, ts: str):
    """Every open event in every in-scope series. Returns (frame, errors, throttled)."""
    import pandas as pd
    from . import collect
    rows, errors, throttled = [], [], False
    for league, series_list in config.SERIES.items():
        for s in series_list:
            try:
                if s not in mults:
                    mults[s] = collect.fee_multiplier(s)
                cursor = None
                while True:
                    p = {"series_ticker": s, "status": "open", "with_nested_markets": "true", "limit": 200}
                    if cursor:
                        p["cursor"] = cursor
                    j = _get("/events", p)
                    for ev in j.get("events", []):
                        rows += collect.flatten(league, s, mults[s], ev, ts)
                    cursor = j.get("cursor")
                    if not cursor:
                        break
            except Throttled:
                throttled = True
                errors.append(f"{s}: 429")
            except Exception as e:  # noqa: BLE001 - log the gap, keep going
                errors.append(f"{s}: {type(e).__name__}")
    return pd.DataFrame(rows), errors, throttled


def hot_pass(events: list[str], meta: dict, mults: dict, ts: str):
    """Only the given events, one GET each. `meta[event] = (league, series)` from the last full pass."""
    import pandas as pd
    from . import collect
    rows, errors, throttled = [], [], False
    for ev_t in events:
        league, s = meta[ev_t]
        try:
            j = _get(f"/events/{ev_t}", {"with_nested_markets": "true"})
            ev = dict(j.get("event") or {})
            if not ev.get("markets"):
                ev["markets"] = j.get("markets", [])
            rows += collect.flatten(league, s, mults.get(s, 1.0), ev, ts)
        except Throttled:
            throttled = True
            errors.append(f"{ev_t}: 429")
            break                                   # stop hammering; back off
        except Exception as e:  # noqa: BLE001
            errors.append(f"{ev_t}: {type(e).__name__}")
    return pd.DataFrame(rows), errors, throttled


def persist(df, ts: str, errors: list[str]) -> Path | None:
    """Same file layout and log as collect.py, so fresh/grade/lag/digest see live snapshots."""
    if df is None or df.empty:
        return None
    t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    out = DATA / "snapshots" / t.strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{t.strftime('%H%M%S')}Z.parquet"
    df.drop(columns=[c for c in ("kickoff_utc", "kickoff_ct", "phase") if c in df], errors="ignore") \
      .to_parquet(f, index=False)
    log = DATA / "collector_log.csv"
    new = not log.exists()
    with log.open("a") as fh:
        if new:
            fh.write("snapshot_ts,rows,errors\n")
        fh.write(f"{ts},{len(df)},\"live: {' ; '.join(errors)}\"\n")
    return f


def fresh_rows(df, ts: str, now: datetime):
    """Rows of this pass only (snapshot_ts == ts), and none at all if ts is older than
    STALE_MINUTES or in the future. Never lets an older quote ride along with a new pass."""
    import pandas as pd
    if df is None or df.empty or "snapshot_ts" not in df:
        return pd.DataFrame()
    t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    age = (_utc(now) - t).total_seconds()
    if age > config.STALE_MINUTES * 60 or age < -60:
        return pd.DataFrame()
    return df[df["snapshot_ts"] == ts]


# ---------- session dedupe ----------
def seed_seen(now: datetime, hours: float = 36) -> set:
    """Keys of flags already logged in the last `hours`, so a chained run doesn't re-log them."""
    import pandas as pd
    from . import scan, sharp
    seen = set()
    cut = pd.Timestamp(_utc(now)) - pd.Timedelta(hours=hours)
    for p in (scan.FLAG_DIR / "flags.csv", scan.FLAG_DIR / "model_shadow.csv", scan.FLAG_DIR / "model_flags.csv",
              sharp.SHADOW, sharp.LIVE):
        try:
            f = pd.read_csv(p)
        except Exception:  # noqa: BLE001 - missing file = nothing seen
            continue
        t = pd.to_datetime(f["snapshot_ts"], utc=True, errors="coerce")
        for r in f[t >= cut].to_dict("records"):
            r = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
            seen.add(scan.flag_key(r))
    return seen


def default_alerts(df, ts: str) -> list:
    """Alert on what scan.run just found (scan.run.last), priced against this pass's quotes."""
    from . import alerts, scan
    last = getattr(scan.run, "last", None)
    if not last:
        return []
    m, sh = last["model"], last["sharp"]
    groups = [(last["flags"], False), (m.get("live", []), True), (m.get("shadow", []), False),
              (sh.get("live", []), True), (sh.get("shadow", []), False)]
    scan.run.last = None                               # never alert twice on the same scan
    return alerts.process(groups, alerts.quotes_of(df), lost_today=scan.lost_today())


# ---------- post steps + git ----------
POST_STEPS = ["lag", "settle", "grade", "weekly", "dashboard", "digest"]


def post_steps() -> list[str]:
    errs = []
    for m in POST_STEPS:
        r = subprocess.run([sys.executable, "-m", f"edge_desk.{m}"], cwd=ROOT)
        if r.returncode:
            errs.append(f"{m} exit {r.returncode}")
    return errs


def git_commit(msg: str) -> bool:
    def g(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    g("add", "data")
    if g("diff", "--cached", "--quiet").returncode == 0:
        return False
    g("commit", "-q", "-m", msg)
    for i in range(3):
        if g("pull", "--rebase", "-q").returncode != 0:
            g("rebase", "--abort")
            if g("pull", "--rebase", "-q", "-X", "theirs").returncode != 0:   # keep our data on conflict
                g("rebase", "--abort")
                time.sleep(5)
                continue
        if g("push", "-q").returncode == 0:
            return True
        time.sleep(5)
    print("live: push failed after 3 tries", file=sys.stderr)
    return False


def append_log(row: dict, path: Path | None = None) -> None:
    path = path or LIVE_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_COLS})


def billed_minutes(start: datetime, end: datetime, setup_min: float = 2) -> int:
    return int(math.ceil((_utc(end) - _utc(start)).total_seconds() / 60 + setup_min))


# ---------- the loop ----------
@dataclass
class Session:
    started: datetime
    stop_at: datetime
    window_end: datetime
    stats: dict = field(default_factory=lambda: {k: 0 for k in
                        ("full_passes", "hot_passes", "snapshots_saved", "commits", "odds_checks", "flags_logged",
                         "alerts")})
    errors: list = field(default_factory=list)


def run_loop(now_fn=lambda: datetime.now(UTC), clock=time.monotonic, sleep=time.sleep,
             fetch_full=full_pass, fetch_hot=hot_pass, odds_fn=None, scan_fn=None,
             persist_fn=persist, post_fn=post_steps, commit_fn=git_commit, used_min: float | None = None,
             seen: set | None = None, alert_fn=None) -> Session:
    """Run until the deadline. All I/O is injectable so the ordering guarantees are testable."""
    from . import kickoffs, odds, scan
    odds_fn = odds_fn or (lambda: odds.run(near_h=config.LIVE_ODDS_NEAR_HOURS,
                                           near_interval_h=odds_interval_hours(odds.MONTHLY_CREDITS)))
    scan_fn = scan_fn or (lambda df, ts, seen, partial: scan.run(df, ts, seen=seen, partial=partial))
    alert_fn = alert_fn or default_alerts
    start = _utc(now_fn())
    w = current_window(start, config.LIVE_EARLY_START_MINUTES)
    if w is None:
        return Session(start, start, start)
    used = minutes_used_this_month(start) if used_min is None else used_min
    sess = Session(start, deadline(start, w[1], used), w[1])
    if start < w[0]:                                   # fired a little early: wait for the window
        sleep((w[0] - start).total_seconds())
    seen = seed_seen(start) if seen is None else seen
    pacer, mults, meta = Pacer(), {}, {}
    last_full_df = None
    last_odds = last_persist = last_commit = None
    while True:
        now = _utc(now_fn())
        if now >= sess.stop_at:
            break
        t0 = clock()
        # 1) sharp odds first: anything they price was fetched before this pass's snapshot_ts
        if last_odds is None or (now - last_odds) >= timedelta(minutes=config.LIVE_ODDS_CHECK_MINUTES):
            try:
                odds_fn()
            except Exception as e:  # noqa: BLE001
                sess.errors.append(f"odds: {type(e).__name__}")
            sess.stats["odds_checks"] += 1
            last_odds = now
        # 2) Kalshi pass, stamped AFTER the odds call
        hot = hot_events(last_full_df, now) if last_full_df is not None else []
        kind = pacer.next_kind(t0, bool(hot))
        ts = _utc(now_fn()).strftime("%Y-%m-%dT%H:%M:%SZ")
        if kind == "full":
            df, errs, thr = fetch_full(mults, ts)
        else:
            df, errs, thr = fetch_hot(hot, meta, mults, ts)
        t1 = clock()
        pacer.done(kind, t0, t1, thr)
        sess.stats[f"{kind}_passes"] += 1
        sess.errors += errs[:3]
        # 3) scan this pass only: rows stamped with this pass's ts, and only if that ts is fresh
        df = fresh_rows(df, ts, _utc(now_fn()))
        if not df.empty:
            if kind == "full" and not thr:
                last_full_df = kickoffs.tag_kalshi(df, ts)
                meta = {r["event_ticker"]: (r["league"], r["series"])
                        for r in last_full_df.drop_duplicates("event_ticker").to_dict("records")}
            before = len(seen)
            try:
                scan_fn(df, ts, seen, kind == "hot" or thr)
            except Exception as e:  # noqa: BLE001 - keep looping
                sess.errors.append(f"scan: {type(e).__name__}: {e}")
            sess.stats["flags_logged"] += len(seen) - before
            try:                                       # #17: push in the same iteration
                sess.stats["alerts"] = sess.stats.get("alerts", 0) + len(alert_fn(df, ts) or [])
            except Exception as e:  # noqa: BLE001
                sess.errors.append(f"alerts: {type(e).__name__}")
            # 4) persist a full snapshot every LIVE_PERSIST_MINUTES
            if kind == "full" and (last_persist is None or now - last_persist >= timedelta(minutes=config.LIVE_PERSIST_MINUTES)):
                if persist_fn(df, ts, errs):
                    sess.stats["snapshots_saved"] += 1
                    last_persist = now
        # 5) grade + commit every LIVE_COMMIT_MINUTES
        if last_commit is None:
            last_commit = now
        elif now - last_commit >= timedelta(minutes=config.LIVE_COMMIT_MINUTES):
            sess.errors += post_fn()
            if commit_fn(f"data: live {ts}"):
                sess.stats["commits"] += 1
            last_commit = now
        # 6) wait for the next tick (never past the deadline)
        left = (sess.stop_at - _utc(now_fn())).total_seconds()
        wait = min(max(pacer.interval - (clock() - t0), 0), max(left, 0))
        if wait > 0:
            sleep(wait)
    sess.errors += post_fn()
    if commit_fn(f"data: live {_utc(now_fn()).strftime('%Y-%m-%dT%H:%MZ')} (end)"):
        sess.stats["commits"] += 1
    return sess


# ---------- CLI ----------
def _gh_output(**kv) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            for k, v in kv.items():
                fh.write(f"{k}={v}\n")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "loop"
    now = datetime.now(UTC)
    if cmd == "guard":
        active = int(argv[1]) if len(argv) > 1 and argv[1].lstrip("-").isdigit() else 0
        chained = len(argv) > 2 and argv[2].lower() == "true"
        go, why = guard(now, active, chained, minutes_used_this_month(now))
        print(f"live guard: {why} (now {now.astimezone(CT):%a %H:%M} CT, active={active}, chained={chained})")
        _gh_output(go=str(go).lower())
        return 0
    if cmd == "loop":
        used = minutes_used_this_month(now)
        sess = run_loop(used_min=used)
        end = datetime.now(UTC)
        mins = billed_minutes(sess.started, end)
        reason = ("window end" if end >= sess.window_end - timedelta(minutes=1)
                  else "job limit / budget" if end >= sess.stop_at else "outside window")
        append_log({"run_id": os.environ.get("GITHUB_RUN_ID", "local"),
                    "started_utc": sess.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "ended_utc": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "minutes": mins,
                    **sess.stats, "errors": " ; ".join(sess.errors[:20]), "exit_reason": reason})
        redo = should_redispatch(end, sess.window_end, used + mins)
        _gh_output(redispatch=str(redo).lower())
        git_commit(f"data: live log {end:%Y-%m-%dT%H:%MZ}")
        print(f"live: {mins} min, {sess.stats}, errors={len(sess.errors)}, redispatch={redo}")
        return 0
    print(f"usage: python -m edge_desk.live [guard <active> <chained>|loop]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
