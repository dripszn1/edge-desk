"""Collector freshness guard (backlog #15). Stdlib only, so it runs before `pip install`.

`python -m edge_desk.fresh 20` prints the newest Kalshi snapshot's age and, inside GitHub
Actions, writes `skip=true` to $GITHUB_OUTPUT when that snapshot is younger than the given
minutes, so an offset cron slot that lands right after a normal run exits without collecting
(no duplicate data, ~no Actions minutes beyond the 1-minute floor).
Snapshot time comes from the file path (data/snapshots/<YYYY-MM-DD>/<HHMMSS>Z.parquet), never
from file mtimes (a fresh checkout resets those).
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = ROOT / "data" / "snapshots"
PAT = re.compile(r"^(\d{4}-\d{2}-\d{2})$"), re.compile(r"^(\d{6})Z\.parquet$")


def latest_snapshot_ts(snap_dir: Path | None = None) -> datetime | None:
    snap_dir = snap_dir or SNAP_DIR
    best = None
    if not snap_dir.exists():
        return None
    for d in snap_dir.iterdir():
        if not (d.is_dir() and PAT[0].match(d.name)):
            continue
        for f in d.iterdir():
            m = PAT[1].match(f.name)
            if m:
                t = datetime.strptime(d.name + m.group(1), "%Y-%m-%d%H%M%S").replace(tzinfo=timezone.utc)
                best = t if best is None or t > best else best
    return best


def age_minutes(snap_dir: Path | None = None, now: datetime | None = None) -> float | None:
    t = latest_snapshot_ts(snap_dir)
    if t is None:
        return None
    return ((now or datetime.now(timezone.utc)) - t).total_seconds() / 60


def should_skip(min_minutes: float, snap_dir: Path | None = None, now: datetime | None = None) -> bool:
    a = age_minutes(snap_dir, now)
    return a is not None and 0 <= a < min_minutes      # a snapshot "from the future" never skips


LIVE_LOG = ROOT / "data" / "live_log.csv"
VPS_FRESH_MINUTES = 25    # a VPS loop checkpoints every 15 min (#66c); older than this -> it stopped


def vps_active(now: datetime, path: Path | None = None, fresh_min: float = VPS_FRESH_MINUTES) -> bool:
    """#66c: is an always-on VPS live loop running? True when data/live_log.csv has a "vps-" row still
    "running" whose checkpoint is < fresh_min old (never a future stamp). Stdlib only."""
    import csv
    path = path or LIVE_LOG
    if not path.exists():
        return False
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if not str(r.get("run_id", "")).startswith("vps-"):
                continue
            if not str(r.get("exit_reason", "")).startswith("running"):
                continue
            try:
                t = datetime.strptime(r.get("ended_utc", ""), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            age = (now - t).total_seconds() / 60
            if 0 <= age < fresh_min:
                return True
    return False


LIVE_FRESH_MINUTES = 45   # live saves a snapshot every 15 min; older than this -> its loop isn't saving


def _now() -> datetime:
    return datetime.now(timezone.utc)


def decide(now: datetime, snap_dir: Path | None, threshold: float, event: str, live_active: int) -> dict:
    """Should this collect run stand down? (#15 + #32)
    - scheduled slot within `threshold` min of the latest snapshot -> skip ("fresh");
    - any run (dispatch too) while a live run is in progress AND the latest snapshot is younger than
      LIVE_FRESH_MINUTES -> skip ("live"): live already pulls odds, scans, alerts and commits, and two
      writers race on data/odds/state.json, alert dedupe and appended CSVs (live keeps its side on a
      rebase conflict, so collect's credit spend and rows were lost; 11-11 04:45/04:48 double CFB pull).
    No snapshot or one stamped in the future never skips."""
    a = age_minutes(snap_dir, now)
    a = None if a is None else round(a, 1)
    ok = a is not None and a >= 0
    if live_active > 0 and ok and a < LIVE_FRESH_MINUTES:
        return {"skip": True, "reason": "live", "age_min": a}
    if event == "schedule" and ok and a < threshold:
        return {"skip": True, "reason": "fresh", "age_min": a}
    return {"skip": False, "reason": "", "age_min": a}


def _opt(argv: list[str], name: str, default: str) -> str:
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    mins = float(argv[0]) if argv and not argv[0].startswith("--") else 20
    event = _opt(argv, "--event", "schedule")
    raw = _opt(argv, "--live-active", "0").strip()
    live = int(raw) if raw.isdigit() else 0          # unknown -> assume no live run, collect (fails safe)
    if vps_active(_now()):                           # #66c: a VPS live loop counts as a live run
        live += 1
    d = decide(_now(), SNAP_DIR, mins, event, live)
    a = d["age_min"]
    print(f"latest snapshot {'none' if a is None else f'{a:.1f} min old'}; event {event}; live runs {live}; "
          f"skip={str(d['skip']).lower()}{' (' + d['reason'] + ')' if d['skip'] else ''} (threshold {mins:g} min)")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"skip={str(d['skip']).lower()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
