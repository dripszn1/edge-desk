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


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    mins = float(argv[0]) if argv else 20
    a = age_minutes()
    skip = should_skip(mins)
    print(f"latest snapshot {'none' if a is None else f'{a:.1f} min old'}; skip={str(skip).lower()} (threshold {mins:g} min)")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"skip={str(skip).lower()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
