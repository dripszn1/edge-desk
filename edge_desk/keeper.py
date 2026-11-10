"""Cron-independent keeper (backlog #21). Stdlib only; read-only (never places or prepares orders).

GitHub's scheduler has not fired a single cron slot on the public repo, while `workflow_dispatch`
works every time. `.github/workflows/keeper.yml` therefore runs one long, cheap job that every
TICK minutes looks at the Actions run list and dispatches:
  collect.yml  inside collect hours when the newest collect run is >= COLLECT_EVERY_MINUTES old
               and none is queued/running, and no live run is active (live persists its own
               snapshots every 15 min, so collecting too would duplicate data);
  live.yml     inside a game window (live.current_window, same early start) when no live run is
               queued/running, at most once per LIVE_RETRY_MINUTES (live's own guard still
               checks window, budget and duplicates).
It stops before the 6 h job limit and asks the workflow to re-dispatch itself (chained=true).
Only one keeper runs at a time (guard + concurrency group). If the chain ever breaks, the
builder's watchdog dispatch of collect.yml restarts it (collect.yml's last step).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

from . import live

UTC = timezone.utc
COLLECT_EVERY_MINUTES = 30
COLLECT_HOURS_UTC = set(range(14, 24)) | set(range(0, 6))   # same as collect.yml's cron
LIVE_RETRY_MINUTES = 20
TICK_SECONDS = 300
MAX_MINUTES = 340
ACTIVE = {"queued", "in_progress", "pending", "waiting", "requested"}


def parse_runs(workflow: str, rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        try:
            t = datetime.strptime(r["createdAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        except (KeyError, TypeError, ValueError):
            continue
        out.append({"workflow": workflow, "status": r.get("status", ""), "id": r.get("databaseId"),
                    "created_at": t})
    return out


def _age_min(now: datetime, t: datetime) -> float:
    return (now - t).total_seconds() / 60


def plan(now: datetime, runs: list[dict], state: dict | None = None) -> tuple[list[str], dict]:
    """Which workflows to dispatch now. `runs` = recent runs of collect.yml and live.yml."""
    state = dict(state or {})
    now = now.astimezone(UTC)
    acts: list[str] = []

    def of(wf):
        return [r for r in runs if r["workflow"] == wf]

    live_active = any(r["status"] in ACTIVE for r in of("live.yml"))
    win = live.current_window(now, live.config.LIVE_EARLY_START_MINUTES)
    last_live = state.get("live_dispatched")
    if win and not live_active and (last_live is None or _age_min(now, last_live) >= LIVE_RETRY_MINUTES):
        acts.append("live.yml")
        state["live_dispatched"] = now

    if now.hour in COLLECT_HOURS_UTC and not live_active:
        col = of("collect.yml")
        busy = any(r["status"] in ACTIVE for r in col)
        # a timestamp from the future is never treated as fresh
        ages = [a for a in (_age_min(now, r["created_at"]) for r in col) if a >= 0]
        newest = min(ages) if ages else None
        last_col = state.get("collect_dispatched")
        just_sent = last_col is not None and 0 <= _age_min(now, last_col) < COLLECT_EVERY_MINUTES
        if not busy and not just_sent and (newest is None or newest >= COLLECT_EVERY_MINUTES):
            acts.append("collect.yml")
            state["collect_dispatched"] = now
    return acts, state


def guard(active_others: int, chained: bool) -> bool:
    return chained or active_others == 0


# ---------- gh plumbing ----------
def _repo() -> str:
    return os.environ.get("GITHUB_REPOSITORY", "")


def gh_list_runs() -> list[dict]:
    out = []
    for wf in ("collect.yml", "live.yml"):
        p = subprocess.run(["gh", "run", "list", "--repo", _repo(), "--workflow", wf, "--limit", "10",
                            "--json", "databaseId,status,createdAt"],
                           capture_output=True, text=True, timeout=60, check=True)
        out += parse_runs(wf, json.loads(p.stdout or "[]"))
    return out


def gh_dispatch(wf: str) -> None:
    subprocess.run(["gh", "workflow", "run", wf, "--repo", _repo(), "--ref",
                    os.environ.get("GITHUB_REF_NAME", "main")], check=True, timeout=60)


def loop(now_fn=lambda: datetime.now(UTC), sleep=time.sleep, list_runs=gh_list_runs,
         dispatch=gh_dispatch, max_minutes: float = MAX_MINUTES, tick_seconds: float = TICK_SECONDS) -> dict:
    start = now_fn()
    stop = start + timedelta(minutes=max_minutes)
    state: dict = {}
    stats = {"ticks": 0, "errors": 0, "dispatched": [], "redispatch": False}
    while now_fn() < stop:
        stats["ticks"] += 1
        t = now_fn()
        try:
            acts, state = plan(t, list_runs(), state)
            for wf in acts:
                dispatch(wf)
                stats["dispatched"].append(f"{t:%H:%M}Z {wf}")
                print(f"keeper {t:%Y-%m-%dT%H:%MZ}: dispatched {wf}", flush=True)
        except Exception as e:                      # never die on a gh hiccup; try next tick
            stats["errors"] += 1
            print(f"keeper {t:%Y-%m-%dT%H:%MZ}: error {e}", flush=True)
        left = (stop - now_fn()).total_seconds()
        if left <= 0:
            break
        sleep(min(tick_seconds, left))
    stats["redispatch"] = True
    return stats


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "loop"
    if cmd == "guard":
        active = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 0
        chained = len(argv) > 2 and argv[2].lower() == "true"
        go = guard(active, chained)
        print(f"keeper guard: go={go} (other active keepers={active}, chained={chained})")
        live._gh_output(go=str(go).lower())
        return 0
    if cmd == "loop":
        stats = loop()
        print(f"keeper done: {stats}")
        live._gh_output(redispatch=str(stats["redispatch"]).lower())
        return 0
    print("usage: python3 -m edge_desk.keeper [guard <active> <chained>|loop]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
