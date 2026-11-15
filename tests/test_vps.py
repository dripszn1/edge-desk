"""#66c: always-on VPS runner. The Actions live job and collect stand down while a VPS loop is
checkpointing; VPS minutes never count against the Actions budget; deploy files carry no secrets."""
import csv
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from edge_desk import fresh, live

UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 15, 18, 0, tzinfo=UTC)


def _log(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=live.LOG_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in live.LOG_COLS})


def _row(run_id, ended, reason="running (checkpoint)", minutes=30):
    return {"run_id": run_id, "started_utc": (ended - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ended_utc": ended.strftime("%Y-%m-%dT%H:%M:%SZ"), "minutes": minutes, "exit_reason": reason}


def test_vps_active_only_on_a_fresh_running_vps_checkpoint(tmp_path):
    p = tmp_path / "live_log.csv"
    assert fresh.vps_active(NOW, p) is False                                    # no log
    _log(p, [_row("123", NOW - timedelta(minutes=2))])                          # an Actions run
    assert fresh.vps_active(NOW, p) is False
    _log(p, [_row("vps-20261115T170000Z", NOW - timedelta(minutes=10))])
    assert fresh.vps_active(NOW, p) is True
    _log(p, [_row("vps-20261115T170000Z", NOW - timedelta(minutes=40))])        # stale: the VPS died
    assert fresh.vps_active(NOW, p) is False
    _log(p, [_row("vps-20261115T170000Z", NOW - timedelta(minutes=5), "window end")])   # it ended cleanly
    assert fresh.vps_active(NOW, p) is False
    _log(p, [_row("vps-20261115T170000Z", NOW + timedelta(minutes=30))])        # future-stamped never counts
    assert fresh.vps_active(NOW, p) is False


def test_collect_stands_down_while_vps_runs(tmp_path):
    d = tmp_path / "2026-11-15"
    d.mkdir()
    (d / "175000Z.parquet").write_bytes(b"")
    assert fresh.decide(NOW, tmp_path, 20, "workflow_dispatch", 0)["skip"] is False
    assert fresh.decide(NOW, tmp_path, 20, "workflow_dispatch", 1)["reason"] == "live"


def test_actions_guard_stands_down_for_vps_even_when_chained():
    t = datetime(2026, 11, 14, 18, 0, tzinfo=UTC)                               # Sat noon CT
    assert live.guard(t, 0, False, 0)[0] is True
    go, why = live.guard(t, 0, True, 0, vps=True)
    assert go is False and "vps" in why


def test_vps_minutes_are_not_actions_minutes(tmp_path):
    p = tmp_path / "live_log.csv"
    _log(p, [_row("123", NOW, "window end", 100), _row("vps-20261115T120000Z", NOW, "window end", 300)])
    assert live.minutes_used_this_month(NOW, p) == 100


def test_checkpoint_row_names_the_runner(monkeypatch):
    s = live.Session(NOW, NOW + timedelta(hours=1), NOW + timedelta(hours=2))
    monkeypatch.setenv("EDGE_RUNNER", "vps")
    assert live.checkpoint_row(s, NOW)["run_id"] == "vps-20261115T180000Z"
    monkeypatch.delenv("EDGE_RUNNER")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    assert live.checkpoint_row(s, NOW)["run_id"] == "42"


class _Clock:
    def __init__(self):
        self.t, self.mono = datetime(2026, 11, 14, 16, 0, tzinfo=UTC), 0.0

    def now(self):
        return self.t

    def clock(self):
        return self.mono

    def sleep(self, s):
        self.t += timedelta(seconds=s)
        self.mono += s


def _loop(monkeypatch, vps_states, runner=None):
    if runner:
        monkeypatch.setenv("EDGE_RUNNER", runner)
    c = _Clock()
    monkeypatch.setattr(live, "deadline", lambda *a, **k: c.t + timedelta(minutes=90))
    it, commits = iter(vps_states), []
    s = live.run_loop(now_fn=c.now, clock=c.clock, sleep=c.sleep,
                      fetch_full=lambda m, ts: (pd.DataFrame(), [], False), odds_fn=lambda: None,
                      scan_fn=lambda *a: None, persist_fn=lambda *a: None, post_fn=lambda: [],
                      commit_fn=lambda m: commits.append(m) or True, used_min=0, seen=set(),
                      alert_fn=lambda df, ts: [], log_fn=lambda row: None, code_fn=None,
                      vps_fn=lambda: next(it, False))
    return s, c, commits


def test_actions_loop_hands_over_when_a_vps_checkpoint_appears(monkeypatch):
    s, c, commits = _loop(monkeypatch, [False, True])
    assert s.restart == "vps active"
    assert c.t < s.stop_at - timedelta(minutes=30)
    assert commits[-1].endswith("(end)")                                       # its data still lands
    assert live.should_redispatch(c.t, s.window_end, 0, restart=s.restart) is False


def test_vps_loop_never_hands_over_to_itself(monkeypatch):
    s, c, _ = _loop(monkeypatch, [True, True, True, True, True, True], runner="vps")
    assert s.restart is None and c.t >= s.stop_at - timedelta(seconds=31)


# ---------- deploy/ files ----------
def test_deploy_files_exist_and_run_as_vps():
    for f in ("Dockerfile", "run.sh", "edge-desk.service", "env.example"):
        assert (ROOT / "deploy" / f).is_file(), f
    run = (ROOT / "deploy" / "run.sh").read_text()
    assert "EDGE_RUNNER=vps" in run and "edge_desk.live loop" in run and "pytest" in run
    assert subprocess.run(["bash", "-n", str(ROOT / "deploy" / "run.sh")]).returncode == 0


def test_env_example_has_no_values_and_secrets_are_ignored():
    for line in (ROOT / "deploy" / "env.example").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            k, _, v = line.partition("=")
            assert v.strip() == "" or k.strip() in ("EDGE_RUNNER", "KALSHI_PRIVATE_KEY_FILE"), line
    gi = (ROOT / ".gitignore").read_text()
    assert "deploy/env" in gi and "*.pem" in gi


def test_deploy_never_touches_orders():
    text = " ".join((ROOT / "deploy" / f).read_text().lower() for f in ("Dockerfile", "run.sh", "edge-desk.service"))
    assert not re.search(r"/portfolio|/orders|create_order|balance", text)
