"""Collector freshness guard (backlog #15)."""
from datetime import datetime, timezone

from edge_desk import fresh

NOW = datetime(2026, 11, 14, 15, 0, tzinfo=timezone.utc)


def _snap(tmp_path, day, hhmmss):
    d = tmp_path / day
    d.mkdir(exist_ok=True)
    (d / f"{hhmmss}Z.parquet").write_bytes(b"")


def test_skip_when_recent(tmp_path):
    _snap(tmp_path, "2026-11-14", "144500")
    assert fresh.should_skip(20, tmp_path, NOW) is True
    assert fresh.should_skip(10, tmp_path, NOW) is False


def test_newest_across_days(tmp_path):
    _snap(tmp_path, "2026-11-13", "235900")
    _snap(tmp_path, "2026-11-14", "010000")
    assert fresh.latest_snapshot_ts(tmp_path) == datetime(2026, 11, 14, 1, 0, tzinfo=timezone.utc)


def test_no_data_or_future_never_skips(tmp_path):
    assert fresh.should_skip(20, tmp_path / "none", NOW) is False
    _snap(tmp_path, "2026-11-14", "153000")       # 30 min in the future
    assert fresh.should_skip(20, tmp_path, NOW) is False


def test_writes_github_output(tmp_path, monkeypatch):
    out = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setattr(fresh, "SNAP_DIR", tmp_path / "none")
    fresh.main(["20"])
    assert out.read_text().strip() == "skip=false"


# ---- #32: collect stands down while a live run owns the data ----
def test_live_active_skips_even_dispatched_runs(tmp_path):
    _snap(tmp_path, "2026-11-14", "145000")       # live saved a snapshot 10 min ago
    d = fresh.decide(NOW, tmp_path, threshold=20, event="workflow_dispatch", live_active=1)
    assert d["skip"] is True and d["reason"] == "live"


def test_no_live_dispatch_still_collects(tmp_path):
    _snap(tmp_path, "2026-11-14", "145000")
    assert fresh.decide(NOW, tmp_path, 20, "workflow_dispatch", 0)["skip"] is False


def test_live_active_but_stale_data_collects_anyway(tmp_path):
    _snap(tmp_path, "2026-11-14", "140000")       # 60 min old: the live loop is not saving
    d = fresh.decide(NOW, tmp_path, 20, "workflow_dispatch", 1)
    assert d["skip"] is False


def test_live_active_no_or_future_data_collects(tmp_path):
    assert fresh.decide(NOW, tmp_path / "none", 20, "schedule", 1)["skip"] is False
    _snap(tmp_path, "2026-11-14", "153000")       # stamped in the future -> never trusted
    assert fresh.decide(NOW, tmp_path, 20, "schedule", 1)["skip"] is False


def test_schedule_freshness_unchanged(tmp_path):
    _snap(tmp_path, "2026-11-14", "144500")
    assert fresh.decide(NOW, tmp_path, 20, "schedule", 0) == {"skip": True, "reason": "fresh", "age_min": 15.0}
    assert fresh.decide(NOW, tmp_path, 20, "workflow_dispatch", 0)["skip"] is False


def test_main_parses_live_flags(tmp_path, monkeypatch):
    out = tmp_path / "out.txt"
    _snap(tmp_path, "2026-11-14", "145000")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setattr(fresh, "SNAP_DIR", tmp_path)
    monkeypatch.setattr(fresh, "_now", lambda: NOW)
    fresh.main(["20", "--event", "workflow_dispatch", "--live-active", "2"])
    assert "skip=true" in out.read_text()
    out.write_text("")
    fresh.main(["20", "--event", "workflow_dispatch", "--live-active", "garbage"])   # unknown -> collect
    assert "skip=false" in out.read_text()
