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
