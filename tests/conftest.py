"""Keep tests from writing into the real data/ folder."""
import pytest

from edge_desk import alerts, sharp


@pytest.fixture(autouse=True)
def _tracked_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp, "REJECTED", tmp_path / "sharp_tracked.csv")


@pytest.fixture(autouse=True)
def _alert_log_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "LOG", tmp_path / "alerts_log.csv")


@pytest.fixture(autouse=True)
def _live_log_to_tmp(tmp_path, monkeypatch):
    from edge_desk import live
    monkeypatch.setattr(live, "LIVE_LOG", tmp_path / "live_log.csv")
