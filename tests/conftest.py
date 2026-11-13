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


# ---------- #34: no test may touch the real data/ folder ----------
# collect.yml runs the tests on the runner and then `git add data`, so a test that writes there
# gets its fixture rows committed as real data (4 future-stamped rows in sharp_tracked.csv came
# from an old test run). Any change under data/ during the session fails the run.
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def data_fingerprint(root: Path = DATA) -> dict:
    out = {}
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                st = p.stat()
                out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


def data_changes(before: dict, after: dict) -> list[str]:
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))


def pytest_sessionstart(session):
    session.config._data_before = data_fingerprint()


def pytest_sessionfinish(session, exitstatus):
    before = getattr(session.config, "_data_before", None)
    if before is None:
        return
    changed = data_changes(before, data_fingerprint())
    if changed:
        print(f"\nTESTS WROTE TO data/ (fix the test, don't commit these): {changed[:10]}")
        session.exitstatus = 1


@pytest.fixture(autouse=True)
def _cap_review_to_tmp(tmp_path, monkeypatch):
    from edge_desk import caps
    monkeypatch.setattr(caps, "OUT", tmp_path / "cap_review_bets.csv")


@pytest.fixture(autouse=True)
def _baseline_to_tmp(tmp_path, monkeypatch):
    """#57: placebo draws live in data/; tests must neither read nor write the real ones."""
    from edge_desk import baseline, settle
    monkeypatch.setattr(settle, "PLACEBOS", tmp_path / "baseline_placebos.csv")
    monkeypatch.setattr(baseline, "PLACEBOS", tmp_path / "baseline_placebos.csv")
    monkeypatch.setattr(baseline, "STATE", tmp_path / "baseline_summary.json")
