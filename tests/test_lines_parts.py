"""#71: data/odds/lines.csv hit GitHub's 100 MB file limit (every push rejected 11-16 21:42Z+).
Odds lines now go to day files data/odds/lines/<YYYY-MM-DD>[-N].csv; readers see the old single
file + every part, so nothing changes for scan/grade/lag; the split is row-for-row verified."""
import pandas as pd
import pytest

from edge_desk import grade, odds


def _rows(*ts):
    return pd.DataFrame([{c: None for c in odds.LINE_COLS} | {"fetched_ts": t, "league": "NFL", "book": "pinnacle",
                          "market": "h2h", "outcome": f"o{i}", "price": -110, "implied": 0.52, "fair": 0.5}
                         for i, t in enumerate(ts)])[odds.LINE_COLS]


@pytest.fixture
def lines_path(monkeypatch, tmp_path):
    p = tmp_path / "lines.csv"
    monkeypatch.setattr(odds, "LINES", p)
    return p


def test_append_writes_day_files_not_the_single_file(lines_path):
    odds.append_lines(_rows("2026-11-16T22:13:42Z", "2026-11-17T00:05:00Z"))
    odds.append_lines(_rows("2026-11-16T23:13:42Z"))
    d = lines_path.with_suffix("")
    assert not lines_path.exists()
    assert sorted(p.name for p in d.glob("*.csv")) == ["2026-11-16.csv", "2026-11-17.csv"]
    assert len(pd.read_csv(d / "2026-11-16.csv")) == 2
    assert len(odds.load_lines()) == 3


def test_reads_legacy_file_plus_parts(lines_path):
    _rows("2026-11-10T19:57:08Z").to_csv(lines_path, index=False)
    odds.append_lines(_rows("2026-11-16T22:13:42Z"))
    got = odds.load_lines()
    assert list(got["fetched_ts"]) == ["2026-11-10T19:57:08Z", "2026-11-16T22:13:42Z"]
    assert grade.recorded_pulls(lines_path) == {"2026-11-10T19:57:08Z", "2026-11-16T22:13:42Z"}


def test_part_rolls_over_before_the_size_limit(lines_path, monkeypatch):
    monkeypatch.setattr(odds, "PART_MAX_BYTES", 200)
    for i in range(6):
        odds.append_lines(_rows(f"2026-11-16T22:{i:02d}:00Z"))
    parts = odds.lines_files()
    assert len(parts) > 1 and parts[0].name == "2026-11-16.csv" and parts[1].name == "2026-11-16-2.csv"
    assert all(p.name.startswith("2026-11-16") for p in parts)
    got = odds.load_lines()
    assert list(got["fetched_ts"]) == [f"2026-11-16T22:{i:02d}:00Z" for i in range(6)]   # write order, none lost


def test_split_legacy_is_verified_and_keeps_every_row(lines_path):
    src = _rows("2026-11-10T19:57:08Z", "2026-11-10T20:00:00Z", "2026-11-14T15:00:00Z")
    src.to_csv(lines_path, index=False)
    n = odds.split_legacy()
    assert n == 3 and not lines_path.exists()
    got = odds.load_lines()
    assert len(got) == 3 and set(got["fetched_ts"]) == set(src["fetched_ts"])
    assert odds.split_legacy() == 0                    # idempotent


def test_split_refuses_when_parts_already_hold_those_days(lines_path):
    _rows("2026-11-16T10:00:00Z").to_csv(lines_path, index=False)
    odds.append_lines(_rows("2026-11-16T22:13:42Z"))   # a part for the same day already exists
    with pytest.raises(RuntimeError):
        odds.split_legacy()
    assert lines_path.exists()                          # nothing moved


def test_collect_commit_step_fails_when_push_never_succeeds():
    from pathlib import Path
    wf = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "collect.yml").read_text()
    step = wf.split("name: Commit data", 1)[1].split("- name:", 1)[0]
    assert "exit 1" in step
