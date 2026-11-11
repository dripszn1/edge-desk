"""Snapshot archive (#25): nothing leaves data/ unless the uploaded asset's checksum matches."""
import shutil
from datetime import date
from pathlib import Path

from edge_desk import archive


def _snaps(tmp: Path, days):
    for d in days:
        p = tmp / d
        p.mkdir(parents=True)
        (p / "120000Z.parquet").write_bytes(d.encode() * 100)
        (p / "130000Z.parquet").write_bytes(b"x" * 50)
    return tmp


def test_due_weeks_only_whole_old_weeks():
    days = [date(2026, 11, d) for d in (2, 3, 8, 9, 10, 16)]
    due = archive.due_weeks(days, today=date(2026, 11, 24))   # cutoff 11-10
    assert list(due) == ["snapshots-2026-W45"]                 # W45 = Nov 2..8; W46 (9..15) ends after cutoff
    assert due["snapshots-2026-W45"] == [date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 8)]
    assert archive.due_weeks(days, today=date(2026, 11, 11)) == {}


def _fake_store():
    store = {}

    def up(tag, tar):
        store[tag] = tar.read_bytes()
        return True

    def down(tag, dest):
        if tag not in store:
            return None
        p = dest / f"{tag}.tar"
        p.write_bytes(store[tag])
        return p
    return store, up, down


def test_archive_then_restore_roundtrip(tmp_path):
    snap = _snaps(tmp_path / "snapshots", ["2026-11-02", "2026-11-03"])
    store, up, down = _fake_store()
    days = [date(2026, 11, 2), date(2026, 11, 3)]
    assert archive.archive_week("snapshots-2026-W45", days, snap, up=up, down=down)
    assert not (snap / "2026-11-02").exists()
    assert "snapshots-2026-W45" in (snap / "ARCHIVE.md").read_text()
    assert archive.restore_day("2026-11-03", snap, down=down)
    assert (snap / "2026-11-03" / "120000Z.parquet").read_bytes() == b"2026-11-03" * 100


def test_checksum_mismatch_keeps_files(tmp_path):
    snap = _snaps(tmp_path / "snapshots", ["2026-11-02"])
    store, up, _ = _fake_store()

    def bad_down(tag, dest):
        p = dest / f"{tag}.tar"
        p.write_bytes(store[tag][:-10] + b"0" * 10)
        return p
    assert not archive.archive_week("snapshots-2026-W45", [date(2026, 11, 2)], snap, up=up, down=bad_down)
    assert (snap / "2026-11-02" / "120000Z.parquet").exists() and not (snap / "ARCHIVE.md").exists()


def test_failed_upload_keeps_files(tmp_path):
    snap = _snaps(tmp_path / "snapshots", ["2026-11-02"])
    assert not archive.archive_week("snapshots-2026-W45", [date(2026, 11, 2)], snap,
                                    up=lambda t, p: False, down=lambda t, d: None)
    assert (snap / "2026-11-02").exists()
