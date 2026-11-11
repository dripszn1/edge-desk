"""Snapshot archive (backlog #25): move snapshot days older than KEEP_DAYS into GitHub Releases.

Runs in Actions only (needs `gh` + GH_TOKEN with contents: write). Nothing is lost:
  1. pick ISO weeks whose every day is older than KEEP_DAYS (a week is archived whole);
  2. tar those day folders -> snapshots-<YYYY>-W<ww>.tar, sha256 it;
  3. upload as the asset of release `snapshots-<YYYY>-W<ww>` (created if missing);
  4. download the asset back and compare sha256; ONLY on a match remove those day folders from
     data/snapshots and add a line to data/snapshots/ARCHIVE.md (week, days, files, bytes, sha256).
Any failure leaves every file in place. Git history keeps the old commits too.

Pull a week back for a backtest/grader: `python -m edge_desk.archive restore 2026-11-03`
(the week holding that day), or `restore_day(day)` from code; restored days are left on disk.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import tarfile
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = ROOT / "data" / "snapshots"
MANIFEST = SNAP_DIR / "ARCHIVE.md"
KEEP_DAYS = 14


def week_tag(d: date) -> str:
    y, w, _ = d.isocalendar()
    return f"snapshots-{y}-W{w:02d}"


def day_dirs(snap_dir: Path = SNAP_DIR) -> list[date]:
    out = []
    for p in snap_dir.iterdir() if snap_dir.exists() else []:
        if p.is_dir():
            try:
                out.append(date.fromisoformat(p.name))
            except ValueError:
                continue
    return sorted(out)


def due_weeks(days: list[date], today: date, keep_days: int = KEEP_DAYS) -> dict[str, list[date]]:
    """{tag: days} for ISO weeks that are entirely older than keep_days (the week's Sunday too)."""
    cutoff = today - timedelta(days=keep_days)
    weeks: dict[str, list[date]] = {}
    for d in days:
        weeks.setdefault(week_tag(d), []).append(d)
    out = {}
    for tag, ds in weeks.items():
        sunday = ds[0] + timedelta(days=6 - ds[0].weekday())
        if sunday < cutoff:
            out[tag] = sorted(ds)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_tar(tag: str, days: list[date], out_dir: Path, snap_dir: Path = SNAP_DIR) -> Path:
    tar = out_dir / f"{tag}.tar"
    with tarfile.open(tar, "w") as t:
        for d in days:
            t.add(snap_dir / d.isoformat(), arcname=d.isoformat())
    return tar


# ---------- GitHub (Actions only) ----------
def gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], capture_output=True, text=True, timeout=600)


def upload(tag: str, tar: Path) -> bool:
    if gh("release", "view", tag).returncode != 0:
        r = gh("release", "create", tag, "--title", tag, "--notes",
               "Kalshi snapshot archive (read-only market data). See data/snapshots/ARCHIVE.md.")
        if r.returncode != 0:
            print(f"archive: release create {tag} failed: {r.stderr.strip()[:200]}")
            return False
    r = gh("release", "upload", tag, str(tar), "--clobber")
    if r.returncode != 0:
        print(f"archive: upload {tag} failed: {r.stderr.strip()[:200]}")
    return r.returncode == 0


def download(tag: str, dest: Path) -> Path | None:
    r = gh("release", "download", tag, "--pattern", f"{tag}.tar", "--dir", str(dest), "--clobber")
    p = dest / f"{tag}.tar"
    return p if r.returncode == 0 and p.exists() else None


# ---------- archive / restore ----------
def archive_week(tag: str, days: list[date], snap_dir: Path = SNAP_DIR, manifest: Path | None = None,
                 up=upload, down=download) -> bool:
    """Upload, verify by checksum, then (only then) remove the day folders. True if moved."""
    manifest = manifest or snap_dir / "ARCHIVE.md"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tar = build_tar(tag, days, tmp, snap_dir)
        want = sha256(tar)
        if not up(tag, tar):
            return False
        back_dir = tmp / "back"
        back_dir.mkdir()
        back = down(tag, back_dir)
        if back is None or sha256(back) != want:
            print(f"archive: {tag} checksum check failed, keeping files")
            return False
        files = sum(1 for d in days for _ in (snap_dir / d.isoformat()).rglob("*") if _.is_file())
        size = tar.stat().st_size
    for d in days:
        for p in sorted((snap_dir / d.isoformat()).rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        (snap_dir / d.isoformat()).rmdir()
    new = not manifest.exists()
    with open(manifest, "a") as f:
        if new:
            f.write("# Archived snapshot weeks\n\nEach week is a GitHub Release asset `<tag>.tar` "
                    "(restore: `python -m edge_desk.archive restore <YYYY-MM-DD>`).\n\n"
                    "| release | days | files | bytes | sha256 | archived (UTC) |\n|---|---|---|---|---|---|\n")
        f.write(f"| {tag} | {days[0]}..{days[-1]} | {files} | {size} | {want} | "
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')} |\n")
    print(f"archive: {tag} moved ({len(days)} days, {files} files, {size} bytes)")
    return True


def restore_day(day: date | str, snap_dir: Path = SNAP_DIR, down=download) -> bool:
    d = date.fromisoformat(str(day)) if not isinstance(day, date) else day
    if (snap_dir / d.isoformat()).exists():
        return True
    tag = week_tag(d)
    with tempfile.TemporaryDirectory() as tmp:
        tar = down(tag, Path(tmp))
        if tar is None:
            print(f"archive: {tag} not found")
            return False
        with tarfile.open(tar) as t:
            safe = [m for m in t.getmembers() if not m.name.startswith(("/", "..")) and ".." not in Path(m.name).parts]
            try:
                t.extractall(snap_dir, members=safe, filter="data")
            except TypeError:                          # Python without extraction filters
                t.extractall(snap_dir, members=safe)
    return (snap_dir / d.isoformat()).exists()


def run(today: date | None = None) -> list[str]:
    today = today or datetime.now(timezone.utc).date()
    moved = []
    for tag, days in sorted(due_weeks(day_dirs(), today).items()):
        if archive_week(tag, days):
            moved.append(tag)
    if not moved:
        print("archive: nothing due")
    return moved


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "restore":
        sys.exit(0 if restore_day(sys.argv[2]) else 1)
    run()
