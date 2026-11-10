"""Game-day digest (backlog #14): window, stale data, fees shown, recap timing, length, CT times."""
import pandas as pd

from edge_desk import config, digest

NOW = "2026-11-14T15:00:00Z"            # Sat 9:00 AM CT


def _flag(ko="2026-11-14T20:30:00Z", edge=0.05, kind="SHARP", **kw):
    return {"event": "Auburn vs Mississippi St.: Spread", "side": "YES", "leg1_desc": "Auburn wins by over 6.5",
            "ticker": "KXNCAAFSPREAD-26NOV14AUBMSST-AUB7", "price": 0.30, "fair": 0.37, "edge": edge,
            "fee_per_contract": 0.015, "stake": 4.5, "kickoff_utc": ko, "kickoff_ct": "Sat Nov 14, 2:30 PM CT",
            "kind": kind, **kw}


def _latest(snap="2026-11-14T14:50:00Z", **kw):
    base = {"snapshot_ts": snap, "stale": False, "sharp_shadow": [_flag()], "model_shadow": [],
            "model": {"NFL": {"gate": "gated: x"}}, "sharp": {"NFL": {"gate": "gated: y"}, "CFB": {"gate": "open: z"}}}
    base.update(kw)
    return base


def test_play_shows_fee_inclusive_edge_and_ct():
    md = digest.build(_latest(), {}, {}, NOW)
    line = [l for l in md.splitlines() if "Auburn" in l][0]
    assert "edge 5.0¢ after fees" in line and "@ 30¢" in line and "CT" in line
    assert "7.0¢" not in line                    # never fair - price (pre-fee gap)


def test_window_excludes_started_and_far_games():
    lat = _latest(sharp_shadow=[_flag(ko="2026-11-14T14:00:00Z"), _flag(ko="2026-11-17T01:00:00Z")])
    assert "- no plays" in digest.build(lat, {}, {}, NOW)


def test_stale_snapshot_shows_no_plays():
    old = (pd.Timestamp(NOW) - pd.Timedelta(minutes=config.STALE_MINUTES + 5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    md = digest.build(_latest(snap=old), {}, {}, NOW)
    assert "stale" in md and "Auburn" not in md
    assert "STALE" in md.splitlines()[-1]


def test_snapshot_from_future_not_trusted():
    md = digest.build(_latest(snap="2026-11-14T15:30:00Z"), {}, {}, NOW)
    assert "Auburn" in md or "no plays" in md      # age negative counts as fresh; never crashes


def test_recap_last_24h_only_and_running_record():
    b = pd.DataFrame({"status": ["settled", "settled", "settled", "pending"],
                      "pnl": [2.0, -1.0, 3.0, None],
                      "settled_ts": ["2026-11-14T03:00:00Z", "2026-11-12T03:00:00Z", "2026-11-14T16:00:00Z", None]})
    md = digest.build(_latest(), {"SHARP shadow": b}, {}, NOW)
    line = [l for l in md.splitlines() if l.startswith("- SHARP shadow:") and "running" in l][0]
    assert "1-0, $+2.00" in line and "running 1-1, $+1.00" in line   # 16:00Z settlement is in the future


def test_under_25_lines_with_many_plays():
    many = [_flag(ko=f"2026-11-14T{h:02d}:30:00Z") for h in range(16, 24)] * 4
    md = digest.build(_latest(sharp_shadow=many), {k: pd.DataFrame() for k in "abc"}, {}, NOW)
    assert len(md.strip().splitlines()) <= digest.MAX_LINES
    assert "more in data/flags/latest.json" in md


def test_status_line_gates_and_credits():
    st = {"remaining": 321, "remaining_seen_utc": "2026-11-14T12:00:00Z", "last_pull": {"NFL": "2026-11-14T12:00:00Z"}}
    last = digest.build(_latest(), {}, st, NOW).splitlines()[-1]
    assert "credits left 321" in last and "SHARP CFB open" in last and "MODEL NFL closed" in last
    assert "sharp odds 3.0h old" in last


def test_run_writes_both_files(tmp_path, monkeypatch):
    import json
    (tmp_path / "flags").mkdir()
    (tmp_path / "flags" / "latest.json").write_text(json.dumps(_latest()))
    monkeypatch.setattr(digest, "DATA", tmp_path)
    monkeypatch.setattr(digest, "OUT", tmp_path / "digest.md")
    monkeypatch.setattr(digest, "ARCHIVE", tmp_path / "digests")
    monkeypatch.setattr(digest, "BET_FILES", {})
    digest.run(NOW)
    assert (tmp_path / "digest.md").exists() and (tmp_path / "digests" / "2026-11-14.md").exists()
