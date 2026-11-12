"""#47: a SHARP play must be priced off a sharp pull we actually recorded.

Four test-fixture rows reached data/flags/sharp_tracked.csv before #34's data guard existed
(odds pulls stamped 2026-11-12T17:00:00Z / 2026-11-14T14:30:00Z that never happened). Once their
snapshot time passed they were graded as real held-back bets on tonight's real NYG/WAS ticker.
The grader now skips any play whose odds_fetched_ts is not a pull in data/odds/lines.csv.
Nothing is deleted from data/: the rows stay in the log, they just never become bets."""
from datetime import datetime, timezone

import pandas as pd

from edge_desk import grade

NOW = datetime(2026, 11, 12, 23, tzinfo=timezone.utc)
COLS = ["snapshot_ts", "league", "event", "ticker", "side", "price", "fair", "edge", "sharp_line",
        "odds_fetched_ts", "reason"]
CAP = "edge above sanity cap"


def tracked(*rows):
    return pd.DataFrame([dict(zip(COLS, r)) for r in rows])


def lines(*ts):
    return pd.DataFrame({"fetched_ts": list(ts), "league": "NFL"})


REAL = ("2026-11-12T15:20:47Z", "NFL", "e", "KXNFLTOTAL-26NOV12WASNYG-33", "NO", 0.22, 0.27, 0.04, 40.5,
        "2026-11-12T14:46:30Z", "sharp line stale")
FAKE = ("2026-11-12T18:00:00Z", "NFL", "e", "KXNFLSPREAD-26NOV12WASNYG-NYG4", "YES", 0.30, 0.44, 0.13, 3.0,
        "2026-11-12T17:00:00Z", CAP)


def test_unrecorded_pull_is_never_graded(tmp_path):
    p = tmp_path / "lines.csv"
    lines("2026-11-10T19:57:08Z", "2026-11-12T14:46:30Z").to_csv(p, index=False)
    pulls = grade.recorded_pulls(p)
    kept = grade.drop_unrecorded(tracked(REAL, FAKE), pulls)
    assert list(kept["ticker"]) == ["KXNFLTOTAL-26NOV12WASNYG-33"]
    b = grade.tracked_bets(kept, pd.DataFrame(columns=["fetched_ts", "ticker", "status", "result"]), NOW)
    assert "YES KXNFLSPREAD-26NOV12WASNYG-NYG4" not in set(b["leg1"])


def test_no_lines_file_means_no_filter(tmp_path):
    pulls = grade.recorded_pulls(tmp_path / "missing.csv")
    assert pulls is None
    assert len(grade.drop_unrecorded(tracked(REAL, FAKE), pulls)) == 2


def test_blank_or_pre_history_pull_kept(tmp_path):
    """MODEL rows carry no pull; rows older than the first recorded pull (an archived/trimmed
    lines file) can't be checked, so they are kept rather than silently dropped."""
    p = tmp_path / "lines.csv"
    lines("2026-11-12T14:46:30Z").to_csv(p, index=False)
    old = REAL[:9] + ("2026-11-01T10:00:00Z",) + REAL[10:]
    blank = REAL[:9] + ("",) + REAL[10:]
    kept = grade.drop_unrecorded(tracked(old, blank, FAKE), grade.recorded_pulls(p))
    assert len(kept) == 2 and FAKE[3] not in set(kept["ticker"])


def test_no_column_passes_through():
    df = pd.DataFrame({"snapshot_ts": ["2026-11-12T15:00:00Z"], "leg1": ["YES X"]})
    assert grade.drop_unrecorded(df, {"2026-11-12T14:46:30Z"}).equals(df)
