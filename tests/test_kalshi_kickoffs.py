"""Kickoff + phase from Kalshi's own occurrence_datetime (kickoff + 3h)."""
import pandas as pd

from edge_desk import config
from edge_desk.kickoffs import tag_kalshi

SNAP = "2026-11-14T18:00:00Z"


def _df(occ):
    return pd.DataFrame({"event_ticker": ["E"] * len(occ), "league": ["CFB"] * len(occ),
                         "occurrence_datetime": occ})


def test_kickoff_is_occurrence_minus_3h():
    out = tag_kalshi(_df(["2026-11-14T23:00:00Z"]), SNAP)
    assert out.kickoff_utc[0] == "2026-11-14T20:00:00Z"
    assert out.kickoff_ct[0] == "Sat Nov 14, 2:00 PM CT"
    assert out.phase[0] == "pregame"


def test_live_after_kickoff():
    assert tag_kalshi(_df(["2026-11-14T20:00:00Z"]), SNAP).phase[0] == "live"  # ko 17:00Z


def test_inside_lead_window_is_unknown():
    ko_in_5 = pd.Timestamp(SNAP) + pd.Timedelta(minutes=5) + pd.Timedelta(hours=3)
    assert config.PREGAME_MIN_LEAD_MINUTES > 5
    assert tag_kalshi(_df([ko_in_5.strftime("%Y-%m-%dT%H:%M:%SZ")]), SNAP).phase[0] == "unknown"


def test_missing_time_is_unknown_and_never_pregame():
    out = tag_kalshi(_df([None, "garbage"]), SNAP)
    assert (out.phase == "unknown").all() and out.kickoff_utc.isna().all()


def test_no_snapshot_means_unknown():
    assert (tag_kalshi(_df(["2026-11-14T23:00:00Z"]), None).phase == "unknown").all()


def test_scope_nfl_cfb_only():                      # CBB removed 11-13 (Emerson)
    assert set(config.SERIES) == {"NFL", "CFB"}
