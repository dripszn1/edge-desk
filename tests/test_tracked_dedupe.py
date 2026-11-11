"""#27: sharp_tracked.csv logs a held-back play only when it is new or its price/fair/line/reason
changed, not once per 30-second live pass (it grew 4 MB in one evening)."""
import pandas as pd

from edge_desk import sharp


def _f(ts, price=0.30, fair=0.36, odds_ts="2026-11-11T00:00:00Z"):
    return {"snapshot_ts": ts, "league": "CFB", "event": "A vs B", "ticker": "KX-T1", "side": "YES",
            "price": price, "fair": fair, "edge": fair - price - 0.02, "sharp_line": 7.0,
            "odds_fetched_ts": odds_ts}


def test_repeat_pass_not_relogged():
    sharp.log_rejected(_f("2026-11-11T01:00:00Z"), "sharp line stale")
    sharp.log_rejected(_f("2026-11-11T01:00:30Z"), "sharp line stale")
    sharp.log_rejected(_f("2026-11-11T01:01:00Z"), "sharp line stale")
    df = pd.read_csv(sharp.REJECTED)
    assert len(df) == 1 and df.snapshot_ts.iloc[0] == "2026-11-11T01:00:00Z"


def test_change_is_logged():
    sharp.log_rejected(_f("2026-11-11T01:00:00Z"), "sharp line stale")
    sharp.log_rejected(_f("2026-11-11T01:00:30Z", price=0.31), "sharp line stale")        # price moved
    sharp.log_rejected(_f("2026-11-11T01:01:00Z", price=0.31), "edge above sanity cap")   # reason changed
    sharp.log_rejected(_f("2026-11-11T01:01:30Z", price=0.31, odds_ts="2026-11-11T01:01:00Z"),
                       "edge above sanity cap")                                          # new sharp pull
    assert len(pd.read_csv(sharp.REJECTED)) == 4


def test_new_process_seeds_from_file():
    sharp.log_rejected(_f("2026-11-11T01:00:00Z"), "sharp line stale")
    sharp._TRACK_SEEN.clear()                      # a fresh collect run starts with an empty cache
    sharp.log_rejected(_f("2026-11-11T01:15:00Z"), "sharp line stale")
    assert len(pd.read_csv(sharp.REJECTED)) == 1


def test_first_sighting_still_graded_at_first_price(tmp_path):
    # grading uses the first sighting; dedupe must keep that row, not the latest
    sharp.log_rejected(_f("2026-11-11T01:00:00Z", price=0.30), "sharp line stale")
    sharp.log_rejected(_f("2026-11-11T01:00:30Z", price=0.30), "sharp line stale")
    df = pd.read_csv(sharp.REJECTED)
    assert df.price.iloc[0] == 0.30 and df.snapshot_ts.iloc[0] == "2026-11-11T01:00:00Z"
