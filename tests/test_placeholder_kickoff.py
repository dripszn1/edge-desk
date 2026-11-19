"""#72: a Kalshi 20:00Z placeholder kickoff gives way to the unique sharp commence when that is later."""
import pandas as pd

from edge_desk import odds, sharp

SNAP = "2026-11-18T21:00:00Z"          # after Kalshi's placeholder, before the real kickoff


def _lines(commence="2026-11-19T00:30:00Z", ts="2026-11-18T20:50:00Z"):
    pay = [{"id": "e1", "commence_time": commence, "home_team": "Buffalo Bulls", "away_team": "Central Michigan Chippewas",
            "bookmakers": [{"key": "pinnacle", "markets": [
                {"key": "spreads", "outcomes": [{"name": "Buffalo Bulls", "price": -110, "point": -3.0},
                                                {"name": "Central Michigan Chippewas", "price": -110, "point": 3.0}]},
                {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": 47.5},
                                                {"name": "Under", "price": -110, "point": 47.5}]}]}]}]
    return odds.parse(pay, "CFB", ts)


def _df(ko="2026-11-18T20:00:00Z", phase="live"):
    rows = []
    for series, tick in (("KXNCAAFSPREAD", "KXNCAAFSPREAD-26NOV18CMUBUFF-BUFF3"),
                         ("KXNCAAFTOTAL", "KXNCAAFTOTAL-26NOV18CMUBUFF-47")):
        rows.append(dict(snapshot_ts=SNAP, league="CFB", series=series, event_ticker=tick.rsplit("-", 1)[0],
                         event_title="Central Michigan vs Buffalo: Spread", event_sub="CMU vs BUFF (Nov 18)",
                         ticker=tick, kickoff_utc=ko, kickoff_ct="", phase=phase))
    return pd.DataFrame(rows)


def test_placeholder_takes_later_sharp_commence():
    out = sharp.fix_placeholder_kickoffs(_df(), _lines(), SNAP)
    assert set(out["kickoff_utc"]) == {"2026-11-19T00:30:00Z"}
    assert set(out["phase"]) == {"pregame"}


def test_non_placeholder_time_untouched():
    df = _df(ko="2026-11-18T20:30:00Z")
    out = sharp.fix_placeholder_kickoffs(df, _lines(), SNAP)
    assert set(out["kickoff_utc"]) == {"2026-11-18T20:30:00Z"} and set(out["phase"]) == {"live"}


def test_never_moves_earlier_or_past_wide_window():
    assert set(sharp.fix_placeholder_kickoffs(_df(), _lines(commence="2026-11-18T19:30:00Z", ts="2026-11-18T18:00:00Z"),
                                              "2026-11-18T18:30:00Z")["kickoff_utc"]) == {"2026-11-18T20:00:00Z"}
    far = sharp.fix_placeholder_kickoffs(_df(), _lines(commence="2026-11-19T09:00:00Z"), SNAP)
    assert set(far["kickoff_utc"]) == {"2026-11-18T20:00:00Z"}


def test_started_game_stays_closed():
    """Once the real game is in play the books' quotes are dropped as in-play: nothing re-opens."""
    out = sharp.fix_placeholder_kickoffs(_df(), _lines(), "2026-11-19T01:00:00Z")
    assert set(out["phase"]) == {"live"}


def test_no_lines_no_change():
    df = _df()
    assert sharp.fix_placeholder_kickoffs(df, _lines().iloc[0:0], SNAP) is df


def test_small_gap_keeps_kalshi_time():
    """A real 20:00Z game the books list at 20:30Z stays at Kalshi's (earlier, stricter) time."""
    out = sharp.fix_placeholder_kickoffs(_df(phase="pregame"), _lines(commence="2026-11-18T20:30:00Z", ts="2026-11-18T18:00:00Z"),
                                         "2026-11-18T18:30:00Z")
    assert set(out["kickoff_utc"]) == {"2026-11-18T20:00:00Z"}
