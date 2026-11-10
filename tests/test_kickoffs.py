"""Backlog #3: ESPN kickoffs -> pregame/live tags; scanner flags pregame only."""
import pandas as pd
import pytest

from edge_desk import config
from edge_desk.espn import parse_scoreboard, load_asof, COLUMNS
from edge_desk.kickoffs import tag, to_ct, norm, kalshi_games, match
from edge_desk.scan import scan

SNAP = "2026-11-15T16:00:00Z"          # Sun 10:00 AM CT


def espn_event(eid, date, home, away, state="pre", hs="0", as_="0"):
    def team(t):
        abbr, loc, short, disp = t
        return {"abbreviation": abbr, "location": loc, "shortDisplayName": short, "displayName": disp}
    return {"id": eid, "date": date,
            "status": {"type": {"state": state, "completed": state == "post", "shortDetail": "x"}},
            "competitions": [{"competitors": [
                {"homeAway": "home", "score": hs, "team": team(home)},
                {"homeAway": "away", "score": as_, "team": team(away)}]}]}


NE = ("NE", "New England", "Patriots", "New England Patriots")
DET = ("DET", "Detroit", "Lions", "Detroit Lions")
BUF = ("BUF", "Buffalo", "Bills", "Buffalo Bills")
NYJ = ("NYJ", "New York", "Jets", "New York Jets")


def espn_games(fetched=SNAP, ne_state="pre", ne_date="2026-11-15T18:00Z", buf_date="2026-11-15T21:25Z"):
    payload = {"events": [espn_event("401", ne_date, DET, NE, state=ne_state),
                          espn_event("402", buf_date, NYJ, BUF)]}
    return pd.DataFrame(parse_scoreboard("NFL", payload, fetched), columns=COLUMNS)


def krow(event, ticker, team, strike, bid, ask, title, sub, snap=SNAP):
    return dict(snapshot_ts=snap, league="NFL", series="KXNFLSPREAD", event_ticker=event,
                event_title=title, event_sub=sub, ticker=ticker, title=ticker, yes_sub_title=ticker,
                strike_type="greater", floor_strike=strike, cap_strike=None, team_key=team,
                yes_bid=bid, yes_ask=ask, yes_bid_size=100, yes_ask_size=100, no_bid=1 - ask,
                no_ask=1 - bid, last_price=bid, volume=1, volume_24h=1, open_interest=1,
                status="active", occurrence_datetime=None, close_time=None, fee_multiplier=1.0)


def broken_ladder(event="KXNFLSPREAD-26NOV15NEDET", title="New England vs Detroit: Spread",
                  sub="NE vs DET (Nov 15)", snap=SNAP):
    # higher rung bid .70 > lower rung ask .55: a real gap after fees on both legs
    return pd.DataFrame([krow(event, event + "-NE3", "NE", 2.5, 0.54, 0.55, title, sub, snap),
                         krow(event, event + "-NE7", "NE", 6.5, 0.70, 0.71, title, sub, snap)])


# --- matching ---
def test_names_normalise():
    assert norm("Kent St.") == norm("Kent State")
    assert norm("Miami (FL)") == "miami"
    assert norm("UConn") == "connecticut"


def test_match_joins_both_teams_order_agnostic():
    kg = kalshi_games(broken_ladder())
    m = match(kg, espn_games())
    assert list(m["espn_id"]) == ["401"]


def test_no_match_when_date_is_wrong():
    kg = kalshi_games(broken_ladder(event="KXNFLSPREAD-26NOV29NEDET"))
    assert match(kg, espn_games()).empty


def test_no_match_when_one_team_differs():
    kg = kalshi_games(broken_ladder(event="KXNFLSPREAD-26NOV15NEBUF",
                                    title="New England vs Buffalo: Spread", sub="NE vs BUF (Nov 15)"))
    assert match(kg, espn_games()).empty


# --- phase tags ---
def test_pregame_tag_and_kickoff():
    t = tag(broken_ladder(), espn_games(), SNAP)
    assert set(t["phase"]) == {"pregame"}
    assert set(t["kickoff_utc"]) == {"2026-11-15T18:00:00Z"}


def test_kickoff_passed_is_live():
    snap = "2026-11-15T18:30:00Z"
    t = tag(broken_ladder(snap=snap), espn_games(fetched="2026-11-15T18:29:00Z"), snap)
    assert set(t["phase"]) == {"live"}


def test_espn_in_progress_is_live_even_before_listed_kickoff():
    t = tag(broken_ladder(), espn_games(ne_state="in"), SNAP)
    assert set(t["phase"]) == {"live"}


def test_inside_lead_window_is_not_pregame():
    snap = "2026-11-15T17:55:00Z"     # 5 min before kickoff
    t = tag(broken_ladder(snap=snap), espn_games(fetched=snap), snap)
    assert set(t["phase"]) == {"unknown"}


def test_unmatched_is_unknown():
    t = tag(broken_ladder(), espn_games().iloc[0:0], SNAP)
    assert set(t["phase"]) == {"unknown"}


# --- stale data ---
def test_stale_espn_means_unknown():
    old = (pd.Timestamp(SNAP) - pd.Timedelta(minutes=config.ESPN_STALE_MINUTES + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    t = tag(broken_ladder(), espn_games(fetched=old), SNAP)
    assert set(t["phase"]) == {"unknown"}


# --- look-ahead ---
def test_tag_refuses_espn_fetched_after_snapshot():
    with pytest.raises(ValueError):
        tag(broken_ladder(), espn_games(fetched="2026-11-15T16:05:00Z"), SNAP)


def test_load_asof_ignores_future_pulls(tmp_path):
    for stamp, state in (("155900", "pre"), ("160500", "in")):
        d = tmp_path / "2026-11-15"
        d.mkdir(exist_ok=True)
        espn_games(fetched=f"2026-11-15T{stamp[:2]}:{stamp[2:4]}:{stamp[4:]}Z", ne_state=state) \
            .to_parquet(d / f"{stamp}Z.parquet", index=False)
    g = load_asof(SNAP, tmp_path)
    assert set(g["state"]) == {"pre"} and set(g["fetched_ts"]) == {"2026-11-15T15:59:00Z"}
    assert load_asof("2026-11-15T15:00:00Z", tmp_path).empty


# --- CT display ---
def test_kickoff_renders_in_central():
    assert to_ct("2026-11-15T18:00:00Z") == "Sun Nov 15, 12:00 PM CT"     # CST
    assert to_ct("2026-09-13T17:00:00Z") == "Sun Sep 13, 12:00 PM CT"     # CDT


# --- scanner: pregame only by default, fees still on every leg ---
def test_scanner_flags_pregame_with_kickoff():
    f = scan(tag(broken_ladder(), espn_games(), SNAP))
    assert len(f) == 1
    assert f[0]["phase"] == "pregame" and f[0]["kickoff_ct"] == "Sun Nov 15, 12:00 PM CT"
    assert f[0]["edge_per_set"] < 1 - (0.55 + 0.30)      # fees came off both legs


def test_scanner_skips_live_and_unknown():
    snap = "2026-11-15T18:30:00Z"
    live = tag(broken_ladder(snap=snap), espn_games(fetched=snap), snap)
    assert scan(live) == []
    assert len(scan(live, include_live=True)) == 1
    assert scan(tag(broken_ladder(), espn_games().iloc[0:0], SNAP)) == []


def test_untagged_snapshot_never_flags():
    assert scan(broken_ladder()) == []


def test_vs_split_handles_at_inside_a_name():
    df = broken_ladder(event="KXNCAAMBGAME-26NOV11DARTALBY", title="Dartmouth vs University at Albany",
                       sub="DART vs ALBY (Nov 11)")
    kg = kalshi_games(df)
    assert list(kg[["t1_name", "t2_name"]].iloc[0]) == ["Dartmouth", "University at Albany"]
