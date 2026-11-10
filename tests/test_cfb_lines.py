"""CFB closing lines (backlog #13) + SHARP for CFB: parsing, sign, look-ahead, name matching, fees."""
import pandas as pd
import pytest

from edge_desk import cfb_lines, config, odds, sharp
from edge_desk.fees import fee_per_contract
from tests.test_sharp import _hist

SNAP = "2026-11-14T15:00:00Z"
KO = "2026-11-14T20:30:00Z"


def _cfbd():
    return [{"id": 401, "season": 2025, "week": 3, "seasonType": "regular", "startDate": "2025-09-13T16:00:00.000Z",
             "homeTeam": "Auburn", "awayTeam": "Mississippi State", "homeScore": 30, "awayScore": 20,
             "lines": [{"provider": "DraftKings", "spread": -7.5, "overUnder": 51.5},
                       {"provider": "consensus", "spread": -7.0, "overUnder": 52.0}]},
            {"id": 402, "season": 2025, "week": 3, "seasonType": "regular", "startDate": "2025-09-13T19:00:00.000Z",
             "homeTeam": "Kent State", "awayTeam": "Akron",
             "lines": [{"provider": "Bovada", "spread": 3.0, "overUnder": None},
                       {"provider": "ESPN Bet", "spread": 4.0, "overUnder": 47.5}]},
            {"id": 403, "season": 2025, "lines": []}]


def test_parse_prefers_consensus_and_flips_sign():
    df = cfb_lines.parse(_cfbd(), "2026-11-10T00:00:00Z")
    a = df[df.game_id == 401].iloc[0]
    assert a.spread_line == 7.0 and a.total_line == 52.0 and a.source == "consensus"   # home -7 -> +7
    b = df[df.game_id == 402].iloc[0]
    assert b.spread_line == -3.5 and b.total_line == 47.5 and b.source == "median:2"
    assert 403 not in set(df.game_id)


def test_attach_by_game_id():
    games = pd.DataFrame({"game_id": [401, 999], "spread_line": [float("nan")] * 2, "total_line": [float("nan")] * 2})
    g = cfb_lines.attach(games, cfb_lines.parse(_cfbd(), "2026-11-10T00:00:00Z"))
    assert g.spread_line.tolist()[0] == 7.0 and pd.isna(g.spread_line.tolist()[1])


def test_missing_key_skips(monkeypatch):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    assert cfb_lines.run() is None


def test_refresh_only_current_season(monkeypatch, tmp_path):
    from datetime import datetime, timezone
    monkeypatch.setattr(cfb_lines, "OUT", tmp_path / "lines.csv")
    now = datetime(2026, 11, 10, tzinfo=timezone.utc)
    assert cfb_lines._due(now) == list(range(cfb_lines.FIRST_SEASON, 2027))
    cfb_lines.parse(_cfbd(), "2026-11-09T23:00:00Z").to_csv(tmp_path / "lines.csv", index=False)
    assert cfb_lines._due(now) is None                       # fetched 1h ago
    assert cfb_lines._due(datetime(2026, 11, 10, 13, tzinfo=timezone.utc)) == [2026]


# ---- CFB name matching + SHARP CFB ----
def _odds(home="Auburn Tigers", away="Mississippi State Bulldogs", home_pt=-7.0, ts="2026-11-14T14:30:00Z", extra=()):
    evs = [(f"e_{home}", home, away, home_pt)] + list(extra)
    pay = [{"id": eid, "commence_time": KO, "home_team": h, "away_team": a, "bookmakers": [{"key": "pinnacle", "markets": [
        {"key": "spreads", "outcomes": [{"name": h, "price": -110, "point": pt}, {"name": a, "price": -110, "point": -pt}]},
        {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": 50.5},
                                        {"name": "Under", "price": -110, "point": 50.5}]}]}]}
           for eid, h, a, pt in evs]
    return odds.parse(pay, "CFB", ts)


def _row(title="Auburn vs Mississippi St.", sub="AUB vs MSST (Nov 14)", ticker="KXNCAAFSPREAD-26NOV14AUBMSST-AUB7",
         strike=6.5, bid=0.28, ask=0.30):
    return dict(snapshot_ts=SNAP, league="CFB", series="KXNCAAFSPREAD", event_ticker=ticker.rsplit("-", 1)[0],
                event_title=f"{title}: Spread", event_sub=sub, ticker=ticker, title="", yes_sub_title="",
                strike_type="greater", floor_strike=strike, yes_bid=bid, yes_ask=ask, yes_bid_size=500,
                yes_ask_size=500, status="active", fee_multiplier=1.0, phase="pregame", kickoff_utc=KO,
                kickoff_ct="Sat Nov 14 2:30 PM CT")


def _fair(lines):
    return odds.fair_lines(lines, SNAP)


@pytest.mark.parametrize("kalshi,book", [("Mississippi St.", "Mississippi State Bulldogs"),
                                         ("Miami (OH)", "Miami (OH) RedHawks"), ("Hawai'i", "Hawaii Rainbow Warriors"),
                                         ("Louisiana-Monroe", "UL Monroe Warhawks"), ("Ole Miss", "Ole Miss Rebels"),
                                         ("San Jose St.", "San José State Spartans"), ("NC St.", "NC State Wolfpack")])
def test_prefix_names(kalshi, book):
    assert sharp._prefix_score(kalshi, book) is not None


def test_prefix_prefers_school_over_longer_school():
    """'Texas' must pick Texas Longhorns over Texas Tech Red Raiders when both are on the board."""
    lines = _odds(home="Texas Longhorns", away="Arkansas Razorbacks",
                  extra=[("e2", "Texas Tech Red Raiders", "Arkansas State Red Wolves", -3.0)])
    sides, fe = sharp.match_event("CFB", _row("Texas vs Arkansas", "TEX vs ARK (Nov 14)"), _fair(lines))
    assert sides == {"TEX": "Texas Longhorns", "ARK": "Arkansas Razorbacks"}
    assert fe["event_id"].nunique() == 1


def test_ambiguous_match_refused():
    lines = _odds(home="Miami Hurricanes", away="Ohio Bobcats",
                  extra=[("e2", "Miami Flames", "Ohio Owls", -3.0)])
    sides, fe = sharp.match_event("CFB", _row("Miami vs Ohio", "MIA vs OHIO (Nov 14)"), _fair(lines))
    assert sides is None and fe.empty


def _cfb_hist():
    h = _hist()
    h["game_type"], h["home_division"], h["away_division"] = "REG", "fbs", "fbs"
    return h


def test_cfb_sharp_flag_fees_and_side():
    """Auburn -7 at the sharp book; Kalshi sells 'Auburn by 6.5+' at 44c -> YES edge after fees.
    (A 30c ask would be a ~20c 'edge': the >12c sanity cap rejects that as a likely data error.)"""
    flags, st = sharp.sharp_flags(pd.DataFrame([_row(bid=0.42, ask=0.44)]), _odds(), _cfb_hist(), SNAP,
                                  now=pd.Timestamp(SNAP), league="CFB")
    assert st["ok"] and len(flags) == 1
    f = flags[0]
    assert f["side"] == "YES" and f["sharp_line"] == 7.0 and f["league"] == "CFB"
    assert f["edge"] == pytest.approx(f["fair"] - f["price"] - fee_per_contract(f["price"], f["contracts"]), abs=1e-4)


def test_cfb_odds_after_snapshot_refused():
    flags, st = sharp.sharp_flags(pd.DataFrame([_row()]), _odds(ts="2026-11-14T15:30:00Z"), _cfb_hist(), SNAP,
                                  now=pd.Timestamp(SNAP), league="CFB")
    assert flags == [] and not st["ok"]


def test_cfb_without_lines_skips():
    h = _cfb_hist()
    h["spread_line"] = float("nan")
    flags, st = sharp.sharp_flags(pd.DataFrame([_row()]), _odds(), h, SNAP, now=pd.Timestamp(SNAP), league="CFB")
    assert flags == [] and "closing lines" in st["reason"]


def test_gate_is_per_league(tmp_path):
    import numpy as np
    p = tmp_path / "b.csv"
    pnl = np.random.default_rng(0).normal(0.5, 0.3, 40)
    pd.DataFrame({"status": "settled", "pnl": pnl, "stake": 5.0, "league": "NFL"}).to_csv(p, index=False)
    assert sharp.gate(p, league="NFL")[0] is True
    assert sharp.gate(p, league="CFB")[0] is False


def test_cfb_edge_sanity_cap_rejects_huge_edge(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp, "REJECTED", tmp_path / "rejected.csv")
    flags, st = sharp.sharp_flags(pd.DataFrame([_row()]), _odds(), _cfb_hist(), SNAP,
                                  now=pd.Timestamp(SNAP), league="CFB")
    assert flags == [] and st.get("rejected_edge_cap") == 1 and (tmp_path / "rejected.csv").exists()


def test_cfb_far_rung_skipped():
    """CFB: a rung 8.5 pts from the sharp line is outside the +-6 CFB cap."""
    flags, st = sharp.sharp_flags(pd.DataFrame([_row(strike=15.5, bid=0.10, ask=0.12,
                                  ticker="KXNCAAFSPREAD-26NOV14AUBMSST-AUB16")]), _odds(), _cfb_hist(), SNAP,
                                  now=pd.Timestamp(SNAP), league="CFB")
    assert flags == [] and st.get("rungs_out_of_range") == 1


def test_stale_odds_guard():
    snap = "2026-11-14T15:00:00Z"
    assert not sharp.odds_too_old(snap, "2026-11-14T14:00:00Z", "2026-11-15T00:00:00Z")  # 60 min, far from KO
    assert sharp.odds_too_old(snap, "2026-11-14T13:00:00Z", "2026-11-15T00:00:00Z")      # 120 min
    assert sharp.odds_too_old(snap, "2026-11-14T13:55:00Z", "2026-11-14T17:00:00Z")      # 65 min, KO in 2h
