"""#65 one kickoff per game: Kalshi's occurrence time differs across a game's series and can be
hours late (11-14 GT/CLEM game-winner 20:00Z vs spread 17:30Z; KU/WVU spread 22:30Z vs a 20:30Z
start), which tagged in-game markets 'pregame' -> in-game SHARP alerts and in-game CLV closes."""
import pandas as pd

from edge_desk import clv, grade, kickoffs, sharp

SPR = "KXNCAAFSPREAD-26NOV14GTCLEM-CLEM7"
GAME = "KXNCAAFGAME-26NOV14GTCLEM-CLEM"


def test_game_key_joins_series_of_one_game():
    assert kickoffs.game_key(SPR) == kickoffs.game_key(GAME) == "KXNCAAF-26NOV14GTCLEM"
    assert kickoffs.game_key("KXNCAAFTOTAL-26NOV14GTCLEM") == "KXNCAAF-26NOV14GTCLEM"
    assert kickoffs.game_key("YES " + SPR) == "KXNCAAF-26NOV14GTCLEM"
    assert kickoffs.game_key("KXNFLSPREAD-26NOV14GTCLEM-X3") != kickoffs.game_key(SPR)   # other league
    assert kickoffs.game_key(None) is None and kickoffs.game_key(float("nan")) is None


def test_late_series_is_live_once_another_series_kicked_off():
    df = pd.DataFrame([
        {"event_ticker": "KXNCAAFSPREAD-26NOV14GTCLEM", "ticker": SPR, "occurrence_datetime": "2026-11-14T20:30:00Z"},
        {"event_ticker": "KXNCAAFGAME-26NOV14GTCLEM", "ticker": GAME, "occurrence_datetime": "2026-11-14T23:00:00Z"},
        {"event_ticker": "KXNCAAFGAME-26NOV14KUWVU", "ticker": "KXNCAAFGAME-26NOV14KUWVU-KU",
         "occurrence_datetime": "2026-11-14T23:00:00Z"},
    ])
    out = kickoffs.tag_kalshi(df, "2026-11-14T18:00:00Z")
    assert list(out["phase"]) == ["live", "live", "pregame"]          # game-winner follows the spread
    assert out["kickoff_utc"].iloc[1] == "2026-11-14T17:30:00Z"
    assert out["kickoff_utc"].iloc[2] == "2026-11-14T20:00:00Z"        # other game untouched


def test_sharp_commence_blocks_in_game_pricing():
    fe = pd.DataFrame({"commence_utc": ["2026-11-14T20:30:00Z"]})
    assert sharp.sharp_started(fe, "2026-11-14T20:44:09Z")              # KU/WVU alert time: in play
    assert sharp.sharp_started(fe, "2026-11-14T20:25:00Z")              # inside the pregame lead window
    assert not sharp.sharp_started(fe, "2026-11-14T17:11:30Z")          # morning play: fine
    assert not sharp.sharp_started(fe.iloc[0:0], "2026-11-14T20:44:09Z")


def test_grader_skips_flags_logged_after_the_games_earliest_kickoff():
    gmap = {"KXNCAAF-26NOV14GTCLEM": pd.Timestamp("2026-11-14T17:30:00Z")}
    f = pd.DataFrame([{"snapshot_ts": "2026-11-14T16:10:01Z", "ticker": GAME},      # pregame: kept
                      {"snapshot_ts": "2026-11-14T17:58:56Z", "ticker": GAME},      # in game: skipped
                      {"snapshot_ts": "2026-11-14T17:58:56Z", "ticker": "KXNFLGAME-26NOV15BUFNYJ-BUF"}])
    out = grade.drop_ingame(f, gmap)
    assert list(out["snapshot_ts"]) == ["2026-11-14T16:10:01Z", "2026-11-14T17:58:56Z"]
    assert list(out["ticker"]) == [GAME, "KXNFLGAME-26NOV15BUFNYJ-BUF"]


def test_clv_close_is_before_the_games_earliest_kickoff():
    def q(ts, t, occ, bid, ask):
        return {"snapshot_ts": ts, "ticker": t, "yes_bid": bid, "yes_ask": ask, "no_bid": round(1 - ask, 2),
                "no_ask": round(1 - bid, 2), "occurrence_datetime": occ}
    snaps = pd.DataFrame([
        q("2026-11-14T15:00:00Z", GAME, "2026-11-14T23:00:00Z", 0.80, 0.82),
        q("2026-11-14T15:00:00Z", SPR, "2026-11-14T20:30:00Z", 0.50, 0.52),
        q("2026-11-14T17:20:00Z", GAME, "2026-11-14T23:00:00Z", 0.79, 0.81),   # last pregame quote
        q("2026-11-14T19:54:05Z", GAME, "2026-11-14T23:00:00Z", 0.13, 0.14),   # mid-game crash
    ])
    bets = pd.DataFrame([{"bet_id": f"SHARP|YES {GAME}", "snapshot_ts": "2026-11-14T15:00:00Z", "league": "CFB",
                          "leg1": f"YES {GAME}", "contracts": 10, "price": 0.82, "fees": 0.0, "source": "shadow"}])
    r = clv.closing_value(bets, snaps, now=pd.Timestamp("2026-11-15T00:00:00Z")).iloc[0]
    assert r["kickoff_utc"] == "2026-11-14T17:30:00Z"
    assert r["close_ts"] == "2026-11-14T17:20:00Z" and abs(r["close_mid"] - 0.80) < 1e-9
