"""#46 distance caps that learn: report only, walk-forward bias, fees kept, no look-ahead."""
import pandas as pd

from edge_desk import caps, config, digest

CAPS = {"NFL": 10.5, "CFB": 6.0, "CBB": 3.0}
SDS = {"CFB": {"SPREAD": (15.0, 9000), "TOTAL": (16.0, 9000)}, "NFL": {"SPREAD": (13.0, 4000), "TOTAL": (13.0, 4000)}}
NOW = pd.Timestamp("2026-11-20T00:00:00Z")


def bt(gap_old=-0.01, gap_new=-0.30):
    """Backtest json with one season ending before the bets and one ending after them."""
    def season(s, last, gap):
        return {"season": s, "last_kickoff_utc": last,
                "bands": [{"band": b, "rungs": 1000, "pred_sum": 300.0, "hit_sum": 300.0 + 1000 * gap}
                          for b in ("0-3", "3-6", "6-10.5")]}
    k = {"by_offset_season": [season(2025, "2026-01-20T00:00:00Z", gap_old),
                              season(2026, "2026-12-10T00:00:00Z", gap_new)]}
    return {"spread": k, "total": k}


def test_strike_and_market_from_ticker():
    assert caps.strike_of("KXNFLTOTAL-26NOV12WASNYG-24") == 23.5
    assert caps.strike_of("KXNCAAFSPREAD-26NOV14AUBMSST-AUB16") == 15.5
    assert caps.market_of("KXNCAAMBGAME-26NOV14AAABBB-AAA") is None


def test_caps_in_sd():
    c = caps.caps_in_sd({"NFL": {"SPREAD": (13.0, 4000)}}, CAPS)
    assert c["NFL"]["SPREAD"]["cap_sd"] == round(10.5 / 13.0, 3)
    assert c["CBB"]["TOTAL"]["cap_sd"] is None          # no history -> no SD, never a guess


def test_outcome_sd_ignores_games_after_now():
    g = pd.DataFrame({"kickoff_utc": ["2025-10-01T00:00:00Z"] * 200 + ["2027-01-01T00:00:00Z"] * 200,
                      "home_score": [10.0, 20.0] * 100 + [100.0, 0.0] * 100, "away_score": [0.0] * 400,
                      "spread_line": [15.0] * 400, "total_line": [15.0] * 400})
    sd, n = caps.outcome_sds({"NFL": g}, NOW)["NFL"]["SPREAD"]
    assert n == 200 and abs(sd - 5.0125) < 0.01


def test_bias_uses_only_seasons_finished_before_entry():
    b = caps.band_bias(bt(), "SPREAD", "2026-11-12T00:00:00Z")
    assert abs(b["6-10.5"] - (-0.01)) < 1e-9             # the 2026 season (ends 12-10) is ignored
    assert caps.band_bias(bt(), "SPREAD", "2025-12-01T00:00:00Z") == {}


def test_adjust_folds_longshot_and_favourite():
    assert abs(caps.adjust(0.20, -0.01) - 0.19) < 1e-9   # longshot side priced too high -> lower
    assert abs(caps.adjust(0.80, -0.01) - 0.81) < 1e-9   # favourite side -> higher


def _bet(i, ts="2026-11-12T00:00:00Z", line=40.0, ticker_n=50, edge=0.035, fair=0.20, price=0.16,
         status="settled", pnl=1.0, reason="far_rung", league="CFB"):
    tk = f"KXNCAAFTOTAL-26NOV14G{i:03d}-{ticker_n}"
    return ({"bet_id": f"SHARP|YES {tk}", "snapshot_ts": ts, "league": league, "leg1": f"YES {tk}",
             "price": price, "fair": fair, "claimed_edge": edge, "status": status, "pnl": pnl,
             "reason_bucket": reason},
            {"snapshot_ts": ts, "ticker": tk, "side": "YES", "sharp_line": line})


def _table(rows, clv_c=1.0, now=NOW, backtest=None):
    bets = pd.DataFrame([r[0] for r in rows])
    flags = pd.DataFrame([r[1] for r in rows])
    clv = pd.DataFrame({"bet_id": bets["bet_id"], "source": "tracked", "clv_c": clv_c, "close_status": "ok"})
    return caps.bet_table({"tracked": bets}, {"tracked": flags}, clv, SDS, {"CFB": backtest or bt()}, now, CAPS)


def test_adjusted_edge_keeps_fees_and_can_drop_a_play():
    t = _table([_bet(0, edge=0.035)])                   # 49.5 vs 40 -> 9.5 pts, band 6-10.5, bias -1c
    r = t.iloc[0]
    assert r["band"] == "6-10.5" and not r["inside_cap"]
    assert abs(r["edge_adj"] - (0.035 - 0.01)) < 1e-9   # logged edge already net of the fee
    assert not r["adj_qualifies"] and r["edge_adj"] < config.MIN_MODEL_EDGE


def test_stale_line_and_future_plays_are_left_out():
    t = _table([_bet(0, reason="far_rung+stale_line"), _bet(1, ts="2026-11-21T00:00:00Z"), _bet(2)])
    assert list(t["bet_id"]) == [_bet(2)[0]["bet_id"]]


def test_widen_proposal_needs_30_plays_positive_clv_and_t():
    rows = [_bet(i) for i in range(30)]
    clv = [1.0 + 0.1 * (i % 3) for i in range(30)]
    rep = caps.review(_table(rows, clv_c=clv), SDS, CAPS)
    assert any(p.startswith("PROPOSE widening CFB total to 10.5") for p in rep["proposals"])
    rep = caps.review(_table(rows[:29], clv_c=clv[:29]), SDS, CAPS)
    assert rep["proposals"] == []
    neg = [-c for c in clv]                             # far plays losing to the close: no widen
    assert caps.review(_table(rows, clv_c=neg), SDS, CAPS)["proposals"] == []


def test_tighten_proposal_inside_cap():
    rows = [_bet(i, ticker_n=45) for i in range(30)]    # 44.5 vs 40 -> 4.5 pts, inside CFB's 6
    clv = [-1.0 - 0.1 * (i % 3) for i in range(30)]
    rep = caps.review(_table(rows, clv_c=clv), SDS, CAPS)
    assert rep["proposals"] and rep["proposals"][0].startswith("PROPOSE tightening CFB total to 3")


def test_digest_shows_proposals_and_ignores_future_report():
    w = {"generated_utc": "2026-11-19T00:00:00Z", "proposals": ["PROPOSE widening CFB total to 10.5 pts"]}
    md = digest.build({}, {}, {}, NOW, weekly=w)
    assert "PROPOSE widening CFB total" in md.splitlines()[2]
    md = digest.build({}, {}, {}, NOW, weekly={**w, "generated_utc": "2026-11-21T00:00:00Z"})
    assert "PROPOSE" not in md
