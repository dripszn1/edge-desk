"""#57 random-entry baseline: deterministic draws, never the real contract, no future snapshot,
fees on both legs, verdict thresholds, settle fetches placebos only after kickoff."""
import pandas as pd
import pytest

from edge_desk import baseline, digest, fees, settle, trades

EV = "KXNCAAFSPREAD-26NOV14MICHORE"
REAL = f"{EV}-ORE6"
ENTRY = "2026-11-14T15:00:00Z"
OCC = "2026-11-14T21:00:00Z"          # kickoff 18:00Z
NOW = pd.Timestamp("2026-11-15T03:00:00Z")


def row(ts, ticker, yb, ya, event=EV, series="KXNCAAFSPREAD"):
    return {"snapshot_ts": ts, "series": series, "event_ticker": event, "ticker": ticker,
            "yes_bid": yb, "yes_ask": ya, "occurrence_datetime": OCC, "fee_multiplier": 1.0}


def entry_snap(ts=ENTRY):
    return pd.DataFrame([
        row(ts, REAL, 0.60, 0.67),                      # the real contract (NO ask 0.40)
        row(ts, f"{EV}-ORE3", 0.55, 0.58),              # NO ask 0.45, YES ask 0.58
        row(ts, f"{EV}-ORE9", 0.30, 0.34),              # YES ask 0.34, NO ask 0.70
        row(ts, f"{EV}-ORE12", 0.20, 0.25),             # YES 0.25 (out of band), NO 0.80 (out)
        row(ts, f"{EV}-ORE15", 0.0, 0.10),              # one-sided book
        row(ts, "KXNCAAFTOTAL-26NOV14MICHORE-50", 0.55, 0.60, series="KXNCAAFTOTAL"),   # other series
        row(ts, "KXNCAAFSPREAD-26NOV14OSUPSU-OSU3", 0.55, 0.60, event="KXNCAAFSPREAD-26NOV14OSUPSU"),
    ])


def bet(bet_id="SHARP|NO " + REAL, price=0.40, n=10, payout=0.0, status="settled", fair=0.45):
    return {"bet_id": bet_id, "league": "CFB", "leg1": f"NO {REAL}", "snapshot_ts": ENTRY, "price": price,
            "fair": fair, "contracts": n, "fees": fees.taker_fee(price, n), "status": status, "payout": payout}


def test_candidates_same_event_series_band_two_sided_never_real():
    pool = baseline.candidates(f"NO {REAL}", 0.40, entry_snap())
    got = {(c["side"], c["ticker"]) for c in pool}
    assert got == {("NO", f"{EV}-ORE3"), ("YES", f"{EV}-ORE9")}
    assert all(c["ticker"] != REAL for c in pool)


def test_draw_is_deterministic_and_row_order_free():
    s = entry_snap()
    a = baseline.draw("b1", baseline.candidates(f"NO {REAL}", 0.40, s))
    b = baseline.draw("b1", baseline.candidates(f"NO {REAL}", 0.40, s.sample(frac=1, random_state=3)))
    assert a == b and len(a) == baseline.K
    assert baseline.seed_of("b1") == baseline.seed_of("b1") != baseline.seed_of("b2")


def test_entry_snapshot_never_later_than_entry():
    times = ["2026-11-14T14:50:00Z", "2026-11-14T15:05:00Z"]
    assert baseline.entry_snapshot(ENTRY, times) == pd.Timestamp("2026-11-14T14:50:00Z")
    assert baseline.entry_snapshot(ENTRY, ["2026-11-14T14:30:00Z"]) is None      # too old
    assert baseline.entry_snapshot(ENTRY, ["2026-11-14T15:00:30Z"]) is None      # after entry


def test_placebo_fees_both_legs_and_no_future_points():
    pre = entry_snap()
    tip = pd.DataFrame([row("2026-11-14T17:45:00Z", f"{EV}-ORE3", 0.50, 0.52),
                        row("2026-11-14T17:45:00Z", f"{EV}-ORE9", 0.36, 0.40),
                        row("2026-11-14T17:45:00Z", REAL, 0.60, 0.62)])
    # a huge spike stamped after `now` must never fill an Auto Sell
    future = pd.DataFrame([row("2026-11-15T04:00:00Z", t, 0.98, 0.99) for t in (f"{EV}-ORE3", f"{EV}-ORE9")]
                          + [row("2026-11-15T04:00:00Z", t, 0.01, 0.02) for t in (f"{EV}-ORE3",)])
    snaps = pd.concat([pre, tip, future], ignore_index=True)
    results = pd.DataFrame([{"fetched_ts": "2026-11-15T01:00:00Z", "ticker": f"{EV}-ORE3", "status": "finalized", "result": "yes"},
                            {"fetched_ts": "2026-11-15T01:00:00Z", "ticker": f"{EV}-ORE9", "status": "finalized", "result": "yes"}])
    plays, rows = baseline.grade_plays(pd.DataFrame([bet()]), snaps, results, NOW)
    assert len(rows) == baseline.K and all(r["ticker"] != REAL for r in rows)
    p = plays[0]
    assert p["real"] is not None and len(p["placebo"]) == baseline.K
    # NO on ORE3 at 0.45, 10 contracts: sold at the tip at NO bid 1-0.52 = 0.48, taker both legs
    i = next(i for i, r in enumerate(rows) if r["ticker"] == f"{EV}-ORE3")
    exp_tip = 10 * (0.48 - 0.45) - fees.taker_fee(0.45, 10) - fees.taker_fee(0.48, 10)
    assert p["placebo"][i]["EXIT_AT_TIP"] == pytest.approx(exp_tip)
    # HOLD: ORE3 settled yes -> NO loses; entry fee only
    assert p["placebo"][i]["HOLD"] == pytest.approx(-10 * 0.45 - fees.taker_fee(0.45, 10))
    # RESTING_15 (target 0.60) never reached before now -> held, not filled by the future spike
    assert p["placebo"][i]["RESTING_15"] == pytest.approx(p["placebo"][i]["HOLD"])


def test_future_snapshot_not_used_as_entry_snapshot():
    snaps = entry_snap("2026-11-15T05:00:00Z")          # only snapshot is after now
    b = bet()
    b["snapshot_ts"] = "2026-11-15T05:00:00Z"
    plays, rows = baseline.grade_plays(pd.DataFrame([b]), snaps, None, NOW)
    assert plays == [] and rows == []


def _plays(n, real, placebo, grp="football"):
    return [{"bet_id": f"b{i}", "group": grp, "event": f"E{i}", "pool": 3,
             "real": {v: real for v in baseline.EXITS},
             "placebo": [{v: placebo + 0.01 * (j - 1) for v in baseline.EXITS} for j in range(3)]}
            for i in range(n)]


def test_verdicts():
    s = baseline.summary(_plays(30, 1.0, 0.0))
    assert s["football"]["exits"]["HOLD"]["verdict"] == "picks add value"
    assert s["football"]["exits"]["HOLD"]["beats_pct"] == 100.0
    s = baseline.summary(_plays(29, 1.0, 0.0))
    assert s["football"]["exits"]["HOLD"]["verdict"].startswith("too few")
    s = baseline.summary(_plays(30, 0.5, 0.5))
    assert "exit, not the pick" in s["football"]["exits"]["RESTING_10"]["verdict"]
    s = baseline.summary(_plays(30, -0.5, -0.5))
    assert "add nothing" in s["football"]["exits"]["HOLD"]["verdict"]
    s = baseline.summary(_plays(30, -1.0, 0.0))
    assert s["football"]["exits"]["HOLD"]["verdict"] == "picks do worse than random entries"
    assert "cbb" in s and s["cbb"]["closed"] == 0                       # never pooled


def test_digest_line_ignores_future_report():
    rep = {"generated_utc": "2026-11-15T02:00:00Z", "baseline": baseline.summary(_plays(5, 0.5, 0.1))}
    line = digest.baseline_line(rep, NOW)
    assert line and "random entries" in line[0] and "football (5 plays)" in line[0]
    rep["generated_utc"] = "2026-11-15T04:00:00Z"
    assert digest.baseline_line(rep, NOW) == []


def test_settle_fetches_placebos_only_after_kickoff():
    p = pd.DataFrame([{"ticker": "A", "kickoff_utc": "2026-11-14T18:00:00Z"},
                      {"ticker": "B", "kickoff_utc": "2026-11-16T18:00:00Z"},
                      {"ticker": "C", "kickoff_utc": None}])
    assert settle.placebo_tickers(p, NOW) == {"A"}
    assert settle.placebo_tickers(None, NOW) == set()
