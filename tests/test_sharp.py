"""SHARP signal (backlog #11): rung distribution, fees, stale data, look-ahead, risk caps, gate."""
import numpy as np
import pandas as pd
import pytest

from edge_desk import config, odds, sharp
from edge_desk.fees import fee_per_contract

SNAP = "2026-11-12T18:00:00Z"
KO = "2026-11-13T01:15:00Z"
ODDS_TS = "2026-11-12T17:00:00Z"


def _hist(n=4000, seed=1, end="2026-10-01"):
    """Synthetic closing lines + outcomes, margins clumped on 3 and 7 like the NFL."""
    rng = np.random.default_rng(seed)
    line = rng.choice(np.arange(-10, 10.5, 0.5), n)
    raw = line + rng.normal(0, 13, n)
    m = np.round(raw)
    snap = (np.abs(np.abs(m) - 3) <= 1) & (rng.random(n) < 0.8)
    m = np.where(snap, np.sign(m) * 3, m)
    snap7 = (np.abs(np.abs(m) - 7) <= 1) & (rng.random(n) < 0.4)
    m = np.where(snap7, np.sign(m) * 7, m)
    m = np.where(m == 0, 1, m)
    tl = rng.choice(np.arange(38, 52, 0.5), n)
    tot = np.round(tl + rng.normal(0, 13, n)).clip(3)
    hs = np.round((tot + m) / 2)
    ko = pd.date_range(end=end, periods=n, freq="1D", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    return pd.DataFrame({"game_id": range(n), "season": 2015, "kickoff_utc": ko, "home_team": "H",
                         "away_team": "A", "home_score": hs, "away_score": hs - m, "spread_line": line,
                         "total_line": tl, "neutral": 0, "fetched_ts": "2026-11-12T12:00:00Z"})


def _odds_payload(home_pt=-3.0, total=44.5, commence=KO):
    return [{"id": "ev1", "commence_time": commence, "home_team": "New York Giants",
             "away_team": "Washington Commanders", "bookmakers": [{"key": "pinnacle", "markets": [
                 {"key": "spreads", "outcomes": [{"name": "New York Giants", "price": -108, "point": home_pt},
                                                  {"name": "Washington Commanders", "price": -102, "point": -home_pt}]},
                 {"key": "totals", "outcomes": [{"name": "Over", "price": -105, "point": total},
                                                 {"name": "Under", "price": -105, "point": total}]}]}]}]


def _lines(ts=ODDS_TS, **kw):
    return odds.parse(_odds_payload(**kw), "NFL", ts)


def _row(ticker, series, strike, bid, ask, size=500, phase="pregame"):
    return dict(snapshot_ts=SNAP, league="NFL", series=series, event_ticker=ticker.rsplit("-", 1)[0],
                event_title="Washington vs New York G", event_sub="WAS vs NYG (Nov 12)", ticker=ticker,
                title="", yes_sub_title=f"over {strike}", strike_type="greater", floor_strike=strike,
                yes_bid=bid, yes_ask=ask, yes_bid_size=size, yes_ask_size=size, status="active",
                fee_multiplier=1.0, phase=phase, kickoff_utc=KO, kickoff_ct="Thu Nov 12 7:15 PM CT")


def _kalshi(nyg_ask=0.30, tot_ask=0.40, **kw):
    """NYG -3 at the sharp book -> P(NYG by 3.5+) ~ 0.45; Kalshi selling it at 30c is an edge."""
    return pd.DataFrame([
        _row("KXNFLSPREAD-26NOV12WASNYG-NYG4", "KXNFLSPREAD", 3.5, nyg_ask - 0.02, nyg_ask, **kw),
        _row("KXNFLTOTAL-26NOV12WASNYG-45", "KXNFLTOTAL", 44.5, tot_ask - 0.02, tot_ask, **kw)])


NOW = pd.Timestamp(SNAP) + pd.Timedelta(minutes=5)


def _flags(df=None, lines=None, games=None, **kw):
    return sharp.sharp_flags(_kalshi() if df is None else df, _lines() if lines is None else lines,
                             _hist() if games is None else games, SNAP, now=NOW, **kw)


# ---- distribution ----
def test_empirical_dist_keeps_key_numbers():
    d = sharp.build_dists(_hist(), SNAP)["spread"]
    jump3 = d.p_greater(3, 2.5) - d.p_greater(3, 3.5)        # mass exactly on 3
    jump5 = d.p_greater(3, 4.5) - d.p_greater(3, 5.5)        # mass exactly on 5
    assert jump3 > 2 * jump5
    normal = sharp._normal_p(3, 2.5, 13.3) - sharp._normal_p(3, 3.5, 13.3)
    assert jump3 > 1.5 * normal


def test_dist_monotone_and_symmetric():
    d = sharp.build_dists(_hist(), SNAP)["spread"]
    ps = [d.p_greater(-3, k) for k in np.arange(-20.5, 21, 1)]
    assert all(a >= b - 1e-12 for a, b in zip(ps, ps[1:]))
    assert d.p_greater(3, 3.5) == pytest.approx(1 - d.p_greater(-3, -3.5), abs=0.02)


def test_dist_uses_only_games_before_snapshot():
    h = _hist()
    future = h.copy()
    future["kickoff_utc"] = "2026-11-20T18:00:00Z"            # after the snapshot
    future["home_score"], future["away_score"] = 100, 0      # would wreck the dist if used
    a = sharp.build_dists(h, SNAP)["spread"].p_greater(-3, 3.5)
    b = sharp.build_dists(pd.concat([h, future]), SNAP)["spread"].p_greater(-3, 3.5)
    assert a == b


def test_backtest_walk_forward_reports_benchmark():
    h = _hist(n=3000)
    h["season"] = pd.to_datetime(h["kickoff_utc"]).dt.year
    r = sharp.backtest(h, seasons=[2025])
    assert r["spread"]["games"] > 100 and "logloss_normal" in r["spread"]
    assert r["method_better_than_normal"]["spread"] in (True, False)


# ---- signal ----
def test_flags_fee_inclusive_edge():
    flags, st = _flags()
    assert st["ok"] and flags
    f = flags[0]
    assert f["kind"] == "SHARP" and f["sharp_source"] == "pinnacle"
    assert f["edge"] == pytest.approx(f["fair"] - f["price"] - fee_per_contract(f["price"], f["contracts"]), abs=1e-4)
    assert f["edge"] >= config.MIN_MODEL_EDGE


def test_no_flag_when_fee_eats_edge():
    # fair ~0.45 for NYG by 3.5+; at 43c the raw gap (~2c) is below fees + 3c
    flags, _ = _flags(df=_kalshi(nyg_ask=0.43, tot_ask=0.50))
    nyg = [f for f in flags if "NYG4" in f["ticker"] and f["side"] == "YES"]
    assert not nyg


def test_spread_side_mapping():
    """WAS rung priced from the WAS point (+3): P(WAS by 3.5+) is small."""
    df = pd.DataFrame([_row("KXNFLSPREAD-26NOV12WASNYG-WAS4", "KXNFLSPREAD", 3.5, 0.40, 0.42)])
    flags, _ = _flags(df=df)
    assert flags and flags[0]["side"] == "NO" and flags[0]["sharp_line"] == -3.0


def test_stale_snapshot_no_flags():
    flags, st = sharp.sharp_flags(_kalshi(), _lines(), _hist(), SNAP,
                                  now=pd.Timestamp(SNAP) + pd.Timedelta(minutes=config.STALE_MINUTES + 1))
    assert flags == [] and "stale" in st["reason"]


def test_stale_odds_no_flags():
    old = pd.Timestamp(SNAP) - pd.Timedelta(hours=odds.ODDS_STALE_HOURS, minutes=1)
    flags, st = _flags(lines=_lines(ts=old.strftime("%Y-%m-%dT%H:%M:%SZ")))
    assert flags == [] and "no usable sharp odds" in st["reason"]


def test_odds_fetched_after_snapshot_refused():
    flags, st = _flags(lines=_lines(ts="2026-11-12T18:30:00Z"))
    assert flags == [] and not st["ok"]


def test_live_markets_never_flagged():
    flags, _ = _flags(df=_kalshi(phase="live"))
    assert flags == []


def test_kickoff_mismatch_not_matched():
    flags, st = _flags(lines=_lines(commence="2026-11-16T18:00:00Z"))
    assert flags == [] and st["events_no_sharp_line"] >= 1


def test_price_cap_and_min_size():
    df = _kalshi()
    df["yes_ask_size"] = df["yes_bid_size"] = config.MIN_SIZE - 1
    assert _flags(df=df)[0] == []
    cheap_no = pd.DataFrame([_row("KXNFLSPREAD-26NOV12WASNYG-NYG4", "KXNFLSPREAD", 3.5, 0.05, 0.07)])
    # NO would cost 95c > MAX_FLAG_PRICE; YES at 7c has no edge vs fair ~0.45? it does -> only YES allowed
    for f in _flags(df=cheap_no)[0]:
        assert f["price"] <= config.MAX_FLAG_PRICE


def test_one_flag_per_event_and_daily_cap():
    flags, _ = _flags()
    evs = [f["event_ticker"] for f in flags]
    assert len(evs) == len(set(evs))
    assert sum(f["stake"] for f in flags) <= config.DAILY_LOSS_CAP
    assert all(f["stake"] <= config.MAX_BET for f in flags)
    assert _flags(lost_today=config.DAILY_LOSS_CAP)[0] == []


def test_cfb_without_cfb_odds_no_flags():
    flags, st = _flags(league="CFB")
    assert flags == [] and "no usable sharp odds" in st["reason"]


def test_no_history_no_flags():
    flags, st = _flags(games=_hist().iloc[:100])
    assert flags == [] and "closing lines" in st["reason"]


# ---- gate ----
def test_gate_closed_without_bets(tmp_path):
    assert sharp.gate(tmp_path / "none.csv")[0] is False


def test_gate_needs_30_bets_positive_t(tmp_path):
    p = tmp_path / "b.csv"
    rng = np.random.default_rng(0)
    pnl = rng.normal(0.5, 0.3, 40)
    pd.DataFrame({"status": "settled", "pnl": pnl, "stake": 5.0}).to_csv(p, index=False)
    assert sharp.gate(p)[0] is True
    pd.DataFrame({"status": "settled", "pnl": pnl[:20], "stake": 5.0}).to_csv(p, index=False)
    assert sharp.gate(p)[0] is False
    pd.DataFrame({"status": "settled", "pnl": -pnl, "stake": 5.0}).to_csv(p, index=False)
    assert sharp.gate(p)[0] is False


def test_far_tail_rungs_never_priced():
    # Over 24.5 with a 44.5 sharp total is 20 points out: outside the backtested range, so even a
    # "mispriced" far-tail rung (NO at 3c) is never flagged; the near rung still is.
    far = _row("KXNFLTOTAL-26NOV12WASNYG-25", "KXNFLTOTAL", 24.5, 0.96, 0.97)
    near = _row("KXNFLTOTAL-26NOV12WASNYG-45", "KXNFLTOTAL", 44.5, 0.38, 0.40)
    flags, st = _flags(df=pd.DataFrame([far, near]))
    assert all(not f["ticker"].endswith("-25") for f in flags)
    assert st.get("rungs_out_of_range") == 1
