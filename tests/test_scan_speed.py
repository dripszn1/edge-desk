"""#44: the structural scan and MODEL matching must be fast enough for the 30-s live cadence,
and the faster code must give exactly the same flags as the old pandas-groupby version."""
import random
import time
from itertools import combinations

import numpy as np
import pandas as pd

from edge_desk import model, scan as scan_mod
from edge_desk.scan import _flag, _quoted, scan

from tests.test_core import row


def _reference_scan(df, include_live=False):
    """Frozen copy of scan.scan before #44 (pandas groupby + to_dict per group)."""
    flags = []
    if df.empty:
        return flags
    q = _quoted(df, include_live)
    for ev_ticker, ev in q.groupby("event_ticker"):
        series = ev["series"].iloc[0]
        greater = ev[ev["strike_type"] == "greater"].dropna(subset=["floor_strike"])
        keys = greater["team_key"].fillna("__total__")
        for _, rung in greater.groupby(keys):
            rung = rung.sort_values("floor_strike", kind="stable")
            rows = rung.to_dict("records")
            for lo, hi in combinations(rows, 2):
                if hi["floor_strike"] <= lo["floor_strike"]:
                    continue
                size = min(lo["yes_ask_size"] or 0, hi["yes_bid_size"] or 0)
                f = _flag("LADDER", lo, "YES", lo["yes_ask"], hi, "NO", 1 - hi["yes_bid"], size)
                if f:
                    flags.append(f)
        if "SPREAD" in series:
            teams = list(greater.groupby("team_key"))
            if len(teams) == 2:
                for a in teams[0][1].to_dict("records"):
                    for b in teams[1][1].to_dict("records"):
                        size = min(a["yes_bid_size"] or 0, b["yes_bid_size"] or 0)
                        f = _flag("EXCLUSIVE", a, "NO", 1 - a["yes_bid"], b, "NO", 1 - b["yes_bid"], size)
                        if f:
                            flags.append(f)
        if series.endswith("GAME") and len(ev) == 2:
            a, b = ev.to_dict("records")
            size = min(a["yes_bid_size"] or 0, b["yes_bid_size"] or 0)
            f = _flag("EXCLUSIVE", a, "NO", 1 - a["yes_bid"], b, "NO", 1 - b["yes_bid"], size)
            if f:
                flags.append(f)
            if ev["league"].iloc[0] in scan_mod.NO_TIE_LEAGUES:
                size = min(a["yes_ask_size"] or 0, b["yes_ask_size"] or 0)
                f = _flag("COVER", a, "YES", a["yes_ask"], b, "YES", b["yes_ask"], size)
                if f:
                    flags.append(f)
    return sorted(flags, key=lambda x: -x["edge_per_set"])


def _messy_board(seed=7, events=120):
    """Random spreads/totals/game-winners with broken prices, NaN sizes, NaN team keys, live rows."""
    rnd = random.Random(seed)
    rows = []
    for e in range(events):
        ev = f"EV{e:03d}"
        kind = e % 3
        if kind == 0:
            for team in ("A", "B"):
                for k in range(rnd.randint(1, 6)):
                    bid = round(rnd.uniform(0.02, 0.95), 2)
                    rows.append(row(f"{ev}-{team}{k}", team, 1.5 + 3 * k + rnd.choice([0, 0, 3]),
                                    bid, min(round(bid + rnd.choice([0.01, 0.02, 0.05]), 2), 0.99),
                                    bsize=rnd.choice([0, 3, 50, np.nan]), asize=rnd.choice([5, 80, np.nan]),
                                    event=ev, league=rnd.choice(["NFL", "CFB"]),
                                    series=rnd.choice(["KXNFLSPREAD", "KXNCAAFSPREAD"])))
        elif kind == 1:
            for k in range(rnd.randint(1, 7)):
                bid = round(rnd.uniform(0.02, 0.95), 2)
                r = row(f"{ev}-T{k}", None, 40.5 + 3 * k, bid, min(round(bid + 0.02, 2), 0.99),
                        asize=rnd.choice([10, 200]), event=ev, series="KXNCAAFTOTAL", league="CFB")
                r["team_key"] = np.nan
                if rnd.random() < 0.1:
                    r["floor_strike"] = np.nan
                if rnd.random() < 0.1:
                    r["phase"] = "live"
                rows.append(r)
        else:
            lg = rnd.choice(["NFL", "CFB", "CBB"])
            for team in ("A", "B"):
                bid = round(rnd.uniform(0.2, 0.6), 2)
                rows.append(row(f"{ev}-{team}", team, np.nan, bid, round(bid + 0.01, 2), event=ev,
                                league=lg, series={"NFL": "KXNFLGAME", "CFB": "KXNCAAFGAME", "CBB": "KXNCAAMBGAME"}[lg],
                                stype=None))
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def test_fast_scan_matches_old_scan():
    for seed in (1, 7, 42):
        df = _messy_board(seed)
        old, new = _reference_scan(df), scan(df)
        assert len(old) > 5, "fixture must produce flags of every kind"
        assert {f["kind"] for f in old} == {"LADDER", "EXCLUSIVE", "COVER"}
        assert pd.DataFrame(new).equals(pd.DataFrame(old))
        assert pd.DataFrame(scan(df, include_live=True)).equals(pd.DataFrame(_reference_scan(df, include_live=True)))


def test_fast_scan_still_fee_and_price_capped():
    # broken ladder whose YES leg is 95c: the >90c rule still blocks it
    df = pd.DataFrame([row("A3", "A", 2.5, 0.94, 0.95), row("A7", "A", 6.5, 0.97, 0.98)])
    assert scan(df) == []


def test_structural_scan_is_fast():
    df = pd.concat([_messy_board(s, 250) for s in range(4)], ignore_index=True)   # ~1,000 events
    t = time.perf_counter()
    scan(df)
    assert time.perf_counter() - t < 1.0   # old groupby code: ~3 s


def test_cfb_name_similarity_is_cached():
    model._sim_cached.cache_clear()
    for _ in range(3):
        model._sim("Western Michigan", "Western Michigan Broncos")
    info = model._sim_cached.cache_info()
    assert info.misses == 1 and info.hits == 2


def test_match_cfb_parses_kickoffs_once_per_frame(monkeypatch):
    games = pd.DataFrame([{"game_id": 1, "home_team": "Akron", "away_team": "Western Michigan",
                           "home_division": "fbs", "away_division": "fbs",
                           "kickoff_utc": "2026-11-11T00:00:00Z"}])
    games = model.with_kickoffs(games)
    calls = {"n": 0}
    real = pd.to_datetime

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(model.pd, "to_datetime", counting)
    for _ in range(5):
        m = model.match_cfb(games, ["Western Michigan", "Akron"], ["WMU", "AKR"], "2026-11-11T00:00:00Z")
        assert m and m[0]["game_id"] == 1 and m[1] == {"WMU": "Western Michigan", "AKR": "Akron"}
    assert calls["n"] <= 5     # only the event's own kickoff, never the whole schedule again


def test_live_log_records_fast_lane_and_pass_times(tmp_path):
    from datetime import datetime, timedelta, timezone
    from edge_desk import live
    t = {"now": datetime(2026, 11, 14, 19, 0, tzinfo=timezone.utc), "mono": 0.0}

    def sleep(s):
        t["mono"] += s
        t["now"] += timedelta(seconds=s)

    calls = []

    def fast(nf, stats):
        calls.append(1)
        if len(calls) == 2:
            raise live.Throttled("x")

    def full(m, ts):
        t["mono"] += 4                                  # a 4-s Kalshi fetch
        return pd.DataFrame(), [], False

    rows = []
    s = live.run_loop(now_fn=lambda: t["now"], clock=lambda: t["mono"], sleep=sleep, fetch_full=full,
                      fetch_hot=full, odds_fn=lambda: None, scan_fn=lambda *a: None,
                      persist_fn=lambda *a: None, post_fn=lambda: [], commit_fn=lambda m: False, used_min=0,
                      seen=set(), alert_fn=lambda df, ts: [], log_fn=rows.append, fast_fn=fast)
    r = live.checkpoint_row(s, t["now"])
    assert r["fast_ticks"] == len(calls) > 2 and r["fast_throttled"] == 1
    assert r["fetch_s_median"] == 4.0 and r["loop_s_median"] == 4.0 and r["loop_s_p90"] == 4.0
    path = tmp_path / "live_log.csv"
    path.write_text("run_id,started_utc,minutes\nold,2026-11-01T00:00:00Z,10\n")   # pre-#44 header
    live.upsert_log(r, path)
    got = pd.read_csv(path)
    assert list(got.columns) == live.LOG_COLS and got["minutes"].tolist()[0] == 10
    assert int(got["fast_ticks"].iloc[-1]) == len(calls)


# ---- SHARP / ML pricing caches (#44): ~21 s -> ~6 s per pass, identical output ----
def test_fair_lines_computed_once_per_frame_and_as_of(monkeypatch):
    from edge_desk import odds
    from tests.test_sharp import _lines, SNAP
    lines = _lines()
    calls = []
    real = odds._fair_lines
    monkeypatch.setattr(odds, "_fair_lines", lambda l, a: calls.append(a) or real(l, a))
    a = odds.fair_lines(lines, SNAP)
    a.loc[:, "fair"] = -1.0                                   # a caller scribbling on its copy...
    b = odds.fair_lines(lines, SNAP)
    assert len(calls) == 1 and (b["fair"] > 0).all()          # ...never reaches the next caller
    odds.fair_lines(lines, "2026-11-12T17:30:00Z")            # other as_of: recomputed (look-ahead guard)
    odds.fair_lines(lines.copy(), SNAP)                       # other frame (next pass's load): recomputed
    assert len(calls) == 3


def test_fair_lines_cache_never_serves_a_look_ahead_line():
    from edge_desk import odds
    from tests.test_sharp import _lines
    lines = _lines(ts="2026-11-12T18:30:00Z")                 # fetched after the first as_of
    assert odds.fair_lines(lines, "2026-11-12T18:35:00Z").shape[0] > 0
    assert odds.fair_lines(lines, "2026-11-12T18:00:00Z").empty


def test_cached_rung_prob_matches_direct_lookup():
    from edge_desk import sharp
    from tests.test_sharp import _flags, _kalshi, _lines, _hist, SNAP, NOW
    f1, st1 = _flags()
    sharp._ID_CACHE.clear()
    f2, st2 = sharp.sharp_flags(_kalshi(), _lines(), _hist(), SNAP, now=NOW)
    assert f1 == f2 and f1 and st1["events_priced"] == st2["events_priced"]
    # the WAS rung must use the WAS point, the total rung the Over point
    fe = pd.DataFrame([{"market": "spreads", "outcome": "New York Giants", "point": -3.0, "source": "pinnacle"},
                       {"market": "spreads", "outcome": "Washington Commanders", "point": 3.0, "source": "pinnacle"},
                       {"market": "totals", "outcome": "Over", "point": 44.5, "source": "pinnacle"},
                       {"market": "totals", "outcome": "Over", "point": 47.5, "source": "late dup"}])

    class D:
        def p_greater(self, line, strike):
            return (line, strike)

    d = {"spread": D(), "total": D()}
    sides = {"WAS": "Washington Commanders", "NYG": "New York Giants"}
    sp = sharp.rung_prob(d, fe, {"floor_strike": 3.5, "series": "KXNFLSPREAD", "ticker": "X-WAS4"}, sides)
    tt = sharp.rung_prob(d, fe, {"floor_strike": 44.5, "series": "KXNFLTOTAL", "ticker": "X-45"}, sides)
    assert sp == ((-3.0, 3.5), -3.0, "pinnacle") and tt == ((44.5, 44.5), 44.5, "pinnacle")
    fe2 = fe.iloc[:2]                                          # a different frame: no totals -> None
    assert sharp.rung_prob(d, fe2, {"floor_strike": 44.5, "series": "KXNFLTOTAL", "ticker": "X-45"}, sides) is None
