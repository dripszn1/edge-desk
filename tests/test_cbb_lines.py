"""CBB history (#22): parse CBBD lines, look-ahead guards, walk-forward SD fit, shadow-only."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from edge_desk import cbb_lines, sharp

FETCH = "2026-11-11T12:00:00Z"


def _payload():
    return [{"gameId": 1, "season": 2026, "startDate": "2025-11-04T00:00:00.000Z", "homeTeam": "Duke",
             "awayTeam": "Kansas", "homeScore": 80, "awayScore": 70,
             "lines": [{"provider": "ESPN BET", "spread": -6.5, "overUnder": 150.5},
                       {"provider": "Bovada", "spread": -5.5, "overUnder": 149.5}]},
            {"gameId": 2, "season": 2026, "startDate": "2025-11-05T00:00:00.000Z", "homeTeam": "A",
             "awayTeam": "B", "homeScore": None, "awayScore": None, "lines": []}]


def test_parse_home_margin_sign_and_median():
    df = cbb_lines.parse(_payload(), FETCH)
    assert len(df) == 1                                   # no lines -> dropped
    r = df.iloc[0]
    assert r.spread_line == 6.0 and r.total_line == 150.0  # home -6 -> home margin line +6
    assert r.source == "median:2" and r.home_score == 80


def _hist(n=1200, sd_m=10.0, sd_t=12.0, seed=1, start="2022-11-07", fetched=FETCH):
    rng = np.random.default_rng(seed)
    ko = pd.date_range(start, periods=n, freq="6h", tz="UTC")
    line = rng.uniform(-15, 15, n).round() + 0.5
    tot = rng.uniform(130, 160, n).round() + 0.5
    m = line + rng.normal(0, sd_m, n)
    t = tot + rng.normal(0, sd_t, n)
    return pd.DataFrame({"game_id": range(n), "season": ko.year + (ko.month >= 8), "kickoff_utc": ko.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "home_score": (t + m) / 2, "away_score": (t - m) / 2, "spread_line": line,
                         "total_line": tot, "fetched_ts": fetched})


def test_history_look_ahead(tmp_path):
    p = tmp_path / "lines.csv"
    h = _hist(n=10)
    h.loc[0, "fetched_ts"] = "2030-01-01T00:00:00Z"        # fetched after as_of -> unusable
    h.to_csv(p, index=False)
    got = cbb_lines.history("2026-11-11T13:00:00Z", p)
    assert 0 not in got["game_id"].tolist() and len(got) == 9
    got = cbb_lines.history("2022-11-08T00:00:00Z", p)      # only games tipped before as_of
    assert (pd.to_datetime(got["kickoff_utc"], utc=True) < pd.Timestamp("2022-11-08", tz="UTC")).all()


def test_sd_fitted_when_enough_games_else_assumed():
    h = _hist()
    d = sharp.cbb_dists(h, "2030-01-01T00:00:00Z")
    assert d["spread"].source.startswith("fitted") and abs(d["spread"].sd - 10.0) < 0.6
    assert abs(d["total"].sd - 12.0) < 0.7 and d["total"].normal and d["spread"].normal
    few = sharp.cbb_dists(h.head(100), "2030-01-01T00:00:00Z")
    assert few["spread"].sd == sharp.CBB_SD["spread"] and few["spread"].source.startswith("assumed")
    assert sharp.cbb_dists(None, "2030-01-01T00:00:00Z")["total"].sd == sharp.CBB_SD["total"]


def test_sd_fit_ignores_future_games():
    """Games after the snapshot (with a wild SD) must not move the fit."""
    past = _hist(n=800, sd_m=10.0)
    fut = _hist(n=800, sd_m=30.0, seed=2, start="2026-12-01")
    d = sharp.cbb_dists(pd.concat([past, fut]), "2026-11-11T00:00:00Z")
    assert abs(d["spread"].sd - 10.0) < 0.7


def test_sd_check_walk_forward_prefers_true_sd():
    h = _hist(n=2400, sd_m=9.0, sd_t=14.0)
    rep = sharp.cbb_sd_check(h)
    assert rep["spread"]["fitted_better"] and rep["total"]["fitted_better"]
    assert all(s["season"] > h["season"].min() for s in rep["spread"]["by_season"])   # first season only trains


def test_due_history_then_refresh():
    now = datetime(2026, 11, 11, 12, tzinfo=timezone.utc)
    full = cbb_lines.due(now, pd.DataFrame(columns=cbb_lines.COLS))
    assert full[0][0].startswith("2020-11-01") and full[-1][1].startswith("2026-11-11")
    fresh = pd.DataFrame({"fetched_ts": ["2026-11-11T06:00:00Z"]})
    assert cbb_lines.due(now, fresh) == []
    old = pd.DataFrame({"fetched_ts": ["2026-11-10T20:00:00Z"]})
    assert len(cbb_lines.due(now, old)) == 1


def test_no_key_skips(monkeypatch):
    monkeypatch.delenv("CBBD_API_KEY", raising=False)
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    assert cbb_lines.run() is None


def test_cbb_never_live_even_if_gate_open(monkeypatch):
    from edge_desk import scan
    monkeypatch.setattr(sharp, "gate", lambda league=None, bets_path=None: (True, "open: test"))
    fake = [{"league": "CBB", "ticker": "X", "stake": 1.0, "kind": "SHARP"}]
    monkeypatch.setattr(sharp, "sharp_flags", lambda *a, **k: (list(fake) if k.get("league") == "CBB" else [], {"ok": True}))
    monkeypatch.setattr(scan, "with_url", lambda f: f)
    monkeypatch.setattr(scan, "_append_csv", lambda *a, **k: None)
    monkeypatch.setattr(sharp, "history", lambda lg: pd.DataFrame())
    out = scan.run_sharp(pd.DataFrame(), "2026-11-11T00:00:00Z", lost=0.0, leagues=("CBB",))
    assert out["live"] == [] and len(out["shadow"]) == 1
    assert out["status"]["CBB"]["gate"].startswith("shadow-only")
