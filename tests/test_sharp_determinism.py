"""#36: rung prices must not depend on input row order (or on the runner's sort algorithm)."""
import numpy as np

from edge_desk import sharp


def _games(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    # half-point lines with lots of exact ties, like real closing lines
    lines = rng.choice(np.arange(-17, 17.5, 0.5), n)
    out = np.round(lines + rng.normal(0, 13, n))
    return lines, out


def test_line_dist_prices_ignore_row_order():
    lines, out = _games()
    base = sharp.LineDist(lines, out, "spread")
    rng = np.random.default_rng(1)
    # a far favourite (fallback branch: MIN_N nearest games, many tied gaps) and a near line
    probes = [(L, k) for L in (-24.0, -20.5, 21.0, -3.0, 0.0) for k in np.arange(L - 10.5, L + 11, 1.0)]
    want = [base.p_greater(L, k) for L, k in probes]
    for _ in range(6):
        p = rng.permutation(len(lines))
        d = sharp.LineDist(lines[p], out[p], "spread")
        got = [d.p_greater(L, k) for L, k in probes]
        assert got == want


def test_fallback_tie_break_is_by_value_not_position():
    # games split between two lines equally far from the probe: the chosen set (all games tied at the
    # MIN_N-th nearest gap) must not depend on the order they came in
    n = sharp.MIN_N
    lines = np.array([10.0] * n + [20.0] * n)
    out = np.concatenate([np.arange(n) % 30 - 5.0, np.arange(n) % 30 + 5.0])
    a = sharp.LineDist(lines, out, "spread").p_greater(15.0, 14.5)
    b = sharp.LineDist(lines[::-1], out[::-1], "spread").p_greater(15.0, 14.5)
    assert a == b


def _hist(n=600, seed=3):
    import pandas as pd
    rng = np.random.default_rng(seed)
    L = rng.choice(np.arange(-10, 10.5, 0.5), n)
    return pd.DataFrame({"season": np.repeat([2017, 2018], n // 2), "spread_line": L,
                         "total_line": 45.0, "home_score": 24 + np.round(-L + rng.normal(0, 13, n)).clip(-24),
                         "away_score": 24.0,
                         "kickoff_utc": pd.date_range("2017-09-01", periods=n, freq="D", tz="UTC").astype(str)})


def test_backtest_reruns_only_when_inputs_change(tmp_path, monkeypatch):
    import json
    hist = {"g": _hist()}
    calls = []
    real = sharp.backtest
    monkeypatch.setattr(sharp, "history", lambda lg: hist["g"] if lg == "NFL" else hist["g"].iloc[0:0])
    monkeypatch.setattr(sharp, "backtest", lambda g, seasons=range(2018, 2026): calls.append(1) or real(g, seasons=[2018]))
    out = tmp_path / "bt.json"
    monkeypatch.setattr(sharp, "BACKTEST_OUT", out)
    sharp.run_backtest()
    assert len(calls) == 1 and json.loads(out.read_text())["inputs_hash"]
    sharp.run_backtest()                      # same games, same code -> kept, not recomputed
    assert len(calls) == 1
    g = hist["g"].copy(); g.loc[5, "home_score"] += 7  # one corrected score -> must re-run
    hist["g"] = g
    sharp.run_backtest()
    assert len(calls) == 2
    sharp.run_backtest(force=True)
    assert len(calls) == 3
    out.write_text("{not json")               # unreadable old result -> re-run, never crash
    sharp.run_backtest()
    assert len(calls) == 4
