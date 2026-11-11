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
