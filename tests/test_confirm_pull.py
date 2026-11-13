"""#28: when a SHARP play is held back ONLY because the sharp line is stale, the next odds check
may refresh that league early (confirm pull), still inside the reserve and the daily allowance."""
import json

import pandas as pd

from edge_desk import moneyline, odds, sharp
from tests import test_cfb_lines as C
from tests import test_moneyline as M

NOW = "2026-11-14T15:00:00Z"
KO = "2026-11-15T12:00:00Z"            # 21h out: game day (4h interval), not "hot"


def _st(last="2026-11-14T13:30:00Z", remaining=200):
    return {"remaining": remaining, "remaining_seen_utc": NOW, "last_pull": {"CFB": last}}


def test_confirm_pulls_early():
    assert odds.plan_pulls(_st(), NOW, {"NFL": None, "CFB": KO}) == []                     # 1.5h < 4h
    assert odds.plan_pulls(_st(), NOW, {"NFL": None, "CFB": KO}, confirm={"CFB"}) == ["CFB"]


def test_confirm_respects_min_gap():
    st = _st(last=(pd.Timestamp(NOW) - pd.Timedelta(hours=odds.CONFIRM_MIN_GAP_H) + pd.Timedelta(minutes=5))
             .strftime("%Y-%m-%dT%H:%M:%SZ"))
    assert odds.plan_pulls(st, NOW, {"NFL": None, "CFB": KO}, confirm={"CFB"}) == []


def test_confirm_respects_allowance_and_reserve():
    st = _st()
    st["spent"] = {"2026-11-14": 999}
    assert odds.plan_pulls(st, NOW, {"NFL": None, "CFB": KO}, confirm={"CFB"}) == []
    low = _st(remaining=odds.ODDS_RESERVE + odds.COST_PER_PULL - 1)
    assert odds.plan_pulls(low, NOW, {"NFL": None, "CFB": KO}, confirm={"CFB"}) == []


def test_confirm_never_for_cbb_or_past_games():
    st = {"remaining": 200, "remaining_seen_utc": NOW}
    assert "CBB" not in odds.plan_pulls(st, NOW, {"NFL": None, "CFB": None, "CBB": None}, confirm={"CBB"})
    assert odds.plan_pulls(_st(), NOW, {"NFL": None, "CFB": "2026-11-14T14:00:00Z"}, confirm={"CFB"}) == []


def _latest(tmp_path, snap, sh):
    p = tmp_path / "latest.json"
    p.write_text(json.dumps({"snapshot_ts": snap, "sharp": sh}))
    return p


def test_confirm_leagues_from_latest(tmp_path):
    sh = {"CFB": {"stale_only": 2}, "NFL": {"stale_only": 0}, "NFL ML": {"stale_only": 1},
          "CBB": {"stale_only": 5}}
    assert odds.confirm_leagues(NOW, _latest(tmp_path, "2026-11-14T14:50:00Z", sh)) == {"CFB", "NFL"}
    # an old scan can't ask for a pull (its plays may be long gone)
    assert odds.confirm_leagues(NOW, _latest(tmp_path, "2026-11-14T13:00:00Z", sh)) == set()
    # a scan stamped after "now" is ignored (look-ahead)
    assert odds.confirm_leagues(NOW, _latest(tmp_path, "2026-11-14T15:10:00Z", sh)) == set()
    assert odds.confirm_leagues(NOW, tmp_path / "missing.json") == set()


def test_sharp_counts_stale_only():
    # fresh line: find this rung's fair, then price the ask ~5c under it so the edge is under the cap
    fresh, _ = sharp.sharp_flags(pd.DataFrame([C._row(strike=6.5, bid=0.28, ask=0.30)]), C._odds(), C._cfb_hist(),
                                 C.SNAP, now=pd.Timestamp(C.SNAP), league="CFB")
    fair = pd.read_csv(sharp.REJECTED)["fair"].iloc[-1] if sharp.REJECTED.exists() else fresh[0]["fair"]
    ask = round(fair - 0.05, 2)
    row = C._row(strike=6.5, bid=round(ask - 0.02, 2), ask=ask)
    stale = (pd.Timestamp(C.SNAP) - pd.Timedelta(minutes=sharp.ODDS_MAX_AGE_MIN + 30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    flags, st = sharp.sharp_flags(pd.DataFrame([row]), C._odds(ts=stale), C._cfb_hist(), C.SNAP,
                                  now=pd.Timestamp(C.SNAP), league="CFB")
    assert flags == [] and st.get("stale_only") == 1


def test_far_rung_stale_is_not_stale_only():
    stale = (pd.Timestamp(C.SNAP) - pd.Timedelta(minutes=sharp.ODDS_MAX_AGE_MIN + 30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    _, st = sharp.sharp_flags(pd.DataFrame([C._row(strike=15.5, bid=0.10, ask=0.12,
                              ticker="KXNCAAFSPREAD-26NOV14AUBMSST-AUB16")]), C._odds(ts=stale), C._cfb_hist(),
                              C.SNAP, now=pd.Timestamp(C.SNAP), league="CFB")
    assert not st.get("stale_only")


def test_moneyline_counts_stale_only():
    _, st = moneyline.ml_flags(pd.DataFrame([M._row()]), M._odds(ts="2026-11-15T12:30:00Z"), M.SNAP,
                               now=pd.Timestamp(M.SNAP))
    assert st.get("stale_only") == 1


def test_month_with_confirm_always_on_fits_free_tier():
    """Worst case: every 10-min live check asks for a confirm pull. Spend still fits the free tier."""
    st, spent = {}, 0
    t = pd.Timestamp("2026-12-01T00:00:00Z")
    while t.month == 12:
        kick = {}
        for lg in odds.SPORTS:
            d = t.normalize()
            nxt = [d + pd.Timedelta(days=k, hours=18) for k in range(8) if (d + pd.Timedelta(days=k)).day_of_week in (3, 5, 6)]
            kick[lg] = next((x for x in nxt if x > t), None)
        for lg in odds.plan_pulls(st, t, kick, near_h=1.5, near_interval_h=1.0, confirm={"NFL", "CFB"}):
            rem = odds.remaining(st, t) - odds.COST_PER_PULL
            st = odds.record_pull(st, lg, t, {"x-requests-remaining": str(rem), "x-requests-last": "3"})
            spent += 3
        t += pd.Timedelta(minutes=10)
    assert spent <= odds.MONTHLY_CREDITS - odds.ODDS_RESERVE
