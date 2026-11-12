"""#39a candles: parse both price formats, never keep a future minute, incremental + key handling."""
import pandas as pd

from edge_desk import candles

NOW = pd.Timestamp("2026-11-14T16:00:00Z")


class R:
    def __init__(self, code, j=None):
        self.status_code, self._j = code, j or {}

    def json(self):
        return self._j


def c(end, bh, bl, ah, al, dollars=False):
    if dollars:
        return {"end_period_ts": int(pd.Timestamp(end).timestamp()),
                "yes_bid": {"high_dollars": str(bh / 100), "low_dollars": str(bl / 100)},
                "yes_ask": {"high_dollars": str(ah / 100), "low_dollars": str(al / 100)}}
    return {"end_period_ts": int(pd.Timestamp(end).timestamp()),
            "yes_bid": {"high": bh, "low": bl}, "yes_ask": {"high": ah, "low": al}}


def test_parse_cents_and_dollars_and_drops_future_minutes():
    j = {"candlesticks": [c("2026-11-14T15:58:00Z", 60, 55, 64, 58),
                          c("2026-11-14T15:59:00Z", 61, 56, 65, 59, dollars=True),
                          c("2026-11-14T16:01:00Z", 90, 90, 95, 95)]}           # future: dropped
    df = candles.parse(j, NOW)
    assert len(df) == 2
    assert df.iloc[0]["yes_bid_hi"] == 0.60 and abs(df.iloc[1]["yes_ask_lo"] - 0.59) < 1e-9


def test_refresh_incremental_and_needs_key(tmp_path):
    bets = pd.DataFrame([{"leg1": "NO KXNCAAFSPREAD-26NOV14MICHORE-ORE6", "snapshot_ts": "2026-11-14T10:00:00Z",
                          "status": "pending", "settled_ts": None}])
    calls = []

    def get(path, params):
        calls.append(params)
        return R(200, {"candlesticks": [c("2026-11-14T10:05:00Z", 60, 55, 64, 58)]})

    s = candles.refresh(NOW, bets, get=get, directory=tmp_path, sleep=lambda x: None)
    assert s["tickers"] == 1 and (tmp_path / "KXNCAAFSPREAD-26NOV14MICHORE-ORE6.csv").exists()
    assert "/series/" not in str(calls[0])                 # params only; series path built by fetch
    assert calls[0]["period_interval"] == 1 and calls[-1]["end_ts"] <= int(NOW.timestamp())
    # next run starts where the last one stopped reading (#43 fetched_to), not the entry again
    calls.clear()
    candles.refresh(NOW, bets, get=get, directory=tmp_path, sleep=lambda x: None)
    assert calls == []
    later = NOW + pd.Timedelta(minutes=30)
    candles.refresh(later, bets, get=get, directory=tmp_path, sleep=lambda x: None)
    assert calls[0]["start_ts"] == int(NOW.timestamp()) and calls[-1]["end_ts"] <= int(later.timestamp())
    # 401 -> remembered, no more requests that day
    s2 = candles.refresh(NOW, bets, get=lambda p, q: R(401), directory=tmp_path / "k", sleep=lambda x: None)
    assert s2["problem"] == "needs_key"
    assert candles.refresh(NOW, bets, get=get, directory=tmp_path / "k", sleep=lambda x: None) == {"skipped": "needs_key"}


def test_refresh_without_network_is_quiet(tmp_path):
    bets = pd.DataFrame([{"leg1": "YES KXNFLTOTAL-26NOV16LACBAL-44", "snapshot_ts": "2026-11-14T10:00:00Z",
                          "status": "pending"}])

    def boom(path, params):
        raise ConnectionError("no route")

    assert candles.refresh(NOW, bets, get=boom, directory=tmp_path, sleep=lambda x: None)["problem"] == "ConnectionError"


# ---- #43: candles must actually reach the graded bets -------------------------------------------

def _bets(rows):
    return pd.DataFrame([{"leg1": f"YES {t}", "snapshot_ts": ts, "status": st, "settled_ts": sts}
                         for t, ts, st, sts in rows])


def test_429_is_retried_with_backoff_not_fatal(tmp_path):
    bets = _bets([("KXNCAAMBTOTAL-26NOV11AB-145", "2026-11-14T10:00:00Z", "pending", None)])
    seq = [R(429), R(429), R(200, {"candlesticks": [c("2026-11-14T10:05:00Z", 60, 55, 64, 58)]})]
    slept = []
    s = candles.refresh(NOW, bets, get=lambda p, q: seq.pop(0) if seq else R(200, {}), directory=tmp_path,
                        sleep=slept.append)
    assert s["problem"] is None and s["candles"] == 1
    assert sum(1 for x in slept if x >= 1) >= 2            # backed off on both 429s


def test_settled_tickers_first_then_least_recently_fetched(tmp_path):
    bets = _bets([("KXNFLTOTAL-26NOV16P-44", "2026-11-13T00:00:00Z", "pending", None),
                  ("KXNFLTOTAL-26NOV16Q-44", "2026-11-13T01:00:00Z", "pending", None),
                  ("KXNCAAMBTOTAL-26NOV11S-145", "2026-11-13T05:00:00Z", "settled", "2026-11-13T08:00:00Z")])
    state = {"tickers": {"KXNFLTOTAL-26NOV16P-44": {"fetched_at": "2026-11-14T15:00:00Z"},
                         "KXNFLTOTAL-26NOV16Q-44": {"fetched_at": "2026-11-14T09:00:00Z"}}}
    order = [t for t, _, _ in candles.todo(bets, state, NOW)]
    assert order == ["KXNCAAMBTOTAL-26NOV11S-145", "KXNFLTOTAL-26NOV16Q-44", "KXNFLTOTAL-26NOV16P-44"]


def test_closed_market_with_no_candles_still_finishes(tmp_path):
    tk = "KXNCAAMBTOTAL-26NOV11S-145"
    bets = _bets([(tk, "2026-11-13T05:00:00Z", "settled", "2026-11-13T08:00:00Z")])
    calls = []

    def get(p, q):
        calls.append(q)
        return R(200, {"candlesticks": []})

    candles.refresh(NOW, bets, get=get, directory=tmp_path, sleep=lambda x: None)
    assert max(q["end_ts"] for q in calls) <= int(pd.Timestamp("2026-11-13T08:00:00Z").timestamp())
    st = candles.load_state(tmp_path / "_state.json")["tickers"][tk]
    assert st["done"] is True
    calls.clear()
    candles.refresh(NOW, bets, get=get, directory=tmp_path, sleep=lambda x: None)
    assert calls == []                                     # finished: never asked again


def test_one_ticker_error_does_not_stop_the_rest(tmp_path):
    bets = _bets([("KXNCAAMBTOTAL-26NOV11BAD-145", "2026-11-13T05:00:00Z", "settled", "2026-11-13T08:00:00Z"),
                  ("KXNCAAMBTOTAL-26NOV11OK-145", "2026-11-13T06:00:00Z", "settled", "2026-11-13T08:00:00Z")])

    def get(p, q):
        if "BAD" in p:
            return R(404)
        return R(200, {"candlesticks": [c("2026-11-13T06:05:00Z", 60, 55, 64, 58)]})

    s = candles.refresh(NOW, bets, get=get, directory=tmp_path, sleep=lambda x: None)
    assert (tmp_path / "KXNCAAMBTOTAL-26NOV11OK-145.csv").exists() and s["errors"] == 1


def test_time_budget_stops_the_refresh(tmp_path):
    bets = _bets([(f"KXNCAAMBTOTAL-26NOV11G{i}-145", "2026-11-13T05:00:00Z", "settled", "2026-11-13T08:00:00Z")
                  for i in range(5)])
    t = [0.0]

    def clock():
        t[0] += 10
        return t[0]

    s = candles.refresh(NOW, bets, get=lambda p, q: R(200, {}), directory=tmp_path, sleep=lambda x: None,
                        budget_s=25, clock=clock)
    assert s["problem"] == "budget" and s["tickers"] < 5


def test_one_cent_in_cents_is_not_a_dollar():
    df = candles.parse({"candlesticks": [c("2026-11-14T15:58:00Z", 1, 1, 2, 1)]}, NOW)
    assert df.iloc[0]["yes_bid_hi"] == 0.01 and df.iloc[0]["yes_ask_hi"] == 0.02
