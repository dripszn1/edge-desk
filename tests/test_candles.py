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
    # next run starts from the last stored minute, not the entry again
    calls.clear()
    candles.refresh(NOW, bets, get=get, directory=tmp_path, sleep=lambda x: None)
    assert calls[0]["start_ts"] == int(pd.Timestamp("2026-11-14T10:05:00Z").timestamp())
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
