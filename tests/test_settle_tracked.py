"""Held-back SHARP plays (sharp_tracked.csv: ticker + side, no leg1) must be settled too."""
import pandas as pd

from edge_desk import settle


def test_flag_tickers_reads_ticker_column_and_legs():
    f = pd.DataFrame([{"leg1": "NO KXA-1", "leg2": None, "ticker": None},
                      {"leg1": None, "leg2": None, "ticker": "KXNCAAMBTOTAL-26NOV10KANUK-150"}])
    assert settle.flag_tickers(f) == {"KXA-1", "KXNCAAMBTOTAL-26NOV10KANUK-150"}


def test_tracked_file_is_in_the_settle_list():
    assert any(p.name == "sharp_tracked.csv" for p in settle.MODEL_FLAGS)


def test_unrecorded_pull_rows_are_not_fetched(tmp_path, monkeypatch):
    """#56: a play priced off a sharp pull we never recorded (leaked test fixture, #47) is never
    graded, so settle must not keep asking Kalshi for its ticker either."""
    tr = tmp_path / "sharp_tracked.csv"
    pd.DataFrame([{"snapshot_ts": "2026-11-12T01:00:00Z", "ticker": "KXREAL-1", "side": "YES",
                   "odds_fetched_ts": "2026-11-12T00:50:00Z"},
                  {"snapshot_ts": "2026-11-12T01:00:00Z", "ticker": "KXFAKE-16", "side": "YES",
                   "odds_fetched_ts": "2026-11-12T00:55:00Z"}]).to_csv(tr, index=False)
    lines = tmp_path / "lines.csv"
    pd.DataFrame({"fetched_ts": ["2026-11-12T00:00:00Z", "2026-11-12T00:50:00Z"]}).to_csv(lines, index=False)
    monkeypatch.setattr(settle, "FLAGS", tmp_path / "none.csv")
    monkeypatch.setattr(settle, "MODEL_FLAGS", (tr,))
    monkeypatch.setattr(settle, "RESULTS", tmp_path / "results.csv")
    monkeypatch.setattr(settle, "ODDS_LINES", lines)
    asked = []
    monkeypatch.setattr(settle, "_get", lambda path, p: asked.append(p["tickers"]) or {"markets": []})
    settle.run()
    assert asked == ["KXREAL-1"]
