"""Held-back SHARP plays (sharp_tracked.csv: ticker + side, no leg1) must be settled too."""
import pandas as pd

from edge_desk import settle


def test_flag_tickers_reads_ticker_column_and_legs():
    f = pd.DataFrame([{"leg1": "NO KXA-1", "leg2": None, "ticker": None},
                      {"leg1": None, "leg2": None, "ticker": "KXNCAAMBTOTAL-26NOV10KANUK-150"}])
    assert settle.flag_tickers(f) == {"KXA-1", "KXNCAAMBTOTAL-26NOV10KANUK-150"}


def test_tracked_file_is_in_the_settle_list():
    assert any(p.name == "sharp_tracked.csv" for p in settle.MODEL_FLAGS)
