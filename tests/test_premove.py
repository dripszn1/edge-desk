import pandas as pd

from edge_desk import grade, premove


def _write(tmp_path, stamp, rows):
    d = tmp_path / stamp[:10]
    d.mkdir(parents=True, exist_ok=True)
    name = stamp[11:13] + stamp[14:16] + stamp[17:19] + "Z.parquet"
    pd.DataFrame(rows).to_parquet(d / name)


def _q(t, bid, ask):
    return {"ticker": t, "yes_bid": bid, "yes_ask": ask, "no_bid": round(1 - ask, 2), "no_ask": round(1 - bid, 2)}


def _flag(t, side="YES"):
    return {"ticker": t, "side": side, "price": 0.42, "fair": 0.49, "sharp_line": 149, "odds_fetched_ts": "x"}


def test_cbb_play_held_back_when_kalshi_moved_first(tmp_path):
    premove._CACHE.clear()
    _write(tmp_path, "2026-11-12T22:00:00Z", [_q("A", .46, .48), _q("B", .40, .42)])
    now = pd.DataFrame([_q("A", .40, .42), _q("B", .40, .42)])
    logged = []
    keep = premove.screen([_flag("A"), _flag("B")], now, "2026-11-12T22:31:00Z", "CBB",
                          log=lambda f, r: logged.append((f["ticker"], r)), snap_dir=tmp_path)
    assert [f["ticker"] for f in keep] == ["B"]                       # B didn't move: still alerts
    assert logged == [("A", "kalshi moved first")]                    # A held back, still graded


def test_no_side_mirrors_and_football_untouched(tmp_path):
    premove._CACHE.clear()
    _write(tmp_path, "2026-11-12T22:00:00Z", [_q("A", .50, .52)])
    now = pd.DataFrame([_q("A", .56, .58)])                            # YES up 6c = NO down 6c
    logged = []
    keep = premove.screen([_flag("A", "NO")], now, "2026-11-12T22:31:00Z", "CBB",
                          log=lambda f, r: logged.append(f["ticker"]), snap_dir=tmp_path)
    assert keep == [] and logged == ["A"]
    keep = premove.screen([_flag("A", "NO")], now, "2026-11-12T22:31:00Z", "NFL", log=None, snap_dir=tmp_path)
    assert len(keep) == 1


def test_fails_open_without_an_earlier_snapshot_and_never_uses_a_later_one(tmp_path):
    premove._CACHE.clear()
    _write(tmp_path, "2026-11-12T22:20:00Z", [_q("A", .60, .62)])     # only 11 min before: too recent
    _write(tmp_path, "2026-11-12T23:00:00Z", [_q("A", .90, .92)])     # after the scan: never used
    now = pd.DataFrame([_q("A", .40, .42)])
    keep = premove.screen([_flag("A")], now, "2026-11-12T22:31:00Z", "CBB",
                          log=lambda f, r: None, snap_dir=tmp_path)
    assert len(keep) == 1


def test_reason_bucket():
    assert grade.reason_bucket("kalshi moved first") == "kalshi_first"
