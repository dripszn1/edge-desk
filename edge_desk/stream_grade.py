"""Grade the instant paper trades from the Kalshi stream (#66), kept apart from the SHARP
alert/held-back scorecards so the pre-registered Verdict reads are not mixed with a new method.

Taker fills in data/stream/fills_*.csv are graded exactly like SHARP bets (grade.model_bets:
first sighting per outcome, taker fee at the paper size, settled only from a final Kalshi result
fetched after the entry), then:
  - CLV vs Kalshi's pregame close after the fee (clv.closing_value, same rules as #31),
  - the same exits as #38 (trades.grade_all: hold, sell at tip, Auto Sell, brackets).
Maker quotes: fill rate only for now.

Outputs: data/stream_bets.csv, data/stream_clv.csv, data/stream_trades.csv,
weekly_report.json["stream"], digest_line().
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FILL_DIR = DATA / "stream"
BETS_OUT = DATA / "stream_bets.csv"
CLV_OUT = DATA / "stream_clv.csv"
TRADES_OUT = DATA / "stream_trades.csv"
LEAGUE = {"KXNFL": "NFL", "KXNCAAF": "CFB", "KXNCAAMB": "CBB"}


def league_of(ticker: str):
    t = str(ticker)
    for k, v in LEAGUE.items():
        if t.startswith(k):
            return v
    return None


def load_fills(directory: Path | None = None) -> pd.DataFrame:
    d = directory or FILL_DIR
    parts = []
    for p in sorted(d.glob("fills_*.csv")) if d.exists() else []:
        try:
            parts.append(pd.read_csv(p))
        except Exception:  # noqa: BLE001
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def stream_tickers(directory: Path | None = None) -> set[str]:
    f = load_fills(directory)
    return set(f["ticker"].dropna().astype(str)) if len(f) and "ticker" in f else set()


def to_flags(fills: pd.DataFrame, now) -> pd.DataFrame:
    """Taker fills -> the flag shape grade.model_bets expects. Never a row stamped after now."""
    if fills is None or fills.empty:
        return pd.DataFrame()
    f = fills[fills["kind"] == "taker"].copy()
    ts = pd.to_datetime(f["decided_utc"], utc=True, errors="coerce")
    f = f[ts.notna() & (ts <= pd.Timestamp(now))]
    if f.empty:
        return pd.DataFrame()
    from .grade import event_of
    out = pd.DataFrame({
        "snapshot_ts": pd.to_datetime(f["decided_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "leg1": f["side"].astype(str) + " " + f["ticker"].astype(str),
        "league": f["ticker"].map(league_of),
        "market": f["ticker"].astype(str).str.split("-").str[0].str.replace(r"^KX(NFL|NCAAF|NCAAMB)", "", regex=True),
        "event": f["ticker"].map(event_of),
        "price": pd.to_numeric(f["price"], errors="coerce"),
        "fair": pd.to_numeric(f["fair"], errors="coerce"),
        "edge": pd.to_numeric(f["edge_c"], errors="coerce") / 100,
        "contracts": pd.to_numeric(f["contracts"], errors="coerce").fillna(1).astype(int),
    })
    return out[out["league"].notna()].reset_index(drop=True)


def _block(bets, c, t) -> dict:
    from . import clv, trades
    out = {"plays": int(len(bets)), "settled": int((bets["status"] == "settled").sum()) if len(bets) else 0}
    if len(bets):
        s = bets[bets["status"] == "settled"]
        out["hold_pnl_usd"] = round(float(pd.to_numeric(s["pnl"]).sum()), 2) if len(s) else 0.0
        out["hold_record"] = f"{int((pd.to_numeric(s['pnl']) > 0).sum())}-{int((pd.to_numeric(s['pnl']) <= 0).sum())}"
    cs = clv.summary(c) if c is not None and len(c) else {}
    out["clv"] = (cs.get("stream") or {}).get("football")
    ts = trades.summary(t) if t is not None and len(t) else {}
    v = (ts.get("football") or {}).get("variants") or {}
    out["exits"] = {k: {x: v[k][x] for x in ("closed", "games", "pnl_usd", "t_by_game")}
                    for k in ("HOLD", "EXIT_AT_TIP", "RESTING_10", "RESTING_FAIR") if k in v}
    return out


def summary(bets, c, t, fills) -> dict:
    out = {"note": "instant paper trades from the Kalshi stream; kept apart from the alert scorecards"}
    if fills is not None and len(fills):
        mq = int((fills["kind"] == "maker_quote").sum())
        mf = int((fills["kind"] == "maker_fill").sum())
        lat = pd.to_numeric(fills.loc[fills["kind"] == "taker", "latency_ms"], errors="coerce").dropna()
        out["maker"] = {"quotes": mq, "fills": mf, "fill_rate_pct": round(100 * mf / mq, 1) if mq else None}
        out["median_latency_ms"] = int(lat.median()) if len(lat) else None
    for lg in ("NFL", "CFB"):
        b = bets[bets["league"] == lg] if len(bets) else bets
        if len(b):
            out[lg] = _block(b, c[c["league"] == lg] if c is not None and len(c) else c,
                             t[t["league"] == lg] if t is not None and len(t) else t)
    return out


def digest_line(s: dict | None) -> list[str]:
    parts = []
    for lg in ("NFL", "CFB"):
        b = (s or {}).get(lg)
        if not b:
            continue
        cell = f"{lg} {b['plays']} plays"
        c = b.get("clv") or {}
        if c.get("n"):
            cell += f", vs close {c['avg_clv_c']:+.1f}c ({c['n']}, {c['events']} games)"
        h = (b.get("exits") or {}).get("HOLD") or {}
        if h.get("closed"):
            cell += (f", hold ${h['pnl_usd']:+.2f}, auto-sell at fair "
                     f"${b['exits'].get('RESTING_FAIR', {}).get('pnl_usd', 0):+.2f} ({h['closed']} done)")
        parts.append(cell)
    if not parts:
        return []
    lat = (s or {}).get("median_latency_ms")
    return ["  instant (stream) paper trades: " + " · ".join(parts) + (f" · median {lat} ms" if lat is not None else "")]


def run(now=None) -> dict:
    from . import clv, grade, trades
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    fills = load_fills()
    flags = to_flags(fills, now)
    if flags.empty:
        return summary(pd.DataFrame(), None, None, fills)
    try:
        results = pd.read_csv(DATA / "results" / "results.csv")
    except Exception:  # noqa: BLE001
        results = pd.DataFrame(columns=["ticker", "status", "result", "fetched_ts"])
    bets = grade.model_bets(flags, results, now, source="stream", kind="STREAM", kickoffs=grade.event_kickoffs())
    bets.to_csv(BETS_OUT, index=False)
    tickers = set(bets["leg1"].astype(str).str.split(" ", n=1).str[1])
    start = pd.to_datetime(bets["snapshot_ts"], utc=True).min()
    snaps = clv.load_snapshots(tickers, start, now)
    c = clv.closing_value(bets, snaps, now)
    c.to_csv(CLV_OUT, index=False)
    t = trades.grade_all(bets, snaps, now, fast=trades.load_fast(tickers, start, now))
    t.to_csv(TRADES_OUT, index=False)
    s = summary(bets, c, t, fills)
    print("stream: " + " | ".join(digest_line(s)))
    return s


if __name__ == "__main__":
    run()
