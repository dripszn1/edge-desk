"""Maker-entry test (#64), REPORT ONLY: would a resting limit BUY beat taking the ask?

Why (11-14): football SHARP picks are ~even with Kalshi's close before fees and lose only after
the taker entry fee, so the cheapest lever is the entry itself. For every graded football SHARP
play (shadow + held back, one per contract/outcome as in #60/#61):

  MAKER_BID  resting buy at our side's BID at flag time
  MAKER_1    resting buy at bid + 1c (skipped when that would reach the ask: then it's a taker)

The flag-time quote = the latest saved snapshot at/before the entry (<= 20 min back, like #57).
A limit FILLS at exactly its price the first time a LATER point (snapshot / fast lane / closed
1-min candle), strictly after the entry and before kickoff and at/before now, shows our side's
ask <= the limit. Never the flag's own quote (no look-ahead). Entry fee = Kalshi maker fee.
Unfilled = no bet (fill rate reported). Filled entries are graded with the same exits as #38
(HOLD, EXIT_AT_TIP, RESTING_10, RESTING_FAIR) from the fill time, plus CLV vs the same close
mid as #31 after the maker fee. Adverse-selection check: the TAKER CLV of plays that would have
filled vs those that wouldn't (if the fills are mostly the losers, maker entries only look cheap).

Output: data/maker_entries.csv, summary() -> weekly_report["maker_entry"], digest_line().
No live alert/pricing change.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from . import fees, kickoffs, trades
from .clv import _event, _t

DATA = trades.DATA
OUT = DATA / "maker_entries.csv"
VARIANTS = ("MAKER_BID", "MAKER_1")
EXITS = ("HOLD", "EXIT_AT_TIP", "RESTING_10", "RESTING_FAIR")
FLAG_LOOKBACK = pd.Timedelta(minutes=20)
COLS = ["bet_id", "source", "league", "leg1", "entry_ts", "taker_price", "variant", "limit",
        "fill_ts", "status", "fee_c", "close_mid", "clv_c", "taker_clv_c"] + \
       [f"{e}_status" for e in EXITS] + [f"{e}_pnl_usd" for e in EXITS] + ["contracts"]


def _ts(x):
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t


def side_ask(side: str, yes_bid, yes_ask):
    """Best price we could BUY our side at: YES -> yes ask, NO -> 1 - yes bid."""
    yb, ya = trades._f(yes_bid), trades._f(yes_ask)
    if str(side).upper() == "YES":
        return ya if ya is not None and 0 < ya < 1 else None
    return round(1 - yb, 4) if yb is not None and 0 < yb < 1 else None


def ask_path(side: str, snaps, candles, fast, entry, now) -> list[tuple]:
    """[(ts, ask_lo)] strictly after entry, at/before now, oldest first."""
    entry, now = _ts(entry), _ts(now)
    pts = []
    for df in (snaps, fast):
        if df is None or df.empty:
            continue
        for r in df.to_dict("records"):
            ts = _ts(r["snapshot_ts"])
            if entry < ts <= now:
                a = side_ask(side, r.get("yes_bid"), r.get("yes_ask"))
                if a is not None:
                    pts.append((ts, a))
    if candles is not None and not candles.empty:
        for r in candles.to_dict("records"):
            ts = _ts(r["end_ts"])
            if not (entry + pd.Timedelta(minutes=1) <= ts <= now):
                continue
            if str(side).upper() == "YES":
                lo = trades._f(r.get("yes_ask_lo"))
            else:
                bh = trades._f(r.get("yes_bid_hi"))
                lo = None if bh is None else round(1 - bh, 4)
            if lo is not None and 0 < lo < 1:
                pts.append((ts, lo))
    pts.sort(key=lambda p: p[0])
    return pts


def flag_bid(side: str, snaps: pd.DataFrame | None, entry):
    """Our side's bid in the latest saved snapshot at/before the entry (<= 20 min back)."""
    if snaps is None or snaps.empty:
        return None
    entry = _ts(entry)
    best = None
    for r in snaps.to_dict("records"):
        ts = _ts(r["snapshot_ts"])
        if entry - FLAG_LOOKBACK <= ts <= entry and (best is None or ts > best[0]):
            best = (ts, r)
    if best is None:
        return None
    return trades.side_bid(side, best[1].get("yes_bid"), best[1].get("yes_ask"))


def limits(bid, taker_price) -> dict:
    """{variant: limit} for the variants that are real resting orders (strictly below the ask)."""
    out = {}
    if bid is None or taker_price is None:
        return out
    if 0 < bid < taker_price:
        out["MAKER_BID"] = round(bid, 4)
        b1 = round(bid + 0.01, 4)
        if b1 < taker_price - 1e-9:
            out["MAKER_1"] = b1
    return out


def grade_play(b: dict, snaps, candles, fast, kickoff, close_mid, taker_clv, now) -> list[dict]:
    side = str(b["leg1"]).split(" ", 1)[0]
    entry = _ts(b["snapshot_ts"])
    now = _ts(now)
    ko = _ts(kickoff) if kickoff is not None else None
    n = max(int(float(b.get("contracts") or 0)), 1)
    price = float(b["price"])
    rows = []
    path = ask_path(side, snaps, candles, fast, entry, now)
    pre = [p for p in path if ko is None or p[0] < ko]
    for variant, lim in limits(flag_bid(side, snaps, entry), price).items():
        base = {"bet_id": b.get("bet_id"), "source": b.get("source"), "league": b.get("league"),
                "leg1": b["leg1"], "entry_ts": entry.strftime("%Y-%m-%dT%H:%M:%SZ"), "taker_price": price,
                "variant": variant, "limit": lim, "contracts": n, "close_mid": close_mid,
                "taker_clv_c": taker_clv}
        fill = next((ts for ts, a in pre if a <= lim + 1e-9), None)
        if fill is None:
            done = ko is not None and now >= ko
            rows.append({**base, "fill_ts": None, "status": "unfilled" if done else "open",
                         **{f"{e}_status": None for e in EXITS}, **{f"{e}_pnl_usd": None for e in EXITS}})
            continue
        fee = trades.maker_fee(lim, n)
        fee_c = round(100 * fee / n, 4)
        clv_c = round(100 * (float(close_mid) - lim) - fee_c, 4) if close_mid is not None and \
            not (isinstance(close_mid, float) and math.isnan(close_mid)) and fill < (ko or now) else None
        mb = {**b, "snapshot_ts": fill.strftime("%Y-%m-%dT%H:%M:%SZ"), "price": lim, "fees": fee,
              "contracts": n}
        bid_path = trades.build_path(side, snaps, candles, fast, fill, now)
        graded = {g["variant"]: g for g in trades.grade_one(mb, bid_path, ko, trades.settle_value(b), now)}
        ex = {}
        for e in EXITS:
            g = graded.get(e) or {}
            ex[f"{e}_status"] = g.get("status")
            ex[f"{e}_pnl_usd"] = g.get("pnl_usd")
        rows.append({**base, "fill_ts": fill.strftime("%Y-%m-%dT%H:%M:%SZ"), "status": "filled",
                     "fee_c": fee_c, "clv_c": clv_c, **ex})
    return rows


def grade_all(bets: pd.DataFrame, snaps: pd.DataFrame, clv_bets: pd.DataFrame | None, now,
              candles_fn=trades.load_candles, fast: pd.DataFrame | None = None) -> pd.DataFrame:
    now = _ts(now)
    if bets is None or bets.empty:
        return pd.DataFrame(columns=COLS)
    from .grade import one_per_contract
    bets = bets[bets["league"].isin(["NFL", "CFB"])]
    bets = one_per_contract(bets)
    s = snaps.copy() if snaps is not None else pd.DataFrame()
    if not s.empty:
        s["_ts"] = pd.to_datetime(s["snapshot_ts"], utc=True, errors="coerce")
        s = s[s["_ts"].notna() & (s["_ts"] <= now)]
        s["_ko"] = kickoffs.snap_kickoff(s, trades.OCC_OFFSET)   # #65
    by_t = {t: g for t, g in s.groupby("ticker")} if not s.empty else {}
    fast_t = {t: g for t, g in fast.groupby("ticker")} if fast is not None and not fast.empty else {}
    closes = {}
    if clv_bets is not None and not clv_bets.empty:
        for r in clv_bets.to_dict("records"):
            if r.get("close_status") in (None, "ok") or (isinstance(r.get("close_status"), float)):
                closes[(r.get("source"), r.get("leg1"))] = (trades._f(r.get("close_mid")), trades._f(r.get("clv_c")))
    rows = []
    for b in bets.to_dict("records"):
        if pd.isna(b.get("snapshot_ts")) or pd.isna(b.get("price")):
            continue
        ticker = str(b["leg1"]).split(" ", 1)[1]
        g = by_t.get(ticker)
        ko = g["_ko"].dropna().iloc[-1] if g is not None and not g.empty and g["_ko"].notna().any() else None
        cm, tc = closes.get((b.get("source"), b["leg1"]), (None, None))
        rows += grade_play(b, g, candles_fn(ticker), fast_t.get(ticker), ko, cm, tc, now)
    return pd.DataFrame(rows, columns=COLS)


def _block(g: pd.DataFrame) -> dict:
    """Stats for one (league group, variant)."""
    decided = g[g["status"].isin(["filled", "unfilled"])]
    filled = g[g["status"] == "filled"]
    unf = g[g["status"] == "unfilled"]
    out = {"plays": int(len(g)), "decided": int(len(decided)), "filled": int(len(filled)),
           "fill_rate_pct": round(100 * len(filled) / len(decided), 1) if len(decided) else None}
    c = filled[pd.to_numeric(filled["clv_c"], errors="coerce").notna()]
    v = pd.to_numeric(c["clv_c"], errors="coerce")
    out["clv"] = {"n": int(len(c)), "games": int(c["leg1"].map(_event).nunique()) if len(c) else 0,
                  "avg_c": round(float(v.mean()), 2) if len(c) else None,
                  "t_by_game": _t(v.groupby(c["leg1"].map(_event)).mean()) if len(c) else None}

    def taker(df):
        x = pd.to_numeric(df["taker_clv_c"], errors="coerce").dropna()
        return {"n": int(len(x)), "avg_c": round(float(x.mean()), 2) if len(x) else None}
    out["adverse_selection"] = {"taker_clv_if_filled": taker(filled), "taker_clv_if_unfilled": taker(unf)}
    final = filled
    for e in EXITS:
        final = final[final[f"{e}_status"] == "closed"]
    ex = {}
    for e in EXITS:
        p = pd.to_numeric(final[f"{e}_pnl_usd"], errors="coerce")
        ex[e] = {"closed": int(len(final)), "games": int(final["leg1"].map(_event).nunique()) if len(final) else 0,
                 "pnl_usd": round(float(p.sum()), 2) if len(final) else 0.0,
                 "t_by_game": _t(p.groupby(final["leg1"].map(_event)).sum()) if len(final) else None}
    out["exits"] = ex
    return out


def summary(m: pd.DataFrame | None) -> dict:
    out = {"note": "report only; resting buys fill at the limit when a later ask reaches it, maker fee"}
    if m is None or m.empty:
        return out
    for grp, g0 in (("football", m), ("NFL", m[m["league"] == "NFL"]), ("CFB", m[m["league"] == "CFB"])):
        if g0.empty:
            continue
        out[grp] = {v: _block(g0[g0["variant"] == v]) for v in VARIANTS if (g0["variant"] == v).any()}
    return out


def digest_line(s: dict | None) -> list[str]:
    parts = []
    for grp in ("NFL", "CFB"):
        b = ((s or {}).get(grp) or {}).get("MAKER_BID")
        if not b or not b.get("decided"):
            continue
        c, ex = b["clv"], b["exits"]
        cell = f"{grp}: {b['filled']}/{b['decided']} filled"
        if c.get("n"):
            cell += f", vs close {c['avg_c']:+.1f}c"
        if ex["HOLD"]["closed"]:
            cell += (f", hold ${ex['HOLD']['pnl_usd']:+.2f}, auto-sell +10c ${ex['RESTING_10']['pnl_usd']:+.2f}"
                     f" ({ex['HOLD']['closed']} done)")
        parts.append(cell)
    return ["  limit-order entries (paper): " + " · ".join(parts)] if parts else []


def run(now=None) -> dict:
    from . import clv
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    frames = []
    for src, p in clv.BET_FILES.items():
        try:
            b = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            continue
        if len(b):
            b["source"] = src
            frames.append(b)
    if not frames:
        pd.DataFrame(columns=COLS).to_csv(OUT, index=False)
        return summary(None)
    bets = pd.concat(frames, ignore_index=True)
    bets = bets[pd.to_datetime(bets["snapshot_ts"], utc=True, errors="coerce").notna()
                & bets["league"].isin(["NFL", "CFB"])]
    if bets.empty:
        return summary(None)
    tickers = set(bets["leg1"].astype(str).str.split(" ", n=1).str[1])
    start = pd.to_datetime(bets["snapshot_ts"], utc=True).min() - FLAG_LOOKBACK
    snaps = clv.load_snapshots(tickers, start, now)
    fast = trades.load_fast(tickers, start, now)
    try:
        cb = pd.read_csv(DATA / "clv_bets.csv")
    except Exception:  # noqa: BLE001
        cb = None
    m = grade_all(bets, snaps, cb, now, fast=fast)
    m.to_csv(OUT, index=False)
    s = summary(m)
    for grp in ("NFL", "CFB"):
        b = (s.get(grp) or {}).get("MAKER_BID")
        if b:
            print(f"maker {grp}: {b['filled']}/{b['decided']} filled, clv {b['clv']}, hold "
                  f"{b['exits']['HOLD']}")
    return s


if __name__ == "__main__":
    run()
