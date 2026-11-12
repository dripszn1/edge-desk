"""Trade-out grading (#38): every SHARP shadow / tracked entry graded several ways, side by side.

Emerson (11-11): "we never ride it till the end, we sell when we reach a certain point", then:
ride by default, sell on a goal; he can't react to in-game swings by hand, but Kalshi's
"Auto Sell" (a resting limit sell placed right after the buy) fills on its own.

Variants (same entry: first sighting, same contracts, entry fee as graded):
  HOLD            settle at $1 / $0                                             MANUAL-OK
  EXIT_AT_TIP     sell at the side's BID in the last snapshot before kickoff    MANUAL-OK
  TRADE           sell when the bid first reaches fair BEFORE kickoff, at fair,
                  else EXIT_AT_TIP                                              MANUAL-OK
  RESTING_<t>     Auto Sell at entry + t (t = 5/10/15¢) or at fair ("RESTING_FAIR"),
                  fills at exactly the target the first time the bid's high
                  reaches it, pregame OR in-game; else HOLD                      MANUAL-OK
  BRACKET_<t>_<s> as RESTING_<t>, plus a stop at entry - s (5/10¢): sells at the
                  bid when the bid's low first reaches it (if goal and stop are
                  hit in the same interval, the stop is assumed first)           API-ONLY
  LINE-STOP is not graded yet: it needs the sharp line's history per contract, which we only
  keep for alerted contracts (#35). Listed as "not_available" in the summary.

Price path per contract = saved snapshots (15 min in live mode, 30 min otherwise) + the
fast-lane quote log (#39, ~3 s for alerted contracts) + Kalshi 1-minute candles (#39, bid
high/low per minute) when cached. Only points strictly after the entry and at/before `now` are
used (no look-ahead). Snapshot-only paths miss brief spikes, so RESTING fills are a lower bound.

Fees: entry fee as graded (taker). Exits that cross the book (EXIT_AT_TIP, TRADE, stops) pay the
taker fee; resting sells (RESTING, BRACKET goals) pay Kalshi's maker fee (fees.MAKER_RATE).
Output: data/sharp_trades.csv (one row per bet x variant) and summary() -> weekly_report["trades"].
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import fees

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "sharp_trades.csv"
CANDLE_DIR = DATA / "candles"
FAST_DIR = DATA / "fastlane"
OCC_OFFSET = pd.Timedelta(hours=3)
TP_GRID = (5, 10, 15)
SL_GRID = (5, 10)
GATE_MIN = 30
GATE_T = 2.0
LABELS = {"HOLD": "MANUAL-OK", "EXIT_AT_TIP": "MANUAL-OK", "TRADE": "MANUAL-OK"}
COLS = ["bet_id", "source", "league", "market", "leg1", "variant", "label", "entry_ts", "price",
        "contracts", "status", "exit_ts", "exit_price", "exit_kind", "pnl_c", "pnl_usd", "path_points",
        "used_candles"]


def variants() -> list[str]:
    v = ["HOLD", "EXIT_AT_TIP", "TRADE"] + [f"RESTING_{t}" for t in TP_GRID] + ["RESTING_FAIR"]
    v += [f"BRACKET_{t}_{s}" for t in TP_GRID for s in SL_GRID]
    return v


def label(variant: str) -> str:
    if variant.startswith("RESTING"):
        return "MANUAL-OK"
    if variant.startswith("BRACKET"):
        return "API-ONLY"
    return LABELS.get(variant, "API-ONLY")


def maker_fee(price: float, contracts: float = 1, multiplier: float = 1.0) -> float:
    if contracts <= 0 or not 0 < price < 1:
        return 0.0
    return fees._round_up_cent(fees.MAKER_RATE * multiplier * contracts * price * (1 - price))


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def side_bid(side: str, yes_bid, yes_ask):
    """Best price we could SELL our side at: YES -> yes bid, NO -> 1 - yes ask."""
    yb, ya = _f(yes_bid), _f(yes_ask)
    if str(side).upper() == "YES":
        return yb if yb is not None and 0 < yb < 1 else None
    return round(1 - ya, 4) if ya is not None and 0 < ya < 1 else None


def build_path(side: str, snaps: pd.DataFrame | None, candles: pd.DataFrame | None,
               fast: pd.DataFrame | None, entry, now) -> list[tuple]:
    """[(ts, bid_hi, bid_lo, src)] strictly after entry, at/before now, oldest first.
    snaps/fast: snapshot_ts, yes_bid, yes_ask. candles: end_ts, yes_bid_hi, yes_bid_lo,
    yes_ask_hi, yes_ask_lo (0-1)."""
    entry, now = pd.Timestamp(entry), pd.Timestamp(now)
    pts = []
    for df, src in ((snaps, "snap"), (fast, "fast")):
        if df is None or df.empty:
            continue
        for r in df.to_dict("records"):
            ts = pd.Timestamp(r["snapshot_ts"])
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
            if not (entry < ts <= now):
                continue
            b = side_bid(side, r.get("yes_bid"), r.get("yes_ask"))
            if b is not None:
                pts.append((ts, b, b, src))
    if candles is not None and not candles.empty:
        for r in candles.to_dict("records"):
            ts = pd.Timestamp(r["end_ts"])
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
            # a minute counts only once it has fully closed after the entry and before now
            if not (entry + pd.Timedelta(minutes=1) <= ts <= now):
                continue
            if str(side).upper() == "YES":
                hi, lo = _f(r.get("yes_bid_hi")), _f(r.get("yes_bid_lo"))
            else:
                ah, al = _f(r.get("yes_ask_hi")), _f(r.get("yes_ask_lo"))
                hi = None if al is None else round(1 - al, 4)
                lo = None if ah is None else round(1 - ah, 4)
            if hi is not None and lo is not None and 0 < lo <= hi < 1:
                pts.append((ts, hi, lo, "candle"))
    pts.sort(key=lambda p: p[0])
    return pts


def grade_one(b: dict, path: list[tuple], kickoff, settle: float | None, now) -> list[dict]:
    """All variants for one bet. settle = 1.0 / 0.0 per contract once known, else None."""
    side, ticker = str(b["leg1"]).split(" ", 1)
    entry = pd.Timestamp(b["snapshot_ts"])
    entry = entry.tz_localize("UTC") if entry.tzinfo is None else entry
    price = float(b["price"])
    n = max(int(float(b.get("contracts") or 0)), 1)
    entry_fee = float(b.get("fees") or 0) or fees.taker_fee(price, n)
    fair = _f(b.get("fair"))
    ko = pd.Timestamp(kickoff) if kickoff is not None else None
    now = pd.Timestamp(now)
    pre = [p for p in path if ko is None or p[0] < ko]
    base = {"bet_id": b.get("bet_id"), "source": b.get("source"), "league": b.get("league"),
            "market": b.get("market"), "leg1": b["leg1"], "entry_ts": entry.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price": price, "contracts": n, "path_points": len(path),
            "used_candles": any(p[3] == "candle" for p in path)}

    def closed(variant, exit_ts, exit_price, kind, exit_fee):
        pnl = n * (exit_price - price) - entry_fee - exit_fee
        return {**base, "variant": variant, "label": label(variant), "status": "closed",
                "exit_ts": exit_ts.strftime("%Y-%m-%dT%H:%M:%SZ") if exit_ts is not None else None,
                "exit_price": round(exit_price, 4), "exit_kind": kind,
                "pnl_usd": round(pnl, 4), "pnl_c": round(100 * pnl / n, 4)}

    def pending(variant, status="pending"):
        return {**base, "variant": variant, "label": label(variant), "status": status, "exit_ts": None,
                "exit_price": None, "exit_kind": None, "pnl_usd": None, "pnl_c": None}

    def hold(variant):
        return closed(variant, None, settle, "settle", 0.0) if settle is not None else pending(variant)

    def at_tip(variant):
        if ko is None or now < ko:
            return pending(variant)
        if not pre:
            return pending(variant, "no_exit")
        ts, _, lo, _ = pre[-1]
        return closed(variant, ts, lo, "tip", fees.taker_fee(lo, n))

    out = [hold("HOLD"), at_tip("EXIT_AT_TIP")]
    # TRADE: pregame take-profit at fair (sold by hand at the bid -> taker), else at tip
    tr = None
    if fair is not None and fair > price:
        for ts, hi, lo, src in pre:
            if hi >= fair - 1e-9:
                tr = closed("TRADE", ts, fair, "take_profit", fees.taker_fee(fair, n))
                break
    out.append(tr or {**at_tip("TRADE"), "variant": "TRADE", "label": label("TRADE")})
    # resting sells (Auto Sell) and brackets: pregame and in-game
    targets = [(f"RESTING_{t}", round(price + t / 100, 4)) for t in TP_GRID]
    if fair is not None and fair > price:
        targets.append(("RESTING_FAIR", math.floor(fair * 100 + 1e-9) / 100))
    for name, tgt in targets:
        hit = next(((ts, src) for ts, hi, lo, src in path if tgt < 1 and hi >= tgt - 1e-9), None)
        out.append(closed(name, hit[0], tgt, "auto_sell", maker_fee(tgt, n)) if hit else hold(name))
    for t in TP_GRID:
        tgt = round(price + t / 100, 4)
        for s in SL_GRID:
            stop = round(price - s / 100, 4)
            name, res = f"BRACKET_{t}_{s}", None
            for ts, hi, lo, src in path:
                if stop > 0 and lo <= stop + 1e-9:          # stop first when both fall in one interval
                    px = stop if src == "candle" else lo
                    res = closed(name, ts, max(px, 0.01), "stop", fees.taker_fee(max(px, 0.01), n))
                    break
                if tgt < 1 and hi >= tgt - 1e-9:
                    res = closed(name, ts, tgt, "auto_sell", maker_fee(tgt, n))
                    break
            out.append(res or hold(name))
    return out


def _t(x) -> float | None:
    x = pd.Series(x, dtype="float64").dropna()
    if len(x) < 2 or x.std(ddof=1) == 0:
        return None
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2)


def _event(leg1: str) -> str:
    parts = str(leg1).split(" ", 1)[-1].split("-")
    return parts[1] if len(parts) > 1 else parts[0]


def summary(t: pd.DataFrame) -> dict:
    """{football|cbb: {variant: {label, closed, pending, pnl_usd, avg_c, win_pct, t_by_game, gate}},
    plus best manual variant} — CBB never pooled with football. Stats use only bets that are final
    under every variant (same bets, same denominator)."""
    out = {"not_available": ["LINE_STOP (needs per-contract sharp-line history)"]}
    if t is None or t.empty:
        return out
    t = t.copy()
    t["_grp"] = t["league"].map(lambda x: "cbb" if x == "CBB" else "football")
    # Same denominator for every variant: only bets where ALL variants are final. Otherwise an
    # auto-sell that filled early (a win by construction) is compared with holds still pending,
    # which flatters RESTING badly (11-11 data: CBB RESTING_10 +$50 on 61 early fills vs 2 settled holds).
    done = t.groupby("bet_id")["status"].apply(lambda x: bool((x == "closed").all()))
    t["_final"] = t["bet_id"].map(done)
    for grp, g0 in t.groupby("_grp"):
        d = {}
        g = g0[g0["_final"]]
        early = g0[(~g0["_final"]) & (g0["status"] == "closed")].groupby("variant").size().to_dict()
        for v, gv in g0.groupby("variant"):
            gv = g[g["variant"] == v]
            c = gv[gv["status"] == "closed"]
            pnl = pd.to_numeric(c["pnl_usd"], errors="coerce")
            tg = _t(pnl.groupby(c["leg1"].map(_event)).sum()) if len(c) else None
            d[v] = {"label": label(v), "closed": int(len(c)),
                    "pending": int((~g0[g0["variant"] == v]["_final"]).sum()),
                    "closed_early_not_counted": int(early.get(v, 0)),
                    "pnl_usd": round(float(pnl.sum()), 2) if len(c) else 0.0,
                    "avg_c": round(float(pd.to_numeric(c["pnl_c"]).mean()), 2) if len(c) else None,
                    "win_pct": round(100 * float((pnl > 0).mean()), 1) if len(c) else None,
                    "games": int(c["leg1"].map(_event).nunique()) if len(c) else 0,
                    "t_by_game": tg,
                    "gate_open": bool(label(v) == "MANUAL-OK" and len(c) >= GATE_MIN
                                      and float(pnl.sum()) > 0 and (tg or 0) >= GATE_T)}
        manual = {k: x for k, x in d.items() if x["label"] == "MANUAL-OK" and x["closed"]}
        best = max(manual, key=lambda k: manual[k]["pnl_usd"]) if manual else None
        out[grp] = {"variants": d, "best_manual": best}
    return out


# ---------- I/O ----------
def load_candles(ticker: str) -> pd.DataFrame | None:
    p = CANDLE_DIR / f"{ticker}.csv"
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001 - not fetched (yet)
        return None


def load_fast(tickers: set[str], start, end) -> pd.DataFrame:
    parts = []
    if not FAST_DIR.exists():
        return pd.DataFrame(columns=["snapshot_ts", "ticker", "yes_bid", "yes_ask"])
    for f in sorted(FAST_DIR.glob("*.csv")):
        try:
            d = pd.Timestamp(f.stem).date()
        except Exception:  # noqa: BLE001
            continue
        if d < pd.Timestamp(start).date() or d > pd.Timestamp(end).date():
            continue
        try:
            df = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            continue
        parts.append(df[df["ticker"].isin(tickers)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["snapshot_ts", "ticker", "yes_bid", "yes_ask"])


def settle_value(b: dict) -> float | None:
    """$ per contract from the HOLD grader (payout / contracts) once settled."""
    if b.get("status") != "settled":
        return None
    n = _f(b.get("contracts")) or 0
    pay = _f(b.get("payout"))
    if n <= 0 or pay is None:
        return None
    return 1.0 if pay / n > 0.5 else 0.0


def grade_all(bets: pd.DataFrame, snaps: pd.DataFrame, now, candles_fn=load_candles,
              fast: pd.DataFrame | None = None) -> pd.DataFrame:
    now = pd.Timestamp(now)
    if bets is None or bets.empty:
        return pd.DataFrame(columns=COLS)
    s = snaps.copy() if snaps is not None else pd.DataFrame()
    if not s.empty:
        s["_ts"] = pd.to_datetime(s["snapshot_ts"], utc=True, errors="coerce")
        s = s[s["_ts"].notna() & (s["_ts"] <= now)]
        s["_ko"] = pd.to_datetime(s["occurrence_datetime"], utc=True, errors="coerce") - OCC_OFFSET
    by_t = {t: g for t, g in s.groupby("ticker")} if not s.empty else {}
    fast_t = {t: g for t, g in fast.groupby("ticker")} if fast is not None and not fast.empty else {}
    rows = []
    for b in bets.to_dict("records"):
        if pd.isna(b.get("snapshot_ts")) or pd.isna(b.get("price")):
            continue
        side, ticker = str(b["leg1"]).split(" ", 1)
        g = by_t.get(ticker)
        ko = None
        if g is not None and not g.empty and g["_ko"].notna().any():
            ko = g["_ko"].dropna().iloc[-1]
        entry = pd.Timestamp(b["snapshot_ts"])
        entry = entry.tz_localize("UTC") if entry.tzinfo is None else entry
        path = build_path(side, g, candles_fn(ticker), fast_t.get(ticker), entry, now)
        rows += grade_one(b, path, ko, settle_value(b), now)
    return pd.DataFrame(rows, columns=COLS)


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
    bets = bets[pd.to_datetime(bets["snapshot_ts"], utc=True, errors="coerce").notna()]
    tickers = set(bets["leg1"].astype(str).str.split(" ", n=1).str[1])
    start = pd.to_datetime(bets["snapshot_ts"], utc=True).min()
    snaps = clv.load_snapshots(tickers, start, now)
    fast = load_fast(tickers, start, now)
    t = grade_all(bets, snaps, now, fast=fast)
    t.to_csv(OUT, index=False)
    s = summary(t)
    for grp in ("football", "cbb"):
        if grp in s:
            v = s[grp]["variants"]
            print(f"trades {grp}: " + " | ".join(
                f"{k} {x['closed']} closed ${x['pnl_usd']:+.2f}" for k, x in v.items()
                if k in ("HOLD", "EXIT_AT_TIP", "TRADE", "RESTING_FAIR", "RESTING_10")) +
                  f" | best manual: {s[grp]['best_manual']}")
    return s


if __name__ == "__main__":
    run()
