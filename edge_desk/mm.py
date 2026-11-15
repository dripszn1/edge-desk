"""Market-making paper test on the Kalshi stream (backlog #67). REPORT ONLY, READ ONLY.

What pro Kalshi traders do instead of taking the ask: rest a buy on BOTH sides of a rung around the
sharp fair, earn the spread when both fill, and carry the inventory risk when only one does.
Replayed from what the live job's stream already records (#66):
  data/stream/quotes_<date>.csv  every best-bid/ask change for contracts we have a fair for
  data/stream/fairs_<date>.csv   every new YES-side sharp fair (logged when set, so no look-ahead)

Rules (one small fixed grid, k = 1 / 2 / 3 c, no tuning beyond it):
  - fair = the newest fair logged at or before the quote; usable only if not stale, its sharp pull is
    <= 90 min old, before kickoff, and 15c <= fair <= 85c (liquid, near-coin-flip rungs; far rungs
    are never logged); the book must be two-sided.
  - quotes: buy YES at floor(fair - k), buy NO at 1 - ceil(fair + k). A quote is placed on an update
    where the book does not already cross it (a resting maker order can't cross), and is re-priced
    when the fair changes.
  - fill: only on a LATER update that trades THROUGH our price (YES: ask < our bid; NO: yes bid > our
    offer). Touching it is not a fill (we may not be first in the queue), so fills are a lower bound
    in count and conservative in price. One fill per crossing; the book must clear before the next.
  - size: CONTRACTS per fill, net inventory per contract capped at +/- MAX_INVENTORY.
  - maker fee on every fill (fees.MAKER_RATE, rounded up to the cent). All quotes cancel at kickoff.
  - P&L: open inventory marked at the last mid before kickoff (always available), and settled
    P&L for fills whose result was fetched after kickoff (settle.py only fetches flagged/stream
    tickers, so this is partial). A YES + NO pair sums to $1, so pairs lock the spread.
  - adverse selection: our side's mid MID_AFTER_MS after each fill minus our price (c); negative =
    the fills are the ones the market then moves against.
  - t by game: per-game P&L sums.
Outputs: data/mm_fills.csv, weekly_report.json["market_making"], digest_line().
Never an order: nothing here talks to Kalshi at all.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import fees

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STREAM_DIR = DATA / "stream"
OUT = DATA / "mm_fills.csv"
GRID_C = (1, 2, 3)
CONTRACTS = 10
MAX_INVENTORY = 50
MAX_PULL_AGE_MS = 90 * 60_000
FAIR_LO, FAIR_HI = 0.15, 0.85
MID_AFTER_MS = 5 * 60_000
DAYS = 8
LEAGUE = {"KXNFL": "NFL", "KXNCAAF": "CFB"}
EPS = 1e-9


def _ms(s) -> int | None:
    if s is None or (isinstance(s, float) and math.isnan(s)) or s == "":
        return None
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    d = d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1000)


def league_of(t: str):
    for k, v in LEAGUE.items():
        if str(t).startswith(k):
            return v
    return None


def event_of(t: str) -> str:
    return str(t).rsplit("-", 1)[0]


def _fee(n: int, p: float) -> float:
    return fees._round_up_cent(fees.MAKER_RATE * n * p * (1 - p))


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def simulate(quotes: pd.DataFrame, fairs: pd.DataFrame, k_c: int = 2, contracts: int = CONTRACTS,
             max_inventory: int = MAX_INVENTORY) -> pd.DataFrame:
    cols = ["k_c", "ticker", "event", "league", "side", "price", "contracts", "fee", "fill_ms", "fair",
            "kickoff_utc"]
    if quotes is None or quotes.empty or fairs is None or fairs.empty:
        return pd.DataFrame(columns=cols)
    fr = {}
    for r in fairs.sort_values("logged_ms", kind="stable").itertuples(index=False):
        fr.setdefault(str(r.ticker), []).append(r)
    q = quotes.sort_values("recv_ms", kind="stable")
    out = []
    state: dict[str, dict] = {}
    k = k_c / 100
    for r in q.itertuples(index=False):
        t = str(r.ticker)
        if t not in fr or league_of(t) is None:
            continue
        now = int(r.recv_ms)
        fl = fr[t]
        f = None
        for x in fl:                                    # newest fair logged at or before this quote
            if int(x.logged_ms) <= now:
                f = x
            else:
                break
        st = state.setdefault(t, {"yes": 0, "no": 0, "q": {}, "fair_ms": None})
        if f is None:
            continue
        ko = _ms(f.kickoff_utc)
        pulled = _ms(f.odds_fetched_ts)
        yb, ya = r.yes_bid, r.yes_ask
        fair = float(f.fair)
        ok = (not _truthy(f.stale) and pulled is not None and now - pulled <= MAX_PULL_AGE_MS
              and (ko is None or now < ko) and FAIR_LO - EPS <= fair <= FAIR_HI + EPS
              and pd.notna(yb) and pd.notna(ya) and 0 < yb < ya < 1)
        if not ok:
            st["q"] = {}                                # cancel everything
            continue
        bid_y = math.floor(round((fair - k) * 100, 6)) / 100
        off_y = math.ceil(round((fair + k) * 100, 6)) / 100
        want = {"YES": round(bid_y, 2), "NO": round(1 - off_y, 2)}
        for side in ("YES", "NO"):
            through = (ya < want["YES"] - EPS) if side == "YES" else (yb > off_y + EPS)
            cur = st["q"].get(side)
            if cur is None or abs(cur["price"] - want[side]) > EPS:     # (re)place on a clear book only
                st["q"][side] = {"price": want[side], "placed_ms": now, "armed": not through}
                continue
            if not cur["armed"]:
                if not through:
                    cur["armed"] = True
                continue
            if not through or now <= cur["placed_ms"]:
                continue
            net = st["yes"] - st["no"]
            if (side == "YES" and net + contracts > max_inventory) or (side == "NO" and -net + contracts > max_inventory):
                continue
            p = cur["price"]
            st["yes" if side == "YES" else "no"] += contracts
            cur["armed"] = False                        # one fill per crossing
            out.append({"k_c": k_c, "ticker": t, "event": event_of(t), "league": league_of(t), "side": side,
                        "price": p, "contracts": contracts, "fee": _fee(contracts, p), "fill_ms": now,
                        "fair": fair, "kickoff_utc": f.kickoff_utc})
    return pd.DataFrame(out, columns=cols)


def _mids(quotes: pd.DataFrame) -> dict:
    m = {}
    if quotes is None or quotes.empty:
        return m
    q = quotes.sort_values("recv_ms", kind="stable")
    for r in q.itertuples(index=False):
        if pd.notna(r.yes_bid) and pd.notna(r.yes_ask) and 0 < r.yes_bid < r.yes_ask < 1:
            m.setdefault(str(r.ticker), []).append((int(r.recv_ms), (r.yes_bid + r.yes_ask) / 2))
    return m


def _side_mid(side: str, ymid: float) -> float:
    return ymid if side == "YES" else 1 - ymid


def adverse(fills: pd.DataFrame, quotes: pd.DataFrame, after_ms: int = MID_AFTER_MS, mids=None) -> pd.Series:
    """Our side's mid `after_ms` after each fill (first quote at/after that, before kickoff, else the
    last one before kickoff) minus our price, in cents."""
    mids = mids if mids is not None else _mids(quotes)
    vals = []
    for r in fills.itertuples(index=False):
        ko = _ms(r.kickoff_utc)
        ms = [x for x in mids.get(r.ticker, []) if x[0] > r.fill_ms and (ko is None or x[0] < ko)]
        if not ms:
            vals.append(None)
            continue
        later = [x for x in ms if x[0] >= r.fill_ms + after_ms]
        y = (later[0] if later else ms[-1])[1]
        vals.append(round(100 * (_side_mid(r.side, y) - r.price), 2))
    return pd.Series(vals, index=fills.index, dtype="float64")


def _mark(r, mids) -> float | None:
    ko = _ms(r.kickoff_utc)
    ms = [x for x in mids.get(r.ticker, []) if ko is None or x[0] < ko]
    return _side_mid(r.side, ms[-1][1]) if ms else None


def _settled_payout(r, results: pd.DataFrame | None):
    if results is None or results.empty:
        return None
    ko = _ms(r.kickoff_utc)
    x = results[(results["ticker"] == r.ticker) & results["status"].isin(["finalized", "settled", "determined"])
                & results["result"].isin(["yes", "no"])]
    for y in x.itertuples(index=False):
        t = _ms(y.fetched_ts)
        if t is not None and (ko is None or t > ko):
            return 1.0 if (y.result == "yes") == (r.side == "YES") else 0.0
    return None


def _t(xs: list[float]):
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return round(m / (sd / math.sqrt(n)), 2) if sd > 0 else None


def _exposure(g: pd.DataFrame) -> float:
    """Max over time of the cost of unpaired inventory in one game."""
    pos, worst = {}, 0.0
    for r in g.sort_values("fill_ms").itertuples(index=False):
        p = pos.setdefault(r.ticker, {"YES": [0, 0.0], "NO": [0, 0.0]})
        p[r.side][0] += r.contracts
        p[r.side][1] += r.contracts * r.price
        tot = 0.0
        for s in pos.values():
            big = "YES" if s["YES"][0] >= s["NO"][0] else "NO"
            n = s[big][0] - s["NO" if big == "YES" else "YES"][0]
            if n > 0 and s[big][0]:
                tot += n * s[big][1] / s[big][0]
        worst = max(worst, tot)
    return round(worst, 2)


def summarize(fills: pd.DataFrame, quotes: pd.DataFrame, results: pd.DataFrame | None) -> dict:
    out = {"note": ("paper market making around the sharp fair; fills only on a later trade-through, "
                    "maker fees, cancel at kickoff; report only"),
           "grid_c": list(GRID_C), "contracts": CONTRACTS, "max_inventory": MAX_INVENTORY, "by_k": {}}
    if fills is None or fills.empty:
        return out
    mids = _mids(quotes)
    f = fills.copy()
    f["adverse_c"] = adverse(f, quotes, mids=mids)
    f["mark"] = [_mark(r, mids) for r in f.itertuples(index=False)]
    f["payout"] = [_settled_payout(r, results) for r in f.itertuples(index=False)]
    f["pnl_mark"] = [None if m is None else round((m - p) * n - fee, 4)
                     for m, p, n, fee in zip(f["mark"], f["price"], f["contracts"], f["fee"])]
    f["pnl_settled"] = [None if v is None or (isinstance(v, float) and math.isnan(v)) else round((v - p) * n - fee, 4)
                        for v, p, n, fee in zip(f["payout"], f["price"], f["contracts"], f["fee"])]
    for k, gk in f.groupby("k_c"):
        blk = {}
        for lg, g in gk.groupby("league"):
            pm = pd.to_numeric(g["pnl_mark"], errors="coerce")
            ps = pd.to_numeric(g["pnl_settled"], errors="coerce")
            games = g.assign(pm=pm.fillna(0)).groupby("event")["pm"].sum()
            paired = sum(min(int(x[x["side"] == "YES"]["contracts"].sum()), int(x[x["side"] == "NO"]["contracts"].sum()))
                         for _, x in g.groupby("ticker"))
            adv = pd.to_numeric(g["adverse_c"], errors="coerce").dropna()
            blk[lg] = {"fills": int(len(g)), "contracts": int(g["contracts"].sum()), "paired_contracts": paired,
                       "tickers": int(g["ticker"].nunique()), "games": int(g["event"].nunique()),
                       "fees_usd": round(float(g["fee"].sum()), 2),
                       "pnl_usd": round(float(pm.sum()), 2), "t_by_game": _t(list(games)),
                       "settled_fills": int(ps.notna().sum()),
                       "settled_pnl_usd": round(float(ps.sum()), 2) if ps.notna().any() else None,
                       "adverse_c": round(float(adv.mean()), 2) if len(adv) else None,
                       "max_exposure_usd": max((_exposure(x) for _, x in g.groupby("event")), default=0.0)}
        out["by_k"][str(int(k))] = blk
    return out


def digest_line(s: dict | None) -> list[str]:
    cells = []
    for k, blk in ((s or {}).get("by_k") or {}).items():
        for lg, b in blk.items():
            if b.get("fills"):
                t = b.get("t_by_game")
                cells.append(f"{lg} ±{k}c {b['fills']} fills/{b['games']} games ${b['pnl_usd']:+.2f}"
                             + (f" t {t:+.1f}" if t is not None else "")
                             + (f", after-fill {b['adverse_c']:+.1f}c" if b.get("adverse_c") is not None else ""))
    return ["  market-making paper: " + " · ".join(cells)] if cells else []


def _load(pattern: str, days: int, now) -> pd.DataFrame:
    parts = []
    if not STREAM_DIR.exists():
        return pd.DataFrame()
    cut = (pd.Timestamp(now) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    for p in sorted(STREAM_DIR.glob(pattern)):
        day = p.stem.split("_", 1)[-1]
        if day < cut:
            continue
        try:
            parts.append(pd.read_csv(p))
        except Exception:  # noqa: BLE001
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def run(now=None) -> dict:
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    now_ms = int(now.timestamp() * 1000)
    quotes = _load("quotes_*.csv", DAYS, now)
    fairs = _load("fairs_*.csv", DAYS, now)
    if quotes.empty or fairs.empty:
        return summarize(pd.DataFrame(), quotes, None) | {"status": "no fair log yet (starts with the next live run)"}
    quotes = quotes[pd.to_numeric(quotes["recv_ms"], errors="coerce") <= now_ms]          # never a future quote
    fairs = fairs[pd.to_numeric(fairs["logged_ms"], errors="coerce") <= now_ms]
    quotes = quotes.drop_duplicates(["recv_ms", "ticker", "yes_bid", "yes_ask"])
    fills = pd.concat([simulate(quotes, fairs, k) for k in GRID_C], ignore_index=True)
    fills.to_csv(OUT, index=False)
    try:
        results = pd.read_csv(DATA / "results" / "results.csv")
    except Exception:  # noqa: BLE001
        results = None
    s = summarize(fills, quotes, results)
    print("mm: " + " | ".join(digest_line(s)))
    return s


if __name__ == "__main__":
    run()
