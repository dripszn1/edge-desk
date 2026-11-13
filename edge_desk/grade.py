"""Grader: settle paper flags from Kalshi results, P&L after fees, daily ledger.

Rules
- One paper bet per distinct flag (kind + both legs); the first time it was seen is the entry.
- Fees are recomputed per leg at the actual paper size (Kalshi rounds each order up to the cent).
- A leg settles only from a result fetched AFTER the flag's snapshot (no look-ahead) whose
  market status is final. Anything else stays pending; pending > STALE_DAYS is marked stale.
- void leg: principal refunded, fee lost (conservative).
- Risk: once a CT day's realized losses (settled before the new flag) reach DAILY_LOSS_CAP,
  later flags that day are skipped. Once drawdown from peak reaches DRAWDOWN_STOP, every later
  flag is skipped until Emerson restarts by hand (SPEC.md).
Outputs: data/paper_bets.csv (one row per bet), data/paper_ledger.csv (one row per CT day).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import config
from .fees import taker_fee
from .settle import FINAL_RESULTS, FINAL_STATUSES

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ROOT / "data" / "flags" / "flags.csv"
RESULTS = ROOT / "data" / "results" / "results.csv"
BETS_OUT = ROOT / "data" / "paper_bets.csv"
LEDGER_OUT = ROOT / "data" / "paper_ledger.csv"
CT = ZoneInfo(config.TIMEZONE)
STALE_DAYS = 7


def _settlement(results: pd.DataFrame, ticker: str, after: pd.Timestamp):
    """(result, settled_ts) from the first final result fetched strictly after `after`."""
    r = results[(results["ticker"] == ticker) & (results["_ts"] > after)
                & results["status"].isin(FINAL_STATUSES) & results["result"].isin(FINAL_RESULTS)]
    if r.empty:
        return None, None
    r = r.sort_values("_ts").iloc[0]
    return r["result"], r["_ts"]


def _leg(spec: str, price: float, sets: int, mult: float, result: str | None) -> dict:
    side, ticker = spec.split(" ", 1)
    fee = taker_fee(price, sets, mult)
    cost = round(price * sets + fee, 4)
    if result is None:
        payout = None
    elif result == "void":
        payout = round(price * sets, 4)
    else:
        payout = float(sets) if result == side.lower() else 0.0
    return {"ticker": ticker, "side": side, "fee": fee, "cost": cost, "payout": payout}


def paper_bets(flags: pd.DataFrame, results: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    cols = ["bet_id", "snapshot_ts", "date_ct", "league", "kind", "event", "leg1", "leg2", "sets",
            "stake", "fees", "payout", "pnl", "status", "settled_ts", "kickoff_ct"]
    if flags.empty:
        return pd.DataFrame(columns=cols)
    results = results.copy()
    results["_ts"] = pd.to_datetime(results.get("fetched_ts"), utc=True)
    f = flags.copy()
    f["_ts"] = pd.to_datetime(f["snapshot_ts"], utc=True)
    f = f.sort_values("_ts").drop_duplicates(["kind", "leg1", "leg2"], keep="first")

    rows = []
    for r in f.to_dict("records"):
        sets = int(r.get("paper_sets") or 0)
        m = r.get("fee_multiplier")
        mult = float(m) if m is not None and pd.notna(m) else 1.0
        settles = [_settlement(results, leg.split(" ", 1)[1], r["_ts"]) for leg in (r["leg1"], r["leg2"])]
        legs = [_leg(r["leg1"], float(r["leg1_price"]), sets, mult, settles[0][0]),
                _leg(r["leg2"], float(r["leg2_price"]), sets, mult, settles[1][0])]
        stake = round(sum(l["cost"] for l in legs), 2)
        fees = round(sum(l["fee"] for l in legs), 2)
        done = all(l["payout"] is not None for l in legs)
        payout = round(sum(l["payout"] for l in legs), 2) if done else None
        if sets <= 0:
            status = "no_size"
        elif done:
            status = "settled"
        elif (now - r["_ts"]).days > STALE_DAYS:
            status = "stale"
        else:
            status = "pending"
        rows.append({
            "bet_id": f"{r['kind']}|{r['leg1']}|{r['leg2']}", "snapshot_ts": r["snapshot_ts"],
            "_ts": r["_ts"], "date_ct": r["_ts"].tz_convert(CT).date().isoformat(),
            "league": r.get("league"), "kind": r["kind"], "event": r.get("event"),
            "leg1": r["leg1"], "leg2": r["leg2"], "sets": sets, "stake": stake, "fees": fees,
            "payout": payout, "pnl": round(payout - stake, 2) if done else None, "status": status,
            "_settled": max(s[1] for s in settles) if done else None,
            "kickoff_ct": r.get("kickoff_ct"),
        })
    bets = pd.DataFrame(rows)
    bets = _apply_risk(bets)
    bets["settled_ts"] = bets["_settled"].map(lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ") if t is not None and pd.notna(t) else None)
    return bets[cols]


def _apply_risk(bets: pd.DataFrame) -> pd.DataFrame:
    """Walk bets in entry order; skip those the daily cap or drawdown stop would have blocked,
    using only results known (settled) before each bet's entry time."""
    bets = bets.sort_values("_ts").reset_index(drop=True)
    taken: list[int] = []
    paused = False
    for i, b in bets.iterrows():
        known = [j for j in taken if bets.at[j, "status"] == "settled" and bets.at[j, "_settled"] < b["_ts"]]
        if not paused and known:
            k = bets.loc[known].sort_values("_settled")
            equity = config.BANKROLL + k["pnl"].cumsum()
            peak = max(config.BANKROLL, equity.max())
            if peak - equity.iloc[-1] >= config.DRAWDOWN_STOP:
                paused = True
        day_loss = -bets.loc[[j for j in known if bets.at[j, "date_ct"] == b["date_ct"]], "pnl"].clip(upper=0).sum() \
            if known else 0.0
        if paused:
            bets.at[i, "status"] = "skipped_drawdown"
        elif day_loss >= config.DAILY_LOSS_CAP:
            bets.at[i, "status"] = "skipped_daily_cap"
        else:
            taken.append(i)
            continue
        bets.at[i, "pnl"] = 0.0
    return bets


def ledger(bets: pd.DataFrame) -> pd.DataFrame:
    cols = ["date_ct", "bets", "settled", "pending", "skipped", "staked", "fees", "pnl",
            "cum_pnl", "bankroll"]
    if bets.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for d, g in bets.groupby("date_ct"):
        live = g[~g["status"].str.startswith("skipped") & (g["status"] != "no_size")]
        s = live[live["status"] == "settled"]
        out.append({"date_ct": d, "bets": len(live), "settled": len(s),
                    "pending": int(live["status"].isin(["pending", "stale"]).sum()),
                    "skipped": int(g["status"].str.startswith("skipped").sum()),
                    "staked": round(live["stake"].sum(), 2), "fees": round(live["fees"].sum(), 2),
                    "pnl": round(s["pnl"].sum(), 2)})
    led = pd.DataFrame(out).sort_values("date_ct")
    led["cum_pnl"] = led["pnl"].cumsum().round(2)
    led["bankroll"] = (config.BANKROLL + led["cum_pnl"]).round(2)
    return led[cols]


MODEL_BET_COLS = ["bet_id", "snapshot_ts", "date_ct", "league", "kind", "market", "event", "leg1",
                  "contracts", "price", "fair", "claimed_edge", "stake", "fees", "payout", "pnl",
                  "status", "settled_ts", "kickoff_ct", "source"]


def model_bets(flags: pd.DataFrame, results: pd.DataFrame, now: datetime | None = None,
               source: str = "shadow", kind: str = "MODEL") -> pd.DataFrame:
    """Grade single-leg MODEL flags: first sighting of each (ticker, side) is the entry, fee
    recomputed at the paper size, settled only from a final result fetched after the entry."""
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if flags is None or flags.empty:
        return pd.DataFrame(columns=MODEL_BET_COLS)
    results = results.copy()
    results["_ts"] = pd.to_datetime(results.get("fetched_ts"), utc=True)
    f = flags.copy()
    f["_ts"] = pd.to_datetime(f["snapshot_ts"], utc=True)
    f = f.sort_values("_ts").drop_duplicates(["leg1"], keep="first")
    rows = []
    for r in f.to_dict("records"):
        n = int(r.get("contracts") or 0)
        m = r.get("fee_multiplier")
        mult = float(m) if m is not None and pd.notna(m) else 1.0
        res, settled = _settlement(results, r["leg1"].split(" ", 1)[1], r["_ts"])
        leg = _leg(r["leg1"], float(r["price"]), n, mult, res)
        done = leg["payout"] is not None
        status = "no_size" if n <= 0 else "settled" if done else \
            "stale" if (now - r["_ts"]).days > STALE_DAYS else "pending"
        rows.append({"bet_id": f"{kind}|{r['leg1']}", "snapshot_ts": r["snapshot_ts"],
                     "date_ct": r["_ts"].tz_convert(CT).date().isoformat(), "league": r.get("league"),
                     "kind": kind, "market": r.get("market"), "event": r.get("event"), "leg1": r["leg1"],
                     "contracts": n, "price": float(r["price"]), "fair": r.get("fair"),
                     "claimed_edge": r.get("edge"), "stake": round(leg["cost"], 2), "fees": round(leg["fee"], 2),
                     "payout": leg["payout"], "pnl": round(leg["payout"] - leg["cost"], 2) if done else None,
                     "status": status,
                     "settled_ts": settled.strftime("%Y-%m-%dT%H:%M:%SZ") if done else None,
                     "kickoff_ct": r.get("kickoff_ct"), "source": source})
    return pd.DataFrame(rows, columns=MODEL_BET_COLS)


MODEL_SHADOW = ROOT / "data" / "flags" / "model_shadow.csv"
MODEL_BETS_OUT = ROOT / "data" / "model_shadow_bets.csv"
SHARP_SHADOW = ROOT / "data" / "flags" / "sharp_shadow.csv"
SHARP_BETS_OUT = ROOT / "data" / "sharp_shadow_bets.csv"


TRACKED = ROOT / "data" / "flags" / "sharp_tracked.csv"
TRACKED_BETS_OUT = ROOT / "data" / "sharp_tracked_bets.csv"
REASON_BUCKETS = (("far from sharp line", "far_rung"), ("line stale", "stale_line"), ("sanity cap", "sanity_cap"),
                  ("kalshi moved first", "kalshi_first"))


def reason_bucket(reason) -> str:
    """'rung far ... ; sharp line stale' -> 'far_rung+stale_line' (stable order)."""
    r = str(reason or "")
    b = [tag for needle, tag in REASON_BUCKETS if needle in r]
    return "+".join(b) or "other"


def tracked_bets(tracked: pd.DataFrame, results: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Grade the low-confidence SHARP plays (far rungs, stale lines, >cap edges) exactly like
    SHARP shadow bets: first sighting of each (side, ticker) is the entry, sized by the same
    quarter-Kelly/$10 rules (at least 1 contract so every play is graded), fees at that size,
    settled only from a final result fetched after the entry. Rows stamped in the future are
    ignored (never look-ahead)."""
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    cols = MODEL_BET_COLS + ["reason", "reason_bucket"]
    if tracked is None or tracked.empty:
        return pd.DataFrame(columns=cols)
    t = tracked.copy()
    t["_ts"] = pd.to_datetime(t["snapshot_ts"], utc=True, errors="coerce")
    t = t[t["_ts"].notna() & (t["_ts"] <= now_ts)]
    t = t[pd.to_numeric(t["price"], errors="coerce").between(0, 1, inclusive="neither")
          & pd.to_numeric(t["fair"], errors="coerce").notna()]
    if t.empty:
        return pd.DataFrame(columns=cols)
    t["leg1"] = t["side"].astype(str) + " " + t["ticker"].astype(str)
    t = t.sort_values("_ts").drop_duplicates(["leg1"], keep="first")
    from .sizing import bet_size
    n = []
    for r in t.to_dict("records"):
        price, fair = float(r["price"]), float(r["fair"])
        fee1 = taker_fee(price, 1)
        stake = bet_size(fair, price, fee1)
        n.append(max(1, int(stake // (price + fee1)) if price + fee1 > 0 else 1))
    t["contracts"] = n
    t["market"] = t["ticker"].astype(str).str.split("-").str[0].str.replace(r"^KX(NFL|NCAAF|NCAAMB)", "", regex=True)
    out = model_bets(t.drop(columns=["_ts"]), results, now, source="tracked", kind="SHARP")
    rmap = dict(zip(t["leg1"], t["reason"]))
    out["reason"] = out["leg1"].map(rmap)
    out["reason_bucket"] = out["reason"].map(reason_bucket)
    return out[cols]


ODDS_LINES = ROOT / "data" / "odds" / "lines.csv"


def recorded_pulls(path=ODDS_LINES) -> set | None:
    """Every sharp pull time we actually recorded (fetched_ts in data/odds/lines.csv); None if
    the file is missing, in which case nothing can be checked and nothing is filtered."""
    try:
        s = pd.read_csv(path, usecols=["fetched_ts"])["fetched_ts"].dropna().astype(str)
    except (FileNotFoundError, ValueError):
        return None
    return set(s) if len(s) else None


def drop_unrecorded(flags: pd.DataFrame, pulls: set | None) -> pd.DataFrame:
    """#47 provenance: skip a SHARP play priced off a sharp pull that is not in our odds history
    (test-fixture rows leaked into data/ before #34). Blank pulls (MODEL rows) and pulls older
    than the first recorded one (history that may have been trimmed) are kept. Rows are only
    skipped for grading; the log itself is never changed."""
    if pulls is None or flags is None or flags.empty or "odds_fetched_ts" not in flags:
        return flags
    first = min(pulls)
    v = flags["odds_fetched_ts"].fillna("").astype(str).str.strip()
    bad = (v != "") & (v >= first) & ~v.isin(pulls)
    if bad.any():
        print(f"provenance: {int(bad.sum())} flag rows priced off an unrecorded sharp pull skipped "
              f"({', '.join(sorted(set(flags.loc[bad, 'ticker'].astype(str))) if 'ticker' in flags else '')})")
    return flags[~bad]


def run() -> pd.DataFrame:
    flags = pd.read_csv(FLAGS) if FLAGS.exists() else pd.DataFrame()
    pulls = recorded_pulls()
    results = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame(
        columns=["fetched_ts", "ticker", "status", "result"])
    bets = paper_bets(flags, results)
    shadow = pd.read_csv(MODEL_SHADOW) if MODEL_SHADOW.exists() else pd.DataFrame()
    mb = model_bets(shadow, results)
    mb.to_csv(MODEL_BETS_OUT, index=False)   # graded on its own; never touches the paper ledger
    sh = pd.read_csv(SHARP_SHADOW) if SHARP_SHADOW.exists() else pd.DataFrame()
    sb = model_bets(drop_unrecorded(sh, pulls), results, kind="SHARP")
    sb.to_csv(SHARP_BETS_OUT, index=False)
    print(f"sharp shadow bets {len(sb)} | " + (str(sb["status"].value_counts().to_dict()) if len(sb) else "{}"))
    tr = pd.read_csv(TRACKED) if TRACKED.exists() else pd.DataFrame()
    tb = tracked_bets(drop_unrecorded(tr, pulls), results)
    tb.to_csv(TRACKED_BETS_OUT, index=False)
    print(f"sharp tracked bets {len(tb)} | " + (str(tb["status"].value_counts().to_dict()) if len(tb) else "{}"))
    print(f"model shadow bets {len(mb)} | " + (str(mb["status"].value_counts().to_dict()) if len(mb) else "{}"))
    led = ledger(bets)
    bets.to_csv(BETS_OUT, index=False)
    led.to_csv(LEDGER_OUT, index=False)
    print(f"paper bets {len(bets)} | " + (bets["status"].value_counts().to_dict().__str__() if len(bets) else "{}"))
    if len(led):
        print(led.tail(7).to_string(index=False))
    return led


if __name__ == "__main__":
    run()
