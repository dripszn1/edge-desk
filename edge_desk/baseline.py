"""Random-entry baseline for the exit comparison (#57, report only).

Question: when an exit (Auto Sell, sell at tip...) makes money on our SHARP plays, is that the
pick or the exit? If random contracts from the same game, entered at the same moment and graded
the same way, do as well, the edge is the exit (Kalshi overreacting in-game), not the pick.

Per graded SHARP play (shadow + tracked):
  entry snapshot  the latest SAVED snapshot at or before the play's entry, at most ENTRY_GAP
                  earlier (live scans every 30 s but saves every 15 min). Never a later one.
  candidates      other contracts of the SAME event and series in that snapshot (the play's own
                  ticker excluded, both sides), two-sided book, our-side ask within +-10c of the
                  play's price and <= 90c (hard rule #9).
  draws           K=20, seeded with sha256(bet_id), so a re-run gives the same placebos;
                  with fewer than K candidates they are drawn with replacement.
  placebo trade   bought at its ask with the play's contract count, taker fee on entry (snapshot
                  fee multiplier); exits as trades.grade_one: HOLD, EXIT_AT_TIP, RESTING_5/10/15,
                  RESTING_FAIR (target = placebo price + the play's own fair - price).
  path            saved snapshots only, for the real play AND the placebos: fast-lane quotes and
                  1-min candles exist only for alerted contracts, so using them for the real
                  play alone would flatter it. Points strictly after entry and at/before now.
  settlement      real play: the HOLD grader; placebo: data/results/results.csv (settle.run
                  fetches placebo tickers once their game has kicked off).

Summary per group (football / cbb, never pooled), per exit, on plays whose real trade AND at
least one placebo are final under every exit: real $ per play, placebo mean and 95% band from
PORTFOLIOS seeded portfolios (one final placebo per play), and the share of portfolios the real
result beats. Verdict needs >= MIN_PLAYS closed plays.
Output: data/baseline_placebos.csv (the draws; settle reads it), weekly_report.json["baseline"].
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import fees, trades

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SNAP_DIR = DATA / "snapshots"
PLACEBOS = DATA / "baseline_placebos.csv"
RESULTS = DATA / "results" / "results.csv"
STATE = DATA / "baseline_summary.json"
K = 20
PRICE_BAND = 0.10
MAX_PRICE = 0.90
ENTRY_GAP = pd.Timedelta(minutes=20)
OCC_OFFSET = pd.Timedelta(hours=3)
PORTFOLIOS = 2000
MIN_PLAYS = 30
BEAT_PCT = 95.0
EXITS = ["HOLD", "EXIT_AT_TIP", "RESTING_5", "RESTING_10", "RESTING_15", "RESTING_FAIR"]
SNAP_COLS = ["snapshot_ts", "series", "event_ticker", "ticker", "yes_bid", "yes_ask",
             "occurrence_datetime", "fee_multiplier"]
PLACEBO_COLS = ["bet_id", "draw", "side", "ticker", "entry_ts", "price", "contracts", "fee",
                "kickoff_utc"]
RERUN_EVERY = pd.Timedelta(minutes=60)


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def seed_of(bet_id: str) -> int:
    return int(hashlib.sha256(str(bet_id).encode()).hexdigest()[:16], 16)


def side_ask(side: str, yes_bid, yes_ask):
    """Price to BUY our side, needing a two-sided book: YES -> yes ask, NO -> 1 - yes bid."""
    yb, ya = trades._f(yes_bid), trades._f(yes_ask)
    if yb is None or ya is None or not (0 < yb <= ya < 1):
        return None
    return ya if side == "YES" else round(1 - yb, 4)


def entry_snapshot(entry, snap_times) -> pd.Timestamp | None:
    """Latest saved snapshot at or before the entry, within ENTRY_GAP. Never a later one."""
    entry = _utc(entry)
    best = None
    for t in snap_times:
        t = _utc(t)
        if entry - ENTRY_GAP <= t <= entry and (best is None or t > best):
            best = t
    return best


def candidates(real_leg1: str, real_price: float, snap: pd.DataFrame) -> list[dict]:
    """Placebo pool from ONE snapshot (rows of all markets at that time), sorted so the draw
    does not depend on row order."""
    side0, ticker0 = str(real_leg1).split(" ", 1)
    me = snap[snap["ticker"] == ticker0]
    if me.empty:
        return []
    ev, ser = me["event_ticker"].iloc[0], me["series"].iloc[0]
    pool = snap[(snap["event_ticker"] == ev) & (snap["series"] == ser) & (snap["ticker"] != ticker0)]
    out = []
    for r in pool.drop_duplicates("ticker").to_dict("records"):
        for side in ("YES", "NO"):
            a = side_ask(side, r.get("yes_bid"), r.get("yes_ask"))
            if a is None or a > MAX_PRICE or abs(a - real_price) > PRICE_BAND + 1e-9:
                continue
            m = trades._f(r.get("fee_multiplier"))
            out.append({"side": side, "ticker": r["ticker"], "price": a,
                        "fee_multiplier": m if m and m > 0 else 1.0})
    return sorted(out, key=lambda c: (c["ticker"], c["side"]))


def draw(bet_id: str, pool: list[dict], k: int = K) -> list[dict]:
    if not pool:
        return []
    rng = random.Random(seed_of(bet_id))
    return rng.sample(pool, k) if len(pool) >= k else [rng.choice(pool) for _ in range(k)]


def placebo_settle(results: pd.DataFrame | None) -> dict:
    """{ticker: 'yes'|'no'} from the first final row; void and unfinished tickers left out."""
    if results is None or results.empty:
        return {}
    r = results[results["status"].isin({"finalized", "settled", "determined"})
                & results["result"].isin({"yes", "no"})]
    if "fetched_ts" in r:
        r = r.sort_values("fetched_ts", kind="stable")
    return dict(zip(r["ticker"].iloc[::-1], r["result"].iloc[::-1]))


def _grade(b: dict, path, ko, settle, now) -> dict:
    rows = trades.grade_one(b, path, ko, settle, now)
    return {r["variant"]: r for r in rows if r["variant"] in EXITS}


def _final(g: dict) -> bool:
    return all(g.get(v, {}).get("status") == "closed" for v in EXITS)


def grade_plays(bets: pd.DataFrame, snaps: pd.DataFrame, results: pd.DataFrame | None, now):
    """-> (plays, placebo rows). plays: [{bet_id, group, event, real: {exit: pnl}, placebo:
    [{exit: pnl}, ...] (final draws only)}]."""
    now = _utc(now)
    placebo_rows, plays = [], []
    if bets is None or bets.empty or snaps is None or snaps.empty:
        return plays, placebo_rows
    s = snaps.copy()
    s["_ts"] = pd.to_datetime(s["snapshot_ts"], utc=True, errors="coerce")
    s = s[s["_ts"].notna() & (s["_ts"] <= now)]                   # never a snapshot from the future
    s["_ko"] = pd.to_datetime(s["occurrence_datetime"], utc=True, errors="coerce") - OCC_OFFSET
    times = sorted(s["_ts"].unique())
    by_time = {t: g for t, g in s.groupby("_ts")}
    by_ticker = {t: g for t, g in s.groupby("ticker")}
    settled = placebo_settle(results)

    def path_for(side, ticker, entry):
        g = by_ticker.get(ticker)
        return trades.build_path(side, g, None, None, entry, now)

    def kickoff(ticker):
        g = by_ticker.get(ticker)
        if g is None or not g["_ko"].notna().any():
            return None
        return g["_ko"].dropna().iloc[-1]

    for b in bets.to_dict("records"):
        if pd.isna(b.get("snapshot_ts")) or pd.isna(b.get("price")) or " " not in str(b.get("leg1")):
            continue
        entry = _utc(b["snapshot_ts"])
        if entry > now:
            continue
        side0, ticker0 = str(b["leg1"]).split(" ", 1)
        price0 = float(b["price"])
        n = max(int(float(b.get("contracts") or 0)), 1)
        ko = kickoff(ticker0)
        t0 = entry_snapshot(entry, times)
        pool = candidates(b["leg1"], price0, by_time[t0]) if t0 is not None else []
        draws = draw(b.get("bet_id"), pool)
        fair0 = trades._f(b.get("fair"))
        gap = (fair0 - price0) if fair0 is not None and fair0 > price0 else None
        real = _grade({**b, "contracts": n}, path_for(side0, ticker0, entry), ko,
                      trades.settle_value(b), now)
        pl = []
        for i, c in enumerate(draws):
            fee = fees.taker_fee(c["price"], n, c["fee_multiplier"])
            pko = kickoff(c["ticker"]) or ko
            placebo_rows.append({"bet_id": b.get("bet_id"), "draw": i, "side": c["side"],
                                 "ticker": c["ticker"], "entry_ts": t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "price": c["price"], "contracts": n, "fee": fee,
                                 "kickoff_utc": pko.strftime("%Y-%m-%dT%H:%M:%SZ") if pko is not None else None})
            res = settled.get(c["ticker"])
            st = None if res is None else (1.0 if (res == "yes") == (c["side"] == "YES") else 0.0)
            pb = {"bet_id": f"{b.get('bet_id')}#{i}", "leg1": f"{c['side']} {c['ticker']}",
                  "snapshot_ts": t0, "price": c["price"], "contracts": n, "fees": fee,
                  "fair": c["price"] + gap if gap is not None else None}
            pl.append(_grade(pb, path_for(c["side"], c["ticker"], t0), pko, st, now))
        lg = str(b.get("league") or "")
        plays.append({"bet_id": b.get("bet_id"), "group": "cbb" if lg == "CBB" else "football",
                      "event": trades._event(b["leg1"]), "pool": len(pool),
                      "real": {v: r["pnl_usd"] for v, r in real.items()} if _final(real) else None,
                      "placebo": [{v: r["pnl_usd"] for v, r in g.items()} for g in pl if _final(g)]})
    return plays, placebo_rows


def verdict(n: int, beat: float | None, lo: float | None, hi: float | None, real: float | None) -> str:
    if n < MIN_PLAYS or beat is None:
        return f"too few closed plays ({n}/{MIN_PLAYS})"
    if beat >= BEAT_PCT:
        return "picks add value"
    if beat <= 100 - BEAT_PCT:
        return "picks do worse than random entries"
    if real is not None and real > 0:
        return "placebo matches real: the edge is the exit, not the pick"
    return "placebo matches real: the picks add nothing"


def summary(plays: list[dict]) -> dict:
    out = {"method": "placebo = other rungs/sides of the same game and series at the same snapshot, "
                     "ask within 10c, same contracts and fees; snapshot paths only for both"}
    for grp in ("football", "cbb"):
        ps = [p for p in plays if p["group"] == grp]
        ok = [p for p in ps if p["real"] is not None and p["placebo"]]
        d = {"plays": len(ps), "closed": len(ok), "games": len({p["event"] for p in ok}),
             "no_placebo": sum(1 for p in ps if p["pool"] == 0), "exits": {}}
        if ok:
            rng = np.random.default_rng(57)
            idx = [rng.integers(0, len(p["placebo"]), PORTFOLIOS) for p in ok]
        for v in EXITS:
            if not ok:
                d["exits"][v] = {"real_per_play": None, "placebo_mean": None, "band95": None,
                                 "beats_pct": None, "verdict": verdict(0, None, None, None, None)}
                continue
            real = float(np.mean([p["real"][v] for p in ok]))
            mat = np.array([[p["placebo"][j][v] for j in ix] for p, ix in zip(ok, idx)])  # plays x R
            pm = mat.mean(axis=0)
            beat = 100 * float(np.mean(pm < real - 1e-12) + 0.5 * np.mean(np.abs(pm - real) <= 1e-12))
            lo, hi = float(np.percentile(pm, 2.5)), float(np.percentile(pm, 97.5))
            d["exits"][v] = {"real_per_play": round(real, 3), "placebo_mean": round(float(pm.mean()), 3),
                             "band95": [round(lo, 3), round(hi, 3)], "beats_pct": round(beat, 1),
                             "verdict": verdict(len(ok), beat, lo, hi, real)}
        out[grp] = d
    return out


def digest_line(rep: dict | None) -> list[str]:
    """One line: real $/play vs random entries, hold and Auto Sell +10c, per group."""
    parts = []
    for grp in ("football", "cbb"):
        d = (rep or {}).get(grp) or {}
        if not d.get("closed"):
            continue
        cells = []
        for v, name in (("HOLD", "hold"), ("RESTING_10", "auto-sell +10c")):
            x = (d.get("exits") or {}).get(v) or {}
            if x.get("real_per_play") is None:
                continue
            cells.append(f"{name} ${x['real_per_play']:+.2f} vs ${x['placebo_mean']:+.2f} "
                         f"(beats {x['beats_pct']:.0f}%)")
        if cells:
            parts.append(f"{'CBB' if grp == 'cbb' else 'football'} ({d['closed']} plays): " + ", ".join(cells))
    return ["  picks vs random entries, $/play: " + " · ".join(parts)] if parts else []


# ---------- I/O ----------
def load_snapshots(events: set[str], start, end) -> pd.DataFrame:
    parts = []
    if not events or not SNAP_DIR.exists():
        return pd.DataFrame(columns=SNAP_COLS)
    for day in sorted(p for p in SNAP_DIR.iterdir() if p.is_dir()):
        try:
            d = pd.Timestamp(day.name).date()
        except Exception:  # noqa: BLE001
            continue
        if d < pd.Timestamp(start).date() or d > pd.Timestamp(end).date():
            continue
        for f in sorted(day.glob("*.parquet")):
            try:
                df = pd.read_parquet(f, columns=SNAP_COLS)
            except Exception:  # noqa: BLE001
                continue
            df = df[df["event_ticker"].isin(events)]
            if len(df):
                parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SNAP_COLS)


def run(now=None) -> dict:
    from . import clv
    now = _utc(now or datetime.now(timezone.utc))
    frames = []
    for src, p in clv.BET_FILES.items():
        try:
            b = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            continue
        if len(b):
            frames.append(b.assign(source=src))
    if not frames:
        return summary([])
    bets = pd.concat(frames, ignore_index=True)
    bets = bets[pd.to_datetime(bets["snapshot_ts"], utc=True, errors="coerce").notna()]
    tick = bets["leg1"].astype(str).str.split(" ", n=1).str[1]
    events = set(tick.str.rsplit("-", n=1).str[0])
    start = pd.to_datetime(bets["snapshot_ts"], utc=True).min() - ENTRY_GAP
    snaps = load_snapshots(events, start, now)
    try:
        results = pd.read_csv(RESULTS)
    except Exception:  # noqa: BLE001
        results = None
    plays, rows = grade_plays(bets, snaps, results, now)
    pd.DataFrame(rows, columns=PLACEBO_COLS).to_csv(PLACEBOS, index=False)
    s = summary(plays)
    print("baseline: " + " | ".join(
        f"{g} {s[g]['closed']}/{s[g]['plays']} closed, hold real {s[g]['exits']['HOLD']['real_per_play']} "
        f"vs placebo {s[g]['exits']['HOLD']['placebo_mean']}" for g in ("football", "cbb") if g in s))
    return s


def code_hash() -> str:
    h = hashlib.sha256()
    for mod in (__file__, trades.__file__, fees.__file__):
        h.update(Path(mod).read_bytes())
    return h.hexdigest()[:16]


def run_if_due(now=None) -> dict:
    """weekly runs every 15 min in live mode; re-grade at most hourly (cache stamped with the
    code that made it, never served from the future)."""
    now = _utc(now or datetime.now(timezone.utc))
    try:
        last = json.loads(STATE.read_text())
        made = _utc(last["made_utc"])
        if made <= now and now - made < RERUN_EVERY and last.get("code") == code_hash():
            return last["summary"]
    except Exception:  # noqa: BLE001
        pass
    s = run(now)
    STATE.write_text(json.dumps({"made_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "code": code_hash(),
                                 "summary": s}, default=str))
    return s


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
