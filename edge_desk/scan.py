"""Structural scanner: prices that break their own logic, after fees.

Checks (all two-leg, both legs bought as a taker, fees on every leg):
  1. LADDER   - same team (or same total), strikes L < H: P(>H) can never exceed P(>L).
                Buy YES on L + NO on H. Pays >= $1 in every outcome.
  2. EXCLUSIVE - two markets that can't both be YES (Team A wins by 3+ / Team B wins by 3+,
                or A wins / B wins). Buy NO on both. Pays >= $1 in every outcome.
  3. COVER    - game-winner markets with no possible tie (CFB, CBB): buy YES on both. Pays exactly $1.
Flag only when $1 - cost >= MIN_EDGE and both legs have >= MIN_SIZE contracts.
Pregame only by default: markets must be tagged phase == 'pregame' (see kickoffs.py);
untagged, live or unknown markets are never flagged unless include_live=True.
Paper mode: writes flags to data/flags/, never trades.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd

from . import cfb_results, config, kickoffs, model, moneyline, nfl_results, odds, premove, sharp
from .fees import fee_per_contract
from .sizing import structural_size

ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = ROOT / "data" / "snapshots"
FLAG_DIR = ROOT / "data" / "flags"

NO_TIE_LEAGUES = {"CFB", "CBB"}


def latest_snapshot() -> tuple[pd.DataFrame, str | None]:
    files = sorted(SNAP_DIR.glob("*/*.parquet"))
    if not files:
        return pd.DataFrame(), None
    df = pd.read_parquet(files[-1])
    return df, (df["snapshot_ts"].iloc[0] if len(df) else None)


def is_stale(snapshot_ts: str | None, now: datetime | None = None) -> bool:
    if not snapshot_ts:
        return True
    now = now or datetime.now(timezone.utc)
    t = datetime.strptime(snapshot_ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (now - t).total_seconds() > config.STALE_MINUTES * 60


def _leg_cost(price: float, size: float, mult: float) -> float:
    return price + fee_per_contract(price, max(1, min(size, 100)), mult)


def _flag(kind, r1, side1, p1, r2, side2, p2, size):
    if max(p1, p2) > config.MAX_FLAG_PRICE:   # audit rule: no contract priced above 90c
        return None
    if size is None or size != size:          # #44: unknown size (NaN) = nothing to buy, never a crash
        return None
    mult = float(r1.get("fee_multiplier") or 1.0)
    cost = _leg_cost(p1, size, mult) + _leg_cost(p2, size, mult)
    edge = round(1 - cost, 4)
    if edge < config.MIN_EDGE or size < config.MIN_SIZE:
        return None
    s = structural_size(cost, size)
    return {
        "snapshot_ts": r1["snapshot_ts"], "league": r1["league"], "kind": kind,
        "event": r1["event_title"], "event_ticker": r1["event_ticker"],
        "leg1": f"{side1} {r1['ticker']}", "leg1_desc": r1["yes_sub_title"], "leg1_price": p1,
        "leg2": f"{side2} {r2['ticker']}", "leg2_desc": r2["yes_sub_title"], "leg2_price": p2,
        "cost_per_set": round(cost, 4), "edge_per_set": edge, "size_available": size,
        "paper_sets": s["sets"], "paper_stake": s["stake"],
        "fee_multiplier": mult,
        "occurrence_datetime": r1.get("occurrence_datetime"),
        "phase": r1.get("phase"), "espn_id": r1.get("espn_id"),
        "kickoff_utc": r1.get("kickoff_utc"), "kickoff_ct": r1.get("kickoff_ct"),
    }


def _quoted(df: pd.DataFrame, include_live: bool = False) -> pd.DataFrame:
    q = df[(df["status"] == "active")].copy()
    phase = q["phase"] if "phase" in q else pd.Series("unknown", index=q.index)
    ok = phase.isin(["pregame", "live"]) if include_live else phase.eq("pregame")
    q = q[ok]
    return q.dropna(subset=["yes_bid", "yes_ask"])


def _isnan(x) -> bool:
    return x is None or x != x


def scan(df: pd.DataFrame, include_live: bool = False) -> list[dict]:
    """Same flags as the original per-event pandas groupby, built from one to_dict pass (#44):
    groupby + to_dict per event/rung cost ~18 s per live pass on a 6,500-market board."""
    flags = []
    if df.empty:
        return flags
    q = _quoted(df, include_live)
    events: dict = {}
    for r in q.to_dict("records"):
        events.setdefault(r["event_ticker"], []).append(r)
    for ev_ticker in sorted(events, key=lambda k: (_isnan(k), k)):
        ev = events[ev_ticker]
        if _isnan(ev_ticker):
            continue                                  # groupby drops NaN keys
        series = ev[0]["series"]
        greater = [r for r in ev if r["strike_type"] == "greater" and not _isnan(r["floor_strike"])]

        # 1. LADDER: same team_key (spreads) or whole event (totals)
        rungs: dict = {}
        for r in greater:
            rungs.setdefault("__total__" if _isnan(r["team_key"]) else r["team_key"], []).append(r)
        for key in sorted(rungs):
            rows = sorted(rungs[key], key=lambda r: r["floor_strike"])
            for lo, hi in combinations(rows, 2):
                if hi["floor_strike"] <= lo["floor_strike"]:
                    continue
                p_yes_lo = lo["yes_ask"]
                p_no_hi = 1 - hi["yes_bid"]
                size = min(lo["yes_ask_size"] or 0, hi["yes_bid_size"] or 0)
                f = _flag("LADDER", lo, "YES", p_yes_lo, hi, "NO", p_no_hi, size)
                if f:
                    flags.append(f)

        # 2. EXCLUSIVE: different teams' spread rungs, or game-winner pairs
        if "SPREAD" in series:
            teams: dict = {}
            for r in greater:
                if not _isnan(r["team_key"]):
                    teams.setdefault(r["team_key"], []).append(r)
            if len(teams) == 2:
                ta, tb = sorted(teams)
                for a in teams[ta]:
                    for b in teams[tb]:
                        size = min(a["yes_bid_size"] or 0, b["yes_bid_size"] or 0)
                        f = _flag("EXCLUSIVE", a, "NO", 1 - a["yes_bid"], b, "NO", 1 - b["yes_bid"], size)
                        if f:
                            flags.append(f)
        if series.endswith("GAME") and len(ev) == 2:
            a, b = ev
            size = min(a["yes_bid_size"] or 0, b["yes_bid_size"] or 0)
            f = _flag("EXCLUSIVE", a, "NO", 1 - a["yes_bid"], b, "NO", 1 - b["yes_bid"], size)
            if f:
                flags.append(f)
            # 3. COVER: no ties possible -> exactly one side pays $1
            if ev[0]["league"] in NO_TIE_LEAGUES:
                size = min(a["yes_ask_size"] or 0, b["yes_ask_size"] or 0)
                f = _flag("COVER", a, "YES", a["yes_ask"], b, "YES", b["yes_ask"], size)
                if f:
                    flags.append(f)
    return sorted(flags, key=lambda x: -x["edge_per_set"])


def phase_summary(df: pd.DataFrame) -> dict:
    if df.empty or "phase" not in df:
        return {}
    c = df.groupby(["league", "phase"]).size()
    return {lg: {ph: int(n) for (l2, ph), n in c.items() if l2 == lg} for lg in c.index.get_level_values(0).unique()}


def unmatched(df: pd.DataFrame, limit: int = 50) -> list[str]:
    """Events with no usable kickoff time (they are never flagged)."""
    if df.empty or "kickoff_utc" not in df:
        return []
    u = df[df["kickoff_utc"].isna()].drop_duplicates("event_ticker")
    return [f"{r.event_ticker} | {r.event_sub}" for r in u.head(limit).itertuples()]


def write_event_map(df: pd.DataFrame, path: Path = ROOT / "data" / "kickoffs" / "event_map.csv") -> None:
    """Upsert Kalshi event -> kickoff, for the grader and model."""
    if df.empty or "kickoff_utc" not in df:
        return
    new = (df.dropna(subset=["kickoff_utc"])
             .drop_duplicates("event_ticker")[["event_ticker", "league", "kickoff_utc", "snapshot_ts"]]
             .rename(columns={"snapshot_ts": "matched_at"}))
    if new.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        new = pd.concat([old, new.astype(str)]).drop_duplicates("event_ticker", keep="last")
    new.sort_values("event_ticker").to_csv(path, index=False)


def flag_key(f: dict) -> tuple:
    """One contract (or two-leg set) per signal: the dedupe key for logs and alerts (#16/#17)."""
    return (f.get("kind"), f.get("leg1"), f.get("leg2") if f.get("kind") in ("LADDER", "EXCLUSIVE", "COVER") else None)


def with_url(flags: list[dict]) -> list[dict]:
    """#18: every flag record carries its Kalshi page (same builder as the phone alerts)."""
    from .alerts import kalshi_url
    tmpl = None
    for f in flags:
        if not f.get("kalshi_url"):
            if tmpl is None:
                from .alerts import load_url_format
                tmpl = load_url_format()
            tick = f.get("ticker") or str(f.get("leg1") or "").split(" ", 1)[-1]
            f["kalshi_url"] = kalshi_url(tick, f.get("event_ticker"), tmpl)
    return flags


def new_only(flags: list[dict], seen: set | None) -> list[dict]:
    """Flags not logged before in this live session. `seen=None` keeps them all (batch runs).
    Marks the returned flags as seen, so a contract is logged once however often it re-appears."""
    if seen is None:
        return flags
    out = []
    for f in flags:
        k = flag_key(f)
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


def _append_csv(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if out.exists():   # concat, not append: keeps columns aligned if the flag schema grows
        new = pd.concat([pd.read_csv(out), new], ignore_index=True)
    new.to_csv(out, index=False)


def lost_today(ledger: Path = ROOT / "data" / "paper_ledger.csv", now: datetime | None = None) -> float:
    """Realized paper loss so far today (CT), for the daily loss cap."""
    try:
        led = pd.read_csv(ledger)
        today = (now or datetime.now(timezone.utc)).astimezone(kickoffs.CT).strftime("%Y-%m-%d")
        pnl = float(led.loc[led["date_ct"].astype(str) == today, "pnl"].sum())
        return max(-pnl, 0.0)
    except Exception:
        return 0.0


RESULT_LOADERS = {"NFL": nfl_results.load, "CFB": cfb_results.load}


def run_model(df: pd.DataFrame, ts: str | None, leagues=("NFL", "CFB"), seen: set | None = None) -> dict:
    """MODEL flags per league. Live only if that league's backtest gate is open; else shadow.
    Leagues share one daily loss cap: each league only gets what the earlier ones left."""
    out = {"status": {}, "live": [], "shadow": []}
    lost = lost_today()
    for lg in leagues:
        try:
            flags, status = model.model_flags(df, RESULT_LOADERS[lg](), ts, lost_today=lost, league=lg)
            flags = with_url(flags)
        except Exception as e:   # one league's bug must not block the other
            flags, status = [], {"ok": False, "reason": f"error: {e}"}
        is_open, why = model.gate(league=lg)
        status["gate"] = why
        out["status"][lg] = status
        _append_csv(new_only(flags, seen), FLAG_DIR / ("model_flags.csv" if is_open else "model_shadow.csv"))
        out["live" if is_open else "shadow"].extend(flags)
        if is_open:
            lost += sum(f["stake"] for f in flags)   # live stakes use up today's room
    return out


def run_sharp(df: pd.DataFrame, ts: str | None, lost: float | None = None, leagues=("NFL", "CFB", "CBB"),
              seen: set | None = None) -> dict:
    """SHARP flags (Kalshi vs sharp sportsbook line), per league. Live only if that league's gate
    is open; else shadow. Leagues share one daily loss cap."""
    lost = lost_today() if lost is None else lost
    out = {"status": {}, "live": [], "shadow": [], "fairs": {}}
    lines = odds.load_lines()
    for lg in leagues:
        try:
            flags, status = sharp.sharp_flags(df, lines, sharp.history(lg), ts, lost_today=lost, league=lg)
            out["fairs"].update(status.pop("_fairs", None) or {})
            flags = with_url(premove.screen(flags, df, ts, lg))     # Emerson 11-12: CBB guard
        except Exception as e:   # never lose other flags over a SHARP bug
            flags, status = [], {"ok": False, "reason": f"error: {e}"}
        is_open, why = sharp.gate(league=lg)
        if lg in sharp.SHADOW_ONLY:
            is_open, why = False, f"shadow-only league ({why})"
        status["gate"] = why
        out["status"][lg] = status
        _append_csv(new_only(flags, seen), sharp.LIVE if is_open else sharp.SHADOW)
        out["live" if is_open else "shadow"].extend(flags)
        if is_open:
            lost += sum(f["stake"] for f in flags)
    for lg in leagues:                                 # moneylines (#23): their own gate per league
        try:
            flags, status = moneyline.ml_flags(df, lines, ts, lost_today=lost, league=lg)
            out["fairs"].update(status.pop("_fairs", None) or {})
            flags = with_url(premove.screen(flags, df, ts, lg))
        except Exception as e:
            flags, status = [], {"ok": False, "reason": f"error: {e}"}
        is_open, why = sharp.gate(league=lg, market="ML")
        if lg in sharp.SHADOW_ONLY:
            is_open, why = False, f"shadow-only league ({why})"
        status["gate"] = why
        out["status"][f"{lg} ML"] = status
        _append_csv(new_only(flags, seen), sharp.LIVE if is_open else sharp.SHADOW)
        out["live" if is_open else "shadow"].extend(flags)
        if is_open:
            lost += sum(f["stake"] for f in flags)
    return out


def alert_fairs(fairs: dict) -> dict:
    """#35: only the fair entries for alerted, not-yet-gone contracts go into latest.json (thousands
    of rungs are priced per scan; collect.yml's alert step needs just these)."""
    from . import alerts
    keys = {alerts.fair_key(a.get("ticker"), a.get("side"))
            for a in alerts.load_state().get("alerts", {}).values() if not a.get("gone")}
    return {k: v for k, v in fairs.items() if k in keys}


def run(df: pd.DataFrame | None = None, ts: str | None = None, seen: set | None = None,
        partial: bool = False) -> list[dict]:
    """Scan one snapshot. Batch runs read the newest snapshot file. The live loop (#16) passes its
    in-memory frame (every row from ONE pass, one snapshot_ts), a `seen` set so each contract is
    logged once, and `partial=True` for hot-games-only passes (event map / latest.json untouched)."""
    if df is None:
        df, ts = latest_snapshot()
    FLAG_DIR.mkdir(parents=True, exist_ok=True)
    stale = is_stale(ts)
    df = kickoffs.tag_kalshi(df, ts)   # Kalshi's own clock; ESPN no longer needed
    flags = [] if stale else with_url(scan(df))
    if not partial:
        write_event_map(df)
    _append_csv(new_only(flags, seen), FLAG_DIR / "flags.csv")
    try:
        m = run_model(df, ts, seen=seen)
    except Exception as e:   # never lose structural flags over a model bug
        m = {"status": {"error": str(e)}, "live": [], "shadow": []}
    live_model = sum(f["stake"] for f in m["live"])
    sh = run_sharp(df, ts, lost_today() + live_model, seen=seen)
    run.last = {"flags": flags, "model": m, "sharp": sh}
    if partial:
        return flags
    (FLAG_DIR / "latest.json").write_text(json.dumps(
        {"snapshot_ts": ts, "stale": stale, "markets_scanned": int(len(df)),
         "kickoff_source": "kalshi occurrence_datetime - 3h",
         "phases": phase_summary(df), "unmatched_events": unmatched(df), "flags": flags,
         "model": m["status"], "model_flags": m["live"], "model_shadow": m["shadow"],
         "sharp": sh["status"], "sharp_flags": sh["live"], "sharp_shadow": sh["shadow"],
         "sharp_fairs": alert_fairs(sh.get("fairs") or {})},
        indent=2, default=str))
    print(f"snapshot {ts} | stale={stale} | markets={len(df)} | phases={phase_summary(df)} | flags={len(flags)}"
          f" | model: " + "; ".join(f"{lg} {st.get('reason')} / {st.get('gate')}" for lg, st in m["status"].items()
                                     if isinstance(st, dict)) + f" | live={len(m['live'])} shadow={len(m['shadow'])}"
          f" | sharp: " + "; ".join(f"{lg} {st.get('reason')} / {st.get('gate')}" for lg, st in sh["status"].items())
          + f" live={len(sh['live'])} shadow={len(sh['shadow'])}")
    for f in flags[:10]:
        print(f"  {f['kind']:9} {f['event']} ({f['kickoff_ct']}): {f['leg1']} @{f['leg1_price']:.2f} + "
              f"{f['leg2']} @{f['leg2_price']:.2f} -> edge {f['edge_per_set']*100:.1f}c x {f['size_available']:.0f}")
    return flags


if __name__ == "__main__":
    run()
