"""Dashboard data: data/dashboard.json — today's flags, paper record, P&L after fees,
splits by league and by signal, risk status. All stored times UTC; *_ct fields for display.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import config
from .kickoffs import to_ct
from .alerts import kalshi_url
from .scan import is_stale

ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "data" / "flags" / "latest.json"
BETS = ROOT / "data" / "paper_bets.csv"
LEDGER = ROOT / "data" / "paper_ledger.csv"
OUT = ROOT / "data" / "dashboard.json"
CT = ZoneInfo(config.TIMEZONE)


def record(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return _empty()
    taken = bets[~bets["status"].astype(str).str.startswith("skipped") & (bets["status"] != "no_size")] \
        if len(bets) else bets
    s = taken[taken["status"] == "settled"] if len(taken) else taken
    staked = float(s["stake"].sum()) if len(s) else 0.0
    pnl = float(s["pnl"].sum()) if len(s) else 0.0
    return {
        "bets": int(len(taken)), "settled": int(len(s)),
        "wins": int((s["pnl"] > 0).sum()) if len(s) else 0,
        "losses": int((s["pnl"] < 0).sum()) if len(s) else 0,
        "pushes": int((s["pnl"] == 0).sum()) if len(s) else 0,
        "pending": int(taken["status"].isin(["pending", "stale"]).sum()) if len(taken) else 0,
        "staked": round(staked, 2), "fees": round(float(s["fees"].sum()) if len(s) else 0.0, 2),
        "pnl": round(pnl, 2), "roi": round(pnl / staked, 4) if staked else None,
    }


def _split(bets: pd.DataFrame, col: str) -> dict:
    if bets.empty:
        return {}
    return {str(k): record(g) for k, g in bets.groupby(col)}


def risk(bets: pd.DataFrame, today: str) -> dict:
    s = bets[bets["status"] == "settled"] if len(bets) else pd.DataFrame()
    lost_today = float(-s.loc[s["date_ct"] == today, "pnl"].clip(upper=0).sum()) if len(s) else 0.0
    if len(s):
        s = s.sort_values("settled_ts")
        eq = config.BANKROLL + s["pnl"].cumsum()
        peak, cur = max(config.BANKROLL, float(eq.max())), float(eq.iloc[-1])
    else:
        peak = cur = config.BANKROLL
    paused = bool(len(bets) and (bets["status"] == "skipped_drawdown").any()) or peak - cur >= config.DRAWDOWN_STOP
    return {"bankroll": round(cur, 2), "peak": round(peak, 2), "drawdown": round(peak - cur, 2),
            "drawdown_stop": config.DRAWDOWN_STOP, "lost_today": round(lost_today, 2),
            "daily_loss_cap": config.DAILY_LOSS_CAP,
            "daily_cap_hit": lost_today >= config.DAILY_LOSS_CAP, "paused": paused,
            "paper_mode": config.PAPER_MODE, "max_bet": config.MAX_BET}


def build(latest: dict, bets: pd.DataFrame, led: pd.DataFrame, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    now_ts = pd.Timestamp(now)
    today = now_ts.tz_convert(CT).date().isoformat()
    if len(bets):   # never show a bet "from the future" relative to the build time
        bets = bets[pd.to_datetime(bets["snapshot_ts"], utc=True) <= now_ts]
    snap_ts = latest.get("snapshot_ts")
    stale = bool(latest.get("stale", True)) or is_stale(snap_ts, now)   # re-check at build time
    flags = [] if stale else [f for f in latest.get("flags", []) if f.get("phase", "pregame") == "pregame"]
    for f in flags:
        f.setdefault("kalshi_url", kalshi_url(str(f.get("leg1", "")).split(" ", 1)[-1], f.get("event_ticker")))
    today_bets = bets[bets["date_ct"] == today] if len(bets) else bets
    cols = ["date_ct", "league", "kind", "event", "leg1", "leg2", "sets", "stake", "fees", "pnl",
            "status", "kickoff_ct", "kalshi_url"]
    if len(today_bets):        # #18: a Kalshi link on every row
        today_bets = today_bets.copy()
        today_bets["kalshi_url"] = [kalshi_url(str(l).split(" ", 1)[-1]) for l in today_bets["leg1"]]
    return {
        "generated_ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_ct": to_ct(now.strftime("%Y-%m-%dT%H:%M:%SZ")),
        "today_ct": today,
        "snapshot": {"ts": snap_ts, "ct": to_ct(snap_ts), "stale": stale,
                     "markets_scanned": latest.get("markets_scanned", 0),
                     "kickoff_source": latest.get("kickoff_source"),
                     "phases": latest.get("phases", {})},
        "flags_now": flags,
        "today": {"record": record(today_bets),
                  "bets": today_bets[cols].to_dict("records") if len(today_bets) else []},
        "record": record(bets),
        "by_league": _split(bets, "league"),
        "by_signal": _split(bets, "kind"),
        "ledger": led.tail(14).to_dict("records") if len(led) else [],
        "risk": risk(bets, today),
    }


def _empty() -> dict:
    return {"bets": 0, "settled": 0, "wins": 0, "losses": 0, "pushes": 0, "pending": 0,
            "staked": 0.0, "fees": 0.0, "pnl": 0.0, "roi": None}


def run() -> dict:
    latest = json.loads(LATEST.read_text()) if LATEST.exists() else {}
    bets = pd.read_csv(BETS) if BETS.exists() and BETS.stat().st_size > 1 else pd.DataFrame()
    led = pd.read_csv(LEDGER) if LEDGER.exists() and LEDGER.stat().st_size > 1 else pd.DataFrame()
    d = build(latest, bets, led)
    OUT.write_text(json.dumps(d, indent=2, default=str))
    r = d["record"]
    print(f"dashboard: {len(d['flags_now'])} flags now | record {r['wins']}-{r['losses']}-{r['pushes']} "
          f"| P&L ${r['pnl']:.2f} after fees | stale={d['snapshot']['stale']}")
    return d


if __name__ == "__main__":
    run()
