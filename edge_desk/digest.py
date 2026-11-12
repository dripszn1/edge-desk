"""Game-day digest (backlog #14): data/digest.md (newest) + data/digests/<CT date>.md.

Plain language, times in CT, under MAX_LINES lines. Read-only; a separate scheduled task reads
data/digest.md and pings Emerson. Sections:
  (a) Upcoming plays: every SHARP / MODEL play (live or shadow) from the latest scan whose
      kickoff is between now and now + WINDOW_H: game, contract, Kalshi price, fair, edge after
      fees in cents, paper stake, kickoff CT. If the scan's snapshot is stale, no plays are shown
      (same rule as the scanner). "No plays" is a valid answer.
  (b) Recap: paper / MODEL shadow / SHARP shadow bets settled in the last RECAP_H hours
      (the digest is rebuilt every workflow run, so "since the last digest" is read as "since
      this time yesterday"), W-L, P&L after fees, plus the running record.
  (c) Status: odds credits left, data freshness (Kalshi snapshot, sharp odds), gates.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .alerts import kalshi_url
from . import config, odds
from .kickoffs import CT, to_ct

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "digest.md"
ARCHIVE = DATA / "digests"
WINDOW_H = 36
RECAP_H = 24
MAX_LINES = 25
MAX_PLAYS = 10
BET_FILES = {"paper": DATA / "paper_bets.csv", "MODEL shadow": DATA / "model_shadow_bets.csv",
             "SHARP shadow": DATA / "sharp_shadow_bets.csv",
             "SHARP tracked (held back)": DATA / "sharp_tracked_bets.csv"}


def _ts(x):
    t = pd.Timestamp(x)
    return t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")


def _age_min(ts, now) -> float | None:
    try:
        return (_ts(now) - _ts(ts)).total_seconds() / 60
    except Exception:
        return None


def _fmt_age(m: float | None) -> str:
    if m is None:
        return "none"
    return f"{m:.0f}m old" if m < 120 else f"{m / 60:.1f}h old"


def plays(latest: dict, now) -> list[dict]:
    if not latest or latest.get("stale"):
        return []
    age = _age_min(latest.get("snapshot_ts"), now)
    if age is None or age > config.STALE_MINUTES:
        return []
    now = _ts(now)
    out = []
    # MODEL failed its backtest: it is graded quietly, never shown as a play
    for key, tag in (("sharp_flags", "SHARP"), ("sharp_shadow", "SHARP shadow")):
        for f in latest.get(key) or []:
            ko = f.get("kickoff_utc")
            if not ko or not (now < _ts(ko) <= now + pd.Timedelta(hours=WINDOW_H)):
                continue
            out.append({**f, "_tag": tag})
    return sorted(out, key=lambda f: (f["kickoff_utc"], -float(f.get("edge") or 0)))


def _play_line(f: dict) -> str:
    game = str(f.get("event") or "").split(":")[0]
    contract = f"{f.get('side')} {f.get('leg1_desc') or f.get('ticker')}"
    return (f"- {f['_tag']}: {game} | {contract} @ {float(f['price']) * 100:.0f}¢, fair "
            f"{float(f['fair']) * 100:.0f}¢, edge {float(f['edge']) * 100:.1f}¢ after fees | "
            f"${float(f['stake']):.2f} paper | {f.get('kickoff_ct') or to_ct(f.get('kickoff_utc'))}"
            f" | [Kalshi]({f.get('kalshi_url') or kalshi_url(f.get('ticker'), f.get('event_ticker'))})")


def _load(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _split_cbb(bets: dict) -> dict:
    """CBB is its own scorecard (#22): never pooled with football."""
    out = {}
    for name, b in bets.items():
        if b is not None and not b.empty and "league" in b and (b["league"] == "CBB").any():
            out[f"{name} football"] = b[b["league"] != "CBB"]
            out[f"{name} CBB"] = b[b["league"] == "CBB"]
        else:
            out[name] = b
    return out


def recap(bets: dict[str, pd.DataFrame], now) -> list[str]:
    now = _ts(now)
    lines = []
    for name, b in _split_cbb(bets).items():
        if b is None or b.empty or "status" not in b:
            continue
        s = b[b["status"] == "settled"].copy()
        if s.empty:
            continue
        s["_st"] = pd.to_datetime(s["settled_ts"], utc=True, errors="coerce")
        s = s[s["_st"] <= now]                      # never report a settlement from the future
        new = s[s["_st"] > now - pd.Timedelta(hours=RECAP_H)]
        w, l = int((s["pnl"] > 0).sum()), int((s["pnl"] <= 0).sum())
        run = f"running {w}-{l}, ${s['pnl'].sum():+.2f}"
        if new.empty:
            lines.append(f"- {name}: nothing settled in {RECAP_H}h ({run})")
        else:
            nw, nl = int((new["pnl"] > 0).sum()), int((new["pnl"] <= 0).sum())
            lines.append(f"- {name}: {nw}-{nl}, ${new['pnl'].sum():+.2f} after fees ({run})")
    return lines or ["- nothing settled yet"]


def held_back_by_reason(b: pd.DataFrame | None, now) -> list[str]:
    """One line: running record of the held-back SHARP plays per reason (far rung, stale line...)."""
    if b is None or b.empty or "reason_bucket" not in b:
        return []
    s = b[b["status"] == "settled"].copy()
    s["_st"] = pd.to_datetime(s["settled_ts"], utc=True, errors="coerce")
    s = s[s["_st"] <= _ts(now)]
    if s.empty:
        return []
    parts = [f"{k} {int((g['pnl'] > 0).sum())}-{int((g['pnl'] <= 0).sum())} ${g['pnl'].sum():+.2f}"
             for k, g in s.groupby("reason_bucket")]
    return ["  held back by reason: " + " · ".join(parts)]


def clv_line(c: pd.DataFrame | None) -> list[str]:
    """One line: did entries beat Kalshi's pregame close after the entry fee (#31)?"""
    from . import clv
    s = clv.summary(c) if c is not None and not c.empty and "close_status" in c else {}
    parts = []
    for src, label in (("shadow", "shadow"), ("tracked", "held back")):
        for grp, v in (s.get(src) or {}).items():
            if v.get("n"):
                parts.append(f"{label} {'CBB' if grp == 'cbb' else 'football'} {v['avg_clv_c']:+.1f}c "
                             f"({v['beat_close_pct']:.0f}% of {v['n']}, {v['events']} games)")
    return ["  vs Kalshi close after fee: " + " · ".join(parts)] if parts else []


def trades_line(t: pd.DataFrame | None) -> list[str]:
    """One line: hold vs sell-at-tip vs Auto Sell at fair vs +10c, same finished bets (#38)."""
    from . import trades
    if t is None or t.empty or "variant" not in t:
        return []
    s = trades.summary(t)
    parts = []
    for grp in ("football", "cbb"):
        v = (s.get(grp) or {}).get("variants") or {}
        h = v.get("HOLD") or {}
        if not h.get("closed"):
            continue
        cells = [f"{name} ${v[k]['pnl_usd']:+.2f}" for k, name in
                 (("HOLD", "hold"), ("EXIT_AT_TIP", "sell at tip"), ("RESTING_FAIR", "auto-sell at fair"),
                  ("RESTING_10", "auto-sell +10c")) if k in v]
        parts.append(f"{'CBB' if grp == 'cbb' else 'football'} ({h['closed']} done): " + ", ".join(cells))
    return ["  exits compared: " + " · ".join(parts)] if parts else []


def status_line(latest: dict, odds_state: dict, now) -> str:
    snap = _age_min((latest or {}).get("snapshot_ts"), now)
    pulls = (odds_state or {}).get("last_pull") or {}
    last_odds = max(pulls.values()) if pulls else None
    credits = odds.remaining(odds_state or {}, now) if odds_state else None
    fresh = "ok" if snap is not None and snap <= config.STALE_MINUTES else "STALE"
    gates = []
    for k, label in (("model", "MODEL"), ("sharp", "SHARP")):
        for lg, st in ((latest or {}).get(k) or {}).items():
            if isinstance(st, dict):
                g = str(st.get("gate", "?"))
                gates.append(f"{label} {lg} {'open' if g.startswith('open') else 'closed'}")
    return (f"Status: Kalshi data {_fmt_age(snap)} ({fresh}); sharp odds {_fmt_age(_age_min(last_odds, now) if last_odds else None)}; "
            f"odds credits left {f'{credits} of {odds.quota(odds_state)}' if credits is not None else 'unknown'}; gates: {', '.join(gates) or 'unknown'}")


def proposals(weekly: dict | None, now) -> list[str]:
    """#46: cap-review proposals from weekly_report.json, at the top of the digest. A report
    stamped after `now` is ignored (never look-ahead)."""
    if not weekly:
        return []
    g = weekly.get("generated_utc")
    try:
        if g is None or _ts(g) > now:
            return []
    except (ValueError, TypeError):
        return []
    return [f"**{p}** (Emerson decides; caps unchanged)" for p in (weekly.get("proposals") or [])[:3]]


def build(latest: dict, bets: dict, odds_state: dict, now, clv: pd.DataFrame | None = None,
          trades_df: pd.DataFrame | None = None, weekly: dict | None = None) -> str:
    now = _ts(now)
    p = plays(latest, now)
    prop = proposals(weekly, now)
    head = [f"# edge-desk digest, {to_ct(now.strftime('%Y-%m-%dT%H:%M:%SZ'))}", ""] + prop + (
        [""] if prop else []) + [f"Paper only, nothing is placed. Plays kicking off in the next {WINDOW_H}h:"]
    if latest and (latest.get("stale") or (_age_min(latest.get("snapshot_ts"), now) or 1e9) > config.STALE_MINUTES):
        body = ["- no plays shown: Kalshi data is stale"]
    elif not p:
        body = ["- no plays"]
    else:
        body = [_play_line(f) for f in p[:MAX_PLAYS]]
        if len(p) > MAX_PLAYS:
            body.append(f"- +{len(p) - MAX_PLAYS} more in data/flags/latest.json")
    rc = (["", f"Settled in the last {RECAP_H}h:"] + recap(bets, now)
          + held_back_by_reason(bets.get("SHARP tracked (held back)"), now)
          + clv_line(clv) + trades_line(trades_df))
    tail = ["", status_line(latest, odds_state, now)]
    lines = head + body + rc + tail
    if len(lines) > MAX_LINES:                       # trim plays first, keep recap + status
        keep = MAX_LINES - len(head) - len(rc) - len(tail) - 1
        lines = head + body[:max(keep, 1)] + [f"- +{len(p) - max(keep, 1)} more in data/flags/latest.json"] + rc + tail
    return "\n".join(lines) + "\n"


def run(now=None) -> str:
    now = _ts(now or datetime.now(timezone.utc))
    try:
        latest = json.loads((DATA / "flags" / "latest.json").read_text())
    except Exception:
        latest = {}
    try:
        weekly = json.loads((DATA / "weekly_report.json").read_text())
    except Exception:
        weekly = {}
    md = build(latest, {k: _load(v) for k, v in BET_FILES.items()}, odds.load_state(), now,
               clv=_load(DATA / "clv_bets.csv"), trades_df=_load(DATA / "sharp_trades.csv"), weekly=weekly)
    OUT.write_text(md)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    (ARCHIVE / f"{now.tz_convert(CT).date().isoformat()}.md").write_text(md)
    print(md)
    return md


if __name__ == "__main__":
    run()
