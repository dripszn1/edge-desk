"""Instant phone alerts (backlog #17) + speed log (#19) + Kalshi links (#18). Read-only: an alert
is a heads-up for Emerson to look; nothing here places or prepares an order.

Channels: ntfy.sh (secret NTFY_TOPIC), optional Pushover (PUSHOVER_TOKEN + PUSHOVER_USER).
Missing secrets -> nothing is sent, the run stays green. The topic is a secret: it is never
printed or written to data/.

Per contract (signal + leg[s]) at most ONE "new" push, plus at most ONE "gone, don't chase" push
when its live price moves past the max price. New pushes respect the daily loss cap: the stakes
already pushed today (CT) plus this one must fit in DAILY_LOSS_CAP - lost_today.

Title: "LIVE" only when that signal's gate is open AND config.PAPER_MODE is off; else "PAPER".
Priority: high (4); 5 (max) when the edge is >= 2x the signal's minimum edge.
State: data/alerts/state.json. Speed log: data/alerts_log.csv
(detected_utc = when this process saw the flag, quote_utc = the Kalshi snapshot time, sent_utc).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from . import config
from .fees import fee_per_contract

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STATE = DATA / "alerts" / "state.json"
URL_FORMAT = DATA / "alerts" / "url_format.json"
LOG = DATA / "alerts_log.csv"
CT = ZoneInfo(config.TIMEZONE)
UTC = timezone.utc
STRUCTURAL = ("LADDER", "EXCLUSIVE", "COVER")
LOG_COLS = ["key", "type", "kind", "title", "ticker", "price", "max_price", "edge", "stake",
            "quote_utc", "detected_utc", "sent_utc", "detect_to_send_s", "quote_to_send_s", "channels"]

# candidate templates, checked in this order by `verify-url` inside Actions
URL_TEMPLATES = ["https://kalshi.com/markets/{series}/{event}",
                 "https://kalshi.com/markets/{event}"]
FALLBACK_TEMPLATE = "https://kalshi.com/markets/{series}"


def _utc(t) -> datetime:
    if isinstance(t, str):
        t = datetime.strptime(t[:19], "%Y-%m-%dT%H:%M:%S")
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def _iso(t) -> str:
    return _utc(t).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------- Kalshi links (#18) ----------
def series_of(ticker: str | None) -> str:
    return str(ticker or "").split("-")[0]


def load_url_format(path: Path | None = None) -> str:
    try:
        j = json.loads((path or URL_FORMAT).read_text())
        return j.get("template") or URL_TEMPLATES[0]
    except Exception:  # noqa: BLE001 - unverified -> default
        return URL_TEMPLATES[0]


def kalshi_url(ticker: str | None, event_ticker: str | None = None, template: str | None = None) -> str:
    """Kalshi web page for a market. Kalshi URLs are lowercase; the event page lists every rung."""
    ev = event_ticker or "-".join(str(ticker or "").split("-")[:2])
    t = template or load_url_format()
    return t.format(series=series_of(ticker or ev).lower(), event=str(ev).lower(), ticker=str(ticker or "").lower())


def verify_url(sample_ticker: str, sample_event: str, get=None) -> dict:
    """Pick the first template that returns HTTP 200 and mentions the event; else the series page."""
    if get is None:
        import requests
        get = lambda u: requests.get(u, timeout=20, headers={"User-Agent": "edge-desk/0.1 (read-only research)"})
    tried = []
    for t in URL_TEMPLATES:
        u = kalshi_url(sample_ticker, sample_event, t)
        try:
            r = get(u)
            ok = r.status_code == 200 and sample_event.lower() in r.text[:2000000].lower()
            tried.append({"url": u, "status": r.status_code, "mentions_event": ok})
            if ok:
                return {"template": t, "verified": True, "tried": tried}
        except Exception as e:  # noqa: BLE001
            tried.append({"url": u, "error": type(e).__name__})
    return {"template": FALLBACK_TEMPLATE, "verified": False, "tried": tried}


# ---------- pricing ----------
def side_price(row: dict, side: str) -> float | None:
    """What it costs right now to buy `side` as a taker."""
    try:
        return float(row["yes_ask"]) if side == "YES" else round(1 - float(row["yes_bid"]), 4)
    except (TypeError, ValueError, KeyError):
        return None


def max_price(fair: float, contracts: int, mult: float = 1.0, min_edge: float = config.MIN_MODEL_EDGE) -> float | None:
    """Highest price (whole cents) where fair - price - fee(price) still clears min_edge."""
    c = max(int(contracts or 1), 1)
    for cents in range(min(int(fair * 100), 99), 0, -1):
        p = cents / 100
        if fair - p - fee_per_contract(p, c, mult) >= min_edge - 1e-9:
            return round(p, 2)
    return None


def structural_max_cost(f: dict) -> float:
    """Highest combined leg price per set that still clears MIN_EDGE after fees on both legs."""
    from .scan import _leg_cost
    mult = float(f.get("fee_multiplier") or 1.0)
    size = float(f.get("paper_sets") or f.get("size_available") or 1)
    p1, p2 = float(f["leg1_price"]), float(f["leg2_price"])
    fees = _leg_cost(p1, size, mult) - p1 + _leg_cost(p2, size, mult) - p2
    return round(1 - config.MIN_EDGE - fees, 4)


def key_of(f: dict) -> str:
    from .scan import flag_key
    return "|".join(str(x) for x in flag_key(f) if x)


# ---------- message ----------
def _plain(side: str, desc: str) -> str:
    return f"Buy YES: {desc}" if side == "YES" else f"Buy NO: {desc} (you win if it does NOT happen)"


def _sent_ct(now) -> str:
    return _utc(now).astimezone(CT).strftime("%H:%M:%S CT")


def build_new(f: dict, gate_open: bool, now, template: str | None = None) -> dict:
    live = gate_open and not config.PAPER_MODE
    game = str(f.get("event") or "").split(":")[0]
    ko = f.get("kickoff_ct") or ""
    if f.get("kind") in STRUCTURAL:
        mc = structural_max_cost(f)
        edge, min_e = float(f["edge_per_set"]), config.MIN_EDGE
        stake, n = float(f.get("paper_stake") or 0), int(f.get("paper_sets") or 0)
        body = (f"{game}\n{f['kind']}: {f['leg1']} ({f.get('leg1_desc')}) @ {f['leg1_price'] * 100:.0f}¢ + "
                f"{f['leg2']} ({f.get('leg2_desc')}) @ {f['leg2_price'] * 100:.0f}¢\n"
                f"Locked edge {edge * 100:.1f}¢/set after fees. Max combined {mc * 100:.0f}¢/set.\n"
                f"Size ${stake:.2f} = {n} sets. Kickoff {ko}")
        mp, price, tick = mc, float(f["leg1_price"]) + float(f["leg2_price"]), f["leg1"].split(" ", 1)[-1]
    else:
        mult = float(f.get("fee_multiplier") or 1.0)
        n = int(f.get("contracts") or 0)
        mp = max_price(float(f["fair"]), n, mult)
        edge, min_e, stake, price, tick = float(f["edge"]), config.MIN_MODEL_EDGE, float(f.get("stake") or 0), float(f["price"]), f.get("ticker")
        body = (f"{game}\n{_plain(f.get('side'), f.get('leg1_desc') or tick)}\n"
                f"Kalshi ask {price * 100:.0f}¢, fair {float(f['fair']) * 100:.0f}¢, edge {edge * 100:.1f}¢ after fees\n"
                f"Max price {'—' if mp is None else f'{mp * 100:.0f}¢'}. Size ${stake:.2f} = {n} contracts. "
                f"Kickoff {ko}")
    body += f"\nsent {_sent_ct(now)}"
    title = f"{'LIVE' if live else 'PAPER'} {f.get('kind')} {f.get('league', '')}".strip()
    return {"type": "new", "key": key_of(f), "kind": f.get("kind"), "title": title, "body": body,
            "priority": 5 if edge >= 2 * min_e else 4, "click": kalshi_url(tick, f.get("event_ticker"), template),
            "ticker": tick, "price": round(price, 4), "max_price": mp, "edge": edge, "stake": stake,
            "quote_utc": f.get("snapshot_ts"), "kickoff_utc": f.get("kickoff_utc"),
            "side": f.get("side"), "legs": [f.get("leg1"), f.get("leg2")] if f.get("kind") in STRUCTURAL else None}


def current_price(alert: dict, quotes: dict) -> float | None:
    """Live price of an alerted contract from {ticker: row}; None if not quoted this pass."""
    if alert.get("legs"):
        tot = 0.0
        for leg in alert["legs"]:
            side, tick = leg.split(" ", 1)
            p = side_price(quotes.get(tick) or {}, side)
            if p is None:
                return None
            tot += p
        return round(tot, 4)
    r = quotes.get(alert.get("ticker"))
    return side_price(r, alert.get("side")) if r else None


def build_gone(alert: dict, now_price: float, now) -> dict:
    return {"type": "gone", "key": alert["key"], "kind": alert.get("kind"),
            "title": f"GONE {alert.get('kind')}: don't chase",
            "body": f"{alert.get('ticker')} now {now_price * 100:.0f}¢, past max "
                    f"{(alert.get('max_price') or 0) * 100:.0f}¢. Skip it.\nsent {_sent_ct(now)}",
            "priority": 3, "click": alert.get("click"), "ticker": alert.get("ticker"), "price": now_price,
            "max_price": alert.get("max_price"), "edge": "", "stake": ""}


# ---------- state ----------
def load_state(path: Path | None = None) -> dict:
    try:
        return json.loads((path or STATE).read_text())
    except Exception:  # noqa: BLE001
        return {"alerts": {}}


def save_state(state: dict, now, path: Path | None = None) -> None:
    path = path or STATE
    cut = _utc(now).timestamp() - 3 * 86400                  # keep 3 days
    state["alerts"] = {k: v for k, v in state.get("alerts", {}).items()
                       if _utc(v.get("sent_utc") or _iso(now)).timestamp() >= cut}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=1, sort_keys=True))


def pushed_stake_today(state: dict, now) -> float:
    day = _utc(now).astimezone(CT).date()
    return sum(float(v.get("stake") or 0) for v in state.get("alerts", {}).values()
               if v.get("sent_utc") and _utc(v["sent_utc"]).astimezone(CT).date() == day)


# ---------- decide ----------
def decide(groups: list[tuple[list[dict], bool]], quotes: dict, state: dict, now,
           lost_today: float = 0.0, template: str | None = None) -> list[dict]:
    """Alerts to send now. `groups` = [(flags, gate_open), ...]. Mutates `state` as if sent."""
    out = []
    alerts = state.setdefault("alerts", {})
    room = config.DAILY_LOSS_CAP - lost_today - pushed_stake_today(state, now)
    cands = [(f, g) for flags, g in groups for f in flags]
    cands.sort(key=lambda x: -float(x[0].get("edge", x[0].get("edge_per_set", 0)) or 0))
    for f, g in cands:
        k = key_of(f)
        if k in alerts:
            continue
        a = build_new(f, g, now, template)
        if f.get("kind") not in STRUCTURAL and a["max_price"] is None:
            continue
        if a["stake"] > room + 1e-9:
            continue                                        # daily cap: skip, don't shrink silently
        room -= a["stake"]
        alerts[k] = {**{x: a[x] for x in ("ticker", "kind", "side", "legs", "max_price", "click", "stake",
                                           "kickoff_utc")}, "key": k, "sent_utc": _iso(now), "gone": False}
        out.append(a)
    for k, a in alerts.items():
        if a.get("gone") or a.get("max_price") is None:
            continue
        ko = a.get("kickoff_utc")
        if ko and _utc(ko) <= _utc(now):
            continue                                        # game started: no more pushes
        p = current_price(a, quotes)
        if p is not None and p > a["max_price"] + 1e-9:
            a["gone"] = True
            out.append(build_gone(a, p, now))
    return out


# ---------- send ----------
def _post(url, **kw):
    import requests
    return requests.post(url, timeout=15, **kw)


def send(alert: dict, post=_post) -> list[str]:
    sent = []
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if topic:
        h = {"Title": alert["title"].encode("utf-8"), "Priority": str(alert["priority"]), "Tags": "football"}
        if alert.get("click"):
            h["Click"] = alert["click"]
            h["Actions"] = f"view, Open Kalshi, {alert['click']}"
        try:
            r = post(f"https://ntfy.sh/{topic}", data=alert["body"].encode("utf-8"), headers=h)
            if getattr(r, "status_code", 200) < 300:
                sent.append("ntfy")
        except Exception as e:  # noqa: BLE001 - never print the topic
            print(f"alerts: ntfy failed ({type(e).__name__})", file=sys.stderr)
    tok, user = os.environ.get("PUSHOVER_TOKEN", "").strip(), os.environ.get("PUSHOVER_USER", "").strip()
    if tok and user:
        try:
            r = post("https://api.pushover.net/1/messages.json", data={
                "token": tok, "user": user, "title": alert["title"], "message": alert["body"],
                "priority": 1 if alert["priority"] >= 4 else 0, "url": alert.get("click") or "",
                "url_title": "Open Kalshi"})
            if getattr(r, "status_code", 200) < 300:
                sent.append("pushover")
        except Exception as e:  # noqa: BLE001
            print(f"alerts: pushover failed ({type(e).__name__})", file=sys.stderr)
    return sent


def log(alert: dict, detected, sent_at, channels: list[str], path: Path | None = None) -> None:
    path = path or LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    q = alert.get("quote_utc")
    row = {**{k: alert.get(k, "") for k in LOG_COLS}, "detected_utc": _iso(detected), "sent_utc": _iso(sent_at),
           "detect_to_send_s": round((_utc(sent_at) - _utc(detected)).total_seconds(), 1),
           "quote_to_send_s": round((_utc(sent_at) - _utc(q)).total_seconds(), 1) if q else "",
           "quote_utc": q or "", "channels": "+".join(channels) or "none"}
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def median_latency(path: Path | None = None) -> float | None:
    path = path or LOG
    try:
        vals = sorted(float(r["detect_to_send_s"]) for r in csv.DictReader(path.open())
                      if r.get("type") == "new" and r.get("detect_to_send_s") not in ("", None))
    except FileNotFoundError:
        return None
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def process(groups, quotes: dict, now=None, lost_today: float = 0.0, post=_post,
            state_path: Path | None = None, log_path: Path | None = None, clock=None) -> list[dict]:
    """Decide, send, log, save. `clock()` gives the send time (defaults to wall clock)."""
    clock = clock or (lambda: datetime.now(UTC))
    detected = _utc(now or clock())
    state = load_state(state_path)
    todo = decide(groups, quotes, state, detected, lost_today)
    for a in todo:
        ch = send(a, post)
        log(a, detected, clock(), ch, log_path)
    save_state(state, detected, state_path)
    return todo


def groups_from_latest(latest: dict, model_gate: bool = False, sharp_gate: bool = False) -> list:
    return [(latest.get("flags") or [], False),
            (latest.get("model_flags") or [], True), (latest.get("model_shadow") or [], False),
            (latest.get("sharp_flags") or [], True), (latest.get("sharp_shadow") or [], False)]


def quotes_of(df) -> dict:
    if df is None or len(df) == 0:
        return {}
    return {r["ticker"]: r for r in df[["ticker", "yes_bid", "yes_ask"]].to_dict("records")}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "verify-url":
        from .scan import latest_snapshot
        df, _ = latest_snapshot()
        if df.empty:
            print("alerts: no snapshot to verify the URL against")
            return 0
        try:
            prev = json.loads(URL_FORMAT.read_text())
            if (datetime.now(UTC) - _utc(prev["checked_utc"])).days < 7:
                print(f"alerts: url template checked {prev['checked_utc']}, verified={prev.get('verified')}")
                return 0
        except Exception:  # noqa: BLE001 - never checked -> check now
            pass
        r = df.iloc[0]
        res = verify_url(r["ticker"], r["event_ticker"])
        res["checked_utc"] = _iso(datetime.now(UTC))
        URL_FORMAT.parent.mkdir(parents=True, exist_ok=True)
        URL_FORMAT.write_text(json.dumps(res, indent=1))
        print(f"alerts: url template {res['template']} verified={res['verified']}")
        return 0
    # batch mode (collect.yml): alert on the latest scan
    from .scan import is_stale, latest_snapshot, lost_today
    try:
        latest = json.loads((DATA / "flags" / "latest.json").read_text())
    except Exception:  # noqa: BLE001
        print("alerts: no latest.json")
        return 0
    if latest.get("stale") or is_stale(latest.get("snapshot_ts")):
        print("alerts: latest scan is stale, nothing sent")
        return 0
    df, _ = latest_snapshot()
    sent = process(groups_from_latest(latest), quotes_of(df), lost_today=lost_today())
    print(f"alerts: {len(sent)} sent ({sum(a['type'] == 'new' for a in sent)} new); "
          f"ntfy={'on' if os.environ.get('NTFY_TOPIC') else 'off'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
