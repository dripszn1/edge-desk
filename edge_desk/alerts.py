"""Instant phone alerts (backlog #17) + speed log (#19) + Kalshi links (#18). Read-only: an alert
is a heads-up for Emerson to look; nothing here places or prepares an order.

Channels: ntfy.sh (secret NTFY_TOPIC), optional Pushover (PUSHOVER_TOKEN + PUSHOVER_USER).
Missing secrets -> nothing is sent, the run stays green. The topic is a secret: it is never
printed or written to data/.

Per contract (signal + leg[s]) at most ONE "new" push, plus at most ONE "gone, don't chase" push
when its live price moves past the max price, or (#35) when a FRESH sharp pull moves the fair value so
the current price no longer clears fee + MIN_MODEL_EDGE ("line moved"). A stale sharp line, or the
same pull the alert was raised on, never fires a line-moved push. New pushes respect the daily loss cap: the stakes
already pushed today (CT) plus this one must fit in DAILY_LOSS_CAP - lost_today.

Title: "LIVE" only when that signal's gate is open AND config.PAPER_MODE is off; else "PAPER".
Priority: high (4); 5 (max) when the edge is >= 2x the signal's minimum edge.
State: data/alerts/state.json. Speed log: data/alerts_log.csv
(detected_utc = when this process saw the flag, quote_utc = the Kalshi snapshot time, sent_utc).
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
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
# Kalshi's real web format is /markets/{series}/{any-slug}/{event} (verified 2026-11-10 from live
# kalshi.com pages, e.g. /markets/kxnflspread/pro-football-spread/kxnflspread-26oct25kcsea and
# /markets/kxnhlspread/event/kxnhlspread-26oct30seacgy: the middle slug is free text).
# The series-only page errors on Kalshi, so it is never used as a fallback.
URL_TEMPLATES = ["https://kalshi.com/markets/{series}/event/{event}"]
FALLBACK_TEMPLATE = URL_TEMPLATES[0]  # never the series-only page (it errors)


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
        t = j.get("template") or URL_TEMPLATES[0]
        return t if "{event}" in t else URL_TEMPLATES[0]
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
                return {"template": t, "verified": True, "inconclusive": False, "tried": tried}
        except Exception as e:  # noqa: BLE001
            tried.append({"url": u, "error": type(e).__name__})
    # 429 / 5xx / network errors say nothing about the URL format -> retry within hours, not a week
    inconclusive = not any(x.get("status") in (200, 404, 410) for x in tried)
    return {"template": FALLBACK_TEMPLATE, "verified": False, "inconclusive": inconclusive, "tried": tried}


URL_RECHECK_HOURS = 7 * 24
URL_RETRY_HOURS = 6


def url_check_due(prev: dict, now) -> bool:
    try:
        age_h = (_utc(now) - _utc(prev["checked_utc"])).total_seconds() / 3600
    except Exception:  # noqa: BLE001 - never checked
        return True
    if age_h < 0:
        return True
    inc = prev.get("inconclusive")
    if inc is None:                     # files written before this field existed
        inc = not prev.get("verified") and not any(x.get("status") in (200, 404, 410) for x in prev.get("tried", []))
    return age_h >= (URL_RETRY_HOURS if inc else URL_RECHECK_HOURS)


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


def autosell_target(fair, price) -> float | None:
    """#38: Kalshi Auto Sell suggestion = the sharp fair price (rounded down to a cent), if above entry."""
    try:
        fair, price = float(fair), float(price)
    except (TypeError, ValueError):
        return None
    t = math.floor(fair * 100 + 1e-9) / 100
    return t if t >= price + 0.01 and t < 1 else None


def side_bid(row: dict, side: str) -> float | None:
    """What we could SELL `side` at right now: YES -> yes bid, NO -> 1 - yes ask."""
    try:
        return float(row["yes_bid"]) if side == "YES" else round(1 - float(row["yes_ask"]), 4)
    except (TypeError, ValueError, KeyError):
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


def _side_word(side: str | None) -> str:
    return {"YES": "Yes", "NO": "No"}.get(str(side or "").upper(), str(side or ""))


def order_lines(game: str | None, desc: str | None, ticker: str | None) -> str:
    """The two lines you see on Kalshi: event title, then the market (rung) title.
    Falls back to the ticker only when no title is known."""
    game = str(game or "").strip()
    desc = str(desc or "").strip() or str(ticker or "")
    return f"{game}\n{desc}" if game else desc


def _sent_ct(now) -> str:
    return _utc(now).astimezone(CT).strftime("%H:%M:%S CT")


def build_new(f: dict, gate_open: bool, now, template: str | None = None) -> dict:
    live = gate_open and not config.PAPER_MODE and f.get("league") != "CBB"   # CBB: shadow test only (#22)
    game = str(f.get("event") or "")
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
        cap = "" if mp is None else f" (pay up to {mp * 100:.0f}¢)"
        body = (f"{order_lines(game, f.get('leg1_desc'), tick)}\n"
                f"Buy {_side_word(f.get('side'))} at {price * 100:.0f}¢{cap}\n"
                f"Fair {float(f['fair']) * 100:.0f}¢ · edge +{edge * 100:.1f}¢ after fees\n"
                f"${stake:.2f} = {n} contracts · kickoff {ko}")
        tgt = autosell_target(f.get("fair"), price)
        if tgt is not None:
            body += f"\nThen set Auto Sell at {tgt * 100:.0f}¢ (sharp fair), or hold"
    body += f"\nsent {_sent_ct(now)}"
    title = f"{'LIVE' if live else 'PAPER'} {f.get('kind')} {f.get('league', '')}".strip()
    if f.get("market") == "ML":
        title += " ML"                                  # "PAPER SHARP NFL ML" (#23)
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


def build_gone(alert: dict, now_price: float, now, row: dict | None = None, max_now: float | None = None) -> dict:
    """`max_now` (#50): the max price from the current fresh sharp fair, when there is one."""
    tick = alert.get("ticker") or ""
    row = row or {}
    mx = max_now if max_now is not None else alert.get("max_price")
    # titles as Kalshi shows them; old state rows lack them, so fall back to the live quote row
    game = alert.get("game") or row.get("event_title")
    desc = alert.get("desc") or row.get("yes_sub_title")
    side = _side_word(alert.get("side"))
    return {"type": "gone", "key": alert["key"], "kind": alert.get("kind"),
            "title": f"GONE {alert.get('kind')}: don't chase",
            "body": f"{order_lines(game, desc, tick)}\n{side} is now {now_price * 100:.0f}¢, past your "
                    f"{(mx or 0) * 100:.0f}¢ max. Skip it.\nsent {_sent_ct(now)}",
            # rebuild the link: old state rows hold the broken series-only URL
            "priority": 3, "click": kalshi_url(tick, alert.get("event_ticker")), "ticker": tick, "price": now_price,
            "max_price": mx, "edge": "", "stake": "",
            "quote_utc": _iso(row["snapshot_ts"]) if row.get("snapshot_ts") else None}


def _gone_keys_logged(path: Path | None = None) -> set:
    """Keys that already got a 'gone' push (any runner). Guards against duplicate pushes when the
    live loop and collect.yml overlap with different copies of state.json."""
    try:
        import csv
        with open(path or LOG, newline="") as fh:
            return {r["key"] for r in csv.DictReader(fh) if r.get("type") == "gone"}
    except Exception:  # noqa: BLE001
        return set()


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
    # Emerson 11-11: PAPER pushes are not capped and don't use up the cap; only LIVE (real-money) ones count
    return sum(float(v.get("stake") or 0) for v in state.get("alerts", {}).values()
               if v.get("sent_utc") and not v.get("paper") and _utc(v["sent_utc"]).astimezone(CT).date() == day)


# ---------- decide ----------
def fair_key(ticker, side) -> str:
    return f"{ticker}|{side}"


def _contracts(alert: dict, p) -> int:
    n = alert.get("contracts")
    if not n:
        try:
            n = max(int(round(float(alert.get("stake") or 0) / float(alert.get("price") or p))), 1)
        except (TypeError, ValueError, ZeroDivisionError):
            n = 1
    return max(int(n), 1)


def _fresh_pull(alert: dict, fe: dict | None, quote_ts=None) -> bool:
    """A usable fair entry: SHARP single-leg alert, not stale, a pull newer than the alert's own, and
    never stamped after the quote it is checked against (no look-ahead)."""
    if not fe or fe.get("stale") or fe.get("fair") is None or not fe.get("odds_fetched_ts"):
        return False
    if alert.get("legs") or alert.get("kind") != "SHARP":
        return False
    try:
        pull = _utc(fe["odds_fetched_ts"])
        ref = alert.get("odds_fetched_ts") or alert.get("quote_utc") or alert.get("sent_utc")
        if ref and pull <= _utc(ref):
            return False
        if fe.get("snapshot_ts") and alert.get("quote_utc") and _utc(fe["snapshot_ts"]) < _utc(alert["quote_utc"]):
            return False
        ref_q = quote_ts or fe.get("snapshot_ts")
        if ref_q and pull > _utc(ref_q):
            return False
    except Exception:  # noqa: BLE001 - unreadable time: not usable
        return False
    return True


def note_fair(alert: dict, fe: dict | None, quote_ts=None) -> None:
    """#50: remember the newest fresh fair for this alert, so later checks (incl. the fast lane, which
    gets no fair values) price against it while it stays fresh."""
    if not _fresh_pull(alert, fe, quote_ts):
        return
    old = alert.get("fair_now_pull")
    if old and _utc(old) >= _utc(fe["odds_fetched_ts"]):
        return
    alert.update(fair_now=float(fe["fair"]), fair_now_pull=_iso(fe["odds_fetched_ts"]),
                 fair_now_line=fe.get("sharp_line"))


def _stored_fe(alert: dict) -> dict:
    return {"fair": alert.get("fair_now"), "odds_fetched_ts": alert.get("fair_now_pull"),
            "sharp_line": alert.get("fair_now_line")}


def current_fair(alert: dict, quote_ts) -> float | None:
    """The stored fresh fair if its pull is still fresh at `quote_ts` (90 min; 60 within 3h of
    kickoff) and not after it; else None (the alert's own max applies)."""
    f, pull = alert.get("fair_now"), alert.get("fair_now_pull")
    if f is None or not pull or not quote_ts:
        return None
    from . import sharp
    try:
        if _utc(pull) > _utc(quote_ts):
            return None
    except Exception:  # noqa: BLE001
        return None
    if sharp.odds_too_old(_iso(quote_ts), pull, alert.get("kickoff_utc")):
        return None
    return float(f)


def line_moved(alert: dict, fe: dict | None, price: float | None, quote_ts=None) -> float | None:
    """#35: the new fair value if a FRESH sharp pull moved it so `price` no longer clears
    fee + MIN_MODEL_EDGE; else None. `fe` = this scan's fair entry for the alerted side."""
    if not fe or fe.get("stale") or fe.get("fair") is None or not fe.get("odds_fetched_ts"):
        return None                                     # never on a stale (>90/60 min) line
    if alert.get("legs") or alert.get("kind") != "SHARP":
        return None
    try:
        new_pull = _utc(fe["odds_fetched_ts"])
        ref = alert.get("odds_fetched_ts") or alert.get("quote_utc") or alert.get("sent_utc")
        if ref and new_pull <= _utc(ref):
            return None                                 # same (or older) pull the alert was raised on
        if fe.get("snapshot_ts") and alert.get("quote_utc") and _utc(fe["snapshot_ts"]) < _utc(alert["quote_utc"]):
            return None
    except Exception:  # noqa: BLE001 - unreadable time: can't say it moved
        return None
    p = price if price is not None else fe.get("price")
    p = p if p is not None else alert.get("price")
    if p is None:
        return None
    fair = float(fe["fair"])
    n = alert.get("contracts")
    if not n:
        try:
            n = max(int(round(float(alert.get("stake") or 0) / float(alert.get("price") or p))), 1)
        except (TypeError, ValueError, ZeroDivisionError):
            n = 1
    mp = max_price(fair, int(n), float(alert.get("fee_multiplier") or 1.0))
    return fair if (mp is None or float(p) > mp + 1e-9) else None


def build_line_gone(alert: dict, fair: float, price: float | None, now, fe: dict | None = None,
                    row: dict | None = None) -> dict:
    tick = alert.get("ticker") or ""
    row, fe = row or {}, fe or {}
    game = alert.get("game") or row.get("event_title")
    desc = alert.get("desc") or row.get("yes_sub_title")
    side = _side_word(alert.get("side"))
    was = f" (was {float(alert['fair']) * 100:.0f}¢)" if alert.get("fair") is not None else ""
    line = ""
    if fe.get("sharp_line") is not None and alert.get("sharp_line") is not None:
        line = f" Sharp line {float(alert['sharp_line']):g} → {float(fe['sharp_line']):g}."
    at = f" {side} at {price * 100:.0f}¢ no longer clears fees + edge." if price is not None else ""
    q = row.get("snapshot_ts") or fe.get("snapshot_ts")
    return {"type": "gone", "key": alert["key"], "kind": alert.get("kind"),
            "title": f"GONE {alert.get('kind')}: line moved, don't chase",
            "body": f"{order_lines(game, desc, tick)}\nLine moved: fair now {fair * 100:.0f}¢{was}.{line}{at} "
                    f"Skip it.\nsent {_sent_ct(now)}",
            "priority": 3, "click": kalshi_url(tick, alert.get("event_ticker")), "ticker": tick,
            "price": price if price is not None else "", "max_price": alert.get("max_price"), "edge": "",
            "stake": "", "quote_utc": _iso(q) if q else None, "reason": "line_moved"}


def decide(groups: list[tuple[list[dict], bool]], quotes: dict, state: dict, now,
           lost_today: float = 0.0, template: str | None = None, fairs: dict | None = None) -> list[dict]:
    """Alerts to send now. `groups` = [(flags, gate_open), ...]. Mutates `state` as if sent.
    `fairs` = {"TICKER|SIDE": {fair, odds_fetched_ts, snapshot_ts, stale, price, sharp_line}} from
    this scan's SHARP pricing (#35, line-moved gone pushes)."""
    fairs = fairs or {}
    out = []
    alerts = state.setdefault("alerts", {})
    room = config.DAILY_LOSS_CAP - lost_today - pushed_stake_today(state, now)
    paper_room = room                                    # only for labelling paper pushes over the cap
    cands = [(f, g) for flags, g in groups for f in flags]
    # open gates first, then structural/SHARP before MODEL (whose backtest is negative), then edge,
    # so the daily cap is spent on the most credible flags
    rank = {"LADDER": 0, "EXCLUSIVE": 0, "COVER": 0, "SHARP": 1, "MODEL": 2}
    cands.sort(key=lambda x: (not x[1], rank.get(x[0].get("kind"), 3),
                              -float(x[0].get("edge", x[0].get("edge_per_set", 0)) or 0)))
    for f, g in cands:
        if f.get("kind") == "MODEL":
            continue                                        # failed its backtest: logged, never pushed
        k = key_of(f)
        if k in alerts:
            continue
        a = build_new(f, g, now, template)
        if f.get("kind") not in STRUCTURAL and a["max_price"] is None:
            continue
        paper = str(a.get("title", "")).startswith("PAPER")
        if paper:
            # Emerson 11-11 ("load as much fake money as you need so it can test"): every qualified PAPER
            # play is pushed; one over the real-money daily cap is labelled, never skipped. LIVE keeps the cap.
            if a["stake"] > paper_room + 1e-9:
                a["body"] = a["body"].replace("\nsent ", f"\nOver the ${config.DAILY_LOSS_CAP:.0f}/day real-money cap (paper only)\nsent ", 1)
            paper_room -= a["stake"]
        else:
            if a["stake"] > room + 1e-9:
                continue                                    # daily cap: skip, don't shrink silently
            room -= a["stake"]
        alerts[k] = {**{x: a[x] for x in ("ticker", "kind", "side", "legs", "max_price", "click", "stake",
                                           "kickoff_utc")}, "key": k, "sent_utc": _iso(now), "gone": False,
                     "desc": f.get("leg1_desc") or "", "game": f.get("event") or "",
                     "event_ticker": f.get("event_ticker"), "quote_utc": f.get("snapshot_ts"),
                     "price": a["price"], "fair": f.get("fair"), "contracts": f.get("contracts"),
                     "fee_multiplier": f.get("fee_multiplier"), "odds_fetched_ts": f.get("odds_fetched_ts"),
                     "sharp_line": f.get("sharp_line"), "paper": paper,
                     "autosell": autosell_target(f.get("fair"), a["price"]) if f.get("kind") not in STRUCTURAL else None}
        out.append(a)
    already_gone = _gone_keys_logged()
    for k, a in alerts.items():
        if a.get("gone") or a.get("max_price") is None or k in already_gone:
            a["gone"] = a.get("gone") or k in already_gone
            continue
        ko = a.get("kickoff_utc")
        if ko and _utc(ko) <= _utc(now):
            continue                                        # game started: no more pushes
        row = quotes.get(a.get("ticker")) or {}
        qts, flag_ts = row.get("snapshot_ts"), a.get("quote_utc")
        if qts and flag_ts and _utc(qts) < _utc(flag_ts):
            continue                                        # quote older than the flag's: can't say it moved
        p = current_price(a, quotes)
        fe = fairs.get(fair_key(a.get("ticker"), a.get("side")))
        note_fair(a, fe, qts)                               # #50: keep the freshest usable fair
        cur = current_fair(a, qts or _iso(now))
        if p is not None and cur is not None:
            mp = max_price(cur, _contracts(a, p), float(a.get("fee_multiplier") or 1.0))
            if mp is None or p > mp + 1e-9:
                a["gone"] = True
                if p > a["max_price"] + 1e-9:
                    out.append(build_gone(a, p, now, row, mp))
                else:
                    out.append(build_line_gone(a, cur, p, now, fe if fe else _stored_fe(a), row))
            continue                                        # a fresh fair decides; old max not used
        if p is not None and p > a["max_price"] + 1e-9:
            a["gone"] = True
            out.append(build_gone(a, p, now, row))
            continue
        new_fair = line_moved(a, fe, p, qts)
        if new_fair is not None:
            a["gone"] = True
            out.append(build_line_gone(a, new_fair, p if p is not None else fe.get("price"), now, fe, row))
    # #38: Auto Sell target reached (pregame or in-game), one push per contract
    for k, a in alerts.items():
        tgt = a.get("autosell")
        if tgt is None or a.get("sell_hit") or a.get("legs"):
            continue
        ko = a.get("kickoff_utc")
        if ko and _utc(now) > _utc(ko) + timedelta(hours=6):
            continue                                        # game long over
        row = quotes.get(a.get("ticker")) or {}
        qts, flag_ts = row.get("snapshot_ts"), a.get("quote_utc")
        if not row or not qts or (flag_ts and _utc(qts) <= _utc(flag_ts)):
            continue                                        # only a quote newer than the alert's
        b = side_bid(row, a.get("side"))
        if b is not None and b >= tgt - 1e-9:
            a["sell_hit"] = True
            out.append(build_sell_hit(a, b, now, row))
    return out


def build_sell_hit(alert: dict, bid: float, now, row: dict | None = None) -> dict:
    row = row or {}
    tick = alert.get("ticker") or ""
    game = alert.get("game") or row.get("event_title")
    desc = alert.get("desc") or row.get("yes_sub_title")
    side = _side_word(alert.get("side"))
    return {"type": "sell", "key": alert["key"], "kind": alert.get("kind"),
            "title": f"AUTO SELL HIT {alert.get('kind')}",
            "body": f"{order_lines(game, desc, tick)}\n{side} bid is {bid * 100:.0f}¢, your Auto Sell at "
                    f"{alert['autosell'] * 100:.0f}¢ should have filled (bought ~{(alert.get('price') or 0) * 100:.0f}¢)."
                    f"\nsent {_sent_ct(now)}",
            "priority": 3, "click": kalshi_url(tick, alert.get("event_ticker")), "ticker": tick, "price": bid,
            "max_price": alert.get("max_price"), "edge": "", "stake": "",
            "quote_utc": _iso(row["snapshot_ts"]) if row.get("snapshot_ts") else None}


# ---------- send ----------
def _post(url, **kw):
    import requests
    return requests.post(url, timeout=15, **kw)


def sport_tag(alert: dict) -> str:
    """#49: ntfy emoji per sport. CBB tickers are KXNCAAMB*; everything else here is football."""
    t = str(alert.get("ticker") or alert.get("key") or "")
    return "basketball" if "KXNCAAMB" in t or alert.get("league") == "CBB" else "football"


def send(alert: dict, post=_post) -> list[str]:
    sent = []
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if topic:
        h = {"Title": alert["title"].encode("utf-8"), "Priority": str(alert["priority"]), "Tags": sport_tag(alert)}
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
            state_path: Path | None = None, log_path: Path | None = None, clock=None,
            fairs: dict | None = None) -> list[dict]:
    """Decide, send, log, save. `clock()` gives the send time (defaults to wall clock)."""
    clock = clock or (lambda: datetime.now(UTC))
    detected = _utc(now or clock())
    state = load_state(state_path)
    todo = decide(groups, quotes, state, detected, lost_today, fairs=fairs)
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
    cols = [c for c in ("ticker", "yes_bid", "yes_ask", "event_title", "yes_sub_title", "snapshot_ts")
            if c in df.columns]
    return {r["ticker"]: r for r in df[cols].to_dict("records")}


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
            if not url_check_due(prev, datetime.now(UTC)):
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
    sent = process(groups_from_latest(latest), quotes_of(df), lost_today=lost_today(),
                   fairs=latest.get("sharp_fairs") or {})
    print(f"alerts: {len(sent)} sent ({sum(a['type'] == 'new' for a in sent)} new); "
          f"ntfy={'on' if os.environ.get('NTFY_TOPIC') else 'off'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
