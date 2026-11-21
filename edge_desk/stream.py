"""Live Kalshi price stream + instant paper engine (#66), READ ONLY.

Emerson 11-14 8:15 PM CT: "do what the others do w the apis ... run alone ... almost instant".
The live loop polls Kalshi every ~30 s and then needs a phone tap; this module holds a WebSocket
to Kalshi's `ticker` channel inside the live job, so every best-bid/ask change reaches us in
milliseconds, and paper-trades on it at once against the SHARP fair values the live loop already
computes (sharp.fair_entry, #35). No phone step, no orders:

  - This module never calls an order, portfolio or balance endpoint (a test greps for it).
  - Paper TAKER fill: the first update where our side's ask leaves fair - ask - taker fee
    >= MIN_MODEL_EDGE (ask <= 90c, fair not stale, before kickoff). One per contract per day.
  - Paper MAKER quote: at the same moment a resting buy at bid + 1c (if < ask) is logged; it fills
    at its price on a LATER update whose ask <= the limit, before kickoff (maker fee).
  - Latency: decision time - Kalshi's ts_ms for the quote that triggered it, in ms.

Auth (docs.kalshi.com, API keys): headers KALSHI-ACCESS-KEY / -TIMESTAMP (ms) / -SIGNATURE =
base64(sign(timestamp + "GET" + path)); RSA keys sign with PSS-SHA256 (salt = digest length),
Ed25519 keys sign directly. Secrets: KALSHI_KEY_ID, KALSHI_PRIVATE_KEY (PEM). Missing -> the
stream stays off and the live loop runs exactly as before.

Ticker messages (docs.kalshi.com/websockets/market-ticker): msg.market_ticker, yes_bid_dollars,
yes_ask_dollars (dollar strings), yes_bid_size_fp, yes_ask_size_fp, ts_ms. Older integer-cent
fields (yes_bid / yes_ask) are read too.

Files: data/stream/fills_<date>.csv (paper taker fills + maker quotes/fills),
data/stream/quotes_<date>.csv (every change for contracts we have a fair for, size-capped),
data/stream/status.json (connected, url, updates, fills, latency).
"""
from __future__ import annotations

import base64
import json
import math
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from . import config, fees

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "stream"
# (url, path that is signed); tried in order, the first that connects is kept
ENDPOINTS = [("wss://api.elections.kalshi.com/trade-api/ws/v2", "/trade-api/ws/v2"),
             ("wss://external-api-ws.kalshi.com/", "/")]
MAX_PRICE = 0.90
QUOTE_LOG_MAX_BYTES = 40_000_000          # per day
SUB_CHUNK = 200                           # tickers per subscribe command
FILL_COLS = ["kind", "decided_utc", "quote_ts_ms", "latency_ms", "ticker", "side", "price", "fair",
             "edge_c", "contracts", "fee", "odds_fetched_ts", "kickoff_utc", "limit", "filled_utc"]


# ---------- auth ----------
def load_key(pem: str):
    """Load the private key however it was pasted into the secret: proper PEM, PEM squashed onto
    one line (spaces or literal \\n instead of line breaks), CRLF, or the bare base64 body."""
    import re
    from cryptography.hazmat.primitives import serialization
    raw = pem.replace("\\n", "\n").replace("\r", "").strip().strip('"').strip("'").strip()
    try:
        return serialization.load_pem_private_key(raw.encode(), password=None)
    except ValueError:
        pass
    m = re.search(r"-----BEGIN ([A-Z ]+)-----(.*?)-----END \1-----", raw, re.S)
    labels = [m.group(1)] if m else ["PRIVATE KEY", "RSA PRIVATE KEY", "EC PRIVATE KEY"]
    body = re.sub(r"[^A-Za-z0-9+/=]", "", m.group(2) if m else raw)
    lines = "\n".join(body[i:i + 64] for i in range(0, len(body), 64))
    for lab in labels:
        try:
            return serialization.load_pem_private_key(
                f"-----BEGIN {lab}-----\n{lines}\n-----END {lab}-----\n".encode(), password=None)
        except ValueError:
            continue
    der = base64.b64decode(body + "=" * (-len(body) % 4))
    return serialization.load_der_private_key(der, password=None)


def sign(key, ts_ms: int, method: str, path: str) -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa
    msg = f"{ts_ms}{method}{path.split('?')[0]}".encode()
    if isinstance(key, ed25519.Ed25519PrivateKey):
        sig = key.sign(msg)
    elif isinstance(key, rsa.RSAPrivateKey):
        sig = key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                        salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
    else:
        raise ValueError("unsupported key type")
    return base64.b64encode(sig).decode()


def headers(key_id: str, key, path: str, now_ms: int | None = None) -> dict:
    ts = int(now_ms if now_ms is not None else time.time() * 1000)
    return {"KALSHI-ACCESS-KEY": key_id, "KALSHI-ACCESS-TIMESTAMP": str(ts),
            "KALSHI-ACCESS-SIGNATURE": sign(key, ts, "GET", path)}


# ---------- messages ----------
def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def parse_ticker(m: dict):
    """-> (ticker, yes_bid, yes_ask, bid_size, ask_size, ts_ms) in dollars, or None."""
    if not isinstance(m, dict) or m.get("type") != "ticker":
        return None
    x = m.get("msg") or {}
    t = x.get("market_ticker")
    if not t:
        return None
    yb, ya = _num(x.get("yes_bid_dollars")), _num(x.get("yes_ask_dollars"))
    if yb is None and _num(x.get("yes_bid")) is not None:
        yb = _num(x.get("yes_bid")) / 100
    if ya is None and _num(x.get("yes_ask")) is not None:
        ya = _num(x.get("yes_ask")) / 100
    ts = x.get("ts_ms")
    if ts is None and x.get("ts") is not None:
        ts = int(x["ts"]) * 1000
    return (t, yb, ya, _num(x.get("yes_bid_size_fp")), _num(x.get("yes_ask_size_fp")),
            int(ts) if ts is not None else None)


def side_quotes(side: str, yb, ya):
    """(bid, ask) for buying/selling our side. YES: (yes_bid, yes_ask); NO: (1-yes_ask, 1-yes_bid)."""
    def ok(v):
        return v is not None and 0 < v < 1
    if side == "YES":
        return (yb if ok(yb) else None, ya if ok(ya) else None)
    return (round(1 - ya, 4) if ok(ya) else None, round(1 - yb, 4) if ok(yb) else None)


def _utc_ms(s) -> int | None:
    if not s:
        return None
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        d = d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        return int(d.timestamp() * 1000)
    except ValueError:
        return None


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ---------- paper engine ----------
class Engine:
    """Event-driven paper trader. Thread-safe: the socket thread calls on_quote, the live loop
    calls set_fairs and flush."""

    def __init__(self, now_ms=lambda: int(time.time() * 1000), max_stale_ms: int = 10 * 60_000,
                 directory: Path | None = None):
        self.now_ms = now_ms
        self.max_stale_ms = max_stale_ms
        self.dir = directory or DIR
        self.lock = threading.Lock()
        self.fairs: dict[str, dict] = {}            # "TICKER|SIDE" -> sharp.fair_entry
        self.taken: set[str] = set()                # contracts already paper-taken today
        self.resting: dict[str, dict] = {}          # contract -> open maker quote
        self.rows: list[dict] = []
        self.quotes: list[tuple] = []
        self.fair_log: list[tuple] = []             # #67: YES-side fair changes, for the market-making replay
        self._fair_sig: dict[str, tuple] = {}
        self.latency: deque = deque(maxlen=2000)
        self.stats = {"updates": 0, "priced_updates": 0, "taker_fills": 0, "maker_quotes": 0,
                      "maker_fills": 0}

    def set_fairs(self, fairs: dict) -> None:
        with self.lock:
            self.fairs.update(fairs or {})
            now = self.now_ms()
            for k, f in (fairs or {}).items():       # #67: log each new YES fair once (not far rungs)
                if not k.endswith("|YES") or not f or f.get("far"):
                    continue
                sig = (f.get("fair"), f.get("odds_fetched_ts"), bool(f.get("stale")))
                if self._fair_sig.get(k) != sig:
                    self._fair_sig[k] = sig
                    self.fair_log.append((now, k.split("|", 1)[0], f.get("fair"), f.get("odds_fetched_ts"),
                                          bool(f.get("stale")), f.get("kickoff_utc")))

    def tickers(self) -> set[str]:
        with self.lock:
            return {k.split("|", 1)[0] for k in self.fairs}

    def _usable(self, f: dict, now: int) -> bool:
        if not f or f.get("stale") or f.get("far"):       # same guards as alerts: no far rungs
            return False
        ko = _utc_ms(f.get("kickoff_utc"))
        if ko is not None and now >= ko:
            return False
        pulled = _utc_ms(f.get("odds_fetched_ts"))
        return pulled is not None and now - pulled <= self.max_stale_ms

    def on_quote(self, ticker, yb, ya, bsz, asz, ts_ms) -> list[dict]:
        now = self.now_ms()
        out = []
        with self.lock:
            self.stats["updates"] += 1
            if f"{ticker}|YES" in self.fairs or f"{ticker}|NO" in self.fairs:
                self.quotes.append((now, ts_ms, ticker, yb, ya, bsz, asz))     # once per update (was once per side)
            for side in ("YES", "NO"):
                k = f"{ticker}|{side}"
                f = self.fairs.get(k)
                if f is None:
                    continue
                self.stats["priced_updates"] += 1
                bid, ask = side_quotes(side, yb, ya)
                # a resting maker quote fills on a LATER update whose ask reaches it, before kickoff
                r = self.resting.get(k)
                if r and ask is not None and ask <= r["limit"] + 1e-9 and now > r["decided_ms"]:
                    ko = _utc_ms(r.get("kickoff_utc"))
                    if ko is None or now < ko:
                        n = r["contracts"]
                        row = {**r["row"], "kind": "maker_fill", "filled_utc": _iso(now),
                               "fee": fees._round_up_cent(fees.MAKER_RATE * n * r["limit"] * (1 - r["limit"]))}
                        self.rows.append(row)
                        out.append(row)
                        self.stats["maker_fills"] += 1
                    del self.resting[k]
                if k in self.taken or ask is None or not self._usable(f, now):
                    continue
                fair = float(f["fair"])
                n = max(1, int(config.MAX_BET // ask)) if ask > 0 else 1
                fee = fees.taker_fee(ask, n)
                edge = fair - ask - fee / n
                from .sharp import in_core_price           # core zone (Emerson 11-15)
                if ask > MAX_PRICE or edge < config.MIN_MODEL_EDGE or not in_core_price(ask):
                    continue
                self.taken.add(k)
                lat = (now - ts_ms) if ts_ms is not None else None
                if lat is not None:
                    self.latency.append(lat)
                row = {"kind": "taker", "decided_utc": _iso(now), "quote_ts_ms": ts_ms, "latency_ms": lat,
                       "ticker": ticker, "side": side, "price": ask, "fair": round(fair, 4),
                       "edge_c": round(100 * edge, 2), "contracts": n, "fee": fee,
                       "odds_fetched_ts": f.get("odds_fetched_ts"), "kickoff_utc": f.get("kickoff_utc"),
                       "limit": None, "filled_utc": _iso(now)}
                self.rows.append(row)
                out.append(row)
                self.stats["taker_fills"] += 1
                if bid is not None and round(bid + 0.01, 4) < ask - 1e-9:
                    lim = round(bid + 0.01, 4)
                    q = {**row, "kind": "maker_quote", "price": lim, "limit": lim, "filled_utc": None,
                         "fee": None}
                    self.resting[k] = {"limit": lim, "decided_ms": now, "contracts": n,
                                       "kickoff_utc": f.get("kickoff_utc"), "row": q}
                    self.rows.append(q)
                    self.stats["maker_quotes"] += 1
        return out

    def summary(self) -> dict:
        with self.lock:
            lat = sorted(self.latency)
        med = lat[len(lat) // 2] if lat else None
        return {**self.stats, "median_latency_ms": med, "open_maker_quotes": len(self.resting)}

    def flush(self, now_ms: int | None = None) -> None:
        """Append new rows to today's files (UTC date)."""
        import csv
        now_ms = now_ms or self.now_ms()
        day = datetime.fromtimestamp(now_ms / 1000, timezone.utc).strftime("%Y-%m-%d")
        with self.lock:
            rows, self.rows = self.rows, []
            quotes, self.quotes = self.quotes, []
            flog, self.fair_log = self.fair_log, []
        self.dir.mkdir(parents=True, exist_ok=True)
        if rows:
            p = self.dir / f"fills_{day}.csv"
            new = not p.exists()
            with p.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FILL_COLS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerows(rows)
        if quotes:
            p = self.dir / f"quotes_{day}.csv"
            if not p.exists() or p.stat().st_size < QUOTE_LOG_MAX_BYTES:
                new = not p.exists()
                with p.open("a", newline="") as fh:
                    w = csv.writer(fh)
                    if new:
                        w.writerow(["recv_ms", "ts_ms", "ticker", "yes_bid", "yes_ask", "bid_size", "ask_size"])
                    w.writerows(quotes)
        if flog:
            p = self.dir / f"fairs_{day}.csv"
            if not p.exists() or p.stat().st_size < QUOTE_LOG_MAX_BYTES:
                new = not p.exists()
                with p.open("a", newline="") as fh:
                    w = csv.writer(fh)
                    if new:
                        w.writerow(["logged_ms", "ticker", "fair", "odds_fetched_ts", "stale", "kickoff_utc"])
                    w.writerows(flog)


# ---------- socket thread ----------
class Stream(threading.Thread):
    def __init__(self, engine: Engine, key_id: str, key, connect=None):
        super().__init__(daemon=True, name="kalshi-stream")
        self.engine, self.key_id, self.key = engine, key_id, key
        self.connect = connect
        self.stop_evt = threading.Event()
        self.status = {"connected": False, "url": None, "connects": 0, "errors": [], "subscribed": 0}
        self._subbed: set[str] = set()
        self._msg_id = 0

    def stop(self):
        self.stop_evt.set()

    def _subscribe(self, ws, tickers):
        new = sorted(set(tickers) - self._subbed)
        for i in range(0, len(new), SUB_CHUNK):
            self._msg_id += 1
            ws.send(json.dumps({"id": self._msg_id, "cmd": "subscribe",
                                "params": {"channels": ["ticker"], "market_tickers": new[i:i + SUB_CHUNK]}}))
        self._subbed |= set(new)
        self.status["subscribed"] = len(self._subbed)

    def _session(self, url, path):
        connect = self.connect
        if connect is None:
            from websockets.sync.client import connect
        with connect(url, additional_headers=headers(self.key_id, self.key, path),
                     open_timeout=15, ping_interval=10, ping_timeout=20, max_size=2**22) as ws:
            self.status.update(connected=True, url=url)
            self.status["connects"] += 1
            self._subbed = set()
            last_sub = 0.0
            while not self.stop_evt.is_set():
                if time.monotonic() - last_sub > 5:          # pick up newly priced contracts
                    self._subscribe(ws, self.engine.tickers())
                    last_sub = time.monotonic()
                try:
                    raw = ws.recv(timeout=1)
                except TimeoutError:
                    continue
                try:
                    m = json.loads(raw)
                except ValueError:
                    continue
                if m.get("type") == "error":
                    self.status["errors"] = (self.status["errors"] + [str(m.get("msg"))[:200]])[-5:]
                    continue
                q = parse_ticker(m)
                if q:
                    self.engine.on_quote(*q)

    def run(self):
        backoff, i = 1.0, 0
        while not self.stop_evt.is_set():
            url, path = ENDPOINTS[i % len(ENDPOINTS)]
            try:
                self._session(url, path)
                backoff = 1.0
            except Exception as e:  # noqa: BLE001 - reconnect, then try the other endpoint
                self.status["errors"] = (self.status["errors"] + [f"{url}: {type(e).__name__}: {str(e)[:150]}"])[-5:]
                if not self.status["connected"]:
                    i += 1
            self.status["connected"] = False
            self.stop_evt.wait(backoff)
            backoff = min(backoff * 2, 30)


def start_from_env(engine: Engine | None = None) -> tuple[Engine, Stream] | None:
    kid, pem = os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_PRIVATE_KEY")
    if not kid or not pem:
        return None
    eng = engine or Engine()
    st = Stream(eng, kid.strip(), load_key(pem))
    st.start()
    return eng, st


def write_status(engine: Engine, stream: Stream | None, directory: Path | None = None) -> None:
    d = directory or DIR
    d.mkdir(parents=True, exist_ok=True)
    st = dict(stream.status) if stream else {"connected": False}
    (d / "status.json").write_text(json.dumps({"utc": _iso(int(time.time() * 1000)), "stream": st,
                                               "engine": engine.summary()}, indent=1))
