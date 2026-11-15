import base64
import json
from pathlib import Path

import pytest

from edge_desk import config, fees, stream

T0 = 1_794_700_000_000                    # ms
KO = "2026-11-15T18:00:00Z"


def fair(f=0.60, stale=False, pulled_ms=T0 - 60_000, ko=KO):
    return {"fair": f, "stale": stale, "kickoff_utc": ko, "odds_fetched_ts": stream._iso(pulled_ms)}


def eng(now):
    return stream.Engine(now_ms=lambda: now[0], directory=Path("/nonexistent"))


def test_sign_rsa_and_ed25519_verify():
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa
    for key in (rsa.generate_private_key(public_exponent=65537, key_size=2048), ed25519.Ed25519PrivateKey.generate()):
        pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption()).decode()
        k = stream.load_key(pem.replace("\n", "\\n"))           # secrets pasted with literal \n still load
        h = stream.headers("kid", k, "/trade-api/ws/v2?x=1", now_ms=T0)
        assert h["KALSHI-ACCESS-KEY"] == "kid" and h["KALSHI-ACCESS-TIMESTAMP"] == str(T0)
        sig = base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"])
        msg = f"{T0}GET/trade-api/ws/v2".encode()              # query string stripped
        if isinstance(key, rsa.RSAPrivateKey):
            key.public_key().verify(sig, msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                                          salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
        else:
            key.public_key().verify(sig, msg)


def test_parse_ticker_dollars_and_cents():
    m = {"type": "ticker", "msg": {"market_ticker": "X-1", "yes_bid_dollars": "0.4500", "yes_ask_dollars": "0.4700",
                                   "yes_bid_size_fp": "10.00", "yes_ask_size_fp": "5.00", "ts_ms": T0}}
    assert stream.parse_ticker(m) == ("X-1", 0.45, 0.47, 10.0, 5.0, T0)
    m2 = {"type": "ticker", "msg": {"market_ticker": "X-1", "yes_bid": 45, "yes_ask": 47, "ts": T0 // 1000}}
    assert stream.parse_ticker(m2)[1:3] == (0.45, 0.47)
    assert stream.parse_ticker({"type": "subscribed"}) is None


def test_no_side_quotes():
    assert stream.side_quotes("NO", 0.45, 0.47) == (0.53, 0.55)
    assert stream.side_quotes("YES", 0.0, 0.47) == (None, 0.47)


def test_taker_fill_once_with_fee_and_latency():
    now = [T0 + 250]
    e = eng(now)
    e.set_fairs({"X-1|YES": fair(0.60)})
    out = e.on_quote("X-1", 0.48, 0.50, 10, 10, T0)
    t = [r for r in out if r["kind"] == "taker"]
    assert len(t) == 1 and t[0]["price"] == 0.50 and t[0]["latency_ms"] == 250
    n = t[0]["contracts"]
    assert n == int(config.MAX_BET // 0.50) and t[0]["fee"] == fees.taker_fee(0.50, n)
    assert t[0]["edge_c"] == round(100 * (0.60 - 0.50 - fees.taker_fee(0.50, n) / n), 2)
    assert not [r for r in e.on_quote("X-1", 0.48, 0.49, 10, 10, T0 + 1) if r["kind"] == "taker"]   # once per contract


def test_no_fill_when_stale_old_pull_after_kickoff_thin_edge_or_pricey():
    now = [T0]
    for f, q in ((fair(stale=True), (0.48, 0.50)), (fair(pulled_ms=T0 - 2 * 3600_000), (0.48, 0.50)),
                 (fair(ko=stream._iso(T0 - 1)), (0.48, 0.50)), (fair(0.52), (0.48, 0.50)),
                 (fair(0.99), (0.90, 0.92))):
        e = eng(now)
        e.set_fairs({"X-1|YES": f})
        assert not e.on_quote("X-1", *q, 10, 10, T0)


def test_maker_quote_fills_only_on_later_update_before_kickoff():
    now = [T0]
    e = eng(now)
    e.set_fairs({"X-1|YES": fair(0.60)})
    e.on_quote("X-1", 0.46, 0.50, 10, 10, T0)                   # taker at 50 + maker quote at 47
    assert "X-1|YES" in e.resting and e.resting["X-1|YES"]["limit"] == 0.47
    out = e.on_quote("X-1", 0.45, 0.47, 10, 10, T0)            # same ms as the decision: no fill
    assert not [r for r in out if r["kind"] == "maker_fill"]
    now[0] = T0 + 5
    out = e.on_quote("X-1", 0.45, 0.47, 10, 10, T0 + 5)
    mf = [r for r in out if r["kind"] == "maker_fill"]
    assert len(mf) == 1 and mf[0]["price"] == 0.47 and mf[0]["fee"] < fees.taker_fee(0.47, mf[0]["contracts"])
    e2 = eng([T0])
    e2.set_fairs({"X-1|YES": fair(0.60, ko=stream._iso(T0 + 10))})
    e2.on_quote("X-1", 0.46, 0.50, 10, 10, T0)
    e2.now_ms = lambda: T0 + 20                                 # after kickoff
    assert not [r for r in e2.on_quote("X-1", 0.40, 0.45, 10, 10, T0 + 20) if r["kind"] == "maker_fill"]


def test_flush_writes_files(tmp_path):
    now = [T0]
    e = stream.Engine(now_ms=lambda: now[0], directory=tmp_path)
    e.set_fairs({"X-1|YES": fair(0.60)})
    e.on_quote("X-1", 0.46, 0.50, 10, 10, T0)
    e.flush()
    files = sorted(p.name for p in tmp_path.iterdir())
    assert any(f.startswith("fills_") for f in files) and any(f.startswith("quotes_") for f in files)


def test_session_subscribes_and_feeds_engine():
    now = [T0 + 100]
    e = eng(now)
    e.set_fairs({"X-1|YES": fair(0.60)})
    sent = []
    msgs = [json.dumps({"type": "ticker", "msg": {"market_ticker": "X-1", "yes_bid_dollars": "0.46",
                                                   "yes_ask_dollars": "0.50", "ts_ms": T0}})]

    class WS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def send(self, s):
            sent.append(json.loads(s))

        def recv(self, timeout=None):
            if msgs:
                return msgs.pop(0)
            st.stop()
            raise TimeoutError

    from cryptography.hazmat.primitives.asymmetric import ed25519
    st = stream.Stream(e, "kid", ed25519.Ed25519PrivateKey.generate(), connect=lambda url, **kw: WS())
    st._session(*stream.ENDPOINTS[0])
    assert sent[0]["cmd"] == "subscribe" and sent[0]["params"]["channels"] == ["ticker"]
    assert sent[0]["params"]["market_tickers"] == ["X-1"]
    assert e.stats["taker_fills"] == 1


def test_stream_never_touches_orders_or_balances():
    src = Path(stream.__file__).read_text().lower()
    for bad in ("/orders", "portfolio/", "/balance", "create_order", "requests.post", '"post"', "'post'"):
        assert bad not in src, bad


def test_start_from_env_off_without_secrets(monkeypatch):
    monkeypatch.delenv("KALSHI_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY", raising=False)
    assert stream.start_from_env() is None


def test_load_key_survives_bad_pastes():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    for fmt in (serialization.PrivateFormat.PKCS8, serialization.PrivateFormat.TraditionalOpenSSL):
        pem = key.private_bytes(serialization.Encoding.PEM, fmt, serialization.NoEncryption()).decode()
        body = "".join(l for l in pem.splitlines() if not l.startswith("-----"))
        for pasted in (pem, pem.replace("\n", " "), pem.replace("\n", "\r\n"), "  " + pem + "  \n",
                       pem.replace("\n", ""), body, '"' + pem + '"'):
            k = stream.load_key(pasted)
            assert k.private_numbers() == key.private_numbers()
