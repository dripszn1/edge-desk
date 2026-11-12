"""Fetch Kalshi settlement results for every ticker a paper flag touched.

Runs inside GitHub Actions (read-only, no API key). The collector only pulls OPEN events,
so settled markets drop out of snapshots; this asks for them by ticker instead.
Output: appends to data/results/results.csv (fetched_ts, ticker, status, result), one row per
ticker per fetch, until the ticker is final.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .collect import _get

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ROOT / "data" / "flags" / "flags.csv"
RESULTS = ROOT / "data" / "results" / "results.csv"
# single-leg MODEL flags (live + shadow) get graded too, so the model's real-price record builds up
MODEL_FLAGS = (ROOT / "data" / "flags" / "model_flags.csv", ROOT / "data" / "flags" / "model_shadow.csv",
               ROOT / "data" / "flags" / "sharp_flags.csv", ROOT / "data" / "flags" / "sharp_shadow.csv",
               # held-back SHARP plays (#21b/#24) are graded too; they carry ticker + side, not leg1
               ROOT / "data" / "flags" / "sharp_tracked.csv")
FINAL_STATUSES = {"finalized", "settled", "determined"}
FINAL_RESULTS = {"yes", "no", "void"}   # void only counts once status is final


def flag_tickers(flags: pd.DataFrame) -> set[str]:
    out = set()
    for col in ("leg1", "leg2"):
        if col in flags:
            out |= {s.split(" ", 1)[1] for s in flags[col].dropna() if " " in s}
    if "ticker" in flags:
        out |= {t for t in flags["ticker"].dropna().astype(str) if t and t != "nan"}
    return out


def final_tickers(results: pd.DataFrame) -> set[str]:
    if results.empty:
        return set()
    f = results[results["status"].isin(FINAL_STATUSES) & results["result"].isin(FINAL_RESULTS)]
    return set(f["ticker"])


def run(now: datetime | None = None) -> pd.DataFrame:
    files = [f for f in (FLAGS, *MODEL_FLAGS) if f.exists()]
    if not files:
        print("no flags yet; nothing to settle")
        return pd.DataFrame()
    flags = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    old = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame(columns=["ticker"])
    pending = sorted(flag_tickers(flags) - final_tickers(old))
    ts = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for i in range(0, len(pending), 50):
        batch = pending[i:i + 50]
        try:
            j = _get("/markets", {"tickers": ",".join(batch), "limit": 100})
            for m in j.get("markets", []):
                rows.append({"fetched_ts": ts, "ticker": m.get("ticker"),
                             "status": m.get("status"), "result": m.get("result") or None})
        except Exception as e:
            print("  ERROR", e)
    new = pd.DataFrame(rows, columns=["fetched_ts", "ticker", "status", "result"])
    if len(new):
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
    print(f"{ts}: {len(pending)} pending tickers, {len(new)} fetched, "
          f"{len(final_tickers(new))} newly final")
    return new


if __name__ == "__main__":
    run()
