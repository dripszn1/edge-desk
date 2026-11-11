"""Weekly skill-or-luck report (backlog #8): data/weekly_report.json.

For each graded bet set (paper bets, MODEL shadow bets, SHARP shadow bets):
  - record: settled bets, staked, fees, P&L after fees, ROI
  - t-stat of per-bet ROI (pnl / stake)
  - coin-flip test: replay every settled bet 10,000 times as a fair coin weighted by the price
    paid (win prob = price, i.e. the market was right and we had zero edge), fees still charged.
    Rank our real P&L inside that no-skill distribution.
  - edge by week (CT, Monday start): bets, P&L, ROI, claimed vs realised edge per contract
Verdict: "not enough data" under MIN_BETS settled; "skill" only if t >= 2 AND we beat >= 95% of
the coin-flip runs; otherwise "luck / no edge". Only settled bets count (pending/stale excluded),
and structural arbs (locked payout, no variance) are reported but never in the coin-flip test.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .kickoffs import to_ct

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "data" / "paper_bets.csv"
SHADOW = ROOT / "data" / "model_shadow_bets.csv"
SHARP_SHADOW = ROOT / "data" / "sharp_shadow_bets.csv"
OUT = ROOT / "data" / "weekly_report.json"
N_SIMS = 10_000
MIN_BETS = 30
SKILL_T = 2.0
SKILL_PCTL = 0.95
SEED = 7


def t_stat(x: pd.Series) -> float | None:
    x = pd.Series(x).dropna().astype(float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return None
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def coin_flips(price: np.ndarray, contracts: np.ndarray, cost: np.ndarray, n_sims: int = N_SIMS,
               seed: int = SEED) -> np.ndarray:
    """Total P&L of each no-skill replay: each bet pays `contracts` dollars with prob = price."""
    rng = np.random.default_rng(seed)
    wins = rng.random((n_sims, len(price))) < price
    return (wins * contracts - cost).sum(axis=1)


def _week(date_ct: str) -> str:
    d = pd.Timestamp(date_ct)
    return (d - pd.Timedelta(days=d.weekday())).date().isoformat()


def assess(bets: pd.DataFrame, n_sims: int = N_SIMS) -> dict:
    s = bets[bets["status"] == "settled"].copy() if len(bets) else bets
    if s.empty:
        return {"settled": 0, "verdict": "not enough data", "why": "no settled bets yet"}
    s["roi"] = s["pnl"] / s["stake"]
    out = {"settled": int(len(s)), "staked": round(float(s["stake"].sum()), 2),
           "fees": round(float(s["fees"].sum()), 2), "pnl": round(float(s["pnl"].sum()), 2),
           "roi": round(float(s["pnl"].sum() / s["stake"].sum()), 4) if s["stake"].sum() else None,
           "t_stat": None if (t := t_stat(s["roi"])) is None else round(t, 2)}
    flips = s[s["kind"].isin(["MODEL", "SHARP"])] if "kind" in s else s.iloc[0:0]
    if len(flips) and {"price", "contracts"} <= set(flips.columns):
        sims = coin_flips(flips["price"].to_numpy(float), flips["contracts"].to_numpy(float),
                          flips["stake"].to_numpy(float), n_sims)
        real = float(flips["pnl"].sum())
        out["coin_flip"] = {"sims": n_sims, "bets": int(len(flips)), "real_pnl": round(real, 2),
                            "sim_median_pnl": round(float(np.median(sims)), 2),
                            "beat_pct": round(float((sims < real).mean()), 4)}
    s["week"] = s["date_ct"].astype(str).map(_week)
    by = []
    for wk, g in s.groupby("week"):
        row = {"week_of": wk, "bets": int(len(g)), "pnl": round(float(g["pnl"].sum()), 2),
               "roi": round(float(g["pnl"].sum() / g["stake"].sum()), 4) if g["stake"].sum() else None}
        if "claimed_edge" in g and "contracts" in g and g["contracts"].sum():
            row["claimed_edge_c"] = round(float((g["claimed_edge"] * g["contracts"]).sum() / g["contracts"].sum() * 100), 2)
            row["realised_edge_c"] = round(float(g["pnl"].sum() / g["contracts"].sum() * 100), 2)
        by.append(row)
    out["by_week"] = by
    cf = out.get("coin_flip", {}).get("beat_pct")
    t = out["t_stat"]
    if len(s) < MIN_BETS:
        out["verdict"], out["why"] = "not enough data", f"{len(s)} settled bets (< {MIN_BETS})"
    elif t is not None and t >= SKILL_T and cf is not None and cf >= SKILL_PCTL:
        out["verdict"], out["why"] = "skill", f"t={t:.2f}, beat {cf:.0%} of {n_sims:,} coin-flip runs"
    else:
        out["verdict"] = "luck / no edge"
        out["why"] = f"t={t if t is None else round(t, 2)}, coin-flip beat {cf if cf is None else f'{cf:.0%}'}"
    return out


def _load(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


SHARP_TRACKED = ROOT / "data" / "sharp_tracked_bets.csv"


def tracked_report(bets: pd.DataFrame, n_sims: int = N_SIMS) -> dict:
    """Low-confidence SHARP plays: overall, and by why they were held back. A bucket with a real
    record (>= MIN_BETS settled, P&L > 0, t >= SKILL_T) is proposed for widening (Emerson decides)."""
    out = assess(bets, n_sims)
    out["by_reason"], out["propose_widen"] = {}, []
    if bets is None or bets.empty or "reason_bucket" not in bets:
        return out
    for b, g in bets.groupby("reason_bucket"):
        a = assess(g, n_sims)
        a.pop("by_week", None)
        out["by_reason"][b] = a
        if a["settled"] >= MIN_BETS and (a.get("pnl") or 0) > 0 and (a.get("t_stat") or 0) >= SKILL_T:
            out["propose_widen"].append(b)
    return out


def run(now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    rep = {"generated_utc": ts, "generated_ct": to_ct(ts), "paper_mode": config.PAPER_MODE,
           "rules": {"min_bets": MIN_BETS, "skill_t": SKILL_T, "skill_beat_pct": SKILL_PCTL, "sims": N_SIMS},
           "paper": assess(_load(PAPER)), "model_shadow": assess(_load(SHADOW)),
           "sharp_shadow": assess(_load(SHARP_SHADOW)), "sharp_tracked": tracked_report(_load(SHARP_TRACKED))}
    OUT.write_text(json.dumps(rep, indent=2, default=str))
    print(f"weekly: paper {rep['paper']['verdict']} ({rep['paper']['settled']}) | "
          f"model shadow {rep['model_shadow']['verdict']} ({rep['model_shadow']['settled']}) | "
          f"sharp shadow {rep['sharp_shadow']['verdict']} ({rep['sharp_shadow']['settled']}) | "
          f"sharp tracked {rep['sharp_tracked']['verdict']} ({rep['sharp_tracked']['settled']})"
          + (f" | PROPOSE WIDEN: {', '.join(rep['sharp_tracked']['propose_widen'])}" if rep['sharp_tracked'].get('propose_widen') else ""))
    return rep


if __name__ == "__main__":
    run()
