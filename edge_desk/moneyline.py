"""SHARP moneylines (backlog #23): Kalshi game-winner markets (KX*GAME) vs the de-vigged sharp
h2h price. Read-only; shadow/paper only unless the ML subset of a league earns its own gate.

fair(YES team wins) = de-vigged sharp win probability (Pinnacle, else sharp consensus; the same
look-ahead / stale / in-play guards as every SHARP price via odds.fair_lines).
NFL ties: a sportsbook 2-way moneyline refunds a tie, so its price is P(win | no tie); a Kalshi
game-winner contract LOSES on a tie. So fair YES = P(win | no tie) * (1 - NFL_TIE_P) and
fair NO = 1 - fair YES (NO wins on a tie). College games can't tie.
Same rules as SHARP: fee on every contract at order size (price_side), <= MAX_FLAG_PRICE, edge >=
MIN_MODEL_EDGE, CBB liquidity filter, stale-line guard and sanity cap (those plays are tracked,
never alerted), one flag per game, shared daily cap.
"""
from __future__ import annotations

import pandas as pd

from . import config, odds, sharp
from .model import price_side
from .kickoffs import to_ct

# nflverse 2012-2025 regular + post season: 14 ties in 3,965 games
NFL_TIE_P = 0.0035
SERIES = {"NFL": "KXNFLGAME", "CFB": "KXNCAAFGAME", "CBB": "KXNCAAMBGAME"}


def team_abbr(ticker: str) -> str | None:
    t = str(ticker or "")
    return t.rsplit("-", 1)[1] if t.count("-") >= 2 else None


def fair_yes(p_win_no_tie: float, league: str) -> float:
    return p_win_no_tie * (1 - NFL_TIE_P) if league == "NFL" else p_win_no_tie


def ml_flags(df: pd.DataFrame, lines: pd.DataFrame, snapshot_ts: str | None, lost_today: float = 0.0,
             now=None, league: str = "NFL") -> tuple[list[dict], dict]:
    from .scan import is_stale
    status = {"ok": False, "reason": "", "events_priced": 0, "events_no_sharp_line": 0}
    if df is None or df.empty or is_stale(snapshot_ts, now):
        status["reason"] = "stale or empty snapshot"
        return [], status
    fair = odds.fair_lines(lines, snapshot_ts)
    fair = fair[(fair["league"] == league) & (fair["market"] == "h2h")] if len(fair) else fair
    if fair.empty:
        status["reason"] = "no usable sharp h2h odds"
        return [], status
    status.update(ok=True, reason="ok", odds_fetched=str(fair["fetched_ts"].max()))
    q = df[(df["league"] == league) & (df["series"] == SERIES[league]) & (df["status"] == "active")
           & (df.get("phase", "unknown") == "pregame")].dropna(subset=["yes_bid", "yes_ask"])
    ev_vol = (df[df["league"] == league].groupby("event_ticker")["volume_24h"].sum()
              if "volume_24h" in df else pd.Series(dtype=float))
    best, seen, fairs = {}, {}, {}
    status["_fairs"] = fairs                            # #35: popped by scan.run_sharp
    for row in q.to_dict("records"):
        ev = row["event_ticker"]
        if league == "CBB" and float(ev_vol.get(ev, 0) or 0) < config.CBB_MIN_EVENT_VOL_24H:
            status["low_volume"] = status.get("low_volume", 0) + 1
            continue
        if ev not in seen:
            seen[ev] = sharp.match_event(league, row, fair)
        sides, fe = seen[ev]
        if fe.empty or not sides:
            continue
        team = sides.get(team_abbr(row["ticker"]))
        r = fe[fe["outcome"] == team] if team else fe.iloc[0:0]
        if len(r) != 1:
            continue
        p_yes = fair_yes(float(r["fair"].iloc[0]), league)
        why = []
        if sharp.odds_too_old(row["snapshot_ts"], fe["fetched_ts"].max(), row.get("kickoff_utc")):
            status["odds_too_old"] = status.get("odds_too_old", 0) + 1
            why.append("sharp line stale")
        mult = float(row.get("fee_multiplier") or 1.0)
        stale_line = "sharp line stale" in why
        for side, price, size, p in (("YES", row["yes_ask"], row.get("yes_ask_size") or 0, p_yes),
                                     ("NO", round(1 - row["yes_bid"], 4), row.get("yes_bid_size") or 0, 1 - p_yes)):
            fairs[f"{row['ticker']}|{side}"] = sharp.fair_entry(row, p, price, fe, None, stale_line)   # #35
            if league == "CBB" and (size or 0) < config.CBB_MIN_SIZE:
                continue
            s = price_side(p, price, size, mult, config.DAILY_LOSS_CAP - lost_today)
            if not s:
                continue
            f = {"snapshot_ts": row["snapshot_ts"], "league": league, "kind": "SHARP", "market": "ML",
                 "event": row["event_title"], "event_ticker": ev, "ticker": row["ticker"],
                 "leg1": f"{side} {row['ticker']}", "leg1_desc": row.get("yes_sub_title"),
                 "side": side, "price": price, "fair": round(p, 4), **s, "size_available": size,
                 "fee_multiplier": mult, "sharp_line": None, "sharp_source": str(r["source"].iloc[0]),
                 "odds_fetched_ts": str(fe["fetched_ts"].max()), "kickoff_utc": row.get("kickoff_utc"),
                 "kickoff_ct": row.get("kickoff_ct") or to_ct(row.get("kickoff_utc"))}
            reasons = list(why)
            if f["edge"] > sharp.EDGE_SANITY_CAP:
                status["rejected_edge_cap"] = status.get("rejected_edge_cap", 0) + 1
                reasons.append("edge above sanity cap")
            if reasons == ["sharp line stale"]:              # #28: a fresh line could make it a play
                status["stale_only"] = status.get("stale_only", 0) + 1
            if reasons:
                sharp.log_rejected(f, "; ".join(reasons))
                continue
            if ev not in best or f["edge"] > best[ev]["edge"]:
                best[ev] = f
    status["events_priced"] = sum(1 for v in seen.values() if not v[1].empty)
    status["events_no_sharp_line"] = sum(1 for v in seen.values() if v[1].empty)
    room, out = config.DAILY_LOSS_CAP - lost_today, []
    for f in sorted(best.values(), key=lambda x: -x["edge"]):
        if f["stake"] > room:
            s = price_side(f["fair"], f["price"], f["size_available"], f["fee_multiplier"], room)
            if not s:
                continue
            f.update(s)
        room -= f["stake"]
        out.append(f)
    return out, status
