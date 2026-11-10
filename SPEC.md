# edge-desk — spec v0.1

Status: **awaiting Emerson's approval.** No code beyond scaffolding until approved.

## The edge in one line
Retail Kalshi bettors price sports contracts off headlines and gut; prices that break their own ladder logic, or drift away from a data-driven projection by more than fees, get corrected before settlement.

## Hard rules (no code can override)
- **Never places orders.** Read-only market data. Emerson places every bet by hand.
- **Paper mode for the first 14 days.** Every flag is logged and graded; no real-money recommendations until the luck test passes.
- Bankroll: **$200**.
- Per-bet cap: **5% ($10)**. Sizing = quarter Kelly, fees included, capped at 5%.
- Daily loss cap: **10% ($20)**. Hit it → no more flags that day.
- Weekly drawdown stop: **25% ($50)** from peak → system pauses, needs a manual restart.
- Combos/parlays are never flagged. Singles only.
- All times shown in **Central (America/Chicago)**; stored in UTC.

## Scope
- Leagues: **NFL and college football (CFB)**. CBB is parked until NFL/CFB show a proven edge.
- Markets: game spreads and totals (point ladders). Moneylines added later.

## Architecture (6 layers)
1. **Data** — GitHub Actions cron (runs on GitHub's servers, since Claude's cloud workspace can't reach Kalshi directly). Pulls Kalshi public market data (events, markets, order books: best bid/ask + size) on a schedule; kickoff times from Kalshi's own `occurrence_datetime` (kickoff + 3h), so no outside schedule feed is needed. Stored as daily Parquet/CSV in `data/`, committed to the repo.
2. **Signal**
   - *Ladder scanner:* within one game, "wins by 7+" can never price above "wins by 3+"; "over 47.5" never above "over 44.5". Flag when a higher rung's best bid > lower rung's best ask, after fees on both legs, with real size.
   - *Model gap (phase 2):* projected margin/total per game (team ratings from public results) → fair probability per rung → flag when fair − ask − fee ≥ 3¢.
3. **Sizing** — EV = p − price − fee; f* = (p − cost)/(1 − cost); bet = ¼ f* × bankroll, capped at $10.
4. **Risk** — the hard rules above, checked on every flag.
5. **Execution** — none. Output is a flag list; Emerson decides.
6. **Monitoring** — dashboard + phone alerts.

## Outputs
- **Pregame alert** (push notification) before each slate: game, contract, price, fair value, edge in ¢, suggested $ size, kickoff in CT.
- **Dashboard** (artifact): today's flags, paper record, P&L after fees, by-league and by-signal splits.
- **Weekly grade** (Monday): skill-or-luck test — t-stat, rank vs 10,000 coin-flip versions, edge by week. Verdict: skill / luck / not enough data.

## Agents (scheduled tasks)
| Agent | When | Job |
|---|---|---|
| Collector | GitHub Actions, every 30 min, 8am–11pm CT | Pull markets + books, commit data |
| Scanner | Before each slate | Run signals, write flags, send alert |
| Grader | Daily, after games | Settle paper flags, update P&L |
| Weekly report | Monday | Skill-or-luck verdict |
| Builder | Overnight | Work the backlog below; commits to a `dev` branch only |

## Tests that must pass before anything goes live
- No data from the future in any signal (timestamp check on every feature).
- Fees applied on every leg of every flag.
- Bet size never exceeds $10; no flags after the daily cap is hit.
- Times render in CT and come from Kalshi's own event clock.
- A stale-data check: if the latest snapshot is >45 min old, no flags.

## Backlog (in order)
1. Collector + GitHub Actions workflow
2. Ladder scanner + paper log
3. Pregame alert
4. Grader + dashboard
5. Team-rating model (NFL first, then CFB)
6. Weekly skill-or-luck report
7. Import Emerson's Kalshi history → audit past bets
