# edge-desk operations: how to pick this up in a fresh chat

Any Claude chat can continue the work from this repo alone. Read this, then CLAUDE.md (rules + backlog),
then the top of REPORTS.md (latest builder runs). Nothing important lives only in a chat.

## Where the state lives
| What | Where |
|---|---|
| Rules, backlog, what's done | `CLAUDE.md` |
| What each builder run shipped / broke | `REPORTS.md` (newest on top) |
| The contract | `SPEC.md` |
| Live prices | `data/snapshots/<date>/*.parquet`, `data/fastlane/<date>.csv`, `data/candles/<ticker>.csv` |
| Plays | `data/flags/sharp_shadow.csv` (qualified), `data/flags/sharp_tracked.csv` (held back, with reason) |
| Results | `data/sharp_shadow_bets.csv`, `data/sharp_tracked_bets.csv` (hold), `data/sharp_trades.csv` (all exit strategies), `data/clv_bets.csv` (vs Kalshi close) |
| Verdicts | `data/weekly_report.json` (gates, CLV, trades), `data/digest.md` |
| Alerts sent | `data/alerts_log.csv`, state in `data/alerts/` |
| Odds credits | `data/odds/state.json` (`quota` = plan size read from The Odds API headers) |

## How Emerson wants it run
- Plain language, phone-short. He reads on his phone.
- Edge-desk operations (merge PRs, start/restart workflows): say what you're about to do, then do it. Don't ask.
- Approvals that truly need him: open them in the built-in browser pane, ready to click.
- Never place or prepare Kalshi orders. Kalshi API/trading is parked (see CLAUDE.md).
- Everything is PAPER until a strategy passes its gate (>=30 closed, P&L>0, t>=2 by game).
- His trading style (11-11): trust the sharp line, don't predict scores. Ride by default, with Kalshi
  Auto Sell (resting limit sell, take-profit only) at a target. No manual in-game stops (Kalshi overreacts
  in-game, and he can't react fast enough). Every play is graded hold vs sell-at-tip vs Auto Sell vs bracket.

## Routine procedures
**Merge dev -> main.** `git fetch origin main +refs/heads/dev:refs/remotes/origin/dev`, test-merge locally.
Data files conflict often because live keeps committing to main: union append-only logs (odds lines, results,
sharp_tracked, collector_log), keep main's copy of state files (latest.json, event_map, fair_latest, digest).
Run `python -m pytest -q`, push the merge to `dev` (pushing straight to main is blocked), then merge the open
PR in the browser pane (github.com/dripszn1/edge-desk/pulls -> Merge status -> Merge -> Confirm).

**Restart live mode on new code** (a running job keeps the code it started with): Actions -> live-game-day ->
open the in-progress run -> Cancel workflow -> wait ~30 s -> Run workflow (main, chained=false).

**Builder lock.** The 3-hourly builder skips a backlog item marked "IN PROGRESS in Emerson's chat session",
and stands down if dev has a non-"Report" commit <15 min old.

## Scheduled tasks (Claude, not GitHub)
- edge-desk builder: every 3h at :43 CT, works the backlog on `dev`, opens a PR.
- edge-desk game-day digest: 11:17 AM / 5:17 PM / 11:17 PM CT, relays data/digest.md as its own chat.
- edge-desk live kicker: 10:05, 11:05, 18:05 CT, dispatches live/collect if missing.
- Daily check-in: 11:30 AM CT in Emerson's main edge-desk chat (each one schedules the next).
GitHub's own cron is unreliable here; keeper.yml dispatches collect/live.

## Open decisions (as of 11-12)
- Distance caps (NFL 10.5 / CFB 6 / CBB 3 pts from the sharp line): evidence-driven review is backlog item; the weekly report proposes widen/tighten, Emerson decides. Don't make him re-raise it.
- Odds plan: free (500 credits). $30 START 20K trial decision pending; code auto-detects the plan.
  Trial rule: keep only if CLV by game t>=2 and paper P&L beats $30/mo; judge ~1 month in.
- Which exit strategy wins (hold / sell at tip / Auto Sell / bracket): data decides after ~30 closed plays.
- Unverified in Actions: Kalshi 1-min candlesticks endpoint (no key?) and GET /markets?tickers= (fast lane).
