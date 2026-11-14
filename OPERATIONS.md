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
- Edge-desk operations (start/restart workflows): say what you're about to do, then do it. Don't ask. Merges are automatic (automerge.yml); no PR links.
- Approvals that truly need him: open them in the built-in browser pane, ready to click.
- Never place or prepare Kalshi orders. Kalshi API/trading is parked (see CLAUDE.md).
- Everything is PAPER until a strategy passes its gate (>=30 closed, P&L>0, t>=2 by game).
- His trading style (11-11): trust the sharp line, don't predict scores. Ride by default, with Kalshi
  Auto Sell (resting limit sell, take-profit only) at a target. No manual in-game stops (Kalshi overreacts
  in-game, and he can't react fast enough). Every play is graded hold vs sell-at-tip vs Auto Sell vs bracket.

## Routine procedures
**Merge dev -> main: automatic since 11-13.** `automerge.yml` runs on every push to dev: merges dev into main,
keeps main's copy of conflicting data/ files, runs the full test suite on the merged tree, pushes main (retries
if live pushed in between). Red tests or a code conflict → no merge; fix on dev. Manual fallback (old way): `git fetch origin main +refs/heads/dev:refs/remotes/origin/dev`, test-merge locally.
Data files conflict often because live keeps committing to main: union append-only logs (odds lines, results,
sharp_tracked, collector_log), keep main's copy of state files (latest.json, event_map, fair_latest, digest).
Run `python -m pytest -q`, push the merge to `dev` (pushing straight to main is blocked), then merge the open
PR in the browser pane (github.com/dripszn1/edge-desk/pulls -> Merge status -> Merge -> Confirm).

**Restart live mode on new code** (a running job keeps the code it started with): Actions -> live-game-day ->
open the in-progress run -> Cancel workflow -> wait ~30 s -> Run workflow (main, chained=false).

**Builder lock.** The 3-hourly builder skips a backlog item marked "IN PROGRESS in Emerson's chat session",
and stands down if dev has a non-"Report" commit <15 min old.

## Where we stand (11-14 1 AM CT, before the big Saturday slate)
- Nothing is proven. Every gate is closed. Scope NFL + CFB only (CBB removed 11-13; out-of-scope alerts get no
  fast lane or follow-up pushes since 11-13 evening).
- Football (one row per contract, #60): 10 closed plays / 6 games. Hold -$19.90 (0-10), sell at tip -$2.43,
  Auto Sell +10c +$5.47 (8 of 10 hit), Auto Sell at fair +$1.91, best bracket (10/10) +$2.94. Entries lose
  ~1.3-1.5c to Kalshi's close. vs random: hold beats 7%, Auto Sell +10c beats 78% (needs 95%).
- Fri 11-13 night games ended after the last save, so not graded yet: HOU/COLO (COLO 42-32) alerted NO Over 48.5
  loses on hold, its Auto Sell (44c) filled; held-back far rungs YES COLO by 2.5/3.5/4.5 all won (~+$27);
  ILL/UCLA (22-14) held-back NO Over 27.5/30.5/33.5 lost, NO Over 36.5 won (~+$1.4). Two games = noise.
- ~186 football plays settle Sat 11-14 / Sun 11-15. Apply the Verdict rules once >=30 closed across >=10 games,
  and report the four exits (hold / sell at tip / Auto Sell / bracket) split NFL vs CFB.
- Open, deliberately undecided: #53 require >=1c room below max price? (about half of alerts go out with 0c room
  and get "gone" within seconds, e.g. MINN/PSU 11-13 7:15 PM). Decide on weekend data.

## Verdict rules (pre-registered 11-13, BEFORE the weekend's football results; don't bend them after)
Read once football has >=30 closed plays across >=10 games (alerted + held back within the caps):
1. **Picks**: football entries vs Kalshi's close after fee, t by game.
   - CLV > 0 and t >= 2 → the sharp-line picks have an edge. Keep going toward the gate.
   - CLV <= 0 and t <= -2 → the picks have no edge on Kalshi. STOP adding SHARP features. Say so plainly.
   - In between → inconclusive: one more week of data, bug fixes only, no new features.
2. **Exits**: an exit counts as working only if it has >=30 closed, P&L > 0, t >= 2 by game AND beats >=95% of the
   random-entry placebo (#57). If placebo Auto Sell does as well as ours, the edge is the exit (Kalshi
   overreacting in-game), not the pick: that is a different, simpler strategy and gets its own plan.
3. **CBB**: resolved 11-13: Emerson removed CBB entirely (picks −2.4c vs close, t −3.7 by game; beat 39% of random).
4. **Odds plan ($30/mo)**: keep after the month only if rule 1 says the picks have an edge. Exits-only (rule 2)
   doesn't need Pinnacle lines.
Every proposal from these rules goes to Emerson; nothing changes live behaviour without his OK.

## What the builder may build (11-13)
Useful = it answers the verdict question, fixes something broken in the data, or Emerson asked for it.
When the backlog is empty the builder does NOT invent features (it added #52-#56 itself on 11-13: reasonable,
but more measurement isn't more edge). It fixes real bugs it finds in the runs/data (with evidence), else
reports "nothing to build" and stops.

## Scheduled tasks (Claude, not GitHub)
- edge-desk builder: every 3h at :43 CT, works the backlog on `dev`, opens a PR.
- edge-desk game-day digest: 11:17 AM / 5:17 PM / 11:17 PM CT, relays data/digest.md as its own chat.
- edge-desk live kicker: 10:05, 11:05, 18:05 CT, dispatches live/collect if missing.
- Daily check-in: ~11:30 AM CT, a FRESH session each day (recurring scheduled task, 11-13): reads this file,
  REPORTS.md, digest and the data, merges any green PR, pushes Emerson a phone-short summary (always incl. the
  four exits: hold / sell at tip / Auto Sell / bracket, per sport), refreshes the dashboard, applies the
  Verdict rules once their thresholds are met. Moved off the long 11-10 chat on purpose (long-context drift).
GitHub's own cron is unreliable here; keeper.yml dispatches collect/live.

## Open decisions (as of 11-13)
- Alerts with 0c room (ask = max price) flip to "gone" on one tick. Decide after the weekend from
  weekly_report.json["alert_room"] (closes, not gone-rate).
- Distance caps (NFL 10.5 / CFB 6 / CBB 3 pts from the sharp line): evidence-driven review is backlog item; the weekly report proposes widen/tighten, Emerson decides. Don't make him re-raise it.
- Odds plan: START 20K ($30/mo) active since 11-12 2:06 PM CT (new key in the secret). Judge per Verdict rule 4.
- Which exit strategy wins (hold / sell at tip / Auto Sell / bracket): data decides after ~30 closed plays.
- Verified in Actions: 1-min candlesticks (no key) and the 3-s fast lane (TNF: 3,347 ticks, 0 throttled).

## Artifacts (built 11-12 in chat)
- Design System "Edge Desk": https://claude.ai/artifact/TP96QAFG92wghWRB9NapGg (slug d5a4f4a5-1e0b-41bd-afc3-1b85c1798f21). Anton headings/numbers, JetBrains Mono body, dark scoreboard palette.
- Dashboard: https://claude.ai/artifact/1xHE6x7nWzTv57vH4DevAR (slug 07b7e82b-edfd-49ee-81e6-20e02063f27e). Data = 5 attached CSVs (records, held_by_reason, exits, clv, gates) built from weekly_report.json / dashboard.json / *_bets.csv / digest.md. Daily check-in: rebuild the CSVs, upload_asset, then db_write `update` each datasets/<id> source.url + updated.at.
- Strategy whiteboard: https://claude.ai/artifact/EEBAKKaKTsggJWTGuRnuyB (slug 6b1dbbb9-1e19-4da5-90fb-27b36dfb28d2).
