"""Single source of truth for scope and risk limits. See SPEC.md."""

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Kalshi series tickers in scope (verified against /series?category=Sports, 2026-11-09)
SERIES = {
    "NFL": ["KXNFLSPREAD", "KXNFLTOTAL", "KXNFLGAME"],
    "CFB": ["KXNCAAFSPREAD", "KXNCAAFTOTAL", "KXNCAAFGAME"],
    # CBB removed 2026-11-13 (Emerson: "skip cbb altogether"): its picks lost to Kalshi's close (t -3.7 by
    # game) and beat only 39% of random entries. Old CBB data stays in data/; nothing new is collected.
}

# Risk (SPEC.md "Hard rules")
BANKROLL = 200.00
KELLY_FRACTION = 0.25
MAX_BET = 10.00            # 5% of bankroll
DAILY_LOSS_CAP = 20.00     # 10%
DRAWDOWN_STOP = 50.00      # 25% from peak
PAPER_MODE = True          # first 14 days

# Signal thresholds
MIN_EDGE = 0.01            # $ per contract after fees, for structural (ladder) flags
MIN_MODEL_EDGE = 0.03      # $ per contract after fees, for model flags (phase 2)
MAX_FLAG_PRICE = 0.90      # never flag a contract priced above 90c (Kalshi audit, CLAUDE.md #9)
MIN_SIZE = 5               # contracts available at the flagged prices
STALE_MINUTES = 45         # no flags if latest snapshot is older than this
ESPN_STALE_MINUTES = 90    # kickoff data older than this -> phase unknown -> no flags
PREGAME_MIN_LEAD_MINUTES = 10  # flag only if kickoff is at least this far after the snapshot

TIMEZONE = "America/Chicago"

# Game-day live mode (backlog #16, .github/workflows/live.yml). Cadence/budget settings, not risk limits.
LIVE_SNAPSHOT_SECONDS = 30        # target loop cadence
LIVE_SLOW_SECONDS = 120           # if a full pass is too slow: full pass this often, hot games every 30 s
LIVE_HOT_HOURS = 3                # "hot" = kickoff within this many hours
LIVE_MAX_BACKOFF_SECONDS = 120    # cadence ceiling while Kalshi returns 429
LIVE_PERSIST_MINUTES = 15         # write a full snapshot to data/snapshots this often (collect.yml then skips)
LIVE_COMMIT_MINUTES = 15          # grade/digest + git commit this often
LIVE_ODDS_CHECK_MINUTES = 10      # ask odds.plan_pulls this often (budgeted; most checks pull nothing)
LIVE_ODDS_NEAR_HOURS = 1.5        # "near kickoff" for the faster odds interval
LIVE_MAX_RUN_MINUTES = 330        # exit before GitHub's 6 h job limit, then re-dispatch
LIVE_FAST_SECONDS = 3            # #39b fast lane: re-quote alerted contracts this often between passes
LIVE_FAST_MAX_TICKERS = 50       # never more than this many contracts in the fast lane
LIVE_FAST_PER_REQUEST = 25       # tickers per GET /markets call (<= 2 calls per tick)
LIVE_EARLY_START_MINUTES = 15     # a run fired this close before a window waits for it
LIVE_MINUTES_BUDGET_MONTH = 15000 # Actions minutes for live.yml per month (windows ~ 14,000 min)

# CBB liquidity rule (Emerson: "games without volume: don't touch")
CBB_MIN_EVENT_VOL_24H = 5000   # contracts traded across the event in 24h
CBB_MIN_SIZE = 50              # contracts available at the flagged price
