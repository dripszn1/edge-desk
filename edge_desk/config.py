"""Single source of truth for scope and risk limits. See SPEC.md."""

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Kalshi series tickers in scope (verified against /series?category=Sports, 2026-11-09)
SERIES = {
    "NFL": ["KXNFLSPREAD", "KXNFLTOTAL", "KXNFLGAME"],
    "CFB": ["KXNCAAFSPREAD", "KXNCAAFTOTAL", "KXNCAAFGAME"],
    # CBB parked until NFL/CFB prove an edge: "CBB": ["KXNCAAMBSPREAD", "KXNCAAMBTOTAL", "KXNCAAMBGAME"],
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
