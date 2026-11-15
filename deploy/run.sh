#!/usr/bin/env bash
# edge-desk always-on live runner (backlog #66c). Read-only: market data + PAPER trades only,
# never an order. Same loop as .github/workflows/live.yml, minus the 6 h job limit:
#   pull new code -> (re)install requirements if they changed -> inside a game window: tests must
#   pass, results/backtests, then `python -m edge_desk.live loop` (ends every <=5.5 h or when a merge
#   brings new code, #54) -> repeat. Outside windows it sleeps.
# While this loop checkpoints (data/live_log.csv row "vps-...", every 15 min), the Actions live job
# and collect.yml stand down (edge_desk/fresh.vps_active); if this server dies, they resume within ~25 min.
set -u
export EDGE_RUNNER=vps
REPO="${REPO:-/srv/edge-desk}"
cd "$REPO" || { echo "no repo at $REPO"; exit 1; }
# the Kalshi key may come from a file (multi-line PEM doesn't fit an env file)
if [ -z "${KALSHI_PRIVATE_KEY:-}" ] && [ -n "${KALSHI_PRIVATE_KEY_FILE:-}" ] && [ -r "$KALSHI_PRIVATE_KEY_FILE" ]; then
  KALSHI_PRIVATE_KEY="$(cat "$KALSHI_PRIVATE_KEY_FILE")"
  export KALSHI_PRIVATE_KEY
fi
git config --global --add safe.directory "$REPO"
git config user.name "edge-desk-vps"
git config user.email "edge-desk-vps@users.noreply.github.com"
last_req=""
while true; do
  if ! git pull --rebase -q; then
    git rebase --abort 2>/dev/null
    git pull --rebase -q -X theirs || echo "run.sh: pull failed, running on the current checkout"
  fi
  req="$(sha256sum requirements.txt | cut -d' ' -f1)"
  if [ "$req" != "$last_req" ]; then
    pip install -q --no-cache-dir -r requirements.txt && last_req="$req"
  fi
  inwin="$(python -c 'from datetime import datetime, timezone
from edge_desk import config, live
print(1 if live.current_window(datetime.now(timezone.utc), config.LIVE_EARLY_START_MINUTES) else 0)' 2>/dev/null)"
  if [ "$inwin" != "1" ]; then
    sleep 120
    continue
  fi
  if ! python -m pytest -q -x -p no:cacheprovider > /tmp/edge-desk-pytest.log 2>&1; then
    echo "run.sh: tests red, not running this code"; tail -5 /tmp/edge-desk-pytest.log
    sleep 300
    continue
  fi
  python -m edge_desk.nfl_results 12 || true
  python -m edge_desk.cfb_results 12 || true
  python -m edge_desk.cfb_lines || true
  python -m edge_desk.backtest --if-stale NFL CFB || true
  python -m edge_desk.alerts verify-url || true
  python -m edge_desk.live loop || sleep 30
done
