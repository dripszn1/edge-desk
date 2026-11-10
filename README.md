# edge-desk

Kalshi prediction-market edge desk for NFL, CFB and CBB. **Read-only, paper mode. Never places orders.**

See [SPEC.md](SPEC.md) for the rules, risk limits and build order.

## Layout
- `edge_desk/config.py`: scope + hard risk limits ($200 bankroll, $10/bet, $20/day)
- `edge_desk/collect.py`: pulls every open spread/total/game market from Kalshi's public API → `data/snapshots/`
- `edge_desk/scan.py`: structural scanner (ladder, exclusive, cover) after fees → `data/flags/`
- `edge_desk/espn.py`: ESPN public scoreboards (NFL, CFB FBS+FCS, CBB D1) → `data/espn/` (kickoffs, state, scores)
- `edge_desk/kickoffs.py`: joins Kalshi events to ESPN games, tags each market pregame/live/unknown; scanner flags pregame only
- `edge_desk/settle.py`: fetches Kalshi results for every flagged ticker → `data/results/results.csv`
- `edge_desk/grade.py`: settles paper flags (fees recomputed per leg, daily cap + drawdown stop) → `data/paper_bets.csv`, `data/paper_ledger.csv`
- `edge_desk/dashboard.py`: today's flags, paper record, P&L after fees, league/signal splits, risk status → `data/dashboard.json`
- `edge_desk/fees.py`, `edge_desk/sizing.py`: fees on every leg, quarter-Kelly with caps
- `.github/workflows/collect.yml`: runs collect + scan every 30 min, 8am–11pm CT

## Run locally
```
pip install -r requirements.txt
python -m pytest -q
python -m edge_desk.espn && python -m edge_desk.collect && python -m edge_desk.scan
```
