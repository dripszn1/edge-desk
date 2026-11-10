"""EV and Kelly sizing with fees inside, capped by SPEC.md hard rules."""
from . import config


def ev_per_contract(p: float, price: float, fee: float) -> float:
    return p - price - fee


def kelly_fraction(p: float, price: float, fee: float) -> float:
    cost = price + fee
    if cost >= 1:
        return 0.0
    return max((p - cost) / (1 - cost), 0.0)


def bet_size(p: float, price: float, fee: float, bankroll: float = config.BANKROLL,
             lost_today: float = 0.0) -> float:
    """Dollar stake. Zero once the daily loss cap is hit."""
    if lost_today >= config.DAILY_LOSS_CAP:
        return 0.0
    stake = config.KELLY_FRACTION * kelly_fraction(p, price, fee) * bankroll
    room = config.DAILY_LOSS_CAP - lost_today      # never risk more than what's left today
    return round(min(stake, config.MAX_BET, room), 2)


def structural_size(cost_per_set: float, available: float, bankroll: float = config.BANKROLL) -> dict:
    """Arb sets have a locked payout >= $1, so size by the $ cap, not Kelly."""
    sets = int(min(available, config.MAX_BET // cost_per_set)) if cost_per_set > 0 else 0
    return {"sets": sets, "stake": round(sets * cost_per_set, 2)}
