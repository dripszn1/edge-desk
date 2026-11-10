"""Kalshi trading fees.

Kalshi's published schedule for 'quadratic' fee series:
    taker fee per order = round_up_to_cent(0.07 * multiplier * C * P * (1 - P))
    maker fee (quadratic_with_maker_fees) = round_up_to_cent(0.0175 * multiplier * C * P * (1 - P))
We assume TAKER fills everywhere (conservative: it's what a hand-placed market order pays).
Verify against https://kalshi.com/fee-schedule if numbers ever look off.
"""
import math

TAKER_RATE = 0.07
MAKER_RATE = 0.0175


def _round_up_cent(x: float) -> float:
    return math.ceil(round(x * 100, 9)) / 100


def taker_fee(price: float, contracts: float = 1, multiplier: float = 1.0) -> float:
    """Total taker fee in dollars for `contracts` at `price` (0-1)."""
    if contracts <= 0 or not 0 < price < 1:
        return 0.0
    return _round_up_cent(TAKER_RATE * multiplier * contracts * price * (1 - price))


def fee_per_contract(price: float, contracts: float = 1, multiplier: float = 1.0) -> float:
    """Fee per contract at a given order size (rounding hurts small orders most)."""
    if contracts <= 0:
        return 0.0
    return taker_fee(price, contracts, multiplier) / contracts
