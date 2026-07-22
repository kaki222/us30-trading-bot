"""
l5_position_sizing.py — Layer 5: Position Sizing

Every Layer 4 strategy used to hardcode size=0.1 on every buy()/sell()
call, regardless of how wide the stop-loss was. That means two trades
with the same `size` but very different SL distances (e.g. a tight ATR
stop in a calm market vs a wide one in a volatile stretch) put very
different dollar amounts at risk - risk per trade wasn't actually
controlled by anything.

risk_based_size() fixes that: given the entry price, the stop-loss
price, and a target risk_pct of equity, it returns the `size` fraction
backtesting.py needs so that if the stop is hit, the loss is exactly
risk_pct of current equity - independent of SL distance, and
independent of leverage (leverage changes how much cash a trade ties
up, not how much it risks if the stop is hit).
"""


def risk_based_size(price: float, sl: float, risk_pct: float, leverage: float,
                     min_size: float = 0.001, max_size: float = 1.0) -> float:
    """
    Returns a `size` fraction (0, 1] for backtesting.py's
    Strategy.buy(size=...)/sell(size=...).

    backtesting.py converts a (0, 1] size fraction to units as:
        units = floor(equity * size * leverage / price)
    so the dollar risk if price moves from `price` to `sl` is:
        units * abs(price - sl) = equity * size * leverage * abs(price - sl) / price

    Setting that equal to risk_pct * equity and solving for size:
        size = risk_pct * price / (leverage * abs(price - sl))

    `leverage` must match whatever the Backtest is actually configured
    with (leverage = 1 / margin) - it's a parameter here rather than a
    hidden assumption so the caller can't get this out of sync with the
    Backtest() call without it being visible at the call site.

    Clamped to [min_size, max_size]: min_size keeps a nonzero position
    on the books when the risk-implied size would otherwise round to
    (near) zero (e.g. an extremely wide stop); max_size caps leverage-
    driven blowups when the stop is unusually tight.
    """
    sl_distance = abs(price - sl)
    if sl_distance <= 0 or leverage <= 0:
        return min_size
    size = risk_pct * price / (leverage * sl_distance)
    return max(min_size, min(max_size, size))
