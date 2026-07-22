"""
l6_risk.py — Layer 6: Risk Overlay

Circuit breaker: after `max_consecutive_losses` losing trades in a row,
force a `cooldown_bars` pause before the strategy will open a new
position.

Extracted out of l4_signal_model.ConfluenceStrategy, where it used to
live inline (see git history before commit that added this file) - the
whole point of pulling it out is that every Layer 4 strategy can share
one implementation instead of reimplementing (or, as with
LiquiditySweepStrategy until now, simply not having one).
"""


class CircuitBreakerMixin:
    """
    Mix into any backtesting.Strategy subclass:

        class MyStrategy(CircuitBreakerMixin, Strategy):
            def init(self):
                ...
                self._cb_init()

            def next(self):
                self._cb_update()
                if self._cb_in_cooldown():
                    return  # or: skip only the entry, let exits/management run
                ...

    The mixin tracks closed-trade P&L itself via self.closed_trades (a
    backtesting.py Strategy attribute), so no other bookkeeping is
    required from the strategy. All methods/attributes are prefixed
    `_cb_`/`cb_` to avoid colliding with backtesting.py's own Strategy
    internals or with strategy-specific state.
    """

    max_consecutive_losses = 3
    cooldown_bars = 20

    def _cb_init(self):
        self._cb_last_closed_count = 0
        self._cb_consecutive_losses = 0
        self._cb_cooldown_until_bar = -1

    def _cb_update(self):
        """Call once per bar (start of next()), before checking _cb_in_cooldown()."""
        closed = self.closed_trades
        if len(closed) > self._cb_last_closed_count:
            for t in closed[self._cb_last_closed_count:]:
                if t.pl < 0:
                    self._cb_consecutive_losses += 1
                else:
                    self._cb_consecutive_losses = 0
            self._cb_last_closed_count = len(closed)
            if self._cb_consecutive_losses >= self.max_consecutive_losses:
                self._cb_cooldown_until_bar = len(self.data) + self.cooldown_bars
                self._cb_consecutive_losses = 0

    def _cb_in_cooldown(self) -> bool:
        return len(self.data) < self._cb_cooldown_until_bar
