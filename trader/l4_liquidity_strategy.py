"""
l4_liquidity_strategy.py — Layer 4: sweep -> displacement -> BOS -> pullback
-> LTF-BOS -> engulfing entry, on a single-timeframe (4H) approximation.
"""
import pandas as pd
from backtesting import Strategy

from .l2_features import premium_discount_zone
from .l5_position_sizing import risk_based_size
from .l6_risk import CircuitBreakerMixin


class LiquiditySweepStrategy(CircuitBreakerMixin, Strategy):
    max_bars_to_bos = 9
    max_pullback_bars = 12
    target_rr = 2.0   # TP = entry +/- (SL distance * target_rr)
    risk_pct = 0.01   # Layer 5: risk 1% of equity per trade, sized off SL distance
    leverage = 30      # must match Backtest(..., margin=1/leverage) - see backtest_harness
    # max_consecutive_losses / cooldown_bars: inherited from
    # CircuitBreakerMixin (Layer 6) - this strategy had no circuit
    # breaker at all before it picked one up from here.

    def init(self):
        self._cb_init()

    def _extra_gates_ok(self, direction: str) -> bool:
        """
        Optional additional entry gate, checked only at the final ARMED-
        phase engulfing trigger (2026-07-26) - a no-op hook (always True)
        by default, so this class's own already-validated behavior is
        unchanged (same pattern/reasoning as l4_signal_model.py's hook
        of the same name - an "and True" that cannot alter existing
        results). `direction` is "long" (bull/buy) or "short" (bear/sell).
        See SMCZoneLiquiditySweepStrategy below for the actual override -
        this is the strategy family the premium/discount zone gate
        turned out to structurally fit: unlike ConfluenceStrategy's
        breakout entries (zone gate tested there first and found
        near-incompatible - see l4_signal_model.py's SMCZoneConfluenceStrategy
        docstring), this strategy's entries fire AFTER a sweep + reversal,
        which is exactly the setup a "trade toward the far side of the
        range" filter was built for.
        """
        return True

    def next(self):
        d = self.data
        c = d.Close[-1]

        self._cb_update()

        if self.position:
            return

        if not hasattr(self, "phase"):
            self.phase = "IDLE"

        if self.phase == "IDLE":
            if d.bsl_swept[-1]:
                self.phase, self.direction, self.bars_in_phase = "SWEPT", "bear", 0
                self.swept_level = d.prior_high[-1]
            elif d.ssl_swept[-1]:
                self.phase, self.direction, self.bars_in_phase = "SWEPT", "bull", 0
                self.swept_level = d.prior_low[-1]
            return

        self.bars_in_phase += 1

        if self.phase == "SWEPT":
            if self.bars_in_phase > self.max_bars_to_bos:
                self.phase = "IDLE"
                return
            if self.direction == "bear" and d.bear_displacement[-1] and c < d.bos_ref_low[-1]:
                self.phase, self.bars_in_phase = "PULLBACK", 0
            elif self.direction == "bull" and d.bull_displacement[-1] and c > d.bos_ref_high[-1]:
                self.phase, self.bars_in_phase = "PULLBACK", 0
            return

        if self.phase in ("PULLBACK", "ARMED"):
            if self.bars_in_phase > self.max_pullback_bars:
                self.phase = "IDLE"
                return
            if self.direction == "bear" and c > self.swept_level:
                self.phase = "IDLE"
                return
            if self.direction == "bull" and c < self.swept_level:
                self.phase = "IDLE"
                return

        if self.phase == "PULLBACK":
            if self.direction == "bear" and c < d.ltf_swing_low[-1]:
                self.phase = "ARMED"
            elif self.direction == "bull" and c > d.ltf_swing_high[-1]:
                self.phase = "ARMED"
            return

        if self.phase == "ARMED":
                    if self.direction == "bear" and d.bear_engulf[-1]:
                        sl = d.ltf_pivot_high[-1]
                        if pd.notna(sl) and sl > c and not self._cb_in_cooldown() and self._extra_gates_ok("short"):
                            sl_distance = sl - c
                            tp = c - sl_distance * self.target_rr
                            size = risk_based_size(c, sl, self.risk_pct, self.leverage)
                            self.sell(sl=sl, tp=tp, size=size)
                        self.phase = "IDLE"
                    elif self.direction == "bull" and d.bull_engulf[-1]:
                        sl = d.ltf_pivot_low[-1]
                        if pd.notna(sl) and sl < c and not self._cb_in_cooldown() and self._extra_gates_ok("long"):
                            sl_distance = c - sl
                            tp = c + sl_distance * self.target_rr
                            size = risk_based_size(c, sl, self.risk_pct, self.leverage)
                            self.buy(sl=sl, tp=tp, size=size)
                        self.phase = "IDLE"


class SMCZoneLiquiditySweepStrategy(LiquiditySweepStrategy):
    """
    LiquiditySweepStrategy + a premium/discount zone gate (2026-07-26,
    mined from the same Journalling/XAUUSD SMC notebook as
    l4_signal_model.SMCZoneConfluenceStrategy - see that class's
    docstring for the full mining story and for why the zone gate did
    NOT fit ConfluenceStrategy's breakout entries). Only takes a SELL
    (bear engulf, ARMED phase) while price sits in the "premium" (upper)
    half of the recent range, only a BUY from "discount" (lower) half -
    l2_features.premium_discount_zone(), structure_high/range_low via
    swing_high()/swing_low() (causal).

    Philosophical fit, stated plainly: this strategy's entries fire
    AFTER a liquidity sweep and a displacement reversal - structurally,
    that means price has just tagged an extreme and reversed, which is
    much more likely to land near the far side of the range than a
    breakout entry ever would. Tested empirically before trusting that
    reasoning (see the ARCHITECTURE.md entry for this class) rather than
    assumed - unlike the ConfluenceStrategy pairing, which failed that
    same empirical test outright.

    No session gate here (unlike SMCZoneConfluenceStrategy) - kept
    deliberately out of scope for this first pass; add one the same way
    if the zone gate alone proves worth keeping.
    """
    zone_lookback = 8

    def init(self):
        super().init()
        d = self.data.df
        self.zone = premium_discount_zone(d, structure_lookback=self.zone_lookback)["zone"]

    def _extra_gates_ok(self, direction: str) -> bool:
        i = len(self.data) - 1
        zone = self.zone.iloc[i]
        if direction == "long" and zone != "discount":
            return False
        if direction == "short" and zone != "premium":
            return False
        return True