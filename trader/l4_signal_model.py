"""
l4_signal_model.py — Layer 4: Signal Model (current: rule-based)

NOTE: this currently also does Layer 3's job (regime recognition, via
the ADX threshold check) since that split hasn't happened yet -
replacing the ADX rule with a learned regime classifier is the next
real piece of work, along with replacing the EMA/MACD/breakout rules
here with a trained return-prediction model. Until that split, this
file is layers 3+4 combined.
"""

import numpy as np
from backtesting import Strategy

from .l2_features import (
    swing_high, swing_low, premium_discount_zone, trading_session,
    build_momentum_structure_features,
)
from .l3_regime import efficiency_ratio
from .l5_position_sizing import risk_based_size
from .l6_risk import CircuitBreakerMixin


class ConfluenceStrategy(CircuitBreakerMixin, Strategy):
    adx_threshold = 22
    swing_lookback = 20
    atr_sl_mult = 1.5
    atr_tp_mult = 2.5
    risk_pct = 0.01   # Layer 5: risk 1% of equity per trade, sized off SL distance
    leverage = 30      # must match Backtest(..., margin=1/leverage) - see backtest_harness
    # max_consecutive_losses / cooldown_bars: inherited from
    # CircuitBreakerMixin (Layer 6). Override here per-strategy if needed.

    def init(self):
        d = self.data.df
        self.ma_360 = self.I(lambda: d["ma_360"], name="MA360")
        self.ma_200 = self.I(lambda: d["ma_200"], name="MA200")
        self.ema_21 = self.I(lambda: d["ema_21"], name="EMA21")
        self.ema_8 = self.I(lambda: d["ema_8"], name="EMA8")
        self.macd_line = self.I(lambda: d["macd"], name="MACD")
        self.macd_signal = self.I(lambda: d["macd_signal"], name="MACDsig")
        self.macd_hist = self.I(lambda: d["macd_hist"], name="MACDhist")
        self.adx_14 = self.I(lambda: d["adx_14"], name="ADX14")
        self.atr_14 = self.I(lambda: d["atr_14"], name="ATR14")
        self.swing_hi = self.I(lambda: swing_high(d["High"], self.swing_lookback), name="SwingHi")
        self.swing_lo = self.I(lambda: swing_low(d["Low"], self.swing_lookback), name="SwingLo")
        self._cb_init()

    def _regime_ok(self) -> bool:
        """Layer 3 gate: is this a tradeable trend? Base version = bare ADX threshold."""
        return self.adx_14[-1] > self.adx_threshold

    def _regime_warmed_up(self) -> bool:
        return not np.isnan(self.adx_14[-1])

    def _extra_gates_ok(self, direction: str) -> bool:
        """
        Optional additional entry gate(s), checked alongside _regime_ok()
        but independent of it (2026-07-26). Base implementation is a
        no-op (always True) - ANDed into next()'s long_signal/
        short_signal below, an "and True" that provably cannot change
        ConfluenceStrategy/RegimeConfluenceStrategy's behavior versus
        before this hook existed. Exists so a subclass (see
        SMCZoneConfluenceStrategy) can layer on extra conditions - e.g.
        a premium/discount zone filter - without touching or duplicating
        next()'s already walk-forward-tested logic. `direction` is
        "long" or "short".
        """
        return True

    def next(self):
        # --- Layer 6 (risk overlay: circuit breaker) ---
        self._cb_update()
        in_cooldown = self._cb_in_cooldown()

        price = self.data.Close[-1]
        atr = self.atr_14[-1]
        if np.isnan(atr) or np.isnan(self.ma_360[-1]) or np.isnan(self.swing_hi[-1]) or not self._regime_warmed_up():
            return

        # --- Layer 3 (regime) ---
        trending = self._regime_ok()

        # --- Layer 4 (signal), currently hand-picked rules ---
        ema_bullish = self.ema_8[-1] > self.ema_21[-1]
        ema_bearish = self.ema_8[-1] < self.ema_21[-1]
        macd_bull = self.macd_hist[-1] > 0
        macd_bear = self.macd_hist[-1] < 0
        bos_up = price > self.swing_hi[-1]
        bos_down = price < self.swing_lo[-1]
        macro_uptrend = price > self.ma_360[-1] and price > self.ma_200[-1]
        macro_downtrend = price < self.ma_360[-1] and price < self.ma_200[-1]

        long_signal = trending and macro_uptrend and ema_bullish and macd_bull and bos_up and self._extra_gates_ok("long")
        short_signal = trending and macro_downtrend and ema_bearish and macd_bear and bos_down and self._extra_gates_ok("short")

        if not self.position:
            if not in_cooldown:
                if long_signal:
                    sl = price - self.atr_sl_mult * atr
                    tp = price + self.atr_tp_mult * atr
                    if sl < price:
                        size = risk_based_size(price, sl, self.risk_pct, self.leverage)
                        self.buy(sl=sl, tp=tp, size=size)
                elif short_signal:
                    sl = price + self.atr_sl_mult * atr
                    tp = price - self.atr_tp_mult * atr
                    if sl > price:
                        size = risk_based_size(price, sl, self.risk_pct, self.leverage)
                        self.sell(sl=sl, tp=tp, size=size)
        else:
            if self.position.is_long and price < self.ema_21[-1] and not macro_uptrend:
                self.position.close()
            elif self.position.is_short and price > self.ema_21[-1] and not macro_downtrend:
                self.position.close()


class RegimeConfluenceStrategy(ConfluenceStrategy):
    """
    Identical Layer 4 signal rules to ConfluenceStrategy. The only change
    is the Layer 3 gate: instead of a bare ADX threshold, "trending" is
    decided by l3_regime's Kaufman Efficiency Ratio (backward-looking,
    range ~[0,1], 1 = clean directional move, 0 = pure chop).

    This is the swap the module docstring above has been flagging since
    Layer 3 was split out - the walk-forward optimizer kept picking the
    loosest available adx_threshold (15) in 19/36 US30 folds, which is
    the regime filter barely filtering anything. ER is a more direct
    measure of "is price actually going somewhere" than ADX.
    """
    er_length = 20
    er_threshold = 0.35

    def init(self):
        super().init()
        d = self.data.df
        self.er = self.I(lambda: efficiency_ratio(d["Close"], self.er_length), name=f"ER{self.er_length}")

    def _regime_ok(self) -> bool:
        return self.er[-1] > self.er_threshold

    def _regime_warmed_up(self) -> bool:
        return not np.isnan(self.er[-1])


class SMCZoneConfluenceStrategy(RegimeConfluenceStrategy):
    """
    RegimeConfluenceStrategy + two additional entry gates mined
    (2026-07-26) from a personal SMC research notebook the user pointed
    me at (Journalling/XAUUSD's "V1006" dashboard - not part of this
    repo; that notebook auto-computes a premium/discount range and
    tags a trading session per bar, for a manual weekly chart journal,
    never backtested there). Both gates are pure ANDs on top of
    ConfluenceStrategy's five original conditions via `_extra_gates_ok()`
    - neither touches or duplicates the already walk-forward-tested
    entry/exit logic itself.

    - Zone gate: only take LONGS while price sits in the "discount"
      (lower) half of the recent range (l2_features.premium_discount_zone,
      structure_high/range_low via swing_high()/swing_low()), only take
      SHORTS from "premium" (upper) half. Idea: only trade toward the
      far side of the range, not while price is already sitting deep in
      the half that would make the trade a "buying high / selling low"
      entry relative to the immediate structure.
    - Session gate (`session_gate_enabled`, default True): only take
      entries during the notebook's London/New York session windows
      (l2_features.trading_session()), skip "Off-session" bars. Ported
      as-is from a filter built for 5m/15m bars onto this project's H4
      production timeframe - worth stating plainly: a 3-5 hour session
      window is a coarse, possibly not very discriminating filter
      against 4-hour bars (many H4 bars will straddle a session
      boundary), and trading_session()'s own docstring flags its
      timezone assumption as unverified for this project's data. Treat
      this gate's walk-forward numbers as informative about whether the
      *zone* gate alone is worth keeping, not as a proven session edge
      at H4 - re-test at a finer timeframe before trusting the session
      half of this class specifically.
    """
    zone_lookback = 8
    session_gate_enabled = True

    def init(self):
        super().init()
        d = self.data.df
        zone_df = premium_discount_zone(d, structure_lookback=self.zone_lookback)
        self.zone = zone_df["zone"]
        self.session = trading_session(d.index)

    def _extra_gates_ok(self, direction: str) -> bool:
        i = len(self.data) - 1
        zone = self.zone.iloc[i]
        if direction == "long" and zone != "discount":
            return False
        if direction == "short" and zone != "premium":
            return False
        if self.session_gate_enabled and self.session.iloc[i] == "Off-session":
            return False
        return True


class MomentumStructureConfluenceStrategy(RegimeConfluenceStrategy):
    """
    RegimeConfluenceStrategy + one additional entry gate, mined
    (2026-07-31) from Michael Oliver's Momentum Structural Analysis (MSA)
    method, described in an interview transcript the user uploaded (not
    code - a discretionary trader's stated method, translated into
    something testable here). Oliver's claim: chart price relative to a
    moving average (an oscillator) instead of price itself, then treat
    THAT oscillator as its own chartable structure - its own swing highs/
    lows, its own breakouts - and that structure often breaks BEFORE
    price's own structure does, i.e. momentum leads price.

    Translated literally: `l2_features.build_momentum_structure_features()`
    re-runs this file's own swing_high()/swing_low() structure logic
    (already used for PRICE breakout detection - `bos_up = price >
    swing_hi` in ConfluenceStrategy.next()) against Close/ma_89 instead of
    against Close itself, producing mom_break_up/mom_break_down. This
    gate then only allows an entry if that MOMENTUM break, in the same
    direction as the trade, already happened within the last
    `mom_lead_bars` bars - not just on the same bar as the price signal
    (that would be redundant with price's own BOS), but at-or-before it,
    which is the actual "leads price" claim made literal and checkable.

    ANDed on top of ConfluenceStrategy's five original conditions via the
    same `_extra_gates_ok()` hook SMCZoneConfluenceStrategy uses - default
    behavior of the base classes is provably unaffected (see that class's
    docstring for the reasoning, unchanged here).
    """
    mom_ma_col = "ma_89"
    mom_structure_lookback = 8
    mom_lead_bars = 10

    def init(self):
        super().init()
        d = self.data.df
        feats = build_momentum_structure_features(
            d, ma_col=self.mom_ma_col, structure_lookback=self.mom_structure_lookback,
        )
        self.mom_break_up = feats["mom_break_up"]
        self.mom_break_down = feats["mom_break_down"]

    def _extra_gates_ok(self, direction: str) -> bool:
        i = len(self.data) - 1
        lo = max(0, i - self.mom_lead_bars)
        if direction == "long":
            return bool(self.mom_break_up.iloc[lo:i + 1].any())
        else:
            return bool(self.mom_break_down.iloc[lo:i + 1].any())