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

from .l2_features import swing_high, swing_low
from .l3_regime import efficiency_ratio


class ConfluenceStrategy(Strategy):
    adx_threshold = 22
    swing_lookback = 20
    atr_sl_mult = 1.5
    atr_tp_mult = 2.5
    max_consecutive_losses = 3
    cooldown_bars = 20

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
        self._last_closed_count = 0
        self._consecutive_losses = 0
        self._cooldown_until_bar = -1

    def _regime_ok(self) -> bool:
        """Layer 3 gate: is this a tradeable trend? Base version = bare ADX threshold."""
        return self.adx_14[-1] > self.adx_threshold

    def _regime_warmed_up(self) -> bool:
        return not np.isnan(self.adx_14[-1])

    def next(self):
        # --- Layer 6 (risk overlay: circuit breaker), temporarily inline ---
        closed = self.closed_trades
        if len(closed) > self._last_closed_count:
            for t in closed[self._last_closed_count:]:
                if t.pl < 0:
                    self._consecutive_losses += 1
                else:
                    self._consecutive_losses = 0
            self._last_closed_count = len(closed)
            if self._consecutive_losses >= self.max_consecutive_losses:
                self._cooldown_until_bar = len(self.data) + self.cooldown_bars
                self._consecutive_losses = 0

        in_cooldown = len(self.data) < self._cooldown_until_bar

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

        long_signal = trending and macro_uptrend and ema_bullish and macd_bull and bos_up
        short_signal = trending and macro_downtrend and ema_bearish and macd_bear and bos_down

        if not self.position:
            if not in_cooldown:
                if long_signal:
                    sl = price - self.atr_sl_mult * atr
                    tp = price + self.atr_tp_mult * atr
                    if sl < price:
                        self.buy(sl=sl, tp=tp, size=0.1)
                elif short_signal:
                    sl = price + self.atr_sl_mult * atr
                    tp = price - self.atr_tp_mult * atr
                    if sl > price:
                        self.sell(sl=sl, tp=tp, size=0.1)
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