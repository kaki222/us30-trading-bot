"""
l4_liquidity_strategy.py — Layer 4: sweep -> displacement -> BOS -> pullback
-> LTF-BOS -> engulfing entry, on a single-timeframe (4H) approximation.
"""
import pandas as pd
from backtesting import Strategy


class LiquiditySweepStrategy(Strategy):
    max_bars_to_bos = 9
    max_pullback_bars = 12

    def init(self):
        pass  # all inputs are precomputed columns from build_liquidity_features

    def next(self):
        d = self.data
        c = d.Close[-1]

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
                sl, tp = d.ltf_pivot_high[-1], d.htf_pivot_low_2back[-1]
                if pd.notna(sl) and pd.notna(tp) and sl > c > tp:
                    self.sell(sl=sl, tp=tp)
                self.phase = "IDLE"
            elif self.direction == "bull" and d.bull_engulf[-1]:
                sl, tp = d.ltf_pivot_low[-1], d.htf_pivot_high_2back[-1]
                if pd.notna(sl) and pd.notna(tp) and sl < c < tp:
                    self.buy(sl=sl, tp=tp)
                self.phase = "IDLE"