"""
l3_regime.py — Layer 3: Regime Recognition

Defines "trending" precisely (Kaufman Efficiency Ratio) rather than
relying on a fixed ADX threshold. Two versions of the same measure:
  - backward-looking: usable as a feature (known at decision time)
  - forward-looking: usable as a training label (what we're trying
    to predict) - never used as a live feature, only for supervision.
"""

import numpy as np
import pandas as pd


def efficiency_ratio(close: pd.Series, length: int) -> pd.Series:
    """
    Backward-looking Kaufman Efficiency Ratio over the trailing
    `length` bars, as of each bar. Range ~[0, 1]. 1 = a clean trend,
    0 = pure chop (net displacement cancels out).
    """
    net_change = (close - close.shift(length)).abs()
    path_length = close.diff().abs().rolling(length).sum()
    return (net_change / path_length).replace([np.inf, -np.inf], np.nan)


def forward_regime_label(close: pd.Series, horizon: int, threshold: float = 0.35) -> pd.Series:
    """
    Training label only: efficiency ratio measured over the NEXT
    `horizon` bars from each point, thresholded into a binary
    "was this actually a tradeable trend" label.
    """
    future_net_change = (close.shift(-horizon) - close).abs()
    future_path_length = close.diff().abs().rolling(horizon).sum().shift(-horizon)
    fwd_er = (future_net_change / future_path_length).replace([np.inf, -np.inf], np.nan)
    return (fwd_er > threshold).astype(float).where(fwd_er.notna())

def atr_expansion(atr: pd.Series, length: int = 50) -> pd.Series:
    """
    Current ATR relative to its own trailing average. >1 means
    volatility is expanding (often precedes/accompanies breakouts),
    <1 means it's contracting (often precedes/accompanies consolidation).
    """
    return atr / atr.rolling(length).mean()


def ema_crossover_count(ema_fast: pd.Series, ema_slow: pd.Series, length: int = 20) -> pd.Series:
    """
    Number of times the fast/slow EMA relationship flipped sign over
    the trailing `length` bars - a direct choppiness proxy. High count
    means the market keeps reversing rather than committing.
    """
    sign = np.sign(ema_fast - ema_slow)
    flips = (sign != sign.shift(1)).astype(int)
    return flips.rolling(length).sum()


def ma_slope(ma: pd.Series, length: int = 20) -> pd.Series:
    """
    Rate of change of a moving average over `length` bars - is the
    underlying trend context itself accelerating, decelerating, or flat.
    """
    return (ma - ma.shift(length)) / ma.shift(length)


def build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assembles the backward-looking (decision-time-safe) feature set a
    regime classifier trains on. `df` must already carry the Layer 2
    indicator columns - i.e. this expects l2_features.build_bt_df() output.
    """
    feats = pd.DataFrame(index=df.index)
    feats["adx_14"] = df["adx_14"]
    feats["er_20"] = efficiency_ratio(df["Close"], length=20)
    feats["atr_expansion_50"] = atr_expansion(df["atr_14"], length=50)
    feats["ema_crossover_count_20"] = ema_crossover_count(df["ema_8"], df["ema_21"], length=20)
    feats["ma200_slope_20"] = ma_slope(df["ma_200"], length=20)
    return feats