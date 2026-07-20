"""
l2_features.py — Layer 2: Feature Engineering

Turns raw OHLCV into the indicators every downstream layer reads:
moving averages, MACD, ATR, ADX, and the swing_high/swing_low
structure helpers used for breakout detection.
"""

import numpy as np
import pandas as pd

from .l1_data import fetch_h4


def sma(series, length):
    return series.rolling(window=length, min_periods=length).mean()


def ema(series, length):
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(high, low, close):
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high, low, close, length=14):
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def adx(high, low, close, length=14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    atr_wilder = true_range(high, low, close).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr_wilder)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr_wilder)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_line = dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return plus_di, minus_di, adx_line


def swing_high(high, n):
    return high.rolling(n).max().shift(1)


def swing_low(low, n):
    return low.rolling(n).min().shift(1)


def build_bt_df(symbol, start=None, period="730d"):
    """
    Layers 1+2 combined: fetch a symbol and attach every indicator
    column, renamed to the Open/High/Low/Close/Volume capitalization
    backtesting.py expects.
    """
    d = fetch_h4(symbol, start=start, period=period)
    d["ma_360"] = sma(d["close"], 360)
    d["ma_200"] = sma(d["close"], 200)
    d["ma_89"] = sma(d["close"], 89)
    d["ema_21"] = ema(d["close"], 21)
    d["ema_8"] = ema(d["close"], 8)
    d["macd"], d["macd_signal"], d["macd_hist"] = macd(d["close"])
    d["atr_14"] = atr(d["high"], d["low"], d["close"])
    d["plus_di"], d["minus_di"], d["adx_14"] = adx(d["high"], d["low"], d["close"])
    return d.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})