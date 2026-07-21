"""
l2_features.py — Layer 2: Feature Engineering

Turns raw OHLCV into the indicators every downstream layer reads:
moving averages, MACD, ATR, ADX, and the swing_high/swing_low
structure helpers used for breakout detection.
"""

import numpy as np
import pandas as pd

from .l1_data import load_h4

# build_bt_df() used to take Yahoo Finance tickers. Keep those working so
# existing notebook cells (build_bt_df("^DJI"), build_bt_df("GC=F")) don't
# need edits — they now resolve to the MT5-backed loaders in l1_data.
_LEGACY_SYMBOL_MAP = {"^DJI": "US30", "GC=F": "GOLD"}


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
    Layers 1+2 combined: load a symbol's H4 bars from its MT5 export and
    attach every indicator column, renamed to the Open/High/Low/Close/Volume
    capitalization backtesting.py expects.

    `start`/`period` are no longer used — MT5 exports are loaded in full
    (kept as accepted-but-ignored params so existing call sites don't
    break) — and now return the entire history instead of yfinance's
    730-day intraday cap.
    """
    d = load_h4(_LEGACY_SYMBOL_MAP.get(symbol, symbol))
    d["ma_360"] = sma(d["close"], 360)
    d["ma_200"] = sma(d["close"], 200)
    d["ma_89"] = sma(d["close"], 89)
    d["ema_21"] = ema(d["close"], 21)
    d["ema_8"] = ema(d["close"], 8)
    d["macd"], d["macd_signal"], d["macd_hist"] = macd(d["close"])
    d["atr_14"] = atr(d["high"], d["low"], d["close"])
    d["plus_di"], d["minus_di"], d["adx_14"] = adx(d["high"], d["low"], d["close"])
    return d.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})

# ---------------------------------------------------------------
# Candle patterns (Layer 2)
# ---------------------------------------------------------------

def is_bearish_engulfing(open_: pd.Series, close: pd.Series) -> pd.Series:
    prev_open, prev_close = open_.shift(1), close.shift(1)
    cond = (
        (prev_close > prev_open) &   # prior candle bullish
        (close < open_) &            # current candle bearish
        (open_ >= prev_close) &      # current open at/above prior close
        (close <= prev_open)         # current close at/below prior open
    )
    return cond.fillna(False)


def is_bullish_engulfing(open_: pd.Series, close: pd.Series) -> pd.Series:
    prev_open, prev_close = open_.shift(1), close.shift(1)
    cond = (
        (prev_close < prev_open) &
        (close > open_) &
        (open_ <= prev_close) &
        (close >= prev_open)
    )
    return cond.fillna(False)


# ---------------------------------------------------------------
# Fractal pivot ladder, for SL/TP (Layer 2)
# ---------------------------------------------------------------

def nth_pivot_price(price: pd.Series, is_high: bool, left: int, right: int, n_back: int = 1) -> pd.Series:
    """
    Price of the n_back-th most recent CONFIRMED fractal pivot as of each bar.
    n_back=1 -> most recent pivot, n_back=2 -> the one before that.
    Causal: a pivot at bar i is only visible `right` bars later.
    """
    window = left + right + 1
    extreme = price.rolling(window).max() if is_high else price.rolling(window).min()
    candidate = price.shift(right)
    confirmed_now = candidate == extreme
    pivot_events = candidate[confirmed_now]
    nth_back = pivot_events.shift(n_back - 1)
    result = pd.Series(index=price.index, dtype=float)
    result.loc[nth_back.index] = nth_back.values
    return result.ffill()


# ---------------------------------------------------------------
# Liquidity-sweep feature bundle (Layer 2) — call AFTER build_bt_df,
# since it needs atr_14 to already exist.
# ---------------------------------------------------------------

def build_liquidity_features(
    df,
    structure_lookback=9,       # the level that gets swept
    bos_lookback=8, bos_exclude_recent=2,   # BOS reference, excl. most recent bars
    sweep_recent=3, sweep_prior=9, sweep_atr_mult=0.20,
    avg_body_lookback=21, disp_body_mult=1.30, disp_atr_mult=0.10,
    ltf_swing_lookback=5,       # local pivot used during pullback (LTF-BOS trigger)
    pivot_ltf_k=2,              # SL pivot window
    pivot_htf_k=4,              # TP pivot window
):
    d = df.copy()

    d["struct_high"] = swing_high(d["High"], structure_lookback)
    d["struct_low"] = swing_low(d["Low"], structure_lookback)

    d["bos_ref_high"] = d["High"].shift(bos_exclude_recent).rolling(bos_lookback).max()
    d["bos_ref_low"] = d["Low"].shift(bos_exclude_recent).rolling(bos_lookback).min()

    d["ltf_swing_high"] = swing_high(d["High"], ltf_swing_lookback)
    d["ltf_swing_low"] = swing_low(d["Low"], ltf_swing_lookback)

    recent_high = d["High"].rolling(sweep_recent).max()
    recent_low = d["Low"].rolling(sweep_recent).min()
    d["prior_high"] = d["High"].shift(sweep_recent).rolling(sweep_prior).max()
    d["prior_low"] = d["Low"].shift(sweep_recent).rolling(sweep_prior).min()
    d["bsl_swept"] = recent_high > (d["prior_high"] + sweep_atr_mult * d["atr_14"])
    d["ssl_swept"] = recent_low < (d["prior_low"] - sweep_atr_mult * d["atr_14"])

    body = (d["Close"] - d["Open"]).abs()
    avg_body = body.rolling(avg_body_lookback).mean()
    d["bull_displacement"] = (
        (d["Close"] > d["Open"]) & (body > avg_body * disp_body_mult) &
        (d["Close"] > d["Close"].shift(1) + disp_atr_mult * d["atr_14"])
    )
    d["bear_displacement"] = (
        (d["Close"] < d["Open"]) & (body > avg_body * disp_body_mult) &
        (d["Close"] < d["Close"].shift(1) - disp_atr_mult * d["atr_14"])
    )

    d["bull_engulf"] = is_bullish_engulfing(d["Open"], d["Close"])
    d["bear_engulf"] = is_bearish_engulfing(d["Open"], d["Close"])

    d["ltf_pivot_high"] = nth_pivot_price(d["High"], True, pivot_ltf_k, pivot_ltf_k, n_back=1)
    d["ltf_pivot_low"] = nth_pivot_price(d["Low"], False, pivot_ltf_k, pivot_ltf_k, n_back=1)
    d["htf_pivot_high_2back"] = nth_pivot_price(d["High"], True, pivot_htf_k, pivot_htf_k, n_back=2)
    d["htf_pivot_low_2back"] = nth_pivot_price(d["Low"], False, pivot_htf_k, pivot_htf_k, n_back=2)

    return d

def kalman_trend(price: pd.Series, process_var: float, measurement_var: float) -> pd.DataFrame:
    """
    Local-linear-trend Kalman filter on `price`. Returns 'level' (smoothed
    price) and 'slope' (estimated per-bar rate of change).

    process_var: how much level+slope are expected to drift bar-to-bar
                 (higher = more responsive, less smooth)
    measurement_var: how noisy each observed price is assumed to be
                 (higher = smoother, laggier)
    """
    prices = price.values
    n = len(prices)

    x = np.array([prices[0], 0.0])          # state: [level, slope]
    P = np.eye(2) * 1.0
    F = np.array([[1.0, 1.0], [0.0, 1.0]])   # level += slope each step
    H = np.array([[1.0, 0.0]])               # we observe level only
    Q = np.eye(2) * process_var
    R = measurement_var

    levels, slopes = np.empty(n), np.empty(n)
    for t in range(n):
        x = F @ x
        P = F @ P @ F.T + Q

        y = prices[t] - (H @ x)[0]
        S = (H @ P @ H.T)[0, 0] + R
        K = (P @ H.T).flatten() / S
        x = x + K * y
        P = (np.eye(2) - np.outer(K, H)) @ P

        levels[t], slopes[t] = x[0], x[1]

    return pd.DataFrame({"level": levels, "slope": slopes}, index=price.index)


def build_kalman_features(
    df,
    fast_process_var=1e-4, fast_measurement_var=1e-2,   # entry-timing horizon
    slow_process_var=1e-7, slow_measurement_var=1e-2,   # macro-bias horizon
):
    d = df.copy()
    fast = kalman_trend(d["Close"], fast_process_var, fast_measurement_var)
    slow = kalman_trend(d["Close"], slow_process_var, slow_measurement_var)
    d["kalman_fast_level"] = fast["level"]
    d["kalman_fast_slope"] = fast["slope"]
    d["kalman_slow_level"] = slow["level"]
    d["kalman_slow_slope"] = slow["slope"]
    return d