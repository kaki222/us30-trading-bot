"""
l1_data.py — Layer 1: Market Data

Fetches raw OHLCV bars and resamples to 4H. This is the only place
that talks to yfinance; every other layer works off the DataFrame
this returns.
"""

import pandas as pd
import yfinance as yf


def fetch_h4(symbol: str, start: str = None, period: str = "730d") -> pd.DataFrame:
    """
    Fetch 60m bars and resample to 4H candles.

    Yahoo limits 60m/intraday history to the last 730 days from *now*,
    measured at call time - prefer `period` (safe default) unless you
    specifically need an explicit start date within that window.
    """
    if start is not None:
        raw = yf.download(symbol, start=start, interval="60m", auto_adjust=True, progress=False)
    else:
        raw = yf.download(symbol, period=period, interval="60m", auto_adjust=True, progress=False)

    if raw.empty:
        raise RuntimeError(f"No data returned for {symbol}.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(columns=str.lower)
    raw = raw[["open", "high", "low", "close", "volume"]]

    h4 = raw.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(how="any")
    h4.index.name = "time"
    return h4