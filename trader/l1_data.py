
"""
l1_data.py — Layer 1: Market Data
 
Loads raw H4 OHLCV bars from MT5 exports. Both US30 and Gold now have
full MT5 H4 history, so this module is MT5-only — no yfinance/Yahoo
fallback.
 
- US30 export: hybrid file — Daily bars before 2016-05-26, true H4 bars
  (~6/day) from 2016-05-26 onward. Rows before the cutoff are dropped.
- Gold export: clean H4 bars (~6/day) from 2013-05-09 onward. Verified
  by checking bar spacing/day across the full range — no Daily/H4 split
  like US30 has, so no cutoff is applied.
 
load_h4(symbol) returns the same shape for either instrument: lowercase
OHLCV columns, index named "time".
"""
 
from pathlib import Path
 
import pandas as pd
 
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
 
# Per-symbol MT5 export config: default file path + cutoff date (or None
# if the export is clean H4 from its first row).
MT5_SOURCES = {
    "US30": {
        "path": DATA_DIR / "us30_h4_mt5.csv",
        "cutoff": "2016-05-26",  # rows before this are Daily, not H4
    },
    "GOLD": {
        "path": DATA_DIR / "gold_h4_mt5.csv",
        "cutoff": None,  # clean H4 from 2013-05-09, no split
    },
}
 
 
def load_mt5_h4(path: str | Path, cutoff: str | None = None) -> pd.DataFrame:
    """
    Load an MT5 CSV/XLSX export (columns: <DATE> <TIME> <OPEN> <HIGH> <LOW>
    <CLOSE> <TICKVOL> <VOL> <SPREAD>) and return H4 bars: lowercase OHLCV
    columns, index named "time".
 
    `cutoff`, if given, drops rows before that date — use this when the
    export is a hybrid (e.g. Daily bars before some date, H4 after).
    """
    path = Path(path)
 
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path, sep="\t")
 
    raw.columns = [c.strip().strip("<>").upper() for c in raw.columns]
 
    required = {"DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "TICKVOL"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Unexpected MT5 export format, missing columns: {missing}")
 
    dt = pd.to_datetime(
        raw["DATE"].astype(str).str.replace(".", "-", regex=False) + " " + raw["TIME"].astype(str),
        format="%Y-%m-%d %H:%M:%S",
    )
 
    h4 = pd.DataFrame(
        {
            # .values (not the bare Series) is required here: a dict of
            # Series passed to pd.DataFrame() aligns each Series by its
            # own index against `index=`, not positionally. raw's OHLC
            # columns still carry the default 0..n RangeIndex, which
            # doesn't match the datetime `dt` index at all — that
            # mismatch silently produces all-NaN columns instead of
            # raising, so it's easy to miss.
            "open": raw["OPEN"].astype(float).values,
            "high": raw["HIGH"].astype(float).values,
            "low": raw["LOW"].astype(float).values,
            "close": raw["CLOSE"].astype(float).values,
            "volume": raw["TICKVOL"].astype(float).values,
        },
        index=dt,
    )
    h4.index.name = "time"
 
    if cutoff is not None:
        h4 = h4[h4.index >= pd.Timestamp(cutoff)]
    h4 = h4[~h4.index.duplicated(keep="first")].sort_index()
 
    return h4
 
 
def load_h4(symbol: str) -> pd.DataFrame:
    """Load H4 bars for a known symbol ("US30" or "GOLD") from its default MT5 export."""
    symbol = symbol.upper()
    if symbol not in MT5_SOURCES:
        raise ValueError(f"Unknown symbol {symbol!r}. Known: {list(MT5_SOURCES)}")
    cfg = MT5_SOURCES[symbol]
    return load_mt5_h4(cfg["path"], cfg["cutoff"])


# H1 export config - same shape as MT5_SOURCES, populated by running
# trader/l1_data_export_h1.py once on the Windows machine (needs a live
# MT5 terminal, not available from this sandbox). No cutoff handling
# here since the export script always writes clean single-timeframe
# files, unlike the hybrid US30 H4 export.
MT5_H1_SOURCES = {
    "US30": DATA_DIR / "us30_h1_mt5.csv",
    "GOLD": DATA_DIR / "gold_h1_mt5.csv",
}


def load_h1(symbol: str) -> pd.DataFrame:
    """
    Load H1 bars for a known symbol from its export - same shape as
    load_h4(). Raises a clear, actionable error (not a bare
    FileNotFoundError) if the export hasn't been run yet.
    """
    symbol = symbol.upper()
    if symbol not in MT5_H1_SOURCES:
        raise ValueError(f"Unknown symbol {symbol!r}. Known: {list(MT5_H1_SOURCES)}")
    path = MT5_H1_SOURCES[symbol]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} doesn't exist yet. H1 bars need a real MT5 export - run "
            f"`python -m trader.l1_data_export_h1` on the Windows machine with "
            f"the MT5 terminal (see that file's docstring), then try again."
        )
    return load_mt5_h4(path, cutoff=None)  # same tab-separated MT5 export format, works unchanged


# Timeframes derivable from the H4 export by resampling (coarser only -
# you can't manufacture H1 bars from H4 data, only aggregate H4 into
# something wider). "H4" itself is the native/no-op case. "H1" is
# handled separately below (native export, not a resample of H4).
RESAMPLE_RULES = {
    "H4": None,   # native, no resampling
    "D1": "1D",   # calendar-day aggregation of H4 bars
}


def resample_ohlcv(h4: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Aggregate H4 OHLCV bars up to a coarser timeframe (e.g. "1D" for
    daily). Standard OHLCV resample: open=first, high=max, low=min,
    close=last, volume=sum. Drops any resulting bar with no underlying
    H4 bars (e.g. weekends) rather than leaving it NaN.
    """
    agg = h4.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return agg.dropna(subset=["open", "high", "low", "close"])


def load_bars(symbol: str, timeframe: str = "H4") -> pd.DataFrame:
    """
    Load bars for a known symbol at the given timeframe: "H1" (native,
    needs trader/l1_data_export_h1.py to have been run once), "H4"
    (native, the original export), or "D1" (resampled from the H4
    export - no separate file needed).
    """
    timeframe = timeframe.upper()
    if timeframe == "H1":
        return load_h1(symbol)
    if timeframe not in RESAMPLE_RULES:
        raise ValueError(
            f"Unknown timeframe {timeframe!r}. Known: 'H1', {list(RESAMPLE_RULES)} "
            f"('H1' needs its own MT5 export - see trader/l1_data_export_h1.py)."
        )
    h4 = load_h4(symbol)
    rule = RESAMPLE_RULES[timeframe]
    if rule is None:
        return h4
    return resample_ohlcv(h4, rule)
 

