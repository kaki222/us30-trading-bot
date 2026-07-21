
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
 

