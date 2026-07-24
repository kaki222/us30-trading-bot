"""
l1_data_export_h1.py — one-off script: pull real H1 history from a
running MT5 terminal and save it as a new raw CSV, in the exact same
format as the existing data/raw/*_h4_mt5.csv exports.

Why this has to be a script you run yourself (not something I can do
from here): H1 bars aren't in data/raw/ at all right now - only H4 - and
you can't manufacture H1 detail out of H4 data by resampling, only go
coarser (which is exactly what l1_data.load_bars() does for D1). Getting
real H1 history means asking a live MT5 terminal for it, and this
sandbox has no terminal to ask - your Windows machine does.

Run it the same way as the other l7_execution test scripts:

    (venv) PS> python -m trader.l1_data_export_h1 "C:\\path\\to\\terminal64.exe"

(path argument optional if a terminal is already logged in and running)

Honesty note, same as l7_execution: this has been written against the
documented MetaTrader5 Python API but not executed - no terminal is
reachable from this sandbox. Read the printed output carefully the
first time you run it.

Caveat that matters: how far back this can actually reach depends on
your broker's history retention and the terminal's own "Max bars in
history" setting (Tools > Options > Charts), not on this script. It
asks for a wide range (2010-01-01 to now) and just takes whatever MT5
actually hands back - check the printed date range in the output before
assuming you got the same multi-year depth the H4 exports have.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from .l7_execution import connect, shutdown, resolve_symbol, account_summary

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# symbol_key -> (candidate names to try via resolve_symbol, output filename)
EXPORT_TARGETS = {
    "US30": (["US30Cash", "US30", "US30.cash", "US30m"], "us30_h1_mt5.csv"),
    "GOLD": (["GOLD", "XAUUSD", "GOLDm", "XAUUSDm"], "gold_h1_mt5.csv"),
}


def export_h1(symbol: str, out_path: Path) -> None:
    # copy_rates_range() rejects timezone-aware datetime objects with a
    # bare (-2, "Invalid params") - no other explanation given. Fix:
    # pass naive datetimes (MT5 treats them as UTC/server time itself).
    # Confirmed via real run 2026-07-23: tz-aware datetime(...,
    # tzinfo=timezone.utc) failed with exactly this error on both
    # symbols; this is the fix, not yet re-confirmed after the fix.
    date_from = datetime(2010, 1, 1)
    date_to = datetime.now(timezone.utc).replace(tzinfo=None)

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, date_from, date_to)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        print(f"  {symbol}: copy_rates_range returned nothing ({err}) - trying copy_rates_from_pos fallback...")
        # Fallback: ask for the most recent N bars by position instead of
        # a date range - simpler call, less prone to whatever the range
        # variant's param validation is rejecting. 200,000 H1 bars is
        # ~22 years if the broker actually has that much history; MT5
        # just returns however many it actually has, doesn't error if
        # fewer are available.
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200_000)
        if rates is None or len(rates) == 0:
            print(f"  {symbol}: fallback also returned nothing ({mt5.last_error()}) - skipping.")
            return

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)

    # Write out in the same <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
    # tab-separated shape l1_data.load_mt5_h4() already knows how to parse,
    # so the new file is a drop-in - no new parser needed.
    out = pd.DataFrame({
        "<DATE>": df["time"].dt.strftime("%Y.%m.%d"),
        "<TIME>": df["time"].dt.strftime("%H:%M:%S"),
        "<OPEN>": df["open"],
        "<HIGH>": df["high"],
        "<LOW>": df["low"],
        "<CLOSE>": df["close"],
        "<TICKVOL>": df["tick_volume"],
        "<VOL>": df["real_volume"] if "real_volume" in df.columns else 0,
        "<SPREAD>": df["spread"] if "spread" in df.columns else 0,
    })
    out.to_csv(out_path, sep="\t", index=False)

    print(f"  {symbol}: {len(out)} H1 bars, {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    print(f"  saved to {out_path}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    connect(path=path)

    acct = account_summary()
    print("Connected account:")
    for k, v in acct.items():
        print(f"  {k}: {v}")
    print()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for symbol_key, (candidates, filename) in EXPORT_TARGETS.items():
        print(f"=== {symbol_key} ===")
        try:
            symbol = resolve_symbol(candidates)
            print(f"  resolved to {symbol}")
        except ValueError as e:
            print(f"  {e}")
            continue
        export_h1(symbol, DATA_DIR / filename)
        print()

    shutdown()
    print("Done. Check the printed date ranges above before assuming full history depth -")
    print("if it's much shorter than the H4 exports, that's your broker's retention limit, not a bug.")


if __name__ == "__main__":
    main()
