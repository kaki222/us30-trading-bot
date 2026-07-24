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
paginates backward in 5,000-bar chunks until the terminal stops
returning full chunks - check the printed date range in the output
before assuming you got the same multi-year depth the H4 exports have.

Fetch method note: an earlier version tried a single copy_rates_range()
call (2010 -> now) and, after that failed, a single huge
copy_rates_from_pos(0, 200_000) call - both failed identically with
(-2, 'Invalid params') on a real run against XM demo 345899957. Since
copy_rates_from_pos doesn't take dates at all, the common factor was
the oversized single request, not a timezone issue - the terminal
appears to reject requests above some undocumented size. Paginating in
5,000-bar chunks (well under the 800-bar count that's worked all
session for live pulls) works around that.
"""

import sys
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


def export_h1(symbol: str, out_path: Path, chunk_size: int = 5000, max_chunks: int = 100) -> None:
    """
    Paginated fetch: copy_rates_from_pos(symbol, H1, start_pos, count)
    walking backward in chunk_size-bar steps, accumulating, until a
    chunk comes back empty (reached the earliest available history) or
    max_chunks is hit. Both the single-shot copy_rates_range(2010->now)
    and a single huge copy_rates_from_pos(0, 200_000) call failed with
    (-2, 'Invalid params') on a real run 2026-07-23 - the range variant
    fails on timezone-aware datetimes (a documented gotcha), but the
    from_pos variant takes no dates at all and still failed identically
    with only `count` unusually large, which points at the terminal
    rejecting oversized single requests rather than a datetime issue.
    chunk_size=5000 is well under that, matching the get_live_bars()
    count=800 default that's worked all session, just chunked to cover
    real history depth instead of only the most recent count.
    """
    all_chunks = []
    start_pos = 0
    for i in range(max_chunks):
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, start_pos, chunk_size)
        if rates is None:
            print(f"  {symbol}: chunk {i} (start_pos={start_pos}) failed: {mt5.last_error()}")
            break
        if len(rates) == 0:
            break  # ran out of history
        all_chunks.append(pd.DataFrame(rates))
        if len(rates) < chunk_size:
            break  # partial chunk = this was the oldest available data
        start_pos += chunk_size

    if not all_chunks:
        print(f"  {symbol}: no H1 bars returned at all - skipping.")
        return

    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
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
