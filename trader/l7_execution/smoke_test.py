"""
smoke_test.py — run this FIRST, before anything else in l7_execution,
on the Windows machine with MT5 installed and a demo account logged in.

    py -m trader.l7_execution.smoke_test
    py -m trader.l7_execution.smoke_test "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

If you have more than one MT5 terminal installed (e.g. a broker-branded
one for a real account, plus a separate generic one for testing), pass
the exact path to the terminal you want THIS script to attach to as an
argument - mt5.initialize() with no path is ambiguous when multiple
terminals are installed, and attaching to the wrong one is exactly the
kind of mistake this script exists to avoid. Find the right path by
right-clicking that terminal's taskbar/desktop icon -> "Open file
location" and copying the full path to terminal64.exe from there.

It does nothing but read: connect, print account info, list symbols
that look like US30/Gold, pull 5 bars, disconnect. No orders are
placed. If any step here fails or prints something unexpected, stop —
don't move on to place_trade() until this passes cleanly.
"""

import sys

from . import connect, shutdown, account_summary, resolve_symbol, get_live_bars


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        print(f"Connecting to MT5 terminal at: {path}")
    else:
        print("Connecting to MT5 terminal (no path given - attaching to "
              "whichever terminal MT5 finds by default; pass an explicit "
              "path if you have more than one installed)...")
    connect(path=path)
    print("Connected.\n")

    print("Account summary:")
    for k, v in account_summary().items():
        print(f"  {k}: {v}")
    print()

    print("Looking for US30...")
    try:
        us30 = resolve_symbol(["US30", "US30Cash", "US30.cash", "US30m"])
        print(f"  found: {us30}")
    except ValueError as e:
        print(f"  {e}")

    print("Looking for Gold...")
    try:
        gold = resolve_symbol(["XAUUSD", "GOLD", "GOLDm", "XAUUSDm"])
        print(f"  found: {gold}")
    except ValueError as e:
        print(f"  {e}")

    print()
    print("If both were found, add them to SYMBOL_MAP in "
          "trader/l7_execution/__init__.py, then re-run this script — "
          "it'll also pull 5 live bars for each as a final check.")

    from . import SYMBOL_MAP
    for key, sym in SYMBOL_MAP.items():
        if sym:
            print(f"\nLast 5 {key} ({sym}) bars:")
            print(get_live_bars(sym, count=5))

    shutdown()
    print("\nDisconnected. Smoke test done.")


if __name__ == "__main__":
    main()
