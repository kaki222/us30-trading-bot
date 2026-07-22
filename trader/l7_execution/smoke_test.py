"""
smoke_test.py — run this FIRST, before anything else in l7_execution,
on the Windows machine with MT5 installed and a demo account logged in.

    py -m trader.l7_execution.smoke_test

It does nothing but read: connect, print account info, list symbols
that look like US30/Gold, pull 5 bars, disconnect. No orders are
placed. If any step here fails or prints something unexpected, stop —
don't move on to place_trade() until this passes cleanly.
"""

from . import connect, shutdown, account_summary, resolve_symbol, get_live_bars


def main():
    print("Connecting to MT5 terminal...")
    connect()
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
