"""
test_place_trade.py — Layer 7 place_trade() test: dry-run first, then
(only if you explicitly confirm) one real tiny order.

SAFETY: this refuses to send a real order unless the connected
account's login matches TEST_ACCOUNT_LOGIN below. That's a hard check
on the account number itself, not just "which terminal path you
passed" - even if you accidentally point this at the wrong terminal,
it won't fire a real order unless the login number matches.

    py -m trader.l7_execution.test_place_trade "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
"""

import sys

from . import connect, shutdown, account_summary, SYMBOL_MAP, place_trade

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

TEST_ACCOUNT_LOGIN = 109989358  # MetaQuotes-Demo throwaway account - the only one this script will trade on


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    connect(path=path)

    acct = account_summary()
    print("Connected account:")
    for k, v in acct.items():
        print(f"  {k}: {v}")

    if acct["login"] != TEST_ACCOUNT_LOGIN:
        print(f"\nREFUSING TO CONTINUE: connected login is {acct['login']}, "
              f"not the test account ({TEST_ACCOUNT_LOGIN}). This script only "
              f"trades on the test account. Aborting - no order sent.")
        shutdown()
        return

    symbol = SYMBOL_MAP["GOLD"]  # "XAUUSD" on the test account
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask
    sl = price - 5.0   # arbitrary small stop, just to exercise the sizing math
    tp = price + 10.0

    print(f"\n{symbol} current ask: {price}")
    print(f"Test SL: {sl}   TP: {tp}")

    print("\n--- DRY RUN ---")
    dry = place_trade(symbol, "long", sl, tp, risk_pct=0.0000001, leverage=100,
                       magic=999001, comment="l7_test", dry_run=True)
    for k, v in dry.items():
        print(f"  {k}: {v}")

    print("\nThe 'lots' value above is exactly what would be sent. Review it.")
    answer = input("Type 'yes' to send this as a REAL order on the TEST account, anything else to stop: ")
    if answer.strip().lower() != "yes":
        print("Stopping - no real order sent.")
        shutdown()
        return

    print("\n--- SENDING REAL ORDER (test account only) ---")
    real = place_trade(symbol, "long", sl, tp, risk_pct=0.0000001, leverage=100,
                        magic=999001, comment="l7_test", dry_run=False)
    for k, v in real.items():
        print(f"  {k}: {v}")

    shutdown()


if __name__ == "__main__":
    main()
