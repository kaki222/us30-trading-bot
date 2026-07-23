"""
test_signal_readonly.py — test the signal-generation path against a
REAL feed (intended for the XM terminal), using only read-only calls.

This script never imports or calls place_trade(), close_position(), or
mt5.order_send() anywhere - there is no code path in this file that
can submit an order, regardless of which account is connected. That's
what makes it safe to point at the real XM terminal (330507861): it
only reads account info, resolves symbol names fresh (doesn't trust
SYMBOL_MAP, which was resolved against the test account and isn't
assumed to carry over), pulls live bars, computes features, and
evaluates the signal rules - all read paths.

    py -m trader.l7_execution.test_signal_readonly "C:\\Program Files (x86)\\XMGlobal MT5\\terminal64.exe"

(path above is a guess at XM's install location - use the real path,
found the same way as before: right-click the XM terminal's taskbar
icon -> Open file location.)
"""

import sys

from . import connect, shutdown, account_summary, resolve_symbol, build_live_features, evaluate_regime_confluence_signal


def report(label, symbol):
    print(f"\n=== {label} ({symbol}) ===")
    feats = build_live_features(symbol, er_length=20, count=800)
    last = feats.iloc[-1]
    print("Last bar features:")
    for col in ["Close", "ma_360", "ma_200", "ema_21", "ema_8", "macd_hist", "atr_14", "adx_14", "er"]:
        val = last[col]
        print(f"  {col}: {val}")

    signal = evaluate_regime_confluence_signal(feats)
    print(f"\nSignal: {signal}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    connect(path=path)

    acct = account_summary()
    print("Connected account:")
    for k, v in acct.items():
        print(f"  {k}: {v}")

    print("\nResolving real symbol names on this account (not using SYMBOL_MAP - "
          "that was resolved against the test account, not this one)...")
    try:
        us30 = resolve_symbol(["US30", "US30Cash", "US30.cash", "US30m"])
        print(f"  US30 -> {us30}")
        report("US30", us30)
    except ValueError as e:
        print(f"  US30: {e}")

    try:
        gold = resolve_symbol(["GOLD", "XAUUSD", "GOLDm", "XAUUSDm"])
        print(f"  GOLD -> {gold}")
        report("GOLD", gold)
    except ValueError as e:
        print(f"  GOLD: {e}")

    shutdown()
    print("\nDone. No orders were placed - this script has no order-sending code path.")


if __name__ == "__main__":
    main()
