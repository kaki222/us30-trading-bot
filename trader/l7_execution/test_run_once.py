"""
test_run_once.py — end-to-end test of the run_once() orchestration:
has_open_position -> circuit breaker -> features -> signal -> (dry-run)
trade, all in one call, against whichever account this is pointed at.

dry_run is never overridden away from True here, so no code path in
this script can call mt5.order_send() - safe against a real account.
Uses SYMBOL_MAP as-is (currently XM's confirmed names), so point this
at an XM-family terminal, not the MetaQuotes-Demo test account (whose
symbol names don't match SYMBOL_MAP right now).

Runs each instrument twice: once with the real strategy params (likely
to hit the "no signal" skip path if current market conditions are
choppy, same as test_signal_readonly.py found), and once with
er_threshold forced to 0.0 to force the regime gate open, to also
exercise the "trade" (dry-run) branch at least once even if real
conditions don't currently produce a signal.

    py -m trader.l7_execution.test_run_once "C:\\path\\to\\terminal64.exe"
"""

import sys

from . import connect, shutdown, account_summary, run_once

MAGIC = 999003  # fresh magic - keeps has_open_position()/circuit breaker checks clean of earlier tests


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    connect(path=path)

    acct = account_summary()
    print("Connected account:")
    for k, v in acct.items():
        print(f"  {k}: {v}")
    print()

    real_params = {"er_threshold": 0.35, "swing_lookback": 20, "atr_sl_mult": 1.5, "atr_tp_mult": 2.5}
    loose_params = {"er_threshold": 0.0, "swing_lookback": 20, "atr_sl_mult": 1.5, "atr_tp_mult": 2.5}

    for key in ["US30", "GOLD"]:
        print(f"=== {key}, real params (er_threshold=0.35) ===")
        result = run_once(key, real_params, risk_pct=0.01, leverage=100, magic=MAGIC, dry_run=True)
        print(f"  {result}\n")

        print(f"=== {key}, forced-open regime gate (er_threshold=0.0) ===")
        result2 = run_once(key, loose_params, risk_pct=0.01, leverage=100, magic=MAGIC, dry_run=True)
        print(f"  {result2}\n")

    shutdown()
    print("Done. dry_run=True throughout - no orders were placed regardless of the results above.")


if __name__ == "__main__":
    main()
