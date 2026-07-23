"""
test_run_once_forced_signal.py — exercises run_once()'s "trade" branch
specifically, since real market conditions didn't satisfy the
confluence rules for either instrument at test time (confirmed by
test_run_once.py returning "no signal" even with the regime gate
forced open via er_threshold=0.0).

Rather than wait for market conditions to align, this monkeypatches
evaluate_regime_confluence_signal() to always return a fixed fake
signal, so we can verify the actual GLUE in run_once() - does it
correctly pass that signal's price/sl/tp into place_trade() and
return the right shape - using a real, live connection for everything
else (has_open_position, circuit breaker check, real account/symbol).

dry_run is never set to False here - this proves the WIRING is
correct, not a real order submission (that part was already proven
separately, for real, in test_place_trade.py).

    py -m trader.l7_execution.test_run_once_forced_signal "C:\\path\\to\\terminal64.exe"
"""

import sys
from unittest.mock import patch

from . import connect, shutdown, account_summary, run_once

MAGIC = 999004  # fresh magic - clean has_open_position()/circuit breaker state


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    connect(path=path)

    acct = account_summary()
    print("Connected account:")
    for k, v in acct.items():
        print(f"  {k}: {v}")

    fake_signal = {"signal": "long", "price": 12345.0, "sl": 12300.0, "tp": 12450.0}
    print(f"\nMonkeypatching evaluate_regime_confluence_signal() to always return: {fake_signal}")

    with patch("trader.l7_execution.evaluate_regime_confluence_signal", return_value=fake_signal):
        result = run_once("US30", {}, risk_pct=0.01, leverage=100, magic=MAGIC, dry_run=True)

    print(f"\nrun_once() result: {result}")

    assert result["action"] == "trade", f"Expected action='trade', got {result['action']!r}"
    assert result["signal"] == fake_signal, "run_once() didn't pass through the exact signal dict"
    assert result["trade"]["dry_run"] is True
    assert result["trade"]["would_send"]["sl"] == fake_signal["sl"], "SL didn't make it into the order request"
    assert result["trade"]["would_send"]["tp"] == fake_signal["tp"], "TP didn't make it into the order request"
    print("\nAll checks passed: run_once() correctly wires a found signal through to place_trade().")

    shutdown()


if __name__ == "__main__":
    main()
