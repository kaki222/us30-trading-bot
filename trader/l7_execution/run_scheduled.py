"""
run_scheduled.py — single-shot script: one run_once() pass over both
instruments, journaled. Meant to be invoked periodically by Windows
Task Scheduler (every 4 hours, to line up with H4 candle closes) rather
than run as a long-lived Python loop - a scheduled task survives
reboots and doesn't depend on a terminal window staying open; a
`while True: sleep(...)` script doesn't.

Why this can't be something I trigger for you automatically: my own
scheduled-tasks tool runs in my own sandbox, which has no MT5 terminal
and never will - MT5 only exists on your Windows machine. So the timer
has to live on your machine too. Task Scheduler setup is in
ARCHITECTURE.md's Layer 7 section.

dry_run stays True here, deliberately hardcoded (not a CLI flag) - this
script's whole purpose right now is building a live paper track record
you can review, not placing real orders. When you're ready to go live
on the demo, that's a deliberate one-line change here, not an accident
waiting to happen from a forgotten flag.

Uses run_once()'s default magic (100001) and timeframe ("H4") -
deliberately the only ones NOT prefixed 999xxx, since every 999xxx
magic across the test scripts was chosen specifically to stay out of
this one's way. H4 because the 2026-07-24 timeframe sweep found H4
clearly the best of H1/H4/D1 for this strategy - see ARCHITECTURE.md.

    (venv) PS> python -m trader.l7_execution.run_scheduled "C:\\path\\to\\terminal64.exe"
"""

import sys

from . import connect, shutdown, account_summary, run_once
from .journal import append_entry

MAGIC = 100001  # the "real" one - see module docstring
TIMEFRAME = "H4"  # confirmed best via the 2026-07-24 timeframe sweep

# RegimeConfluenceStrategy's own class defaults (l4_signal_model.py) -
# using these rather than re-deriving "the latest optimized fold's
# params" keeps this script simple and matches exactly what
# test_run_once.py already exercised as "real_params". Revisit if/when
# a process exists for picking up the latest walk-forward fold's
# optimized values automatically instead.
PARAMS = {"er_threshold": 0.35, "swing_lookback": 20, "atr_sl_mult": 1.5, "atr_tp_mult": 2.5}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    connect(path=path)

    acct = account_summary()
    print(f"Connected: login={acct['login']} server={acct['server']} equity={acct['equity']}")

    for symbol_key in ["US30", "GOLD"]:
        result = run_once(
            symbol_key, PARAMS,
            risk_pct=0.01, leverage=30, magic=MAGIC,
            dry_run=True,  # see module docstring - deliberate, not a flag
            timeframe=TIMEFRAME,
        )
        append_entry(symbol_key, TIMEFRAME, MAGIC, result)
        print(f"  {symbol_key}: {result['action']}"
              + (f" ({result.get('reason')})" if result["action"] == "skip" else f" - {result['signal']['signal']}"))

    shutdown()


if __name__ == "__main__":
    main()
