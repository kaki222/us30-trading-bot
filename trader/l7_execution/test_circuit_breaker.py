"""
test_circuit_breaker.py — LiveCircuitBreaker test against real deal
history on the MetaQuotes-Demo TEST account only.

Generates real losing deals on purpose (open a tiny position, then
immediately close it — the bid/ask spread guarantees a small loss
without needing to guess market direction), then checks that
LiveCircuitBreaker.in_cooldown() flips True after max_consecutive_losses
in a row, and stays False before that.

Same account-login safety check as the other test scripts — refuses to
run against anything but the test account.

    py -m trader.l7_execution.test_circuit_breaker "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
"""

import sys
import time

from . import (
    connect, shutdown, account_summary, SYMBOL_MAP,
    place_trade, close_position, has_open_position, LiveCircuitBreaker,
)

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

TEST_ACCOUNT_LOGIN = 109989358
MAGIC = 999002  # separate magic from the earlier place_trade test, so deal history is clean for this test


def close_all_open(symbol, magic):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    for p in positions:
        if p.magic == magic:
            r = close_position(p.ticket)
            print(f"  closed pre-existing position {p.ticket}: retcode={r['result'].retcode}")
            time.sleep(1)


def open_and_close_one(symbol):
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask
    sl = price - 5.0
    tp = price + 500.0  # far away - we're closing manually right after, TP shouldn't matter

    opened = place_trade(symbol, "long", sl, tp, risk_pct=0.01, leverage=100,
                          magic=MAGIC, comment="l7_cb_test", dry_run=False)
    print(f"  opened: retcode={opened['result'].retcode} lots={opened['request']['volume']}")
    time.sleep(1.5)

    positions = mt5.positions_get(symbol=symbol)
    mine = [p for p in positions if p.magic == MAGIC] if positions else []
    if not mine:
        raise RuntimeError("Could not find the position just opened - check retcode above.")
    ticket = mine[0].ticket

    closed = close_position(ticket)
    print(f"  closed ticket {ticket}: retcode={closed['result'].retcode}")
    time.sleep(1.5)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    connect(path=path)

    acct = account_summary()
    if acct["login"] != TEST_ACCOUNT_LOGIN:
        print(f"REFUSING TO CONTINUE: connected login is {acct['login']}, "
              f"not the test account ({TEST_ACCOUNT_LOGIN}). Aborting.")
        shutdown()
        return

    symbol = SYMBOL_MAP["GOLD"]
    breaker = LiveCircuitBreaker(symbol=symbol, magic=MAGIC, max_consecutive_losses=3, cooldown_bars=20)

    print(f"Testing on {symbol}, magic={MAGIC}\n")

    print("Clearing any pre-existing open positions under this test magic...")
    close_all_open(symbol, MAGIC)

    print(f"\nBaseline in_cooldown(): {breaker.in_cooldown()}  (expect False - no losses yet)\n")

    for i in range(1, 4):
        print(f"--- Loss #{i}: open + immediately close (spread loss) ---")
        open_and_close_one(symbol)
        status = breaker.in_cooldown()
        expected = "False (not yet at threshold)" if i < 3 else "True (threshold hit)"
        print(f"  in_cooldown() = {status}   expected: {expected}\n")

    print("Done. If the last line above showed True, the breaker correctly "
          "detected 3 consecutive losses and is now enforcing a cooldown.")

    shutdown()


if __name__ == "__main__":
    main()
