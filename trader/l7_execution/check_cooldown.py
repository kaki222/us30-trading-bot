"""
check_cooldown.py — check LiveCircuitBreaker.in_cooldown() against
EXISTING deal history, without opening any new positions. Use this
after the earlier test_circuit_breaker.py run to see whether the
timestamp-skew fix in in_cooldown() correctly picks up the 3 losses
already sitting in history under magic=999002.

    py -m trader.l7_execution.check_cooldown "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
"""

import sys

from . import connect, shutdown, account_summary, SYMBOL_MAP, LiveCircuitBreaker

TEST_ACCOUNT_LOGIN = 109989358
MAGIC = 999002  # same magic test_circuit_breaker.py used


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    connect(path=path)

    acct = account_summary()
    if acct["login"] != TEST_ACCOUNT_LOGIN:
        print(f"REFUSING TO CONTINUE: connected login is {acct['login']}, not the test account.")
        shutdown()
        return

    symbol = SYMBOL_MAP["GOLD"]
    breaker = LiveCircuitBreaker(symbol=symbol, magic=MAGIC, max_consecutive_losses=3, cooldown_bars=20)

    print(f"Checking in_cooldown() for {symbol}, magic={MAGIC} against existing history...")
    result = breaker.in_cooldown()
    print(f"in_cooldown() = {result}   (expect True - 3 losses already exist on this magic)")

    shutdown()


if __name__ == "__main__":
    main()
