"""
debug_deals.py — inspect real deal history to debug why
LiveCircuitBreaker.in_cooldown() isn't seeing what we expect.

Prints every deal from the last 24h, both with and without the group
filter in_cooldown() uses, so we can see exactly what MT5 is returning
and compare against what the code assumes.

    py -m trader.l7_execution.debug_deals "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
"""

import sys
from datetime import datetime, timedelta, timezone

from . import connect, shutdown, account_summary, SYMBOL_MAP

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

TEST_ACCOUNT_LOGIN = 109989358


def dump(label, deals):
    print(f"\n{label}: {len(deals) if deals else 0} deals")
    if not deals:
        return
    for d in deals:
        t = datetime.fromtimestamp(d.time, tz=timezone.utc)
        print(f"  ticket={d.ticket} symbol={d.symbol!r} magic={d.magic} "
              f"entry={d.entry} profit={d.profit} time={t} type={d.type}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    connect(path=path)

    acct = account_summary()
    if acct["login"] != TEST_ACCOUNT_LOGIN:
        print(f"REFUSING TO CONTINUE: connected login is {acct['login']}, not the test account.")
        shutdown()
        return

    symbol = SYMBOL_MAP["GOLD"]

    tick = mt5.symbol_info_tick(symbol)
    server_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
    local_time = datetime.now(timezone.utc)
    skew = (local_time - server_time).total_seconds()
    print(f"Server time (from tick): {server_time}")
    print(f"Local Python time (UTC): {local_time}")
    print(f"Skew (local - server):   {skew:.0f} seconds")

    # Window built from SERVER time (matches deal.time's clock), not
    # local time - that mismatch was the actual bug. Padded generously.
    date_from = server_time - timedelta(days=7)
    date_to = server_time + timedelta(hours=1)

    total = mt5.history_deals_total(date_from, date_to)
    print(f"\nhistory_deals_total() over 7-day window: {total}")

    all_deals = mt5.history_deals_get(date_from, date_to)
    dump("ALL deals, no group filter, 7-day window", all_deals)

    grouped = mt5.history_deals_get(date_from, date_to, group=f"*{symbol}*")
    dump(f'Deals with group="*{symbol}*", 7-day window', grouped)

    print("\nENUM_DEAL_ENTRY reference: IN=0, OUT=1, INOUT=2, OUT_BY=3")

    shutdown()


if __name__ == "__main__":
    main()
