"""
journal_summary.py — reads data/journal.jsonl (written by
run_scheduled.py) and prints a weekly-style review: how often each
instrument skipped and why, how often a signal actually fired, and -
once dry_run is eventually flipped to False and real trades start
closing - realized win rate/expectancy compared against what the
walk-forward backtest expected, which is the whole point of keeping
this journal in the first place (catching live drift from backtest
assumptions early, not months in).

Key-level context (2026-07-25): if an entry carries a "context" field
(the key_levels this cycle read via manual_overrides.py - e.g. GOLD's
4,180/3,958 discretionary invalidation levels), each trade line shows
where price sat relative to that level at decision time, and each
symbol block gets an above/below tally across the whole window - turns
"I think it broke the shelf around Tuesday" into something you can
actually check.

Manual override change log (2026-07-26): prints every bias/pause/key-
level change made via live_monitor.py's dashboard widgets (or a direct
edit of data/manual_overrides.json) within the lookback window, before
the per-symbol breakdown - see manual_overrides.py.

    (venv) PS> python -m trader.l7_execution.journal_summary
    (venv) PS> python -m trader.l7_execution.journal_summary --days 30
"""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone

from .journal import read_entries
from .manual_overrides import read_change_log


def _context_str(ctx: dict) -> str:
    """One-line summary of where price sat vs. the discretionary key levels, for a single entry."""
    bits = [f"close={ctx['close']:.2f}"]
    if ctx.get("above_invalidation_up"):
        bits.append(f"ABOVE up-invalidation {ctx['invalidation_up']}")
    elif ctx.get("invalidation_up") is not None:
        bits.append(f"below up-invalidation {ctx['invalidation_up']}")
    if ctx.get("below_invalidation_down"):
        bits.append(f"BELOW down-invalidation {ctx['invalidation_down']}")
    elif ctx.get("invalidation_down") is not None:
        bits.append(f"above down-invalidation {ctx['invalidation_down']}")
    return ", ".join(bits)


def _print_override_changes(days: int) -> None:
    """
    Manual bias/pause/key-level changes (2026-07-26) made via
    live_monitor.py's dashboard widgets (or a direct edit of
    data/manual_overrides.json) - printed separately from the journal
    itself since these are config changes, not run_once() results.
    Shown first, before the per-symbol trade/skip breakdown, since a
    bias flip or pause toggle is usually the reason the numbers below
    it look different from the week before.
    """
    log = read_change_log()
    if not log:
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    recent = [e for e in log if datetime.fromisoformat(e["timestamp"]) >= cutoff]
    if not recent:
        return
    print(f"=== Manual override changes ({len(recent)} in the last {days} day(s)) ===")
    for e in recent:
        print(f"  {e['timestamp']}: {e['symbol_key']} {e['field']}: {e['old']} -> {e['new']}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="lookback window (default 7, i.e. weekly)")
    args = parser.parse_args()

    _print_override_changes(args.days)

    entries = read_entries()
    if not entries:
        print("Journal is empty - run_scheduled.py hasn't run yet, or hasn't run since the journal path changed.")
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    recent = [e for e in entries if datetime.fromisoformat(e["timestamp"]) >= cutoff]

    print(f"Journal: {len(entries)} total entries, {len(recent)} in the last {args.days} day(s).\n")
    if not recent:
        print("Nothing in the lookback window.")
        return

    by_symbol = {}
    for e in recent:
        by_symbol.setdefault(e["symbol_key"], []).append(e)

    real_trades = []
    dry_run_trades = []

    for symbol_key, runs in sorted(by_symbol.items()):
        actions = Counter(r["result"]["action"] for r in runs)
        skip_reasons = Counter(r["result"]["reason"] for r in runs if r["result"]["action"] == "skip")
        print(f"=== {symbol_key} ({len(runs)} runs) ===")
        print(f"  actions: {dict(actions)}")
        if skip_reasons:
            print(f"  skip reasons: {dict(skip_reasons)}")

        contexts = [r["context"] for r in runs if r.get("context")]
        if contexts:
            above = sum(1 for c in contexts if c.get("above_invalidation_up"))
            below = sum(1 for c in contexts if c.get("below_invalidation_down"))
            lvl = contexts[-1]
            print(f"  key levels: up={lvl.get('invalidation_up')} down={lvl.get('invalidation_down')}  "
                  f"({above}/{len(contexts)} runs above up-shelf, {below}/{len(contexts)} below down-shelf)")

        for r in runs:
            res = r["result"]
            ctx = r.get("context")
            ctx_suffix = f"  [{_context_str(ctx)}]" if ctx else ""
            if res["action"] == "trade":
                direction = res["signal"]["signal"]
                is_dry = res["trade"]["dry_run"]
                (dry_run_trades if is_dry else real_trades).append(r)
                tag = "DRY-RUN" if is_dry else "REAL"
                print(f"  [{tag}] {r['timestamp']}: {direction} signal, "
                      f"sl={res['signal']['sl']:.2f} tp={res['signal']['tp']:.2f}{ctx_suffix}")
            elif res.get("reason") == "manual_pause_window":
                print(f"  [PAUSED] {r['timestamp']}: manual_pause_window{ctx_suffix}")
        print()

    print(f"Totals: {len(dry_run_trades)} dry-run signal(s), {len(real_trades)} real trade(s) in this window.")
    if real_trades:
        print("Real trades exist - once positions from these have closed, compare realized win rate/")
        print("expectancy against the walk-forward backtest's numbers (see ARCHITECTURE.md) to check for live drift.")
    else:
        print("No real trades yet (dry_run=True throughout) - this is purely a signal-frequency check so far.")


if __name__ == "__main__":
    main()
