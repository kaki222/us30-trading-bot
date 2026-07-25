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

Manual bias override (2026-07-25, made live-editable 2026-07-25):
lets you layer your own discretionary chart read (Elliott/Wyckoff or
otherwise) on top of the mechanical RegimeConfluenceStrategy signal,
WITHOUT touching l4_signal_model.py or
evaluate_regime_confluence_signal() at all — the strategy's actual
entry rule stays exactly as walk-forward-tested. Set to "long" or
"short" when you have a real discretionary view (e.g. watching GOLD
test the 4,165/4,180 shelf); leave it at its default None to let the
mechanical signal run untouched, which is the default and what it
should be most of the time.

_preview_signal() peeks at what the mechanical strategy would do this
cycle (read-only, no orders) purely to compare its direction against
your bias. If they agree, or there's no signal, or no bias is set,
nothing changes. If they disagree, BIAS_MODE decides what happens:
  "mute"     - skip the trade entirely, log why (default, safest)
  "downsize" - still take it, but at BIAS_DOWNSIZE_FACTOR x risk_pct
Either way it's logged in the journal so journal_summary.py shows when
and why a bias override fired.

Key-level logging (2026-07-25): records where price sat relative to a
manually-set discretionary level (e.g. GOLD's 4,180 upper-shelf/4,165-
peak invalidation and 3,958 (v)/(C) low invalidation from the
Elliott/Wyckoff read worked through in chat) on EVERY journal entry,
whether or not that run led to a trade. This is observational only -
it never gates or mutes anything (BIAS already does that job); it
exists purely so journal_summary.py can show "was price above/below
your shelf" alongside every decision after the fact, instead of the
levels only ever living in your head or in chat history.

Manual pause windows (2026-07-25): PAUSE_WINDOWS (below, still
hardcoded here - it's calendar dates, not something worth a live
toggle) skips a symbol entirely - no bias check, no run_once() call at
all - during a stretch you already expect to be noisy/whipsaw-prone
(e.g. the spring scenario's predicted flush window), rather than
relying on the circuit breaker, which only reacts AFTER it's already
eaten a string of losses. `paused_now` (below) is the same idea but an
immediate on/off you flip yourself, not a pre-set date range. Either
way it's journaled (reason "manual_pause_window" or "paused_now") so
it shows up in the weekly review like everything else.

Bias/key-levels/paused-now went from hardcoded dicts in this file to
manual_overrides.py's data/manual_overrides.json (2026-07-25) so
live_monitor.py's dashboard widgets can actually change them with a
click - this script now calls load_overrides() fresh at the top of
every cycle instead of reading a BIAS/KEY_LEVELS dict defined here.
See manual_overrides.py's module docstring for the full design (atomic
writes, the change-log file, why PAUSE_WINDOWS itself stayed out of it).

See live_monitor.py for a live desktop view of ER/regime per symbol,
plus the bias/pause/key-level controls themselves - useful to have
open alongside this while you're forming or watching a bias call.

    (venv) PS> python -m trader.l7_execution.run_scheduled "C:\\path\\to\\terminal64.exe"
"""

import sys
from datetime import datetime, timezone

from . import (
    connect, shutdown, account_summary, run_once, get_live_bars,
    build_live_features, evaluate_regime_confluence_signal, SYMBOL_MAP,
)
from .journal import append_entry
from .manual_overrides import load_overrides

MAGIC = 100001  # the "real" one - see module docstring
TIMEFRAME = "H4"  # confirmed best via the 2026-07-24 timeframe sweep

# RegimeConfluenceStrategy's own class defaults (l4_signal_model.py) -
# using these rather than re-deriving "the latest optimized fold's
# params" keeps this script simple and matches exactly what
# test_run_once.py already exercised as "real_params". Revisit if/when
# a process exists for picking up the latest walk-forward fold's
# optimized values automatically instead.
PARAMS = {"er_threshold": 0.35, "swing_lookback": 20, "atr_sl_mult": 1.5, "atr_tp_mult": 2.5}

# --- manual discretionary bias / key levels / paused_now (see module
# docstring) - now live in data/manual_overrides.json via
# manual_overrides.load_overrides(), NOT hardcoded here anymore. BIAS_MODE
# and BIAS_DOWNSIZE_FACTOR stay here since they're about HOW an override
# is applied, not WHAT the override currently is - no reason those need
# to be clickable.
BIAS_MODE = "mute"           # "mute" or "downsize"
BIAS_DOWNSIZE_FACTOR = 0.5   # only used when BIAS_MODE == "downsize"

# --- manual pause windows (calendar dates - stays hardcoded, see module
# docstring for why this is different from paused_now) ------------------
PAUSE_WINDOWS = {
    "US30": [],
    "GOLD": [
        (datetime(2026, 8, 7, 0, 0, tzinfo=timezone.utc), datetime(2026, 8, 10, 23, 59, tzinfo=timezone.utc)),
    ],
}


def _preview_signal(symbol_key: str) -> dict:
    """
    Read-only peek at what evaluate_regime_confluence_signal() would
    return right now for symbol_key, computed exactly the way
    run_once() computes it internally. Does NOT place anything and does
    NOT check open positions or the circuit breaker - it exists purely
    to compare the mechanical signal's direction against BIAS before
    deciding whether to let run_once() proceed normally.
    """
    symbol = SYMBOL_MAP[symbol_key]
    df = build_live_features(symbol, er_length=PARAMS.get("er_length", 20), timeframe=TIMEFRAME)
    return evaluate_regime_confluence_signal(
        df,
        er_threshold=PARAMS["er_threshold"],
        swing_lookback=PARAMS["swing_lookback"],
        atr_sl_mult=PARAMS["atr_sl_mult"],
        atr_tp_mult=PARAMS["atr_tp_mult"],
    )


def _key_level_context(symbol_key: str, levels: dict | None) -> dict | None:
    """
    Read-only annotation: where does the latest close sit relative to
    `levels` (this cycle's key_levels[symbol_key] from
    manual_overrides.load_overrides())? None if no levels are set (both
    invalidation_up/down are None). One extra live_bars fetch (count=1)
    - cheap, and kept independent of _preview_signal()/run_once() so
    this never has to change if either of those does.
    """
    if not levels or (levels.get("invalidation_up") is None and levels.get("invalidation_down") is None):
        return None
    symbol = SYMBOL_MAP[symbol_key]
    bars = get_live_bars(symbol, timeframe=TIMEFRAME, count=1)
    close = float(bars["close"].iloc[-1])
    up, down = levels.get("invalidation_up"), levels.get("invalidation_down")
    return {
        "close": close,
        "invalidation_up": up,
        "invalidation_down": down,
        "above_invalidation_up": (close > up) if up is not None else None,
        "below_invalidation_down": (close < down) if down is not None else None,
    }


def _in_pause_window(symbol_key: str, now: datetime) -> bool:
    return any(start <= now <= end for start, end in PAUSE_WINDOWS.get(symbol_key, []))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    connect(path=path)

    acct = account_summary()
    print(f"Connected: login={acct['login']} server={acct['server']} equity={acct['equity']}")

    now = datetime.now(timezone.utc)
    overrides = load_overrides()  # fresh read every cycle - see manual_overrides.py

    for symbol_key in ["US30", "GOLD"]:
        context = _key_level_context(symbol_key, overrides["key_levels"].get(symbol_key))

        paused_now = overrides["paused_now"].get(symbol_key, False)
        if _in_pause_window(symbol_key, now) or paused_now:
            reason = "paused_now" if paused_now and not _in_pause_window(symbol_key, now) else "manual_pause_window"
            result = {"action": "skip", "reason": reason}
            append_entry(symbol_key, TIMEFRAME, MAGIC, result, context=context)
            print(f"  {symbol_key}: skip ({reason})")
            continue

        bias = overrides["bias"].get(symbol_key)
        risk_pct = 0.01

        if bias:
            preview = _preview_signal(symbol_key)
            fights_bias = preview["signal"] is not None and preview["signal"] != bias

            if fights_bias and BIAS_MODE == "mute":
                result = {"action": "skip", "reason": f"bias_override(bias={bias}, signal={preview['signal']})"}
                append_entry(symbol_key, TIMEFRAME, MAGIC, result, context=context)
                print(f"  {symbol_key}: skip (bias_override - {bias} bias vs {preview['signal']} signal)")
                continue

            if fights_bias and BIAS_MODE == "downsize":
                risk_pct *= BIAS_DOWNSIZE_FACTOR
                print(f"  {symbol_key}: bias_override downsize ({bias} bias vs {preview['signal']} signal, "
                      f"risk_pct 0.01 -> {risk_pct})")

        result = run_once(
            symbol_key, PARAMS,
            risk_pct=risk_pct, leverage=30, magic=MAGIC,
            dry_run=True,  # see module docstring - deliberate, not a flag
            timeframe=TIMEFRAME,
        )
        append_entry(symbol_key, TIMEFRAME, MAGIC, result, context=context)
        print(f"  {symbol_key}: {result['action']}"
              + (f" ({result.get('reason')})" if result["action"] == "skip" else f" - {result['signal']['signal']}")
              + (f"  [close={context['close']:.2f} vs up={context['invalidation_up']} down={context['invalidation_down']}]"
                 if context else ""))

    shutdown()


if __name__ == "__main__":
    main()
