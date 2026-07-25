"""
live_monitor.py — desktop dashboard: one live candlestick chart per
instrument, refreshed on an interval, each titled with its current
Efficiency Ratio / regime classification, your manual BIAS setting
(from run_scheduled.py), and the circuit breaker's live cooldown
status. Also prints the same info as a one-line-per-symbol log to the
console each cycle, so you have a plain-text history even with the
chart window closed.

Read-only: this never calls place_trade() or touches an order, and
never runs run_once() - it only reads bars and computes the same
features run_once() would, on its own timer, independent of the
scheduled trading cycle. Safe to leave open all day.

BIAS/TIMEFRAME/PARAMS/MAGIC are imported from run_scheduled.py rather
than redefined here, so this always reflects whatever that script is
actually configured to trade - editing your bias in one place updates
what both scripts see.

Needs mplfinance in addition to what run_scheduled.py already needs:
    (venv) PS> pip install mplfinance

Close the chart window or Ctrl+C in the terminal to stop.

    (venv) PS> python -m trader.l7_execution.live_monitor
    (venv) PS> python -m trader.l7_execution.live_monitor "C:\\path\\to\\terminal64.exe"
    (venv) PS> python -m trader.l7_execution.live_monitor --refresh 30 --bars 150
    (venv) PS> python -m trader.l7_execution.live_monitor --timeframe H1
"""

import argparse
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

from . import (
    connect, shutdown, SYMBOL_MAP, TIMEFRAME_SECONDS,
    build_live_features, LiveCircuitBreaker,
)
from .run_scheduled import BIAS, TIMEFRAME, PARAMS, MAGIC


def _status_line(symbol_key: str, symbol: str, df: pd.DataFrame, timeframe: str) -> tuple[str, dict]:
    """One symbol's current readout: close, ER, regime, bias, breaker status."""
    row = df.iloc[-1]
    er = row["er"]
    trending = bool(er > PARAMS["er_threshold"]) if pd.notna(er) else False
    regime = "TRENDING" if trending else "CHOP" if pd.notna(er) else "warming up"
    bias = BIAS.get(symbol_key) or "neutral"

    breaker = LiveCircuitBreaker(symbol=symbol, magic=MAGIC, bar_seconds=TIMEFRAME_SECONDS[timeframe.upper()])
    cooldown = breaker.in_cooldown()

    line = (f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  "
            f"{symbol_key:6s} close={row['Close']:>10.2f}  "
            f"ER={er:.3f}  regime={regime:9s} bias={bias:7s} "
            f"breaker={'COOLDOWN' if cooldown else 'clear'}")
    info = {"close": row["Close"], "er": er, "regime": regime, "bias": bias, "cooldown": cooldown}
    return line, info


def run(refresh_seconds: int = 60, bars: int = 150, timeframe: str | None = None, mt5_path: str | None = None):
    tf = timeframe or TIMEFRAME
    connect(path=mt5_path)

    symbols = list(SYMBOL_MAP.items())
    plt.ion()
    fig, axes = plt.subplots(len(symbols), 1, figsize=(11, 4.2 * len(symbols)))
    if len(symbols) == 1:
        axes = [axes]

    print(f"live_monitor: {tf} bars, refreshing every {refresh_seconds}s. Ctrl+C or close the window to stop.\n")

    try:
        while True:
            for ax, (symbol_key, symbol) in zip(axes, symbols):
                ax.clear()
                df = build_live_features(symbol, er_length=PARAMS.get("er_length", 20), timeframe=tf)
                chart_df = df[["Open", "High", "Low", "Close", "Volume"]].tail(bars)

                line, info = _status_line(symbol_key, symbol, df, tf)
                print(line)

                mpf.plot(chart_df, type="candle", ax=ax, style="yahoo", volume=False, show_nontrading=False)
                title_color = "crimson" if info["cooldown"] else ("seagreen" if info["regime"] == "TRENDING" else "gray")
                ax.set_title(
                    f"{symbol_key} ({tf})  close={info['close']:.2f}  ER={info['er']:.2f} "
                    f"[{info['regime']}]  bias={info['bias']}  "
                    f"breaker={'COOLDOWN' if info['cooldown'] else 'clear'}",
                    fontsize=10, loc="left", color=title_color,
                )
            print()
            fig.tight_layout()
            fig.canvas.draw()
            plt.pause(0.1)
            time.sleep(refresh_seconds)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mt5_path", nargs="?", default=None, help="optional path to terminal64.exe")
    parser.add_argument("--refresh", type=int, default=60, help="seconds between refreshes (default 60)")
    parser.add_argument("--bars", type=int, default=150, help="bars shown on each chart (default 150)")
    parser.add_argument("--timeframe", default=None, help="override run_scheduled.py's TIMEFRAME for this view only")
    args = parser.parse_args()
    run(refresh_seconds=args.refresh, bars=args.bars, timeframe=args.timeframe, mt5_path=args.mt5_path)


if __name__ == "__main__":
    main()
