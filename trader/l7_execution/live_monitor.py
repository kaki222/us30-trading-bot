"""
live_monitor.py — one persistent, live-refreshing dashboard window:
a column per instrument, each with three stacked panels:
  1. Candlesticks + the actual indicators RegimeConfluenceStrategy's
     entry rule checks — EMA8/EMA21 (trend cross), MA200/MA360 (macro
     trend filter), and the rolling swing-high/low levels price has to
     break for a BOS entry. If it's not one of these, it isn't part of
     why the bot would or wouldn't fire here.
  2. Efficiency Ratio over time, with the er_threshold line and the
     "tradeable trend" zone shaded — the regime gate, visible as a
     history instead of a single instantaneous number.
  3. MACD histogram (green/red by sign) — the other half of the entry
     rule's momentum check.
Each column's price panel is titled with close, ER, TRENDING/CHOP,
your BIAS (from run_scheduled.py), and circuit-breaker cooldown status,
color-coded. The same info also prints as a one-line log per symbol
per cycle, so there's a plain-text record even with the window closed.

Read-only: never calls place_trade() or run_once() - it's on its own
refresh timer, independent of the scheduled trading cycle. Safe to
leave open all day.

BIAS/TIMEFRAME/PARAMS/MAGIC come from run_scheduled.py, not redefined
here, so this always reflects whatever that script is actually
configured to trade - edit your bias there, this picks it up.

If the window opens but stays blank: that's a matplotlib GUI backend
issue on your machine, not this script's logic (the print log will
still be updating in the terminal even if the chart isn't drawing).
Try `pip install PyQt5`, then add these two lines at the very top of
this file, before the `matplotlib.pyplot` import:
    import matplotlib
    matplotlib.use("QtAgg")

Needs mplfinance in addition to what run_scheduled.py already needs:
    (venv) PS> pip install mplfinance

Close the window or Ctrl+C in the terminal to stop.

    (venv) PS> python -m trader.l7_execution.live_monitor
    (venv) PS> python -m trader.l7_execution.live_monitor "C:\\path\\to\\terminal64.exe"
    (venv) PS> python -m trader.l7_execution.live_monitor --refresh 30 --bars 150
    (venv) PS> python -m trader.l7_execution.live_monitor --timeframe H1
"""

import argparse
from datetime import datetime, timezone

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

from . import (
    connect, shutdown, resolve_symbol, SYMBOL_MAP, TIMEFRAME_SECONDS,
    build_live_features, LiveCircuitBreaker, _swing_high, _swing_low,
)
from .run_scheduled import BIAS, TIMEFRAME, PARAMS, MAGIC

# Extra instruments live_monitor can watch purely to see live movement -
# these are NOT traded by run_scheduled.py and never touch BIAS or the
# real SYMBOL_MAP (adding a symbol here doesn't make the bot trade it).
# Candidate names are guesses, same as SYMBOL_MAP's own comment explains
# brokers vary - resolve_symbol() finds whichever one is actually in
# your Market Watch. Pass --extra btc,eth to add them for one run.
WATCHLIST_CANDIDATES = {
    "BTC": ["BTCUSD", "BTCUSDm", "BTCUSD.cash", "Bitcoin"],
    "ETH": ["ETHUSD", "ETHUSDm", "ETHUSD.cash", "Ethereum"],
}


def _regime_info(symbol_key: str, symbol: str, df: pd.DataFrame, timeframe: str) -> dict:
    row = df.iloc[-1]
    er = row["er"]
    trending = bool(er > PARAMS["er_threshold"]) if pd.notna(er) else False
    regime = "TRENDING" if trending else ("CHOP" if pd.notna(er) else "warming up")
    bias = BIAS.get(symbol_key) or "neutral"

    breaker = LiveCircuitBreaker(symbol=symbol, magic=MAGIC, bar_seconds=TIMEFRAME_SECONDS[timeframe.upper()])
    cooldown = breaker.in_cooldown()

    return {"close": row["Close"], "er": er, "regime": regime, "bias": bias, "cooldown": cooldown}


def _log_line(symbol_key: str, info: dict) -> str:
    return (f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  "
            f"{symbol_key:6s} close={info['close']:>10.2f}  "
            f"ER={info['er']:.3f}  regime={info['regime']:9s} bias={info['bias']:7s} "
            f"breaker={'COOLDOWN' if info['cooldown'] else 'clear'}")


def _redraw_column(symbol_key: str, symbol: str, price_ax, er_ax, macd_ax,
                    bars: int, timeframe: str) -> dict:
    df = build_live_features(symbol, er_length=PARAMS.get("er_length", 20), timeframe=timeframe)
    swing_hi = _swing_high(df["High"], PARAMS["swing_lookback"]).tail(bars)
    swing_lo = _swing_low(df["Low"], PARAMS["swing_lookback"]).tail(bars)
    chart_df = df.tail(bars)

    info = _regime_info(symbol_key, symbol, df, timeframe)
    print(_log_line(symbol_key, info))

    for ax in (price_ax, er_ax, macd_ax):
        ax.clear()

    er_threshold_line = pd.Series(PARAMS["er_threshold"], index=chart_df.index)
    macd_colors = ["seagreen" if v > 0 else "firebrick" for v in chart_df["macd_hist"].fillna(0)]

    addplots = [
        mpf.make_addplot(chart_df["ema_8"], ax=price_ax, color="dodgerblue", width=1.1),
        mpf.make_addplot(chart_df["ema_21"], ax=price_ax, color="darkorange", width=1.1),
        mpf.make_addplot(chart_df["ma_200"], ax=price_ax, color="slategray", width=1.0, linestyle="--"),
        mpf.make_addplot(chart_df["ma_360"], ax=price_ax, color="dimgray", width=1.0, linestyle=":"),
        mpf.make_addplot(swing_hi, ax=price_ax, color="seagreen", width=0.8, linestyle=":"),
        mpf.make_addplot(swing_lo, ax=price_ax, color="firebrick", width=0.8, linestyle=":"),
        mpf.make_addplot(chart_df["er"], ax=er_ax, color="purple", width=1.2, ylabel="ER"),
        mpf.make_addplot(er_threshold_line, ax=er_ax, color="crimson", width=0.9, linestyle="--"),
        mpf.make_addplot(chart_df["macd_hist"], type="bar", ax=macd_ax, color=macd_colors, width=0.7, ylabel="MACD"),
    ]

    mpf.plot(
        chart_df[["Open", "High", "Low", "Close", "Volume"]],
        type="candle", ax=price_ax, style="yahoo", volume=False,
        show_nontrading=False, addplot=addplots,
    )

    er_ax.axhspan(PARAMS["er_threshold"], 1.0, color="seagreen", alpha=0.08)
    er_ax.set_ylim(0, 1)
    macd_ax.axhline(0, color="gray", linewidth=0.6)

    # mplfinance only date-formats the ax it draws candles into (price_ax) -
    # er_ax/macd_ax get plain 0..N integer ticks by default even though their
    # data lines up with price_ax bar-for-bar. Force a draw so price_ax's tick
    # labels are actually computed, then copy them onto the bottom panel and
    # hide the redundant copies above it, so only one (correct) date axis shows.
    price_ax.figure.canvas.draw()
    date_ticks = price_ax.get_xticks()
    date_labels = [t.get_text() for t in price_ax.get_xticklabels()]
    price_ax.tick_params(labelbottom=False)
    er_ax.set_xlim(price_ax.get_xlim())
    er_ax.tick_params(labelbottom=False)
    macd_ax.set_xlim(price_ax.get_xlim())
    macd_ax.set_xticks(date_ticks)
    macd_ax.set_xticklabels(date_labels, rotation=45, ha="right", fontsize=8)

    title_color = "crimson" if info["cooldown"] else ("seagreen" if info["regime"] == "TRENDING" else "gray")
    price_ax.set_title(
        f"{symbol_key} ({timeframe})  close={info['close']:.2f}  ER={info['er']:.2f} "
        f"[{info['regime']}]  bias={info['bias']}  breaker={'COOLDOWN' if info['cooldown'] else 'clear'}",
        fontsize=10, loc="left", color=title_color,
    )
    return info


def run(refresh_seconds: int = 60, bars: int = 150, timeframe: str | None = None,
        mt5_path: str | None = None, extra: list[str] | None = None):
    tf = timeframe or TIMEFRAME
    connect(path=mt5_path)

    symbols = list(SYMBOL_MAP.items())
    for key in extra or []:
        key = key.strip().upper()
        if key not in WATCHLIST_CANDIDATES:
            raise ValueError(f"--extra {key!r} not recognized. Known: {list(WATCHLIST_CANDIDATES)}")
        symbols.append((key, resolve_symbol(WATCHLIST_CANDIDATES[key])))
    n = len(symbols)

    plt.ion()
    fig, axgrid = plt.subplots(
        nrows=3, ncols=n, figsize=(8 * n, 8.5),
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )
    if n == 1:
        axgrid = axgrid.reshape(3, 1)
    try:
        fig.canvas.manager.set_window_title("us30-trading-bot — live monitor")
    except AttributeError:
        pass  # backend doesn't support a custom window title - cosmetic only

    def _on_resize(_event):
        # Without this, resizing/snapping the window only re-flows text
        # and axes at the NEXT scheduled data refresh (up to refresh_seconds
        # away), so it looks broken/off-scale in between. Re-fitting on the
        # resize event itself makes it correct immediately instead.
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("resize_event", _on_resize)
    plt.show(block=False)

    print(f"live_monitor: {tf} bars, refreshing every {refresh_seconds}s. Ctrl+C or close the window to stop.\n")

    try:
        while plt.fignum_exists(fig.number):
            for col, (symbol_key, symbol) in enumerate(symbols):
                price_ax, er_ax, macd_ax = axgrid[0, col], axgrid[1, col], axgrid[2, col]
                _redraw_column(symbol_key, symbol, price_ax, er_ax, macd_ax, bars, tf)
            fig.suptitle(f"Last refreshed {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", fontsize=9, color="gray")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.canvas.draw_idle()
            print()
            _wait_responsively(fig, refresh_seconds)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        shutdown()


def _wait_responsively(fig, seconds: float, step: float = 0.15):
    """
    Waits `seconds` between refreshes WITHOUT a plain time.sleep(). A
    single long time.sleep() blocks Python entirely, so the window's
    event loop never runs during that stretch and the window can't be
    dragged, resized, or even redrawn by the OS for the whole interval -
    it only looked "stuck" between the brief moments it happened to
    wake up. plt.pause() is what actually processes those OS window
    events, so looping it in small steps keeps the window responsive
    the entire time it's waiting, not just right after a redraw.
    """
    elapsed = 0.0
    while elapsed < seconds:
        if not plt.fignum_exists(fig.number):
            return
        plt.pause(step)
        elapsed += step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mt5_path", nargs="?", default=None, help="optional path to terminal64.exe")
    parser.add_argument("--refresh", type=int, default=60, help="seconds between refreshes (default 60)")
    parser.add_argument("--bars", type=int, default=150, help="bars shown on each chart (default 150)")
    parser.add_argument("--timeframe", default=None, help="override run_scheduled.py's TIMEFRAME for this view only")
    parser.add_argument("--extra", default=None,
                         help=f"comma-separated watch-only symbols to add, not traded by the bot. "
                              f"Known: {','.join(WATCHLIST_CANDIDATES)} (e.g. --extra btc,eth)")
    args = parser.parse_args()
    extra = args.extra.split(",") if args.extra else None
    run(refresh_seconds=args.refresh, bars=args.bars, timeframe=args.timeframe,
        mt5_path=args.mt5_path, extra=extra)


if __name__ == "__main__":
    main()
