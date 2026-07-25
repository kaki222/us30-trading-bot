"""
live_monitor.py — one persistent, live-refreshing dashboard window:
a column per instrument, each with three stacked panels:
  1. Candlesticks + the actual indicators RegimeConfluenceStrategy's
     entry rule checks — EMA8/EMA21 (trend cross), MA200/MA360 (macro
     trend filter), and the rolling swing-high/low levels price has to
     break for a BOS entry. If it's not one of these, it isn't part of
     why the bot would or wouldn't fire here. A small legend on the
     price panel says which line is which.
  2. Efficiency Ratio over time, with the er_threshold line and the
     "tradeable trend" zone shaded — the regime gate, visible as a
     history instead of a single instantaneous number.
  3. MACD histogram (green/red by sign) — the other half of the entry
     rule's momentum check.
Each column's price panel is titled with close, ER, TRENDING/CHOP,
your BIAS (from run_scheduled.py), and circuit-breaker cooldown status,
color-coded. The same info also prints as a one-line log per symbol
per cycle, so there's a plain-text record even with the window closed.

Dark theme (TradingView-ish palette) - built once as DASH_STYLE below
and reapplied every redraw, since ax.clear() resets an axes' facecolor
back to matplotlib's default each cycle. Change the color constants
near the top of this file to retheme it.

Default view is the most recent ~18 candles (--visible-bars to change).
A wider window (--bars, default 150) is still fetched and plotted
underneath that - use the toolbar's Pan/Zoom buttons (the hand and
magnifying-glass icons) or scroll out to see it. Panning/zooming is
preserved across refreshes: only the FIRST draw and any refresh where
you haven't touched the view snap to the latest ~18 candles - once you
manually pan or zoom, that view sticks through future refreshes instead
of snapping back every cycle.

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
    (venv) PS> python -m trader.l7_execution.live_monitor --visible-bars 50
"""

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplfinance as mpf

from . import (
    connect, shutdown, SYMBOL_MAP, TIMEFRAME_SECONDS,
    build_live_features, LiveCircuitBreaker, _swing_high, _swing_low,
)
from .run_scheduled import BIAS, TIMEFRAME, PARAMS, MAGIC

# Silences "findfont: Font family 'X' not found" - printed once per text
# element per redraw (Rajdhani/Orbitron/Share Tech Mono aren't installed
# system fonts yet, see HEADER_FONT/MONO below) via matplotlib's own
# logger, not a real error - it already falls back correctly on its own.
# Left noisy, this buries the one-line-per-symbol log this script is
# also meant to leave behind. Installing the fonts doesn't need this
# line removed - matplotlib just won't have anything to warn about then.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ---------------------------------------------------------------------
# Look and feel - lifted directly from the user's own "Mini Digital
# Twin" HMI palette (their :root CSS vars) rather than an approximation
# of it. Mapped by role, not just by matching original position:
#   their SETPOINT amber (dashed reference line)  -> ER_THRESH
#   their LEVEL/PV blue (the live measured value)  -> EMA_FAST
#   their PUMP-running green / FAULT red            -> UP / DOWN
#   their DRAIN purple accent                       -> ER_LINE
#   their header cyan                                -> ACCENT_CYAN
# Edit these to retheme; everything below reads from these constants
# rather than hardcoding colors inline, so this is the one place to change.
# ---------------------------------------------------------------------
BG = "#0a0d12"
PANEL_EDGE = "#1e2d45"
GRID = "#1e2d45"
TEXT = "#c8d8f0"
TEXT_MUTED = "#4a6080"
UP = "#00e676"
DOWN = "#ff1744"
EMA_FAST = "#29b6f6"
EMA_SLOW = "#7c4dff"
MA_MACRO_1 = "#7c93b8"
MA_MACRO_2 = "#4a6080"
ER_LINE = "#ce93d8"
ER_THRESH = "#ffab00"
ACCENT_CYAN = "#00e5ff"  # header banner / "// TAG" captions

# Their web fonts (Orbitron/Share Tech Mono/Rajdhani) aren't installed
# system fonts, so matplotlib can't see them out of the box - falls back
# to Segoe UI/Consolas below until/unless you install them (they're free
# on Google Fonts; after installing, matplotlib picks them up on its next
# font-cache rebuild, which can need a restart). HEADER_FONT tries
# Orbitron first for the banner title specifically, since that's the
# single most visible place it'd read as "their" font if installed.
MONO = ["Share Tech Mono", "Consolas", "Cascadia Mono", "DejaVu Sans Mono", "monospace"]
HEADER_FONT = ["Orbitron", "Segoe UI", "Arial", "sans-serif"]

plt.rcParams["font.family"] = ["Rajdhani", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["font.size"] = 9.5

DASH_STYLE = mpf.make_mpf_style(
    base_mpl_style="dark_background",
    marketcolors=mpf.make_marketcolors(up=UP, down=DOWN, edge="inherit", wick="inherit", volume="inherit"),
    facecolor=BG,
    edgecolor=PANEL_EDGE,
    figcolor=BG,
    rc={"font.family": ["Rajdhani", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"], "text.color": TEXT},
)

_LEGEND_HANDLES = [
    Line2D([0], [0], color=EMA_FAST, lw=1.2, label="EMA8"),
    Line2D([0], [0], color=EMA_SLOW, lw=1.2, label="EMA21"),
    Line2D([0], [0], color=MA_MACRO_1, lw=1.0, linestyle="--", label="MA200"),
    Line2D([0], [0], color=MA_MACRO_2, lw=1.0, linestyle=":", label="MA360"),
    Line2D([0], [0], color=UP, lw=0.9, linestyle=":", label="swing hi/lo"),
]


def _style_axes(ax):
    """Reapply the dark theme to one axes - needed every cycle since ax.clear() resets facecolor/spines/grid."""
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(PANEL_EDGE)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8.5)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.label.set_color(TEXT_MUTED)


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
                    bars: int, timeframe: str, visible_bars: int, view_state: dict) -> dict:
    df = build_live_features(symbol, er_length=PARAMS.get("er_length", 20), timeframe=timeframe)
    swing_hi = _swing_high(df["High"], PARAMS["swing_lookback"]).tail(bars)
    swing_lo = _swing_low(df["Low"], PARAMS["swing_lookback"]).tail(bars)
    chart_df = df.tail(bars)

    info = _regime_info(symbol_key, symbol, df, timeframe)
    print(_log_line(symbol_key, info))

    # Preserve manual pan/zoom across refreshes: compare the CURRENT view
    # against the default view WE set on the last redraw (not just "did
    # xlim change at all", which would always be true after set_xlim below).
    # If they differ, you moved it yourself - keep that view. If not (or
    # this is the first draw), fall through to the default latest-N view.
    prior_default = view_state.get(symbol_key)
    prior_xlim = price_ax.get_xlim() if price_ax.lines or price_ax.patches else None
    user_adjusted = (
        prior_default is not None and prior_xlim is not None
        and (abs(prior_xlim[0] - prior_default[0]) > 0.5 or abs(prior_xlim[1] - prior_default[1]) > 0.5)
    )

    for ax in (price_ax, er_ax, macd_ax):
        ax.clear()
        _style_axes(ax)

    er_threshold_line = pd.Series(PARAMS["er_threshold"], index=chart_df.index)
    macd_colors = [UP if v > 0 else DOWN for v in chart_df["macd_hist"].fillna(0)]

    addplots = [
        mpf.make_addplot(chart_df["ema_8"], ax=price_ax, color=EMA_FAST, width=1.2),
        mpf.make_addplot(chart_df["ema_21"], ax=price_ax, color=EMA_SLOW, width=1.2),
        mpf.make_addplot(chart_df["ma_200"], ax=price_ax, color=MA_MACRO_1, width=1.0, linestyle="--"),
        mpf.make_addplot(chart_df["ma_360"], ax=price_ax, color=MA_MACRO_2, width=1.0, linestyle=":"),
        mpf.make_addplot(swing_hi, ax=price_ax, color=UP, width=0.8, linestyle=":"),
        mpf.make_addplot(swing_lo, ax=price_ax, color=DOWN, width=0.8, linestyle=":"),
        mpf.make_addplot(chart_df["er"], ax=er_ax, color=ER_LINE, width=1.3, ylabel="ER"),
        mpf.make_addplot(er_threshold_line, ax=er_ax, color=ER_THRESH, width=0.9, linestyle="--"),
        mpf.make_addplot(chart_df["macd_hist"], type="bar", ax=macd_ax, color=macd_colors, width=0.7, ylabel="MACD"),
    ]

    mpf.plot(
        chart_df[["Open", "High", "Low", "Close", "Volume"]],
        type="candle", ax=price_ax, style=DASH_STYLE, volume=False,
        show_nontrading=False, addplot=addplots,
    )

    price_ax.legend(handles=_LEGEND_HANDLES, loc="upper left", fontsize=7,
                     framealpha=0.35, facecolor=BG, edgecolor=PANEL_EDGE, labelcolor=TEXT_MUTED)

    er_ax.axhspan(PARAMS["er_threshold"], 1.0, color=UP, alpha=0.10)
    er_ax.set_ylim(0, 1)
    macd_ax.axhline(0, color=TEXT_MUTED, linewidth=0.7)

    # SCADA-style panel captions (ax.text, not set_title, so they read as
    # instrument labels rather than headings) - cleared with the rest of
    # the axes every cycle by ax.clear() above, so no separate bookkeeping.
    er_ax.text(0.005, 1.06, "// ER — REGIME GATE", transform=er_ax.transAxes,
               fontsize=7.5, fontfamily=MONO, color=ACCENT_CYAN, ha="left", va="bottom")
    macd_ax.text(0.005, 1.06, "// MACD — MOMENTUM", transform=macd_ax.transAxes,
                 fontsize=7.5, fontfamily=MONO, color=ACCENT_CYAN, ha="left", va="bottom")

    # Default view: latest `visible_bars` candles, not the full `bars`
    # window mpf.plot laid out - the rest is still there, just scrolled
    # out until you pan/zoom to it.
    n_total = len(chart_df)
    default_xlim = (n_total - min(visible_bars, n_total) - 0.5, n_total - 0.5)
    view_state[symbol_key] = default_xlim
    active_xlim = prior_xlim if user_adjusted else default_xlim
    price_ax.set_xlim(active_xlim)

    # mpf.plot() auto-scaled the y-axis to the FULL `chart_df` (bars=150
    # by default), not just the ~18 candles now visible via xlim above -
    # that's what actually made candles look flattened/stretched (wide
    # slots, but a y-range sized for 150 bars' worth of movement, not 18).
    # Re-fit y to only the bars actually in view, same idea on the MACD
    # panel (ER stays fixed 0-1, already correct either way).
    lo_idx = max(0, int(active_xlim[0]))
    hi_idx = min(n_total, int(active_xlim[1]) + 2)
    visible = chart_df.iloc[lo_idx:hi_idx]
    if len(visible):
        y_lo, y_hi = visible["Low"].min(), visible["High"].max()
        y_pad = (y_hi - y_lo) * 0.08 or y_hi * 0.001
        price_ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

        m_vals = visible["macd_hist"].dropna()
        if len(m_vals):
            m_lo, m_hi = min(0.0, m_vals.min()), max(0.0, m_vals.max())
            m_pad = (m_hi - m_lo) * 0.15 or 1.0
            macd_ax.set_ylim(m_lo - m_pad, m_hi + m_pad)

    # mplfinance only date-formats the ax it draws candles into (price_ax) -
    # er_ax/macd_ax get plain 0..N integer ticks by default even though their
    # data lines up with price_ax bar-for-bar. Force a draw so price_ax's tick
    # labels are actually computed, then copy them onto the bottom panel and
    # hide the redundant copies above it, so only one (correct) date axis shows.
    price_ax.figure.canvas.draw()
    date_ticks = price_ax.get_xticks()
    date_labels = [t.get_text() for t in price_ax.get_xticklabels()]
    price_ax.tick_params(labelbottom=False)
    er_ax.set_xlim(active_xlim)
    er_ax.tick_params(labelbottom=False)
    macd_ax.set_xlim(active_xlim)
    macd_ax.set_xticks(date_ticks)
    macd_ax.set_xticklabels(date_labels, rotation=45, ha="right", fontsize=8, color=TEXT_MUTED)

    title_color = DOWN if info["cooldown"] else (UP if info["regime"] == "TRENDING" else TEXT_MUTED)
    price_ax.set_title(
        f"{symbol_key} ({timeframe})  close={info['close']:.2f}  ER={info['er']:.2f} "
        f"[{info['regime']}]  bias={info['bias']}  breaker={'COOLDOWN' if info['cooldown'] else 'clear'}",
        fontsize=11, fontweight="bold", loc="left", color=title_color,
    )
    return info


def run(refresh_seconds: int = 60, bars: int = 150, timeframe: str | None = None,
        mt5_path: str | None = None, visible_bars: int = 18):
    tf = timeframe or TIMEFRAME
    connect(path=mt5_path)

    symbols = list(SYMBOL_MAP.items())
    n = len(symbols)
    view_state: dict = {}  # symbol_key -> last default (latest-N) xlim, for pan/zoom preservation

    plt.ion()
    fig, axgrid = plt.subplots(
        nrows=3, ncols=n, figsize=(8 * n, 8.5),
        gridspec_kw={"height_ratios": [3, 1, 1]},
        sharex="col",  # BUG FIX: without this, toolbar zoom/pan on price_ax only
        # moved price_ax in real time - er_ax/macd_ax only caught up at the next
        # scheduled redraw (up to refresh_seconds later), which is what "not
        # synchronized at all" was. sharex makes matplotlib itself keep every
        # axes in a column locked to the same x-range the instant any one of
        # them changes, whether that's my code or your mouse doing it.
    )
    fig.patch.set_facecolor(BG)
    if n == 1:
        axgrid = axgrid.reshape(3, 1)
    try:
        fig.canvas.manager.set_window_title("us30-trading-bot — live monitor")
    except AttributeError:
        pass  # backend doesn't support a custom window title - cosmetic only

    HEADER_RECT = (0, 0, 1, 0.93)  # leaves room for the two-line banner below

    def _on_resize(_event):
        # Without this, resizing/snapping the window only re-flows text
        # and axes at the NEXT scheduled data refresh (up to refresh_seconds
        # away), so it looks broken/off-scale in between. Re-fitting on the
        # resize event itself makes it correct immediately instead.
        fig.tight_layout(rect=HEADER_RECT)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("resize_event", _on_resize)

    # SCADA-style header banner: static title left, live connection/clock
    # right. fig.text() has no built-in "replace the last one" behavior
    # like fig.suptitle() does, so the right side is created once here and
    # updated in place via set_text() each cycle - calling fig.text() again
    # in the loop would just keep stacking new overlapping text objects.
    fig.text(0.01, 0.975, "US30-TRADING-BOT  //  LIVE MONITOR", fontsize=12, fontweight="bold",
             fontfamily=HEADER_FONT, color=ACCENT_CYAN, ha="left", va="top")
    header_right = fig.text(0.99, 0.975, "", fontsize=9, fontfamily=MONO, color=UP, ha="right", va="top")

    plt.show(block=False)

    print(f"live_monitor: {tf}, {visible_bars} candles visible by default ({bars} available - pan/zoom to see "
          f"more), refreshing every {refresh_seconds}s. Ctrl+C or close the window to stop.\n")

    try:
        while plt.fignum_exists(fig.number):
            for col, (symbol_key, symbol) in enumerate(symbols):
                price_ax, er_ax, macd_ax = axgrid[0, col], axgrid[1, col], axgrid[2, col]
                _redraw_column(symbol_key, symbol, price_ax, er_ax, macd_ax, bars, tf, visible_bars, view_state)
            header_right.set_text(f"● MT5 LIVE   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            fig.tight_layout(rect=HEADER_RECT)
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
    parser.add_argument("--bars", type=int, default=150, help="bars fetched/available to scroll into (default 150)")
    parser.add_argument("--visible-bars", type=int, default=18, help="candles shown by default before you pan/zoom (default 18)")
    parser.add_argument("--timeframe", default=None, help="override run_scheduled.py's TIMEFRAME for this view only")
    args = parser.parse_args()
    run(refresh_seconds=args.refresh, bars=args.bars, timeframe=args.timeframe,
        mt5_path=args.mt5_path, visible_bars=args.visible_bars)


if __name__ == "__main__":
    main()
