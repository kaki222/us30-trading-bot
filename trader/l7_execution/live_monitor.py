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

Read-only about ORDERS: never calls place_trade() or run_once() - it's
on its own refresh timer, independent of the scheduled trading cycle.
Safe to leave open all day. No buttons, no order entry, no way to move
money from this window - that's a deliberately separate, bigger
decision (a real GUI framework, not matplotlib widgets) not taken here.

Right-side panel: top block (2026-07-25) is account/position readout -
only ever reads account_summary()/get_position_info() and writes text
(equity/balance/margin, and per symbol a one-line flat/direction+P&L
summary). Below that (2026-07-26): clickable controls for the
DISCRETIONARY layer only - bias (long/neutral/short radio buttons),
an immediate pause toggle, and the two key-level text boxes. These
write to data/manual_overrides.json (via manual_overrides.py) which
run_scheduled.py reads fresh every cycle - so a click here changes
what the NEXT scheduled cycle does, same as hand-editing that file
used to, just without the edit. This is fundamentally different from
order-entry buttons: BIAS/pause/key-levels can only ever mute, downsize,
or skip a trade the mechanical strategy would otherwise take - none of
them can place, size, or close a real order themselves, so a click here
carries the same risk profile as editing a config file, not the risk
profile of a trade button.

TIMEFRAME/PARAMS/MAGIC come from run_scheduled.py, not redefined here,
so this always reflects whatever that script is actually configured to
trade. Bias/key-levels/pause come from manual_overrides.load_overrides()
instead, read fresh each redraw - not from run_scheduled.py's module
globals, since those no longer hold the live values (see run_scheduled.py's
module docstring, "made live-editable 2026-07-25" section).

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
from matplotlib.widgets import RadioButtons, Button, TextBox
import mplfinance as mpf

from . import (
    connect, shutdown, SYMBOL_MAP, TIMEFRAME_SECONDS,
    build_live_features, LiveCircuitBreaker, _swing_high, _swing_low,
    account_summary, get_position_info,
)
from .run_scheduled import TIMEFRAME, PARAMS, MAGIC
from .manual_overrides import load_overrides, set_bias, set_key_level, set_paused_now

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


def _regime_info(symbol_key: str, symbol: str, df: pd.DataFrame, timeframe: str, bias: str | None) -> dict:
    """`bias` is passed in (from manual_overrides.load_overrides(), read once per
    redraw cycle in run()) rather than read here, so every column of a given
    cycle sees the exact same override snapshot a click might have just changed."""
    row = df.iloc[-1]
    er = row["er"]
    trending = bool(er > PARAMS["er_threshold"]) if pd.notna(er) else False
    regime = "TRENDING" if trending else ("CHOP" if pd.notna(er) else "warming up")
    bias_label = bias or "neutral"

    breaker = LiveCircuitBreaker(symbol=symbol, magic=MAGIC, bar_seconds=TIMEFRAME_SECONDS[timeframe.upper()])
    cooldown = breaker.in_cooldown()

    return {"close": row["Close"], "er": er, "regime": regime, "bias": bias_label, "cooldown": cooldown}


def _log_line(symbol_key: str, info: dict) -> str:
    return (f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  "
            f"{symbol_key:6s} close={info['close']:>10.2f}  "
            f"ER={info['er']:.3f}  regime={info['regime']:9s} bias={info['bias']:7s} "
            f"breaker={'COOLDOWN' if info['cooldown'] else 'clear'}")


def _redraw_column(symbol_key: str, symbol: str, price_ax, er_ax, macd_ax,
                    bars: int, timeframe: str, visible_bars: int, view_state: dict, bias: str | None) -> dict:
    df = build_live_features(symbol, er_length=PARAMS.get("er_length", 20), timeframe=timeframe)
    swing_hi = _swing_high(df["High"], PARAMS["swing_lookback"]).tail(bars)
    swing_lo = _swing_low(df["Low"], PARAMS["swing_lookback"]).tail(bars)
    chart_df = df.tail(bars)

    info = _regime_info(symbol_key, symbol, df, timeframe, bias)
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
    breaker_txt = "COOLDOWN" if info["cooldown"] else "clear"
    line1 = f"{symbol_key} ({timeframe})  close={info['close']:.2f}  ER={info['er']:.2f} [{info['regime']}]"
    line2 = f"bias={info['bias']}  breaker={breaker_txt}"

    # Column titles used to be one fixed-size line regardless of how wide
    # the actual window was - fine at this file's original full-size
    # design width, but resizing/docking the window to something
    # narrower (get_size_inches() genuinely shrinks on an interactive
    # backend resize, unlike the axes' figure-fraction position) left the
    # same point-sized text taking up a growing share of a shrinking
    # column, until neighboring columns' titles started overlapping -
    # exactly what showed up when docked to half a screen (2026-07-26).
    # Recomputed every redraw (not just once) so it tracks whatever the
    # CURRENT window size is, not the size at startup.
    fig_width_in = price_ax.figure.get_size_inches()[0]
    col_width_in = fig_width_in * price_ax.get_position().width
    if col_width_in >= 5.5:
        title_text, fontsize = f"{line1}  {line2}", 11
    elif col_width_in >= 3.0:
        title_text, fontsize = f"{line1}  {line2}", 9
    else:
        title_text, fontsize = f"{line1}\n{line2}", 8

    price_ax.set_title(title_text, fontsize=fontsize, fontweight="bold", loc="left", color=title_color)
    return info


PANEL_WIDTH_IN = 3.4  # fixed-width inches for the right-side panel (info + controls)

BIAS_OPTIONS = ["Long", "Neutral", "Short"]  # RadioButtons labels -> set_bias() value below
_BIAS_LABEL_TO_VALUE = {"Long": "long", "Neutral": None, "Short": "short"}
_BIAS_VALUE_TO_LABEL = {"long": "Long", None: "Neutral", "short": "Short"}


def _draw_side_panel(ax, acct: dict | None, positions: dict) -> None:
    """
    Read-only text readout - account equity/balance, then one compact
    line per symbol (flat, or direction/volume/P&L). Every value comes
    straight from account_summary()/get_position_info() (themselves
    read-only mt5.account_info()/positions_get() wrappers) - no widgets,
    no callbacks, nothing that could place, modify, or close a trade.
    Kept deliberately brief (unlike the first cut of this panel) - it
    only owns the top slice of the side panel now; the bias/pause/
    key-level controls below it (_build_controls) need the rest.
    """
    ax.clear()
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(PANEL_EDGE)

    y = [0.95]

    def line(text, color=TEXT, size=8.7, bold=False, dy=0.11):
        ax.text(0.06, y[0], text, transform=ax.transAxes, color=color, fontsize=size,
                 fontfamily=MONO, fontweight=("bold" if bold else "normal"), va="top")
        y[0] -= dy

    line("// ACCOUNT", ACCENT_CYAN, size=8.5, dy=0.15)
    if acct is None:
        line("unavailable this cycle", TEXT_MUTED, dy=0.15)
    else:
        line(f"equity   {acct['equity']:>12,.2f} {acct['currency']}")
        line(f"balance  {acct['balance']:>12,.2f} {acct['currency']}", dy=0.15)

    for symbol_key, pos in positions.items():
        if pos is None:
            line(f"{symbol_key:6s} flat", TEXT_MUTED)
            continue
        pnl_color = UP if pos["profit"] >= 0 else DOWN
        dir_color = UP if pos["direction"] == "long" else DOWN
        pct = f" ({pos['pnl_pct'] * 100:+.2f}%)" if pos["pnl_pct"] is not None else ""
        ax.text(0.06, y[0], f"{symbol_key:6s} {pos['direction'].upper():5s}", transform=ax.transAxes,
                 color=dir_color, fontsize=8.7, fontfamily=MONO, fontweight="bold", va="top")
        y[0] -= 0.11
        line(f"   vol={pos['volume']}  P&L {pos['profit']:+,.2f}{pct}", pnl_color)


def _panel_rect(panel_left: float, panel_width: float, y_top: float, height: float, x_inset: float = 0.025):
    """One [left, bottom, width, height] rect, in figure fraction, inset within the panel column."""
    return [panel_left + x_inset, y_top - height, panel_width - 2 * x_inset, height]


# Right-side panel vertical layout, in figure fraction - shared between
# run()'s panel_ax (account/position text) and _build_controls() (bias/
# pause/key-level widgets) so the two can't drift out of sync the way an
# earlier version of this did (hardcoded numbers in both places disagreed
# by enough for the controls' "// SYMBOL" label to overlap the text box's
# bottom border).
#
# 2026-07-26: compressed from the original PANEL_BOTTOM=0.04 (content ran
# almost to the literal bottom edge of the window) so that EVERYTHING -
# header, charts, and this side panel - lives in the upper ~70% of the
# window (y >= RESERVED_BAND_TOP), leaving the bottom ~30% deliberately
# blank and reserved (per the user's request) for a future addition -
# candidate: a running-positions table. All the constants below were
# scaled down together by the same ~0.69 factor from their original
# values so nothing inside a control block (radio/pause/two textboxes)
# overlaps - shrinking the outer two (TEXT_BLOCK_HEIGHT/CONTROL_BLOCK_HEIGHT)
# alone without shrinking _build_controls' internal widget heights would
# have reintroduced the exact overlap bug this layout was fixed for
# before (see the "live_monitor.py dashboard rebuild" ARCHITECTURE.md entry).
RESERVED_BAND_TOP = 0.30  # y < this is the reserved-for-later bottom 30%
PANEL_TOP = 0.90
PANEL_BOTTOM = RESERVED_BAND_TOP
TEXT_BLOCK_HEIGHT = 0.18
CONTROL_BLOCK_HEIGHT = 0.18
BLOCK_GAP = 0.02


def _build_controls(fig, panel_left: float, panel_width: float, symbols: list[tuple[str, str]], y_top: float) -> dict:
    """
    Builds the bias/pause/key-level widgets ONCE (unlike price/ER/MACD,
    these must NOT be recreated every redraw - matplotlib widgets lose
    focus/typed state and their event bindings if torn down and rebuilt,
    and there'd be no reason to anyway since only a click changes them).
    Seeds each widget's initial state from manual_overrides.load_overrides()
    at startup. Returns {symbol_key: {"radio": ..., "pause_btn": ..., "tb_up": ..., "tb_down": ...}}
    - caller (run()) must keep this dict alive for the widgets to keep working.

    Every callback here does exactly one thing: call the matching
    manual_overrides.set_*() (which itself journals the change), then
    update the clicked widget's own on-screen label/color so the click
    reads back immediately instead of waiting for the next 60s redraw.
    None of these touch price/ER/MACD axes, run_once(), or place_trade().
    """
    overrides = load_overrides()
    controls = {}

    for symbol_key, _symbol in symbols:
        block_top = y_top
        y_top -= (CONTROL_BLOCK_HEIGHT + BLOCK_GAP)

        fig.text(panel_left + 0.025, block_top, f"// {symbol_key} — MANUAL OVERRIDE",
                  fontsize=8, fontfamily=MONO, color=ACCENT_CYAN, ha="left", va="top")

        cursor = block_top - 0.022  # clears the label row above before the first widget

        radio_h = 0.07
        radio_ax = fig.add_axes(_panel_rect(panel_left, panel_width, cursor, radio_h))
        radio_ax.set_in_layout(False)
        radio_ax.set_facecolor(BG)
        for spine in radio_ax.spines.values():
            spine.set_color(PANEL_EDGE)
        current_bias_label = _BIAS_VALUE_TO_LABEL.get(overrides["bias"].get(symbol_key), "Neutral")
        radio = RadioButtons(radio_ax, BIAS_OPTIONS, active=BIAS_OPTIONS.index(current_bias_label),
                              activecolor=ACCENT_CYAN)
        for lbl in radio.labels:
            lbl.set_color(TEXT)
            lbl.set_fontsize(8)
            lbl.set_fontfamily(MONO)
        radio.on_clicked(_make_bias_callback(symbol_key))
        cursor -= (radio_h + 0.006)

        pause_h = 0.022
        pause_ax = fig.add_axes(_panel_rect(panel_left, panel_width, cursor, pause_h))
        pause_ax.set_in_layout(False)
        is_paused = overrides["paused_now"].get(symbol_key, False)
        pause_btn = Button(pause_ax, "RESUME" if is_paused else "PAUSE NOW",
                            color=(DOWN if is_paused else PANEL_EDGE), hovercolor=ACCENT_CYAN)
        pause_btn.label.set_color(TEXT)
        pause_btn.label.set_fontsize(8)
        pause_btn.label.set_fontfamily(MONO)
        pause_btn.on_clicked(_make_pause_callback(symbol_key, pause_btn))
        cursor -= (pause_h + 0.006)

        tb_h = 0.022
        levels = overrides["key_levels"].get(symbol_key, {})
        up_val = levels.get("invalidation_up")
        tb_up_ax = fig.add_axes(_panel_rect(panel_left, panel_width, cursor, tb_h))
        tb_up_ax.set_in_layout(False)
        tb_up = TextBox(tb_up_ax, "inv-up  ", initial=("" if up_val is None else str(up_val)),
                         color=BG, hovercolor=PANEL_EDGE, label_pad=0.02)
        tb_up.label.set_color(TEXT_MUTED)
        tb_up.label.set_fontsize(7.5)
        tb_up.label.set_fontfamily(MONO)
        tb_up.text_disp.set_color(TEXT)
        tb_up.text_disp.set_fontfamily(MONO)
        tb_up.on_submit(_make_key_level_callback(symbol_key, "invalidation_up", tb_up))
        cursor -= (tb_h + 0.004)

        down_val = levels.get("invalidation_down")
        tb_down_ax = fig.add_axes(_panel_rect(panel_left, panel_width, cursor, tb_h))
        tb_down_ax.set_in_layout(False)
        tb_down = TextBox(tb_down_ax, "inv-dn  ", initial=("" if down_val is None else str(down_val)),
                           color=BG, hovercolor=PANEL_EDGE, label_pad=0.02)
        tb_down.label.set_color(TEXT_MUTED)
        tb_down.label.set_fontsize(7.5)
        tb_down.label.set_fontfamily(MONO)
        tb_down.text_disp.set_color(TEXT)
        tb_down.text_disp.set_fontfamily(MONO)
        tb_down.on_submit(_make_key_level_callback(symbol_key, "invalidation_down", tb_down))

        controls[symbol_key] = {"radio": radio, "pause_btn": pause_btn, "tb_up": tb_up, "tb_down": tb_down}

    return controls


def _make_bias_callback(symbol_key: str):
    def _on_bias_clicked(label: str):
        set_bias(symbol_key, _BIAS_LABEL_TO_VALUE[label])
        print(f"  [override] {symbol_key} bias -> {label}")
    return _on_bias_clicked


def _make_pause_callback(symbol_key: str, button):
    def _on_pause_clicked(_event):
        now_paused = not load_overrides()["paused_now"].get(symbol_key, False)
        set_paused_now(symbol_key, now_paused)
        button.label.set_text("RESUME" if now_paused else "PAUSE NOW")
        button.color = DOWN if now_paused else PANEL_EDGE
        button.ax.set_facecolor(button.color)
        button.ax.figure.canvas.draw_idle()
        print(f"  [override] {symbol_key} paused_now -> {now_paused}")
    return _on_pause_clicked


def _make_key_level_callback(symbol_key: str, which: str, textbox):
    def _on_submit(text: str):
        text = text.strip()
        if not text:
            set_key_level(symbol_key, which, None)
            print(f"  [override] {symbol_key} {which} -> cleared")
            return
        try:
            value = float(text)
        except ValueError:
            print(f"  [override] {symbol_key} {which}: '{text}' isn't a number - ignored, box left as-is")
            return
        set_key_level(symbol_key, which, value)
        print(f"  [override] {symbol_key} {which} -> {value}")
    return _on_submit


def run(refresh_seconds: int = 60, bars: int = 150, timeframe: str | None = None,
        mt5_path: str | None = None, visible_bars: int = 18):
    tf = timeframe or TIMEFRAME
    connect(path=mt5_path)

    symbols = list(SYMBOL_MAP.items())
    n = len(symbols)
    view_state: dict = {}  # symbol_key -> last default (latest-N) xlim, for pan/zoom preservation

    plt.ion()
    grid_width_in = 8 * n
    total_width_in = grid_width_in + PANEL_WIDTH_IN
    fig, axgrid = plt.subplots(
        nrows=3, ncols=n, figsize=(total_width_in, 8.5),
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

    # Right-side panel (2026-07-25/26) - plain fig.add_axes() rects at
    # manually-computed coords rather than an extra subplots() column, so
    # this whole area lives outside the price/ER/MACD gridspec entirely
    # and can't get reflowed by tight_layout()'s column-sizing logic.
    # set_in_layout(False) on every one of these axes keeps tight_layout()
    # (called every redraw for the resize fix below) from trying to fit
    # them at all - they're positioned once, here. Top slice is the
    # account/position text (redrawn every cycle, see _draw_side_panel);
    # below that, one bias/pause/key-level control block per symbol
    # (built once - see _build_controls's docstring for why those can't
    # be torn down and rebuilt like the text above them).
    panel_left = grid_width_in / total_width_in + 0.01
    panel_width = 1 - panel_left - 0.015
    text_bottom = PANEL_TOP - TEXT_BLOCK_HEIGHT
    panel_ax = fig.add_axes([panel_left, text_bottom, panel_width, TEXT_BLOCK_HEIGHT])
    panel_ax.set_in_layout(False)
    controls = _build_controls(fig, panel_left, panel_width, symbols, y_top=text_bottom - BLOCK_GAP)

    # tight_layout only owns the chart grid, and only the upper ~70% of the
    # window (bottom=RESERVED_BAND_TOP, not 0 as before 2026-07-26) - the
    # rest is deliberately blank, reserved for a future addition (candidate:
    # a running-positions table). Squeezing the full 3-row grid into
    # whatever height was available regardless of window shape was also
    # the direct cause of the growing-dead-space-between-panels look when
    # the window got resized to an unusual (e.g. tall, half-screen) aspect
    # ratio - giving tight_layout a smaller, fixed vertical budget to work
    # with makes its auto-spacing choices less extreme in those cases too.
    HEADER_RECT = (0, RESERVED_BAND_TOP, grid_width_in / total_width_in, 0.93)

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

    # Bottom ~30% (y < RESERVED_BAND_TOP) is deliberately empty - see
    # HEADER_RECT/PANEL_TOP comments above. Labeled rather than left as
    # unexplained blank space, so it reads as "reserved for later" instead
    # of looking like a layout bug.
    fig.text(0.01, RESERVED_BAND_TOP - 0.03, "// RESERVED — running positions (planned)",
              fontsize=8, fontfamily=MONO, color=TEXT_MUTED, ha="left", va="top")

    plt.show(block=False)

    print(f"live_monitor: {tf}, {visible_bars} candles visible by default ({bars} available - pan/zoom to see "
          f"more), refreshing every {refresh_seconds}s. Ctrl+C or close the window to stop.\n")

    try:
        while plt.fignum_exists(fig.number):
            # One overrides read per cycle, shared by every column's title AND
            # the panel - so a click that lands mid-cycle doesn't show a
            # different bias in the chart title than what the panel just wrote.
            cycle_bias = load_overrides()["bias"]
            for col, (symbol_key, symbol) in enumerate(symbols):
                price_ax, er_ax, macd_ax = axgrid[0, col], axgrid[1, col], axgrid[2, col]
                _redraw_column(symbol_key, symbol, price_ax, er_ax, macd_ax, bars, tf, visible_bars,
                                view_state, cycle_bias.get(symbol_key))

            try:
                acct = account_summary()
            except RuntimeError as e:
                acct = None
                print(f"  [panel] account_summary() unavailable this cycle: {e}")
            positions = {}
            for symbol_key, symbol in symbols:
                try:
                    positions[symbol_key] = get_position_info(symbol, MAGIC)
                except Exception as e:
                    positions[symbol_key] = None
                    print(f"  [panel] get_position_info({symbol_key}) unavailable this cycle: {e}")
            _draw_side_panel(panel_ax, acct, positions)

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
