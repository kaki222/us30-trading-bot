"""
l7_execution — Layer 7: Execution (MT5 live/demo)

Windows-only. Requires:
  - a running MT5 terminal, already logged into an account (start with
    a DEMO account, not live) at your broker (XM or otherwise) — this
    module talks to that running terminal, it does not launch one
  - `pip install MetaTrader5` (a Windows-only package — it will not
    import in this project's Linux dev sandbox; see ARCHITECTURE.md)

Honesty note: nothing in this file has been executed. It was written
against the documented MetaTrader5 Python API, not run against a live
terminal, because no MT5 terminal is reachable from the environment
this was written in. Run `python -m trader.l7_execution.smoke_test`
yourself before trusting any of this with even a demo account — see
that file and ARCHITECTURE.md's Layer 7 section for the exact steps.

Design: rather than re-deriving signals from scratch, this bridges live
MT5 bars into the *same* Layer 2/3 feature functions used in
backtesting (l2_features.sma/ema/macd/atr/adx, l3_regime.efficiency_ratio),
so live features are computed identically to backtest features. The
entry RULES themselves (long_signal/short_signal) are a hand-port of
RegimeConfluenceStrategy.next() in l4_signal_model.py — backtesting.py's
Strategy class is a backtest-loop construct and can't be driven live
directly, so this is a second copy of that logic, not a shared one. If
you change the rules in l4_signal_model.py, update
evaluate_regime_confluence_signal() here too — they will silently drift
apart otherwise. That duplication is the main structural weak point of
this file; collapsing it into one shared rule definition is the
natural next step once this has been smoke-tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # lets this module be imported/read on Linux; connect() raises clearly below

from ..l2_features import sma, ema, macd, atr, adx
from ..l3_regime import efficiency_ratio
from ..l5_position_sizing import risk_based_size

# ---------------------------------------------------------------------
# Timeframe helpers - lets callers pass friendly strings ("H1"/"H4"/"D1",
# matching l1_data.RESAMPLE_RULES naming) instead of raw mt5.TIMEFRAME_*
# constants, and gives LiveCircuitBreaker a matching bar_seconds without
# hardcoding it. Plain dict (no mt5 dependency) so it's safe to read on
# Linux too; the mt5 constant itself is only resolved lazily, at call
# time, since mt5 is None outside Windows.
# ---------------------------------------------------------------------
TIMEFRAME_SECONDS = {
    "M1": 60, "M5": 5 * 60, "M15": 15 * 60, "M30": 30 * 60,
    "H1": 3600, "H4": 4 * 3600, "D1": 24 * 3600,
}


def _resolve_timeframe(timeframe):
    """Accepts a friendly string ("H1", "H4", "D1", ...) or a raw mt5.TIMEFRAME_* constant already."""
    if isinstance(timeframe, str):
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package not installed - can't resolve a timeframe string without it.")
        attr = f"TIMEFRAME_{timeframe.upper()}"
        if not hasattr(mt5, attr):
            raise ValueError(f"Unknown timeframe {timeframe!r} - no mt5.{attr}.")
        return getattr(mt5, attr)
    return timeframe  # already a raw mt5 constant (or None, handled by the caller)

# ---------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------

def connect(login: int | None = None, password: str | None = None,
            server: str | None = None, path: str | None = None) -> bool:
    """
    Attach to an already-running MT5 terminal. If the terminal is already
    logged in manually, you can call connect() with no arguments — MT5
    will attach to that session. Only pass login/password/server if you
    want this script to log in itself (avoid hardcoding real credentials
    in source; read them from an environment variable or a local
    untracked file instead).
    """
    if mt5 is None:
        raise RuntimeError(
            "MetaTrader5 package not installed or not on Windows. "
            "Run `pip install MetaTrader5` on the Windows machine with "
            "the MT5 terminal — this package does not work on Linux/Mac."
        )
    kwargs = {}
    if path:
        kwargs["path"] = path
    if login and password and server:
        kwargs.update(login=int(login), password=password, server=server)
    ok = mt5.initialize(**kwargs)
    if not ok:
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    return True


def shutdown():
    if mt5 is not None:
        mt5.shutdown()


def account_summary() -> dict:
    info = mt5.account_info()
    if info is None:
        raise RuntimeError(f"account_info() failed: {mt5.last_error()}")
    return {
        "login": info.login,
        "server": info.server,
        "balance": info.balance,
        "equity": info.equity,
        "margin": info.margin,
        "margin_free": info.margin_free,
        "leverage": info.leverage,
        "currency": info.currency,
    }


def resolve_symbol(candidates: list[str]) -> str:
    """
    Brokers name instruments differently (e.g. "US30", "US30Cash",
    "US30.cash", "XAUUSD", "GOLD", "GOLDm"). I don't know XM's exact
    naming and won't guess — pass a few likely candidates and this
    returns whichever one actually exists in your Market Watch. If none
    match, it raises and prints every symbol MT5 knows about containing
    a hint from your candidates, so you can find the real name and hardcode
    it in SYMBOL_MAP below.
    """
    all_symbols = {s.name for s in mt5.symbols_get()}
    for c in candidates:
        if c in all_symbols:
            return c
    hint = candidates[0][:3].lower()
    close_matches = sorted(s for s in all_symbols if hint in s.lower())
    raise ValueError(
        f"None of {candidates} found in this account's Market Watch. "
        f"Symbols containing {hint!r}: {close_matches or '(none)'}. "
        f"Open Market Watch in MT5, find the real symbol name, and add "
        f"it to SYMBOL_MAP below."
    )


# Fill these in yourself after running resolve_symbol() or checking
# Market Watch directly — symbol names are broker-specific, so these
# must be re-confirmed for whichever account is actually in use.
#
# Values below are XM's REAL symbol names, confirmed 2026-07-23 via
# resolve_symbol() run against an XM demo account (345899957, same
# broker/server family as the real account 330507861 - "XMGlobal-MT5
# 10" vs "XMGlobal-MT5 9") through test_signal_readonly.py (read-only,
# no orders). These are what to use for anything eventually pointed at
# XM. Note this is DIFFERENT from the generic MetaQuotes-Demo test
# account (109989358), which resolved to "US30"/"XAUUSD" instead —
# different broker, different feed, different names. If reusing that
# test account for anything, override these two values first.
SYMBOL_MAP = {
    "US30": "US30Cash",  # confirmed on XM (both demo 345899957 and, by broker match, presumably 330507861)
    "GOLD": "GOLD",       # confirmed on XM
}


# ---------------------------------------------------------------------
# Live data -> same feature pipeline as backtesting (l2_features, l3_regime)
# ---------------------------------------------------------------------

def get_live_bars(symbol: str, timeframe=None, count: int = 800) -> pd.DataFrame:
    """
    Pull the most recent `count` bars (default H4) for `symbol` from the
    running terminal, shaped exactly like l1_data.load_h4()'s output
    (lowercase OHLCV, index named "time") so it's a drop-in feed into the
    same feature functions build_bt_df() uses for backtesting.

    `timeframe` accepts a friendly string ("H1", "H4", "D1", ...), a raw
    mt5.TIMEFRAME_* constant, or None (defaults to H4, unchanged
    behavior). This is also how you'd pull real H1 history for backtesting
    - point this at count=however many bars you want and dump the result
    to CSV; see trader/l1_data_export_h1.py.
    """
    tf = _resolve_timeframe(timeframe) if timeframe is not None else mt5.TIMEFRAME_H4
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"copy_rates_from_pos({symbol}) returned nothing: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df.rename(columns={"tick_volume": "volume"})[["open", "high", "low", "close", "volume"]]
    return df


def build_live_features(symbol: str, er_length: int = 20, count: int = 800, timeframe="H4") -> pd.DataFrame:
    """Live equivalent of l2_features.build_bt_df() + the ER column RegimeConfluenceStrategy adds."""
    d = get_live_bars(symbol, timeframe=timeframe, count=count)
    d["ma_360"] = sma(d["close"], 360)
    d["ma_200"] = sma(d["close"], 200)
    d["ema_21"] = ema(d["close"], 21)
    d["ema_8"] = ema(d["close"], 8)
    d["macd"], d["macd_signal"], d["macd_hist"] = macd(d["close"])
    d["atr_14"] = atr(d["high"], d["low"], d["close"])
    _, _, d["adx_14"] = adx(d["high"], d["low"], d["close"])
    d["er"] = efficiency_ratio(d["close"], er_length)
    return d.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})


# swing_high/swing_low need a lookback window matching the strategy's
# swing_lookback param — computed inline here rather than imported,
# since the value depends on the optimized param from the last
# walk-forward fold (see run_once()'s params argument).
def _swing_high(high: pd.Series, n: int) -> pd.Series:
    return high.rolling(n).max().shift(1)


def _swing_low(low: pd.Series, n: int) -> pd.Series:
    return low.rolling(n).min().shift(1)


def evaluate_regime_confluence_signal(df: pd.DataFrame, er_threshold: float = 0.35,
                                       swing_lookback: int = 20, atr_sl_mult: float = 1.5,
                                       atr_tp_mult: float = 2.5) -> dict:
    """
    Hand-port of RegimeConfluenceStrategy.next()'s entry logic
    (l4_signal_model.py) evaluated on the LAST closed bar of `df`
    (from build_live_features()). Returns
    {"signal": "long"|"short"|None, "price", "sl", "tp"}.

    Only entry logic is ported — exit logic (EMA21 cross with trend
    loss) and circuit-breaker gating are handled separately in
    run_once(), since they need live position/trade-history state this
    function doesn't have.
    """
    swing_hi = _swing_high(df["High"], swing_lookback)
    swing_lo = _swing_low(df["Low"], swing_lookback)

    row = df.iloc[-1]
    atr_val = row["atr_14"]
    if pd.isna(atr_val) or pd.isna(row["ma_360"]) or pd.isna(swing_hi.iloc[-1]) or pd.isna(row["er"]):
        return {"signal": None}

    trending = row["er"] > er_threshold
    ema_bullish = row["ema_8"] > row["ema_21"]
    ema_bearish = row["ema_8"] < row["ema_21"]
    macd_bull = row["macd_hist"] > 0
    macd_bear = row["macd_hist"] < 0
    price = row["Close"]
    bos_up = price > swing_hi.iloc[-1]
    bos_down = price < swing_lo.iloc[-1]
    macro_uptrend = price > row["ma_360"] and price > row["ma_200"]
    macro_downtrend = price < row["ma_360"] and price < row["ma_200"]

    long_signal = trending and macro_uptrend and ema_bullish and macd_bull and bos_up
    short_signal = trending and macro_downtrend and ema_bearish and macd_bear and bos_down

    if long_signal:
        sl = price - atr_sl_mult * atr_val
        tp = price + atr_tp_mult * atr_val
        return {"signal": "long", "price": price, "sl": sl, "tp": tp}
    if short_signal:
        sl = price + atr_sl_mult * atr_val
        tp = price - atr_tp_mult * atr_val
        return {"signal": "short", "price": price, "sl": sl, "tp": tp}
    return {"signal": None}


# ---------------------------------------------------------------------
# Layer 6 (circuit breaker), live version
# ---------------------------------------------------------------------

@dataclass
class LiveCircuitBreaker:
    """
    Same policy as l6_risk.CircuitBreakerMixin (N losses in a row ->
    cooldown), but sourced from MT5's own deal history for this
    symbol+magic number instead of backtesting.py's self.closed_trades,
    since there's no backtest engine live to track that for us.
    """
    symbol: str
    magic: int
    max_consecutive_losses: int = 3
    cooldown_bars: int = 20
    bar_seconds: int = 4 * 3600  # H4

    def in_cooldown(self) -> bool:
        """
        Bug fixed 2026-07-23, found via real testing (not the mock):
        MT5 deal timestamps are in the broker's SERVER time, which is
        commonly offset ~3h from true UTC (varies by broker/DST) - not
        the same clock as datetime.now(timezone.utc)) on this machine.
        The original version compared server-time deal timestamps
        against local-machine UTC "now", so recent deals could appear
        to be timestamped in the future relative to that boundary and
        get silently excluded from the query window - confirmed via
        debug_deals.py, which measured a ~10,799s (~3h) skew directly.

        Fix: get "now" from a live tick instead of the local clock, so
        every timestamp compared here - the query window, the deal
        times, and the cooldown-expiry check - is in the same server-
        time domain. Never mix local-clock time with deal.time again.
        """
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return False  # can't determine server time - fail open rather than block trading on a data error
        server_now = tick.time  # epoch seconds, server time - same clock deal.time uses

        date_from = datetime.fromtimestamp(server_now - 365 * 24 * 3600, tz=timezone.utc)
        date_to = datetime.fromtimestamp(server_now + 3600, tz=timezone.utc)  # +1h pad against residual clock skew

        deals = mt5.history_deals_get(date_from, date_to, group=f"*{self.symbol}*")
        if not deals:
            return False
        closes = sorted(
            [d for d in deals if d.magic == self.magic and d.entry == 1],  # entry=1 -> DEAL_ENTRY_OUT (closes a position)
            key=lambda d: d.time,
        )
        if not closes:
            return False

        consecutive = 0
        for d in reversed(closes):
            if d.profit < 0:
                consecutive += 1
            else:
                break

        if consecutive < self.max_consecutive_losses:
            return False

        cooldown_until = closes[-1].time + self.cooldown_bars * self.bar_seconds  # both in server-time epoch seconds
        return server_now < cooldown_until


# ---------------------------------------------------------------------
# Sizing: backtesting.py's size fraction -> real MT5 lot volume
# ---------------------------------------------------------------------

def size_fraction_to_lots(symbol: str, size_fraction: float, price: float, leverage: float) -> float:
    """
    l5_position_sizing.risk_based_size() returns a (0, 1] fraction of
    equity, matching backtesting.py's Strategy.buy(size=...) semantics.
    MT5 orders need an actual lot volume instead. Converts using the
    same relationship backtesting.py uses internally
    (units = equity * size * leverage / price), then divides by the
    symbol's contract size to get lots, and rounds/clamps to what the
    broker actually allows (volume_step, volume_min, volume_max).
    """
    acc = mt5.account_info()
    sym = mt5.symbol_info(symbol)
    if acc is None or sym is None:
        raise RuntimeError(f"Missing account_info/symbol_info for {symbol}: {mt5.last_error()}")

    units = acc.equity * size_fraction * leverage / price
    lots = units / sym.trade_contract_size

    step = sym.volume_step
    lots = round(lots / step) * step
    lots = max(sym.volume_min, min(sym.volume_max, lots))
    return round(lots, 2)


# ---------------------------------------------------------------------
# Order placement
# ---------------------------------------------------------------------

def place_trade(symbol: str, direction: str, sl: float, tp: float, risk_pct: float,
                 leverage: float, magic: int, comment: str = "l4_regime_confluence",
                 dry_run: bool = True) -> dict:
    """
    direction: "long" or "short". Computes lot size via l5's
    risk_based_size() + size_fraction_to_lots(), then submits a market
    order with SL/TP attached.

    dry_run=True by default (deliberately) — it computes and returns
    everything it WOULD send without calling mt5.order_send(). Flip to
    False only after you've read back what a dry run prints and it
    looks right, on a DEMO account.
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"symbol_info_tick({symbol}) failed: {mt5.last_error()}")
    price = tick.ask if direction == "long" else tick.bid

    size_fraction = risk_based_size(price, sl, risk_pct, leverage)
    lots = size_fraction_to_lots(symbol, size_fraction, price, leverage)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lots,
        "type": mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    if dry_run:
        return {"dry_run": True, "would_send": request, "size_fraction": size_fraction, "lots": lots}

    result = mt5.order_send(request)
    return {"dry_run": False, "request": request, "result": result}


def close_position(ticket: int, deviation: int = 20) -> dict:
    """
    Close an open position by its ticket - sends the opposite-direction
    order for the same volume, tagged with `position` so MT5 nets it
    against the existing position rather than opening a new one. Used
    by test scripts (e.g. deliberately opening and immediately closing
    a few tiny positions to generate real closed-loss deals for
    LiveCircuitBreaker to read) as well as any future manual/EA exit.
    """
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        raise RuntimeError(f"No open position with ticket {ticket}: {mt5.last_error()}")
    pos = positions[0]

    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None:
        raise RuntimeError(f"symbol_info_tick({pos.symbol}) failed: {mt5.last_error()}")

    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": close_type,
        "position": ticket,
        "price": price,
        "deviation": deviation,
        "magic": pos.magic,
        "comment": "l7_close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    return {"request": request, "result": result}


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def has_open_position(symbol: str, magic: int) -> bool:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return False
    return any(p.magic == magic for p in positions)


def run_once(symbol_key: str, params: dict, risk_pct: float = 0.01, leverage: float = 30,
             magic: int = 100001, dry_run: bool = True, timeframe="H4") -> dict:
    """
    One polling cycle: fetch live bars -> compute features -> evaluate
    RegimeConfluenceStrategy's entry rule -> check circuit breaker and
    existing position -> place (or dry-run print) a trade if everything
    clears. `params` are the optimized values from the latest
    walk-forward fold (er_threshold, swing_lookback, atr_sl_mult,
    atr_tp_mult) — see ARCHITECTURE.md for how to get current ones.

    `timeframe` accepts a friendly string ("H1", "H4", "D1", ...) or a
    raw mt5.TIMEFRAME_* constant, and drives BOTH the live bar pull
    (build_live_features) and the circuit breaker's cooldown math
    (LiveCircuitBreaker.bar_seconds) - these two used to be independently
    hardcoded to H4, which would have silently mismatched if only one
    were changed. Whatever timeframe you pick here must match the
    timeframe the walk-forward validation for `params` was actually run
    on - see ARCHITECTURE.md's timeframe-sweep section.
    """
    symbol = SYMBOL_MAP.get(symbol_key)
    if not symbol:
        raise ValueError(f"SYMBOL_MAP[{symbol_key!r}] is not set — run resolve_symbol() first.")

    if has_open_position(symbol, magic):
        return {"action": "skip", "reason": "position already open"}

    bar_seconds = TIMEFRAME_SECONDS[timeframe.upper()] if isinstance(timeframe, str) else 4 * 3600
    breaker = LiveCircuitBreaker(symbol=symbol, magic=magic, bar_seconds=bar_seconds)
    if breaker.in_cooldown():
        return {"action": "skip", "reason": "circuit breaker cooldown"}

    df = build_live_features(symbol, er_length=params.get("er_length", 20), timeframe=timeframe)
    signal = evaluate_regime_confluence_signal(
        df,
        er_threshold=params.get("er_threshold", 0.35),
        swing_lookback=params.get("swing_lookback", 20),
        atr_sl_mult=params.get("atr_sl_mult", 1.5),
        atr_tp_mult=params.get("atr_tp_mult", 2.5),
    )
    if signal["signal"] is None:
        return {"action": "skip", "reason": "no signal"}

    trade = place_trade(
        symbol, signal["signal"], signal["sl"], signal["tp"],
        risk_pct=risk_pct, leverage=leverage, magic=magic, dry_run=dry_run,
    )
    return {"action": "trade", "signal": signal, "trade": trade}
