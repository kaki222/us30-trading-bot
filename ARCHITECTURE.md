# Architecture — Seven-Layer Pipeline

Status snapshot as of 2026-07-22, taken by reading the actual code (every
`grep -i "Layer [0-9]"` hit across `trader/`), not from a separate spec —
none existed before this file. Layers 5 and 7 don't appear anywhere in
code; their names/scope below are inferred from the one Layer 6 comment
plus the shape of the pipeline (signal → sizing → risk → execution) and
the existing "MT5 live/demo execution" backlog item. Treat those two as
a proposal to confirm, not an established fact, until this doc is
edited to say otherwise.

| # | Name | File(s) | Status |
|---|------|---------|--------|
| 1 | Market Data | `trader/l1_data.py` | ✅ Done |
| 2 | Feature Engineering | `trader/l2_features.py` | ✅ Done |
| 3 | Regime Recognition | `trader/l3_regime.py` | ⚠️ Partially wired in |
| 4 | Signal Model | `trader/l4_signal_model.py`, `trader/l4_liquidity_strategy.py` | ✅ Rule-based, done; not yet ML |
| 5 | Position Sizing | `trader/l5_position_sizing.py` | ✅ Risk-based, wired into all strategies |
| 6 | Risk Overlay | `trader/l6_risk.py` | ✅ Extracted, shared by all Layer 4 strategies |
| 7 | Execution | *(none — inferred)* | ❌ Not started |

---

## Layer 1 — Market Data (`l1_data.py`)

MT5-only. No yfinance/Yahoo dependency (removed in commit `7e8ba3f`).

- `load_h4(symbol)` — `"US30"` or `"GOLD"`, returns lowercase OHLCV,
  index named `"time"`.
- `MT5_SOURCES` registry: per-symbol default CSV path + cutoff date.
  - `US30`: `data/raw/us30_h4_mt5.csv`, cutoff `2016-05-26` (rows before
    that are Daily bars, not H4 — the export is a hybrid).
  - `GOLD`: `data/raw/gold_h4_mt5.csv`, no cutoff — verified by checking
    bar-spacing/day across the full range, clean H4 (~6 bars/day) from
    the first row, 2013-05-09.
- `load_mt5_h4(path, cutoff)` is the generic loader underneath, if a
  third instrument's export ever needs wiring in.

## Layer 2 — Feature Engineering (`l2_features.py`)

- `build_bt_df(symbol)` — Layers 1+2 combined: loads H4 bars, attaches
  SMA(360/200/89), EMA(21/8), MACD, ATR(14), ADX(14)/+DI/-DI. Renames to
  `Open/High/Low/Close/Volume` for `backtesting.py`.
  - Accepts legacy Yahoo tickers (`"^DJI"`, `"GC=F"`) via
    `_LEGACY_SYMBOL_MAP`, mapped to `"US30"`/`"GOLD"` — old notebook
    cells didn't need edits when Layer 1 moved to MT5.
- Candle patterns: `is_bearish_engulfing`, `is_bullish_engulfing`.
- `nth_pivot_price` — fractal pivot ladder for SL/TP.
- `build_liquidity_features(df)` — sweep/BOS/displacement/engulf columns
  for `LiquiditySweepStrategy`. Must run after `build_bt_df`.
- `build_kalman_features(df)` — fast + slow local-linear-trend Kalman
  filter (level + slope), used by the liquidity-sweep notebook.

## Layer 3 — Regime Recognition (`l3_regime.py`)

Built as a standalone feature module, not a trained classifier ("Build
Layer 3: learned regime recognition module" — "learned" was aspirational;
what exists is engineered features, no fitted model).

- `efficiency_ratio(close, length)` — Kaufman ER, backward-looking,
  ~[0,1], 1 = clean trend. **The only piece currently wired into a
  strategy** (`RegimeConfluenceStrategy`, see Layer 4).
- `forward_regime_label(close, horizon, threshold)` — forward-looking ER,
  training-label only, never a live feature. Unused — no training
  pipeline consumes it yet.
- `atr_expansion`, `ema_crossover_count`, `ma_slope` — unused by any
  strategy so far.
- `build_regime_features(df)` — assembles all of the above into one
  feature set for a future classifier. Unused.

## Layer 4 — Signal Model

Two independent strategy families, both rule-based (the docstring's
"trained return-prediction model" goal hasn't happened):

**`l4_signal_model.py`**
- `ConfluenceStrategy` — EMA cross + MACD + swing breakout + macro
  MA trend, gated by `_regime_ok()`. Base version gates on a bare ADX
  threshold (`adx_14 > adx_threshold`).
- `RegimeConfluenceStrategy(ConfluenceStrategy)` — same signal rules,
  overrides `_regime_ok()` to use Layer 3's `efficiency_ratio` instead.
  **This is now the default** in `backtest_harness.run_fold`/
  `walk_forward` (commit `ee6a36b`). Walk-forward result vs the ADX
  version, full MT5 history:
  - US30 (36 folds, 2016–2026): total return −13.7% → **+55.8%**, max
    drawdown 39.8% → **18.6%**, fold win rate 50% → 55.6%. Clean
    improvement on every axis.
  - Gold (48 folds, 2014–2026): total return 49.5% → **+87.5%**, but max
    drawdown also grew 57.6% → **65.0%**. More return for more risk, not
    a free upgrade — most of the gain is concentrated in the last two
    folds during the current gold rally.
  - Full fold-by-fold data: `reports/walk_forward_*_regime_er.csv` and
    `reports/*_adx_vs_er_comparison.png`.
- `ADX_OPTIMIZE_KWARGS` / `REGIME_OPTIMIZE_KWARGS` in `backtest_harness.py`
  — param grids for each variant. `DEFAULT_OPTIMIZE_KWARGS` is a
  back-compat alias for `ADX_OPTIMIZE_KWARGS`; don't use it in new code.

**`l4_liquidity_strategy.py`**
- `LiquiditySweepStrategy` — sweep → displacement → BOS → pullback →
  LTF-BOS → engulfing entry, 4H single-timeframe approximation. State
  machine (`IDLE → SWEPT → PULLBACK → ARMED`), fixed 2.0 R:R target off
  a pivot-based SL. No regime gate at all currently — trades this
  pattern whenever it appears, regardless of Layer 3.
- Walk-forward tested (`optimize_kwargs=None` path in `run_fold`, i.e.
  strategy's own hardcoded defaults, no per-fold optimization) — see
  `notebooks/04_liquidity_sweep.ipynb`. Not yet re-run against the full
  MT5 history the way the two `l4_signal_model` strategies were.

## Layer 5 — Position Sizing (`trader/l5_position_sizing.py`)

`risk_based_size(price, sl, risk_pct, leverage)`: replaces the old
hardcoded `size=0.1` on every entry across both Layer 4 files. Returns
the `size` fraction such that if the stop-loss is hit, the loss equals
exactly `risk_pct` of current equity (default 1%) — independent of how
wide that particular trade's stop happens to be, and independent of
leverage (which only changes cash tied up, not risk). Every strategy
now carries `risk_pct` and `leverage` class attributes; `leverage` must
match `Backtest(..., margin=1/leverage)` — `backtest_harness.py` now
derives `margin` from `strategy_cls.leverage` instead of hardcoding it,
so the two can't drift apart.

**Real gotcha found while wiring this in:** backtesting.py only trades
whole units — no fractional contracts. Since
`units = floor(cash * risk_pct / sl_distance)` (price and leverage
cancel out of that formula entirely), a fixed `cash=25_000` meant that
once US30's price grew past the mid-$20ks, any setup with a stop wider
than roughly $250 silently rounded down to 0 units and the trade just
never executed — no error, no warning, it's just gone.
`LiquiditySweepStrategy`, whose stops are structural/pivot-based (wider
than the ATR-based stops the other two strategies use), lost **74%** of
its trade count to this (50 → 13) and stopped trading entirely after
March 2024. `ConfluenceStrategy`/`RegimeConfluenceStrategy` lost ~23%
each. Gold was essentially unaffected (its price never got close to
$25k). Fixed by raising the walk-forward harness's default `cash` from
25,000 to **100,000** (`backtest_harness.run_fold`/`walk_forward`) —
verified trade counts recover to at least the pre-Layer-5 fixed-size
levels across the full 2016–2026 range, and $150k/$250k don't move the
numbers further, so $100k isn't an arbitrary bump.

## Layer 6 — Risk Overlay (`trader/l6_risk.py`)

`CircuitBreakerMixin`: tracks consecutive losing trades via
`self.closed_trades`; after `max_consecutive_losses` (default 3) in a
row, forces a `cooldown_bars` (default 20) pause before the strategy
will open a new position. Mix-in pattern — a strategy calls `_cb_init()`
from `init()`, `_cb_update()` once per bar in `next()`, and gates entries
with `_cb_in_cooldown()`.

Wired into all three Layer 4 strategies:
- `ConfluenceStrategy` / `RegimeConfluenceStrategy` — extracted from what
  used to be inline logic in `next()`; verified behavior-preserving
  (identical trade counts/returns before and after extraction).
- `LiquiditySweepStrategy` — had **no** circuit breaker before this; now
  gated at the two entry points (bear/bull engulf in the `ARMED` phase)
  rather than freezing the whole sweep/BOS/pullback state machine, so
  cooldown blocks new positions without discarding in-progress setup
  tracking.

Confirmed the breaker actually engages (not just present but inert) for
all three strategies × both instruments.

## Layer 7 — Execution *(not started)*

No code. Maps to the standing backlog item "Draft MT5 live/demo
execution skeleton for XM" — connecting to a running MT5 terminal and
actually placing trades, as opposed to backtesting them. Nothing in
`trader/` talks to a live or demo account yet.

---

## Reproducing the current numbers

```python
from trader.l2_features import build_bt_df
from trader.backtest_harness import walk_forward  # defaults to RegimeConfluenceStrategy now

df = build_bt_df("US30")   # or "GOLD"
folds = walk_forward(df, cash=25_000)
```

Walk-forward over full history is slow (grid-search re-optimized every
quarter, anchored/growing training window) — expect ~5–35s per fold,
accelerating as the training window grows, ~36 folds for US30 and ~48
for Gold.
