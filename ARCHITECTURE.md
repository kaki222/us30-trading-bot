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
| 7 | Execution | `trader/l7_execution/` | ⚠️ Written, **not yet run** — needs a Windows smoke test |

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
  `walk_forward` (commit `ee6a36b`).
  - ER gate vs ADX gate, both on the OLD fixed-size=0.1 sizing at $25k
    cash (`reports/*_adx_vs_er_comparison.png`): US30 (36 folds,
    2016–2026) total return −13.7% → +55.8%, max drawdown 39.8% →
    18.6%, fold win rate 50% → 55.6% — clean improvement on every axis
    (the ADX filter was barely filtering; optimizer picked the loosest
    available threshold in 19/36 folds). Gold (48 folds, 2014–2026)
    total return 49.5% → +87.5%, but max drawdown also grew 57.6% →
    65.0% — more return for more risk, not a free upgrade.
  - **Superseded by the Layer 5 sizing re-run below** — those numbers
    ran on the old uncontrolled fixed-size sizing (see Layer 5). Same
    ER-gated `RegimeConfluenceStrategy`, re-run on $100k cash + 1%-risk
    sizing (`reports/walk_forward_*_sized.csv`,
    `reports/walk_forward_sized_equity_curves.png`):
    - US30 (36 folds): total return **+24.7%**, max drawdown **8.2%**
      (was 18.6%), fold win rate **61.1%** (was 55.6%), 251 trades.
      Return/drawdown ratio ~3.0x, basically unchanged — the smaller
      total-return number buys a much safer ride.
    - Gold (48 folds): total return **+28.7%**, max drawdown **17.9%**
      (was 65.0% — a >3.6x improvement), fold win rate **52.1%** (was
      47.9%), 235 trades. Return/drawdown ratio improved 1.35x → 1.60x.
    - Takeaway: the earlier "ER beats ADX" total-return numbers were
      partly an artifact of sizing that would have been genuinely
      dangerous to trade live (see the whole-unit-flooring bug under
      Layer 5). Properly risk-sized, the ER gate still wins — just a
      more modest, more honest win.
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

## Layer 7 — Execution (`trader/l7_execution/`)

Windows-only, since it wraps the `MetaTrader5` Python package (COM/DLL
interop with a running MT5 terminal — doesn't exist on Linux/Mac).
Originally written and reasoned through against the documented MT5 API
in the Linux sandbox everything else in this doc was built in, with no
way to execute it there.

**Update 2026-07-22: the read path is now verified against a real
terminal, not just a mock.** `connect()`, `account_summary()`,
`resolve_symbol()`, and `get_live_bars()` were all run via
`smoke_test.py` against a real MT5 desktop terminal (generic
MetaQuotes build, not XM's) logged into a throwaway MetaQuotes-Demo
account (109989358) — connected successfully, printed real account
info, resolved `"US30"` and `"XAUUSD"` as that account's actual symbol
names, and pulled 5 real live H4 bars for each. That's a materially
higher confidence level than "mock-tested" for those four functions
specifically.

**Still unverified against a real terminal:** `build_live_features()`
beyond the raw bar pull, `evaluate_regime_confluence_signal()`,
`size_fraction_to_lots()`, `place_trade()` (dry-run or real),
`LiveCircuitBreaker`, and `run_once()`. None of those have been
exercised outside the mock. The test account above is fully
consequence-free (fake $5,000,000, isolated MetaQuotes demo server, no
connection to anything real) — it's the right place to verify those
next, including a real `dry_run=False` order, before this ever goes
near XM.

- `connect()` / `shutdown()` / `account_summary()` — attach to an
  already-running, already-logged-in MT5 terminal. **Verified working.**
- `resolve_symbol(candidates)` — XM's exact instrument names for US30
  and Gold still aren't confirmed (brokers vary: `"US30Cash"`,
  `"US30.cash"`, `"XAUUSD"`, `"GOLDm"`, etc. — the XM desktop terminal's
  Market Watch tab shows `"US30Cash"` as a hint, but that hasn't been
  run through `resolve_symbol()` against that account yet). This
  searches the account's actual Market Watch instead of hardcoding a
  guess. `SYMBOL_MAP` currently holds `{"US30": "US30", "GOLD":
  "XAUUSD"}` — **but those values are confirmed only for the
  MetaQuotes-Demo test account (109989358), not XM.** Don't assume
  they carry over; re-run `resolve_symbol()` against the XM terminal
  before pointing anything at account 330507861.
- `get_live_bars()` / `build_live_features()` — pull live H4 bars and
  run them through the *same* `l2_features`/`l3_regime` functions used
  in backtesting, so live features are computed identically to backtest
  features. `get_live_bars()` itself is **verified** (real 5-bar pull
  succeeded for both symbols above); the indicator-attaching part of
  `build_live_features()` still isn't. This part reuses real, tested
  code — lower risk than the
  rest of this layer.
- `evaluate_regime_confluence_signal()` — a **hand-port** of
  `RegimeConfluenceStrategy.next()`'s entry rules from
  `l4_signal_model.py`. `backtesting.py`'s `Strategy` class is a
  backtest-loop construct and can't be driven bar-by-bar live directly,
  so this is a second, separate copy of the same rules. **This will
  silently drift out of sync if `l4_signal_model.py` changes and this
  isn't updated too** — the single biggest structural weak point of
  this layer. Collapsing both into one shared rule definition both
  backtest and live can call is the natural next step, once this has
  been smoke-tested at all.
- `LiveCircuitBreaker` — same N-losses-in-a-row → cooldown policy as
  `l6_risk.CircuitBreakerMixin`, but sourced from MT5's own deal history
  (`mt5.history_deals_get`) instead of `backtesting.py`'s
  `self.closed_trades`, since there's no backtest engine live to supply
  that.
- `size_fraction_to_lots()` — converts `l5_position_sizing`'s `(0, 1]`
  size fraction into an actual MT5 lot volume, using the symbol's
  contract size and rounding/clamping to the broker's `volume_step`/
  `volume_min`/`volume_max`. This is the one piece of real, new
  arithmetic in this layer (everything else is plumbing) — verify it
  by hand against a couple of known prices before trusting it.
- `place_trade()` — **defaults to `dry_run=True`**, meaning it computes
  and returns the exact MT5 order request without calling
  `mt5.order_send()`. Only flip to `False` after reading a dry-run
  output and confirming it looks right, on a **demo account**.
- `run_once()` — one full polling cycle: skip if a position is already
  open, skip if the circuit breaker is in cooldown, else fetch → feature
  → signal → size → (dry-run or real) trade.

### Testing this yourself (I cannot do this part)

On the Windows machine with the MT5 terminal, logged into an XM demo
account:

```powershell
pip install MetaTrader5
cd path\to\us30-trading-bot
py -m trader.l7_execution.smoke_test
```

That script only reads — connects, prints account info, searches for
US30/Gold's real symbol names, pulls 5 bars, disconnects. No orders.
Fix `SYMBOL_MAP` in `trader/l7_execution/__init__.py` based on what it
finds, re-run it to confirm bars come back, and only then try
`run_once()` with `dry_run=True` (the default) to see what order it
*would* place, before ever setting `dry_run=False`.

### What "written, not run" actually means here

Ran every pure-logic function (`get_live_bars`, `build_live_features`,
`evaluate_regime_confluence_signal`, `size_fraction_to_lots`,
`place_trade` dry-run, `LiveCircuitBreaker`, `run_once`) against a
hand-built fake `MetaTrader5` module with synthetic bar data, deal
history, and account/symbol info — catches real bugs (wrong field
names, shape mismatches, broken imports) beyond what `py_compile`
would. All of it ran clean; the sizing math checks out by hand too
(1% of $100k risk / a $500.5 stop ≈ 2.0 lots, which is what
`size_fraction_to_lots` returned). What this does **not** verify: the
*real* MT5 API's actual field names/semantics, real broker symbol
naming, real fill behavior, or anything about a live terminal — a
mock only proves the code does what I intended, not that what I
intended matches reality. The Windows smoke test is still the first
real test.

### Not done yet

- No `run_loop()` bar-close scheduler wired up — `run_once()` exists,
  calling it on a timer (aligned to H4 candle closes, not naive
  polling) is the next piece.
- No wiring for `LiquiditySweepStrategy` (only `RegimeConfluenceStrategy`
  is ported) or for the un-optimized `ConfluenceStrategy` variant.
- `LiveCircuitBreaker`'s `history_deals_get` window and grouping filter
  are untested against real MT5 deal records — the deal object's exact
  field names/semantics (`entry`, `magic`, `profit`) are drawn from the
  MT5 API docs, not verified against live output.

---

## Reproducing the current numbers

```python
from trader.l2_features import build_bt_df
from trader.backtest_harness import walk_forward  # defaults to RegimeConfluenceStrategy now

df = build_bt_df("US30")   # or "GOLD"
folds = walk_forward(df)   # cash defaults to 100_000 - see Layer 5
```

Walk-forward over full history is slow (grid-search re-optimized every
quarter, anchored/growing training window) — expect ~5–35s per fold,
accelerating as the training window grows, ~36 folds for US30 and ~48
for Gold.
