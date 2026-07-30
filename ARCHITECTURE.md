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
| 7 | Execution | `trader/l7_execution/` | ✅ Fully verified against real MT5 (test account + XM feed) |

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
  `notebooks/04_liquidity_sweep.ipynb`.
- **Update 2026-07-23: re-run on full MT5 history with Layer 5/6
  sizing** (`reports/walk_forward_{US30,GOLD}_liquidity_sized.csv`,
  `reports/liquidity_sweep_sized_equity_curves.png`) — the piece
  flagged as outstanding since Layer 5 was built. US30 (36 folds):
  total compounded return **+0.54%**, 41 trades, 39.0% win rate, 13/36
  folds (36%) positive. Gold (48 folds): total compounded return
  **+15.86%**, 70 trades, 47.1% win rate, 22/48 folds (46%) positive.
  Both markedly weaker than `RegimeConfluenceStrategy`'s numbers on the
  same instruments — expected, since this strategy has no regime gate
  at all (trades the sweep/BOS/pullback pattern whenever it appears,
  regardless of Layer 3), unlike the ER-gated strategy.
  - **Update 2026-07-23: root-caused.** Read `backtesting.py`'s own
    `_Broker._process_orders()` (site-packages, not our code): for a
    proportional `size` in (-1, 1) it converts to whole units as
    `int(margin_available * leverage * size // adjusted_price_plus_commission)`
    and cancels the order with an "insufficient margin" warning if that
    truncates to 0. Substituting `risk_based_size`'s formula
    (`size = risk_pct * price / (leverage * sl_distance)`) makes
    `leverage` and `price` cancel out almost entirely, leaving
    `units ≈ floor(equity * risk_pct / sl_distance)` — units hit 0
    whenever `sl_distance > equity * risk_pct` **at that moment in the
    fold**, not at the fold's starting $100k. Two things combine to
    make that threshold trip on US30 but never on Gold: (1) this
    strategy's stops are structural/pivot-based, not ATR-based, and run
    much wider on US30 (mean SL ~495 pts, max ~988) than the naive
    $100k*1% = 1000-pt threshold suggests should ever fail — but (2)
    equity drifts *within* a fold from that fold's own earlier trades,
    so by the time a wide-stop setup shows up later in the fold, the
    live threshold (current equity * 1%) can already be well under
    1000, making a 495-988 pt stop enough to floor to 0 units. Gold's
    stops (mean ~28 pts on a much smaller-priced instrument) never get
    close regardless of equity drift, hence 0 rejections there. Not a
    bug — this is `risk_based_size` and `backtesting.py`'s own
    whole-unit rounding correctly agreeing that a 1%-of-current-equity
    loss on that particular wide a stop rounds to less than 1 tradeable
    US30 unit, so the broker sim (correctly) declines the trade rather
    than under- or over-risking it. Same root shape as the earlier
    flooring bug, smaller blast radius (16% of attempts vs. 74% of
    trades), and self-limiting: it only ever fires on the widest-stop,
    already-drawn-down tail, which is arguably the trade population you
    least want sized up anyway.

## Timeframe sweep (2026-07-24)

Timeframe was hardcoded to H4 everywhere (data loading, feature
building, live bar pulls, circuit breaker cooldown math) until now.
`l1_data.load_bars(symbol, timeframe)` / `l2_features.build_bt_df(...,
timeframe=)` / `l7_execution.run_once(..., timeframe=)` all accept
"H1"/"H4"/"D1" now (see those modules' docstrings). D1 is resampled
from the existing H4 export (no new data). H1 needed a real MT5 export
- `trader/l1_data_export_h1.py`, run by the user against XM demo
345899957: US30 61,244 H1 bars (2011-08-10 -> present), Gold 81,444 H1
bars (2001-06-04 -> present). That script's first two attempts both
failed with `(-2, 'Invalid params')` - once from passing timezone-aware
datetimes to `copy_rates_range()`, and again from a single oversized
`copy_rates_from_pos(0, 200_000)` call; fixed by paginating in
5,000-bar chunks.

**RegimeConfluenceStrategy across timeframes** (same walk-forward
harness, same $100k/1%-risk sizing everywhere):

- H4 (production baseline, full-grid optimized, already on record
  above): US30 **+24.7%**, 251 trades, 61.1% fold win rate, 8.2% max
  drawdown. Gold **+28.7%**, 235 trades, 52.1% fold win rate, 17.9% max
  drawdown.
- D1 (small 16-combo grid, not the full 288-combo
  `REGIME_OPTIMIZE_KWARGS` - sandbox time constraints): US30 **-2.29%**,
  41 trades, 31% folds positive. Gold **+24.96%**, 62 trades, 42% folds
  positive. (Fixed-defaults-only pass, before optimizing: US30 -7.66%,
  Gold +15.87% - optimization narrowed the US30 gap and pushed Gold
  above its own fixed-default number, but neither beats H4.)
- H1 (fixed defaults only - the small-grid optimizer that worked for
  D1 didn't finish even one fold within the sandbox's time limit at H1's
  bar count, ~23x D1's): US30 **-45.26%**, 1,382 trades, 42% folds
  positive. Gold **-76.05%**, 1,830 trades, 36% folds positive.

**Conclusion: H4 stays the production timeframe.** D1 is a mixed
downgrade (competitive on Gold's total return but on a third of the
trades, so noisier; a clear loss on US30). H1 is decisively worse on
both instruments even before any optimization - the size of the loss
(-45%/-76%) combined with trade counts 5-8x higher than H4 points at
the strategy's parameters (`ma_360`, ATR multipliers, `swing_lookback`)
being calibrated for H4's noise/signal ratio and simply not
transferring to hourly bars, not a "needs better params" gap that
optimization would likely close. Not pursued further given how
decisive the fixed-default result already was - optimizing H1 properly
would need to run outside the sandbox (e.g. the user's own venv, no
45-second cap) if this is ever revisited.

---

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

**Update 2026-07-23: `place_trade()` and `LiveCircuitBreaker` now
verified too — and real testing caught a real bug the mock couldn't.**

`place_trade()` dry-run and a real `dry_run=False` order both worked
correctly on the test account (a GOLD long, ticket 9678509436, filled
and tracked normally). The dry-run lot-size math checked out by hand.
One practical lesson from that run: sizing scales with account equity,
so the same `risk_pct` that looks reasonable on a normal-sized account
can produce very large lot counts on a $5,000,000 test account (one
run hit the broker's `volume_max` clamp at 100 lots) — not a bug, just
a reminder to sanity-check the resulting lot size against whatever
account is actually connected before trusting a dry-run number.

`LiveCircuitBreaker.in_cooldown()` failed its first real test — after
3 real losing deals (deliberately generated by opening and immediately
closing positions, since the bid/ask spread guarantees a small loss),
it returned `False` when it should have returned `True`. Root cause,
found with a dedicated debug script rather than guessed: **MT5 deal
timestamps are in the broker's server time, not true UTC** — measured
skew was ~10,799 seconds (~3h) on this account. The original code
compared server-time deal timestamps against `datetime.now(timezone.utc)`
on the local machine, so very recent deals could land *after* the
query's "now" boundary in true-UTC terms and get silently excluded.
Fixed by deriving "now" from a live tick's `.time` field instead of
the local clock, so the query window, the deal timestamps, and the
cooldown-expiry check are all in the same server-time domain — never
mixed with local-machine time again. Re-verified against the exact
same 3 losses already in history: `in_cooldown()` now correctly
returns `True`.

This is the concrete payoff of testing against a real terminal instead
of stopping at the mock: the mock's fake `history_deals_get` had no
real timestamp semantics to get wrong, so this bug was structurally
invisible to it. It would have shipped silently — the breaker would
have looked like it worked in every check *except* the one that
matters, catching a genuine losing streak in real time.

**Update 2026-07-23 (later same day): full signal-generation path
verified against XM's real live feed — and the correct XM symbol
names are now confirmed, not guessed.**

Realized partway through testing that the MetaQuotes-Demo test account
uses a *different price feed* than XM (different liquidity providers →
different quotes, different bar prints, a few dollars off even for the
"same" instrument) — expected and normal across brokers, but it means
signal *values* computed against that test account's data were never
going to be representative of what the same code sees on XM, even
though the mechanics were fine to prove there. `evaluate_regime_
confluence_signal()` specifically needed testing against XM's actual
feed to mean anything.

Opened a second, real XM demo account (345899957, "XMGlobal-MT5 10" —
same broker/server family as the real account 330507861, just a
different demo number) via XM's own web signup, logged it into the XM
desktop terminal (that build turned out to be single-instance — can't
run two windows of it at once, so this meant temporarily switching the
one XM terminal window's login rather than running two side by side;
safe here specifically because `test_signal_readonly.py` has no
order-sending code path at all, regardless of which account is
connected), and ran `test_signal_readonly.py` — a strictly read-only
script (`connect`, `resolve_symbol`, `build_live_features`,
`evaluate_regime_confluence_signal`; no `place_trade`, no
`order_send`) against it.

Results: `resolve_symbol()` found XM's real names directly —
**`"US30Cash"`** (confirmed exactly what the Market Watch tab label had
hinted at, now actually verified) and **`"GOLD"`** (plain, not
`"XAUUSD"` like the test account) — `SYMBOL_MAP` now holds these as the
values to use for XM. Both instruments returned `{"signal": None}` on
the live bar at test time, which checks out by hand: US30's
efficiency ratio was 0.055 (deep chop, nowhere near the 0.35 trending
threshold) and its EMA8/EMA21 relationship didn't match its macro
trend either; Gold's ER was 0.30 (still under threshold) with the same
kind of EMA/MACD-vs-macro-trend mismatch. Correct behavior, not a gap
— confirms the rule logic is internally consistent on real current
data, not just non-crashing.

**Update 2026-07-23 (final): `run_once()` verified end-to-end — Layer 7
is now fully proven, not just plumbed.**

Real market conditions at test time didn't satisfy the confluence
rules on either instrument, even with the regime gate forced open
(`er_threshold=0.0`) — confirmed both instruments correctly returned
`{"action": "skip", "reason": "no signal"}` via `test_run_once.py`,
consistent with the real ER/EMA/MACD readings already checked by hand
above. That proved the skip path but not the "found a signal" path,
since live conditions wouldn't cooperate. Rather than wait, monkey-
patched `evaluate_regime_confluence_signal()` to return a fixed fake
signal (`test_run_once_forced_signal.py`) and ran `run_once()` for
real against XM with everything else live (`has_open_position`, the
circuit breaker check, real account/symbol) — asserted the signal's
SL/TP actually made it into the constructed order request, not just
that something executed without an exception.

One correct-but-worth-confirming behavior this surfaced: the
constructed order's `price` field was the real current market price,
not the fake signal's price — because `place_trade()` never uses a
signal's `price` at all, it re-fetches its own live tick internally at
execution time and only takes `sl`/`tp` from the caller. Right design
(a fresh price beats a stale one from feature-building time), now
confirmed by a real test rather than assumed from reading the code.

With this, every function in `trader/l7_execution/` has been run for
real at least once — the read path, the sizing math, real order
placement (dry-run and `dry_run=False`), the circuit breaker (after
fixing its real bug), the signal-generation path against XM's actual
feed, and now the full orchestration. `dry_run=False` has only ever
been exercised on the throwaway MetaQuotes-Demo account (109989358) —
that should stay true; nothing here has sent a real order to XM, by
design, and shouldn't until there's a specific reason to.

- `connect()` / `shutdown()` / `account_summary()` — attach to an
  already-running, already-logged-in MT5 terminal. **Verified working.**
- `resolve_symbol(candidates)` — **verified**, and XM's real symbol
  names are now confirmed rather than guessed: `SYMBOL_MAP` holds
  `{"US30": "US30Cash", "GOLD": "GOLD"}`, resolved directly against an
  XM demo account (345899957, same broker/server family as the real
  account). Different from the MetaQuotes-Demo test account's names
  (`"US30"`/`"XAUUSD"`) — different broker, different naming; that
  test account's values would need overriding if it's reused.
- `get_live_bars()` / `build_live_features()` — pull live H4 bars and
  run them through the *same* `l2_features`/`l3_regime` functions used
  in backtesting, so live features are computed identically to backtest
  features. **Both verified** — real bars and a full feature row (MA,
  EMA, MACD, ATR, ADX, ER) were pulled and printed against XM's real
  feed and checked by hand for internal consistency.
- `evaluate_regime_confluence_signal()` — a **hand-port** of
  `RegimeConfluenceStrategy.next()`'s entry rules from
  `l4_signal_model.py`. `backtesting.py`'s `Strategy` class is a
  backtest-loop construct and can't be driven bar-by-bar live directly,
  so this is a second, separate copy of the same rules. **This will
  silently drift out of sync if `l4_signal_model.py` changes and this
  isn't updated too** — the single biggest structural weak point of
  this layer, still true even now that it's **verified working**
  (correctly returned `None` on both US30 and GOLD given their real
  live regime readings, checked by hand — see the 2026-07-23 update
  above). Collapsing both into one shared rule definition both
  backtest and live can call is the natural next step.
- `LiveCircuitBreaker` — same N-losses-in-a-row → cooldown policy as
  `l6_risk.CircuitBreakerMixin`, but sourced from MT5's own deal history
  (`mt5.history_deals_get`) instead of `backtesting.py`'s
  `self.closed_trades`, since there's no backtest engine live to supply
  that. **Verified working** — but only after fixing a real
  server-time-vs-local-time bug found via testing (see above);
  `in_cooldown()` now derives "now" from a live tick, never the local
  clock, and correctly detects a real 3-loss streak.
- `size_fraction_to_lots()` — converts `l5_position_sizing`'s `(0, 1]`
  size fraction into an actual MT5 lot volume, using the symbol's
  contract size and rounding/clamping to the broker's `volume_step`/
  `volume_min`/`volume_max`. **Verified** — hand-checked against a real
  fill (1% risk, $5.00 stop, $5,000,000 equity → 1.2 lots, matched the
  formula exactly).
- `place_trade()` — **defaults to `dry_run=True`**, meaning it computes
  and returns the exact MT5 order request without calling
  `mt5.order_send()`. **Verified** — both a dry run and a real
  `dry_run=False` order were tested on the MetaQuotes-Demo test
  account and worked correctly. Still only ever run with `dry_run=True`
  as the default; still never run against XM.
- `run_once()` — one full polling cycle: skip if a position is already
  open, skip if the circuit breaker is in cooldown, else fetch → feature
  → signal → size → (dry-run or real) trade. **Verified** — both the
  real "no signal" skip path and (via a monkeypatched signal, since
  real conditions didn't cooperate) the "found a signal → dry-run
  trade" path, with the SL/TP asserted to actually reach the
  constructed order request.

### Test scripts added during real verification (`trader/l7_execution/`)

- `smoke_test.py` — read-only connectivity check (see above).
- `test_place_trade.py` — dry-run then (with explicit `yes` confirmation)
  a real order, hard-gated to only run against the test account login.
- `test_circuit_breaker.py` — deliberately generates real losing deals
  (open + immediate close, spread loss) and checks `in_cooldown()`
  against them.
- `debug_deals.py` — dumps raw deal history plus server-time-vs-local-
  time skew; this is what actually found the timestamp bug rather than
  guessing at a fix.
- `check_cooldown.py` — checks `in_cooldown()` against whatever deal
  history already exists, without opening new positions.
- `test_signal_readonly.py` — tests the signal-generation path
  (`build_live_features`, `evaluate_regime_confluence_signal`) against
  whatever real feed it's pointed at. Has no `place_trade`/`order_send`
  code path anywhere in the file — safe to point at a real account
  (used against the XM demo, 345899957) since there's no way for it to
  submit an order regardless of which account is connected.

The five order-capable scripts share the same safety pattern: read
`account_summary()` first and refuse to proceed if the connected login
isn't the designated test account (109989358) — kept as a reusable
pattern for any future Layer 7 test script that can send orders.
`test_signal_readonly.py` doesn't need that gate, by construction.

### Scheduling + journal (2026-07-24)

`run_once()` existed but nothing ever called it more than once, and no
call's result was kept anywhere - both gaps closed now:

- `run_scheduled.py` — single-shot script: one `run_once()` pass over
  both US30 and GOLD, `dry_run=True` (hardcoded, not a flag - see its
  docstring for why), `timeframe="H4"` (confirmed the best of H1/H4/D1
  by the timeframe sweep above), `magic=100001` (the one magic number
  *not* prefixed `999xxx` - every test script's magic was chosen
  specifically to stay out of this one's way). Needs no admin rights to
  run - it's a plain script, nothing Task-Scheduler-specific baked in.
  **Decision 2026-07-24: run it manually, at will**, from PowerShell or
  VS Code's terminal (`python -m trader.l7_execution.run_scheduled
  "path\to\terminal64.exe"`), not via Windows Task Scheduler. Task
  Scheduler's "Create Task" dialog turned out to require a UAC
  elevation prompt on this machine, which computer-use automation
  can't click through (a hard OS security boundary, not a settings
  issue) - would have needed the user to babysit every setup click
  anyway, so simpler to just skip it. Trade-off, stated plainly: manual
  invocation means the journal only fills in when someone actually runs
  it, not an even every-4-hours drumbeat - `journal_summary.py`'s
  "weekly" review will have gaps on days it wasn't run. Acceptable
  trade for staying in control of it. Task Scheduler setup steps are
  kept below in case the unattended version is ever wanted later. Not
  **Update 2026-07-24: run for real** — `python -m
  trader.l7_execution.run_scheduled "...\terminal64.exe"` against XM
  demo 345899957, connected fine, evaluated both US30 and GOLD,
  correctly returned "skip (no signal)" for both (matches the regime
  gate being picky by design, same as every other real signal check
  this week).
- `journal.py` — append-only JSON-Lines log (`data/journal.jsonl`), one
  line per `run_once()` result. JSONL over CSV because "skip" and
  "trade" results have different, nested shapes that don't flatten into
  one schema cleanly. Append is a single atomic write, so concurrent
  runs can't corrupt each other. **Verified** — round-tripped fake
  entries matching `run_once()`'s real result shapes through
  `append_entry`/`read_entries` in the sandbox first, **then for real**:
  the run above appended 2 real entries without error.
- `journal_summary.py` — reads the journal, defaults to the last 7 days
  (`--days N` to change), breaks down skip reasons and signal fires per
  instrument. Right now (before `dry_run` is ever flipped to `False`
  here) this is just a signal-frequency check - "how often would this
  actually have fired this week." Once real trades exist, the intent is
  the same script compares realized win rate/expectancy against the
  walk-forward backtest's numbers, to catch live drift early rather
  than months in. **Verified** — sandbox fake-entry round-trip first,
  **then for real**: `python -m trader.l7_execution.journal_summary`
  read the 2 real entries back and rendered the report correctly (2
  total, 2 in window, per-symbol skip-reason breakdown, 0/0 dry-run/real
  trade totals - all matching what was actually journaled).

The full loop - `run_once()` → `run_scheduled.py` → `journal.py` →
`journal_summary.py` - is now verified against real MT5 data end to
end, not just sandbox-simulated. What's left isn't more code, it's
time: run it periodically (manually, per the decision above) and let a
real week of entries accumulate before the weekly review means anything.

### Manual bias override + live monitor (2026-07-25)

Grew out of a live discretionary GOLD chart read (Elliott/Wyckoff,
worked through in chat) that landed on a genuine either/or: price either
clears the ~4,164-4,180 shelf and keeps going (bullish), or fails there
and eventually sweeps the ~3,958 low first (bearish). The mechanical
strategy has no concept of that kind of discretionary read — it only
sees ER/EMA/MACD/BOS. Two additions bridge the two without touching the
strategy's actual rules:

Chart reads themselves (the wave counts, levels, and confirm/invalidate
conditions worked out in chat, session by session) are now recorded in
**`DISCRETIONARY_ANALYSIS.md`** (2026-07-27, repo root) rather than only
living in scrollback — a dated log, not executed by any code, that gives
the reasoning behind whatever bias/key-levels are currently set a paper
trail. See that file for the full geometry; this section covers only the
mechanism that lets a read there turn into an actual override.

- `run_scheduled.py`'s `BIAS` dict — set `BIAS[symbol_key]` to `"long"`
  or `"short"` from your own chart read; leave `None` (the default) to
  let the mechanical signal run untouched, same as before this change.
  `_preview_signal()` peeks at what `evaluate_regime_confluence_signal()`
  would return this cycle — read-only, no position/breaker checks, no
  order — purely to compare its direction against `BIAS`. If they
  agree, or there's no signal, nothing changes. If they fight,
  `BIAS_MODE` decides: `"mute"` (default) skips the trade and journals
  why (`reason: "bias_override(bias=..., signal=...)"`); `"downsize"`
  still takes it at `BIAS_DOWNSIZE_FACTOR` (default 0.5) × risk_pct.
  Neither mode edits `l4_signal_model.py` or
  `evaluate_regime_confluence_signal()` — the strategy stays exactly
  as walk-forward-tested; the override lives entirely in the
  orchestration script. **Verified**: mocked `run_once`/`_preview_signal`/
  `append_entry` in the sandbox and exercised both modes directly —
  mute correctly skipped and journaled the fighting symbol while
  letting the unbiased one trade normally; downsize correctly halved
  `risk_pct` on both symbols and still called `run_once`. Not yet run
  against real MT5 data with a live bias set (both default to `None`
  right now).
- `live_monitor.py` — read-only desktop dashboard, no relation to the
  trading cycle's timer. On its own refresh loop (default 60s) it pulls
  live bars per symbol via the same `build_live_features()` `run_once()`
  uses, plots a live-updating `mplfinance` candlestick chart per
  instrument, and titles each chart with close price, current
  Efficiency Ratio, TRENDING/CHOP classification, the symbol's `BIAS`
  (imported straight from `run_scheduled.py`, so the two scripts can't
  drift apart), and the circuit breaker's live cooldown status. Same
  info also prints as a one-line log per symbol per cycle, so there's a
  plain-text record even with the window closed. Needs
  `pip install mplfinance` (added to `requirements.txt`). **Verified**:
  `_status_line()` exercised against a synthetic OHLC+ER DataFrame with
  a stubbed circuit breaker (correct TRENDING/CHOP/bias/cooldown
  readout), and the `mpf.plot(..., ax=ax, ...)` external-axes call
  exercised directly with synthetic bars — rendered and saved without
  error. Not yet run against a real MT5 feed.

    (venv) PS> python -m trader.l7_execution.live_monitor "C:\path\to\terminal64.exe"
    (venv) PS> python -m trader.l7_execution.live_monitor --refresh 30 --bars 150 --timeframe H1

### live_monitor.py dashboard rebuild + key-level/pause-window logging (2026-07-25)

Same-day follow-on, in three parts, driven by using the dashboard live
and finding the first version wasn't actually usable, plus closing the
loop on two of the three items proposed when BIAS above first shipped.

- **Dashboard rebuild.** The first cut was a static-looking single
  price chart with no separate indicator view. Rebuilt to a 3-row-per-
  symbol layout (price+overlays / ER "regime gate" / MACD "momentum"),
  default 18-bar visible window (was showing the full 150-bar pull,
  too zoomed out to read individual candles), dark SCADA/digital-twin
  theme with colors and fonts matched directly to the user's own HMI
  dashboard CSS (`#0a0d12` background, `#00e676`/`#ff1744` up/down,
  Share Tech Mono / Orbitron / Rajdhani font stack), and a responsive
  event loop (`_wait_responsively()` looping small `plt.pause()` calls
  instead of one long `time.sleep()`) so the window no longer freezes
  and can be dragged/resized mid-cycle. A `resize_event` callback
  re-runs `tight_layout()` immediately instead of waiting for the next
  60s data refresh. Y-axis on the price/MACD panels is now recomputed
  from only the bars inside the active visible window each redraw
  (previously `mpf.plot()` was auto-scaling to the full 150-bar
  DataFrame regardless of the narrowed `xlim`, which flattened/
  stretched the candles once zoomed to 18 bars).
- **Cross-panel zoom sync fix.** Zooming/panning the price panel via
  the matplotlib toolbar wasn't propagating to the ER/MACD panels below
  it — the old code only copied `xlim` across panels once per 60s
  redraw, so interactive zoom looked completely unsynced until the next
  cycle caught up. Fixed at the root with `sharex="col"` on the
  `plt.subplots()` call instead of manual `xlim` copying — matplotlib's
  native axis-sharing propagates instantly and, confirmed by direct
  test, survives the `ax.clear()` every redraw does each cycle.
- **Key-level logging.** `run_scheduled.py`'s `KEY_LEVELS` dict (e.g.
  GOLD's 4,180 invalidation-up / 3,958 invalidation-down from the
  Elliott/Wyckoff read) is now recorded on every journal entry via
  `_key_level_context()` — one extra `get_live_bars(count=1)` call,
  kept independent of `_preview_signal()`/`run_once()`. Purely
  observational: never gates or mutes anything (`BIAS` already owns
  that job) — it exists so `journal_summary.py` can show "was price
  above/below your shelf" alongside every decision after the fact.
  `journal.py`'s `append_entry()` gained an optional `context` param,
  kept as its own top-level JSON key rather than merged into `result`
  so the mechanical result is untouched and no trading logic ever
  reads it back.
- **Manual pause windows.** `run_scheduled.py`'s `PAUSE_WINDOWS` dict
  skips a symbol entirely (no bias check, no `run_once()` call at all)
  during a stretch expected to be noisy/whipsaw-prone — first use is
  GOLD's Aug 7-10 2026 window, the spring scenario's predicted flush
  period. Distinct from the circuit breaker, which only reacts after
  losses have already happened; this pre-empts a stretch already
  expected to be bad rather than waiting to get hurt by it first.
  Still journaled (`reason: "manual_pause_window"`), still shown in
  `journal_summary.py`.

**Verified**: all four changes tested via mocked calls / synthetic
data in the sandbox — `_in_pause_window()` correct for GOLD inside/
outside the window and US30 never paused; `_key_level_context()`
returns the right dict for GOLD and `None` for US30; a full `main()`
run with a mocked "now" inside the pause window correctly skipped
GOLD's `run_once()` while still journaling context, and ran US30
normally; `journal_summary.py`'s per-entry and per-symbol aggregate
key-level output and the new `[PAUSED]` line checked against a
synthetic journal file covering skip/trade/paused/no-context entries;
the dashboard rebuild and sync fix rendered to PNG against synthetic
OHLC data across several iterations and the `sharex` link tested
directly for instant same-column propagation, cross-column
independence, and survival through `ax.clear()`. None of this has run
yet against a live MT5 feed or a real multi-hour dashboard session.

### Right-side account/P&L panel (2026-07-25)

Used the dashboard's previously-empty right margin for a read-only
account/position readout, per your call to keep this demo display-only
rather than adding clickable order-entry controls (a bigger, separate
decision - matplotlib widgets are a poor fit for real order entry and
it would need its own dry-run safety discipline built in deliberately).

- `l7_execution/__init__.py`'s new `get_position_info(symbol, magic)` -
  a read-only wrapper around `mt5.positions_get()`/`mt5.account_info()`,
  returns direction/volume/entry/current price/SL/TP/profit/pnl_pct for
  this bot's magic number on that symbol, or `None` if flat.
- `live_monitor.py`'s `_draw_side_panel()` renders that alongside
  `account_summary()` (equity/balance/margin) as plain monospace text a
  fixed-width `fig.add_axes()` panel to the right of the existing
  price/ER/MACD grid — positioned once outside the grid's own
  `gridspec`/`tight_layout()` management (`set_in_layout(False)`) so it
  can't get reflowed by the resize-fix's `tight_layout()` call.
  Long positions/positive P&L render in the up-green, short/negative in
  down-red, flat symbols show "flat - no open position" in muted text.
  No buttons, no callbacks — same read-only guarantee as the rest of
  this script.

**Verified**: `_draw_side_panel()` rendered to PNG against synthetic
account/position dicts (long position, short position, flat, and
`account_summary()` unavailable) — correct color-coding and layout in
all cases; the full 2-column chart grid + panel rendered together
against synthetic OHLC data confirmed the panel doesn't distort or
overlap the price/ER/MACD columns. Not yet run against a real MT5
account or a real open position.

### Clickable bias/pause/key-level controls (2026-07-26)

You asked, after the read-only P&L panel: "shouldn't there be an
inputbox with options that feeds back to the bot engine" for
bias/key-levels/pause, instead of hand-editing run_scheduled.py's
source every time. This is that - but scoped deliberately narrower
than order-entry buttons (which you separately said to hold off on for
the demo): bias/pause/key-levels can only ever mute, downsize, or skip
a trade the mechanical strategy would otherwise take on its own -
nothing here can place, size, or close a real order, so a click
carries the same risk profile as editing a config file used to, not
the risk profile of a trade button.

- **`manual_overrides.py`** (new file) - `data/manual_overrides.json`
  is now the live, shared state for `bias`/`key_levels`/`paused_now`
  per symbol. `load_overrides()` reads it (seeding it with the old
  hardcoded defaults the first time it doesn't exist);
  `set_bias()`/`set_key_level()`/`set_paused_now()` write it via an
  atomic `os.replace()` and append one line to
  `data/manual_overrides_log.jsonl` (timestamp, field, symbol, old,
  new) - an audit trail for this config the same way `journal.py` is
  one for trade decisions. `PAUSE_WINDOWS` (the pre-set calendar dates)
  deliberately stayed hardcoded in `run_scheduled.py` - it's dates, not
  something worth a click; `paused_now` is the new immediate on/off
  layered on top of it.
- **`run_scheduled.py`** - `BIAS`/`KEY_LEVELS` module-level dicts are
  gone; `main()` now calls `load_overrides()` fresh at the top of every
  cycle instead. `_key_level_context()` takes `levels` as an argument
  rather than reading a module global. `_in_pause_window()` (calendar
  dates) and the new `paused_now` check are both applied - either one
  skips the symbol, journaled with the matching reason
  ("manual_pause_window" or "paused_now").
- **`live_monitor.py`** - the right-side panel now has, below the
  read-only account/position text, one control block per symbol built
  ONCE at startup (`_build_controls()` - unlike the price/ER/MACD
  panels, these can't be torn down and rebuilt every redraw without
  losing typed/focus state): `RadioButtons` for bias (Long/Neutral/
  Short), a `Button` that toggles `paused_now` and relabels itself
  PAUSE NOW ↔ RESUME immediately on click, and two `TextBox`es for the
  invalidation-up/down key levels. Every callback does exactly one
  thing - call the matching `manual_overrides.set_*()` - then updates
  its own on-screen state so the click reads back without waiting for
  the next 60s redraw. The price panel title's `bias=` now comes from
  `load_overrides()` read once per redraw cycle (shared with the
  panel), not a stale module-level import.
- **`journal_summary.py`** - prints every override change in the
  lookback window (`_print_override_changes()`, reading
  `manual_overrides.read_change_log()`) before the per-symbol trade/
  skip breakdown, since a bias flip or pause toggle is usually why the
  numbers below it look different from the week before.

**Verified**: `manual_overrides.py`'s load/save/set_* functions and
atomic-write behavior tested directly against a scratch directory
(seeding, round-trip, no-op suppression when a value doesn't change,
change-log entries, final-file-is-valid-JSON check).
`run_scheduled.py`'s `main()` re-tested end-to-end with `paused_now`
set via `manual_overrides` (mocked MT5 calls) - confirmed GOLD's
`run_once()`/`build_live_features()` were never called while
`paused_now` was set, and the journal entry still carried key-level
context. `live_monitor.py`'s widget construction, callbacks, and full
`run()` (one cycle, mocked MT5 + synthetic OHLC data) all rendered to
PNG without error or overlap. The core round-trip was verified two
ways in one script: (1) called `controls["GOLD"]["radio"].set_active(0)`
- matplotlib's own internal method a real click invokes, not my
callback function directly - and confirmed `manual_overrides.json`
updated; (2) ran a second `_redraw_column()` cycle afterward and
confirmed the price panel's title changed from `bias=neutral` to
`bias=long`, proving a dashboard click actually reaches the next
redraw. Combined with the `run_scheduled.py` test above, both halves
of "a click reaches the bot" are verified (dashboard → file, and file →
next scheduled cycle) - just not against a live MT5 terminal yet, and
not with a real mouse click through the actual GUI (only via the
widgets' own programmatic entry points, which is what a real click
calls internally).

**Windows Task Scheduler setup** (not the chosen path - see decision
above - kept here only in case the unattended version is wanted later;
do this yourself, I have no presence on your machine to do it for you,
and "Create Task" needs a UAC prompt I genuinely cannot click through):
1. Task Scheduler → Create Task (not "Basic Task" - need the Conditions
   tab). General tab: run whether user is logged in or not, if you want
   it to survive a locked screen.
2. Triggers tab → New → Daily, recur every 1 day, **Repeat task every:
   4 hours**, for a duration of 1 day (so it re-fires every 4h
   indefinitely) - this is what approximates H4 candle-close alignment
   without hand-maintaining 6 separate daily trigger times.
3. Actions tab → New → Program: your venv's `python.exe` (full path,
   e.g. `C:\Users\milat\Documents\us30-trading-bot\venv\Scripts\python.exe`)
   → Arguments: `-m trader.l7_execution.run_scheduled "C:\path\to\terminal64.exe"`
   → Start in: `C:\Users\milat\Documents\us30-trading-bot`.
4. The MT5 terminal itself still needs to be running and logged into
   the XM demo (345899957) for this to work - Task Scheduler runs the
   script, not the terminal.

### Mobile app (PWA) — `mobile_api.py` (2026-07-27)

User asked to "test the trading bot on my Android phone." Reality check
that shaped this: MetaTrader5's Python package is Windows-only, wired to a
locally-running MT5 terminal via COM/DLL calls — there is no MT5 Python
API for Android (or Linux/Mac), so the bot's actual logic (connect(),
build_live_features(), the scheduled cycle) has no way to run on a phone.
An Android "app" can only ever be a *client* to the bot, not a port of it.

Given that, and the user's choice of scope ("home WiFi only," not exposed
to the open internet), the buildable path is: a small Flask server
(`trader/l7_execution/mobile_api.py`) running as a third independent
MT5-connected process alongside `run_scheduled.py`'s Task Scheduler job
and `live_monitor.py` — same pattern as those two, its own `connect()`
call, reading/writing the same `data/manual_overrides.json` and
`data/journal.jsonl` — plus a mobile-friendly frontend
(`trader/l7_execution/mobile_app/`) that Chrome on Android can
"Add to Home Screen" into a real app icon (a PWA — `manifest.json` +
icons + a no-op service worker, no Play Store needed).

**What it exposes**: `GET /api/status` (account equity/balance, and per
symbol: close/ER/regime/MACD/cooldown/bias/paused/key-levels/open
position), `GET /api/journal` (recent entries, most-recent-first). And
the *exact same safe controls* `live_monitor.py`'s buttons already call —
`POST /api/bias`, `/api/pause`, `/api/key_level` — thin wrappers around
`manual_overrides.set_bias()/set_paused_now()/set_key_level()`. Nothing
new was added to what these can do: still mute/downsize/skip only, never
place or size a real order — `place_trade()`/`run_once()` are never
called from this file.

**Charts** (added same day, after the user asked for the ER/MACD/price
charts too, not just numbers): `GET /api/chart/<symbol_key>.png` renders
the same price+EMA8/EMA21/MA89/MA200/MA360+swing-hi/lo, ER, and MACD
panels `live_monitor.py`'s desktop window draws — by calling
`live_monitor.py`'s own `_redraw_column()` against a throwaway Agg-backend
figure and returning the PNG bytes, rather than reimplementing the
drawing logic a second time. matplotlib's backend is set to `"Agg"`
(headless, buffer-only) at the very top of `mobile_api.py`, before
`live_monitor` gets imported anywhere — this process never opens a real
GUI window, unlike running `live_monitor.py` directly. The frontend polls
this on the same ~10s cadence as `/api/status`, pre-loading each new PNG
into a throwaway `Image()` before swapping it in so the visible chart
doesn't flash blank while the next one renders — it's a snapshot, not
interactive (no pinch-zoom/pan like the desktop window's toolbar), by
design: simplicity and drawing-code reuse over a client-side charting
library.

**Auth**: a random token, generated once into `data/mobile_token.txt`
(gitignored) on first run, required (via `?token=` or an `X-Auth-Token`
header) on the three POST endpoints only — not meant to withstand a real
attacker, just to stop another device on the same WiFi from flipping bias/
pause without it, matching the "home WiFi only" scope this was explicitly
built for. GET endpoints are unauthenticated (read-only numbers off a
demo account, same sensitivity as what's already visible in
`live_monitor.py`'s window). If this is ever opened to the open internet
instead of just home WiFi, that token is not sufficient on its own — needs
a real auth layer (and a tunnel like Tailscale, not port-forwarding)
first, not something to default into.

**Run it** (own process, own terminal window, alongside the others):

    (venv) PS> pip install flask
    (venv) PS> python -m trader.l7_execution.mobile_api "C:\path\to\terminal64.exe"

Prints the token and a URL. On the phone (same WiFi): open that URL in
Chrome, then menu → "Add to Home Screen." Whether Chrome offers full
chrome-less standalone install (a WebAPK) vs. a plain bookmark-style
shortcut can depend on install-criteria details on plain HTTP (no TLS,
deliberately, given the home-WiFi-only scope) — either way it's a one-tap
icon, no URL typing needed after the first visit; the token is saved to
the phone's `localStorage` on first load so it isn't needed again unless
that storage is cleared.

**Verified**: `test_mobile_api.py` (same mocked-MT5 pattern as
`test_run_once.py` etc., runs on Linux, no real terminal needed) drives
the actual Flask app through its test client — status/journal shape, all
three POST endpoints rejecting a missing/wrong token then actually writing
through to `manual_overrides.json` when given the right one, bad
symbol/value rejected with 400, the chart endpoint returning real PNG
bytes (magic-byte checked) for a known symbol and 404 for an unknown one,
and the static frontend/manifest/icon files actually being served. Note
the chart path needed mocking `live_monitor`'s *own* `build_live_features`/
`LiveCircuitBreaker` bindings too, separately from `mobile_api`'s — each
module's `from . import X` binds its own local name, so patching one
doesn't reach code running inside the other. All passing as of this
writing. The frontend itself (`index.html`'s JS) has not been exercised in
a real browser — only the backend it talks to.

### Auto/Manual mode + manual order placement (2026-07-30)

The one deliberate exception to "nothing here places or sizes a real
order" above. User's ask: a per-symbol Auto/Manual switch, and when
Manual, a way to configure and actually send a real order from the
running app itself — "engine should reside on the live app not
somewhere else or needing you to do anything once i have run it and
monitoring the market." Confirmed scope before writing any of this:
demo account only (345899957), for the next couple of months, while the
user's real-money account continues to be traded by them personally,
entirely outside this app.

**`manual_mode`** (`manual_overrides.py`): a new per-symbol bool next to
bias/key_levels/paused_now, same atomic-write + change-log pattern,
`set_manual_mode(symbol_key, value)`. `False` (Auto, default) is every
behavior this file already had. `True` (Manual) does two things:
`run_scheduled.py`'s loop skips that symbol entirely (checked first,
before `paused_now` — see its module docstring), and the two manual-order
endpoints below refuse to act on that symbol unless it's set. This is
the mechanism that keeps the mechanical engine and a hand-placed order
from ever fighting over the same symbol at the same time.

**`POST /api/manual_mode`** — token-gated toggle, thin wrapper around
`set_manual_mode()`, same shape as `/api/pause`.

**`POST /api/manual_order/validate`** — always `dry_run=True`, a pure
preview (no token, no manual_mode check — lets the app show live sizing
as the user types). Reuses `place_trade()` from `l7_execution/__init__.py`
directly rather than re-deriving lot-size math a second time. Body:
`symbol_key`, `direction` (long/short), `sl`, `tp`, `risk_pct` (capped at
0.05 — a fat-finger guard, not a strategy choice). Also sanity-checks
sl/tp landed on the correct side of the live tick price for the given
direction (long needs sl < price < tp, short the reverse) — a wrong-side
SL isn't a risk-pct problem, it's a "this order is backwards" problem,
caught before place_trade() ever runs.

**`POST /api/manual_order/send`** — the only call anywhere in this repo
that can reach `place_trade(dry_run=False)`. Every gate below is checked,
in order, before that call; any failure returns first, no order sent:
token auth → `confirm: true` present in the body → `manual_mode=True`
already set for that symbol → connected account's login equals
`DEMO_LOGIN` (345899957) — a hard-coded equality check, not a setting,
so the user's real account (330507861) is structurally unreachable
through this endpoint regardless of what the request asks for → no
existing open position under `MANUAL_MAGIC` on that symbol already → the
same sl/tp-vs-live-price sanity check `/validate` does, re-run rather
than trusted from an earlier preview (price may have moved). Uses
`MANUAL_MAGIC = 100002`, distinct from the mechanical engine's
`MAGIC = 100001`, so a manual fill is invisible to the mechanical
engine's `has_open_position()`/`LiveCircuitBreaker`/position-tracking
and vice versa — the journal entry is tagged with `MANUAL_MAGIC` and
`result.action = "manual_trade"` so `journal_summary.py` can tell manual
fills apart from mechanical ones.

**Frontend** (`mobile_app/index.html`): each symbol card gets an
AUTO/MANUAL toggle; MANUAL reveals a form (direction, SL, TP, risk %)
with VALIDATE and SEND buttons. SEND stays disabled until VALIDATE
succeeds, and any change afterward — a field edit, switching direction —
immediately disables it again (`invalidateOrderValidation()`), so a
send body always exactly matches what was last validated, never a stale
one. SEND itself is gated behind a native `confirm()` dialog stating
it's a real order on the demo account before the request ever goes out.

**Verified**: extended `test_mobile_api.py` with a mocked `place_trade`/
`has_open_position` — manual_mode toggle auth, validate accepting a
correctly-sided order and rejecting a backwards one, risk_pct-over-cap
rejection, send rejecting missing token/confirm/manual_mode-still-Auto/
wrong-account-login/existing-open-position, then a full successful send
with the journal entry checked for `MANUAL_MAGIC` and `manual_trade`.
All passing. The frontend's order form has not been exercised in a real
browser or against a real MT5 terminal — only `node --check`'d for
syntax and reviewed by hand; the user should validate-then-send one
small test order on the demo account themselves before relying on it.

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

- `run_scheduled.py` + Windows Task Scheduler setup exist now (see
  above) but the script itself hasn't actually been run yet - same as
  everything else in this file, written and reasoned through but not
  yet exercised for real. That's the next actual step, not more code.
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

---

## SMC zone/session gate — mined from external research, tested to a final verdict (2026-07-26)

User granted access to a personal folder (`Journalling/`, outside this
repo) of Smart Money Concepts (SMC) chart-journalling notebooks - LaTeX
weekly reports plus ~25 iterations of a Jupyter dashboard, culminating in
`XAUUSD/___V1006____smc_xauusd_dashboard_upgraded.ipynb`. That notebook
auto-computes (no manual level entry, unlike earlier iterations) a
premium/discount trading range from rolling price extremes, classifies
each bar's zone, and tags a trading session (London/New York/Off-session)
by hour-of-day - built for a manual weekly journal, never backtested
there. Mined for concrete, codeable rules per the user's direction
("mine the SMC logic for new signal ideas" / goal: "better risk-adjusted
returns").

**New, reusable features** (`l2_features.py`):
- `premium_discount_zone(df, structure_lookback=8, liquidity_lookback=60)`
  — structure_high/range_low via the existing `swing_high()`/`swing_low()`
  (causal, `.shift(1)`'d) rather than the source notebook's unshifted
  rolling max/min - a bar can't use its own extreme to classify its own
  zone, consistent with every other lookback feature in this file.
  Returns structure_high/range_low/eq/liquidity_high/liquidity_low/zone.
- `trading_session(index, tz="Asia/Dubai")` — London/New York/Off-session
  per bar. **Caveat flagged in its own docstring, not silently trusted**:
  this project's OHLCV index is timezone-naive and its true origin
  (broker/server time vs UTC) has never been confirmed - the function's
  `tz_localize()` assumption is unverified. Built for 5m/15m bars in the
  source notebook; ported onto this project's H4 bars, where a 3-5 hour
  session window is coarse against 4-hour candles.

**New gate hook, provably behavior-preserving**: `ConfluenceStrategy`
and `LiquiditySweepStrategy` both gained a `_extra_gates_ok(direction)`
method - default `return True`, ANDed into each strategy's existing
entry conditions. Since the default is an unconditional `True`, this is
an `and True` that cannot change either strategy's behavior unless a
subclass overrides it - confirmed by re-running both against real
history and getting the same class of results as before this change
existed (see below; also directly reasoned through by code inspection,
same standard used for the Layer 6 circuit-breaker extraction).

**`SMCZoneConfluenceStrategy(RegimeConfluenceStrategy)`** — adds the zone
gate (longs only from "discount", shorts only from "premium") and the
session gate. **Finding: structurally incompatible with this strategy
family, confirmed empirically, not assumed.** Tested the zone gate alone
across 7 lookback widths (8 to 200 bars) against US30's full history -
long entries satisfying `ConfluenceStrategy`'s breakout condition
(`price > swing_high`) almost never land in "discount" at *any* width
(0 qualifying signals every time); shorts fared only slightly better
(3-9 signals). Reason, once seen: breaking out to a new high **is**
pushing price toward the top of whatever range you measure it against -
the zone concept (buy cheap, sell expensive at range extremes) and
breakout/trend-continuation entries are close to logically incompatible
when simply ANDed together. Session-gate-only (dropping the zone gate),
fixed defaults, full history, no optimization: US30 +10.31% → **-10.71%**
(247 trades, worse on every axis), Gold +4.87% → +5.82% (315 trades,
marginal improvement). Mixed and not optimized - not a final verdict,
but not an obvious win either.

**`SMCZoneLiquiditySweepStrategy(LiquiditySweepStrategy)`** — same zone
gate, paired with the strategy family it's the actual conceptual fit
for: entries here fire *after* a liquidity sweep + reversal, which
structurally lands much closer to a range extreme than a breakout does.
Fixed defaults, full history, no optimization:
- US30: 47 trades/+0.03%/34.0% win → **5 trades/-4.42%/0% win**. Sample
  size (5 trades) is too small to conclude anything from directly.
- Gold: 70 trades/+3.39%/37.1% win → 15 trades/+1.90%/40.0% win. Win
  rate improved, total return and trade count both dropped.

**Verdict, stated plainly: inconclusive.** The zone gate is a much
better philosophical fit for the reversal strategy than the breakout
one (confirmed, not assumed), but neither quick fixed-defaults test
shows a clean improvement - trade counts collapse by 80-90% in every
pairing, and returns don't clearly improve to compensate. This was
fixed-defaults testing only (matching how `LiquiditySweepStrategy`
itself was first evaluated, per the Layer 4 section above) - a proper
walk-forward optimization pass (same rigor as every other headline
number in this document) is the only way to give this a final verdict,
and hasn't been run - the fixed-defaults signal so far didn't clearly
justify the time cost (walk-forward over full history takes minutes per
instrument, per the section above).

**Full walk-forward optimization pass (2026-07-26, same day, later)** — the
proper test flagged as missing above. No `optimize_kwargs` grid existed
yet for this strategy family (`LiquiditySweepStrategy` had, until now,
only ever been walk-forward tested with `optimize_kwargs=None` - fixed
defaults, no per-fold tuning, unlike `ConfluenceStrategy`'s
`REGIME_OPTIMIZE_KWARGS`). Defined one and ran real per-fold optimization
- same anchored walk-forward harness, `Backtest.optimize()` re-fit every
fold on the training slice only, applied out-of-sample - for **both**
the baseline and the zone-gated variant, on both instruments, so the
comparison is optimized-vs-optimized rather than the old fixed-defaults
number against a newly-optimized one:

```python
BASE_GRID = dict(max_bars_to_bos=[6, 9, 12], max_pullback_bars=[8, 12, 16],
                  target_rr=[1.5, 2.0, 2.5, 3.0], maximize="SQN")
ZONE_GRID = dict(BASE_GRID, zone_lookback=[8, 15, 25])
```

Compounded return across all out-of-sample folds:
- **US30**: baseline **-7.56%** (36 folds, 41 trades, 9/36 folds positive)
  vs. zone-gated **-3.60%** (36 folds, only 7 trades total, 1/36 folds
  positive). Both lose money once genuinely optimized - a separate
  finding from the zone gate itself: the strategy's earlier
  "fixed-defaults" numbers for US30 (+0.03%/+0.54% in the sections
  above) don't survive real per-fold optimization at all. With roughly
  1 trade per fold, SQN-based grid search on US30 has almost nothing to
  generalize from and appears to be overfitting noise rather than
  finding real structure - worth flagging as its own caveat, independent
  of the SMC gate question.
- **Gold**: baseline **+20.16%** (48 folds, 54 trades, 21/48 folds
  positive, 41.4% avg win rate on traded folds) vs. zone-gated **-1.84%**
  (48 folds, only 18 trades, 6/48 folds positive). The zone gate turns a
  genuinely solid optimized baseline into a loser, while cutting trade
  count by two-thirds.

**Final verdict: the zone gate does not earn a place in production.** It
never improves on its own baseline once both are optimized to the same
standard, and on Gold - the one instrument where the baseline is real
and profitable - it actively destroys the edge. This replaces
"inconclusive" with a real answer; the decision below (off by default)
doesn't change, it's now backed by a rigorous result instead of a quick
fixed-defaults read. `SMCZoneConfluenceStrategy`/
`SMCZoneLiquiditySweepStrategy` stay in the codebase as opt-in,
documented dead ends - kept for the reusable feature code
(`premium_discount_zone()`/`trading_session()`) and as a worked example
of "mine an idea, test it properly, reject it cleanly," not for their
own trading value.

**Decision (2026-07-26, user's call)**: commit the infrastructure -
`premium_discount_zone()`/`trading_session()`/both `_extra_gates_ok()`
hooks/both new SMC-gated classes - since they're real, tested, and
reusable regardless of this specific idea's verdict. Leave the zone
gate **off by default**: `run_scheduled.py` and `live_monitor.py` keep
using `RegimeConfluenceStrategy`, unaffected and unchanged by any of
this. `SMCZoneConfluenceStrategy`/`SMCZoneLiquiditySweepStrategy` exist
as opt-in classes for a future proper walk-forward evaluation, not
production defaults.
