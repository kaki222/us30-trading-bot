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
