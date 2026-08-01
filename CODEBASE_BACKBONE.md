# Backbone of the US30/Gold trading bot

Purpose of this file: a map, not a manual. One line per file, grouped by
layer, with the underlying skill each layer actually teaches — so you can
pick a piece and go deep on it. Full narrative history (why each decision
was made, every bug fixed, every backtest number) lives in
`ARCHITECTURE.md`; this is the skeleton underneath it.

---

## Layer 1 — Market Data
**Skill: data plumbing.** Getting real price history into a clean, consistent shape.

- `trader/l1_data.py` — loads raw H4 OHLCV bars from MT5 CSV exports (`data/raw/*.csv`), resamples to D1 on request. No live connection needed to run this layer.
- `trader/l1_data_export_h1.py` — one-off script you run against a live MT5 terminal to pull H1 history and save it in the same CSV format.

## Layer 2 — Feature Engineering
**Skill: vectorized technical analysis in pandas.** Every indicator here is a pure function: series in, series out, no lookahead.

- `trader/l2_features.py` — the big one. Moving averages, MACD, ATR, ADX, swing-high/low structure, candle patterns, the SMC premium/discount zone + session tagging, a Kalman trend filter, and the newest addition: the MSA momentum-oscillator/structure-break functions. If you want to understand "how do you turn raw candles into a signal," this file is the whole vocabulary.

## Layer 3 — Regime Recognition
**Skill: statistics, not just indicators.** Answering "is this market actually trending right now" rigorously instead of by eyeballing a chart.

- `trader/l3_regime.py` — Kaufman Efficiency Ratio (net displacement ÷ total path length — the single most-used gate in this whole system), plus ATR expansion, EMA-crossover-count choppiness proxy, MA slope.

## Layer 4 — Signal Model (the strategies themselves)
**Skill: `backtesting.py`'s `Strategy` class — turning rules into code that can be backtested.**

- `trader/l4_signal_model.py` — `ConfluenceStrategy` (the original 5-condition entry rule: macro trend + EMA cross + MACD + breakout, ADX-gated) → `RegimeConfluenceStrategy` (same rules, ER-gated instead — **this is the current production default**) → `SMCZoneConfluenceStrategy` (adds a zone/session filter, tested and rejected) → `MomentumStructureConfluenceStrategy` (adds the MSA momentum-break gate, tested and just proven to beat the baseline on both instruments — decision on production use still pending).
- `trader/l4_liquidity_strategy.py` — a completely different entry model: sweep → displacement → break-of-structure → pullback → engulfing. Good file to read if you want to see a second, structurally different strategy next to the confluence one.

## Layer 5 — Position Sizing
**Skill: risk math.** One function, worth understanding cold.

- `trader/l5_position_sizing.py` — `risk_based_size()`: given entry price, stop price, and a target % of equity to risk, returns how many units to trade. This is what stops "same size every trade" from silently putting wildly different dollar risk on the table depending on stop width.

## Layer 6 — Risk Overlay
**Skill: state machines.** Simple but easy to get subtly wrong.

- `trader/l6_risk.py` — `CircuitBreakerMixin`: after N losing trades in a row, force a cooldown before the next entry. Shared by every Layer 4 strategy via mixin, not copy-pasted.

## Walk-forward validation (cuts across every layer above)
**Skill: the actual discipline of quant trading — proving a strategy isn't overfit before trusting it.**

- `trader/backtest_harness.py` — anchored walk-forward: train window grows, test window rolls forward, re-optimized every fold, only ever judged on out-of-sample results. This is the file that turned "I think this rule works" into "here's what it actually did on data it never saw during tuning." If you master one file for the *trading* side (not the *engineering* side) of this project, make it this one — it's the difference between a real system and a curve-fit story, and it's exactly what the Michael Oliver / SMC-gate / momentum-structure tests all ran through.

## Layer 7 — Execution (the live system)
**Skill: systems engineering — API integration, scheduling, shared state, a backend, a frontend.** This is where "backtest" becomes "actually running."

- `trader/l7_execution/__init__.py` — the MT5 bridge itself: `connect()`, `get_live_bars()`, `place_trade()`, `account_summary()`, `LiveCircuitBreaker` (the live version of Layer 6, reading real deal history instead of backtest trades).
- `trader/l7_execution/run_scheduled.py` — the single-shot script Windows Task Scheduler runs every 4 hours: one `run_once()` pass per instrument, journaled. `dry_run` is hardcoded `True` here — this script can never place a real trade on its own, by design.
- `trader/l7_execution/manual_overrides.py` — the shared live-editable state file (bias, key levels, pause flags, auto/manual mode) both the dashboard and the scheduled script read/write, so a click in one reaches the other without a code edit.
- `trader/l7_execution/journal.py` / `journal_summary.py` — append-only JSONL log of every decision (skip or trade, with reason), plus a script that summarizes it into a weekly review.
- `trader/l7_execution/live_monitor.py` — the desktop dashboard: live candlestick charts with every Layer 2/3 indicator overlaid, P&L panel, clickable bias/pause/key-level controls. `_redraw_column()` is the core drawing function, shared with the mobile chart renderer.
- `trader/l7_execution/mobile_api.py` — Flask backend serving the phone dashboard: status/journal/chart-PNG endpoints, token auth, and the manual order placement gate chain (demo-account-only, multiple safety checks before ever calling `place_trade(dry_run=False)`).
- `trader/l7_execution/mobile_app/` — the actual PWA frontend: `index.html` (all the JS/CSS inline, single-file), `manifest.json`/`service-worker.js` (what makes it installable on a phone).
- `check_cooldown.py`, `debug_deals.py` — small diagnostic scripts for inspecting real MT5 deal history against what the circuit breaker expects.
- `test_*.py` (7 files) — the test suite, all mocking MT5 so they run without a live terminal. Good place to see the system's behavior specified precisely, if you learn better from tests than from prose.

---

## If you're picking ONE place to go deep

- **Want to understand markets/strategy design**: Layer 2 (`l2_features.py`) + Layer 4 (`l4_signal_model.py`) + `backtest_harness.py`. This is the actual trading logic.
- **Want to understand risk**: Layer 5 + Layer 6, both short files, both worth reading end to end in one sitting.
- **Want to understand "how does this become a real running system"**: Layer 7, starting with `__init__.py` (the MT5 bridge) then `run_scheduled.py` (how a cycle actually executes).
- **Want the single highest-leverage skill**: `backtest_harness.py` and the walk-forward methodology itself — it's language-agnostic, transfers to any strategy or platform (including EasyLanguage/C# if you go that route later), and it's the one thing separating this project from a strategy that just *looks* good on a chart.
