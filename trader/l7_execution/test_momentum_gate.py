"""
test_momentum_gate.py — offline unit test for the 2026-08-01 production
strategy flip (evaluate_momentum_structure_signal() + run_once()'s new
use_momentum_gate branch). Unlike test_run_once*.py (which need a real
MT5 terminal, since they exercise has_open_position/circuit-breaker/live
connection), this needs nothing MT5-related: evaluate_momentum_structure_signal()
and evaluate_regime_confluence_signal() are both pure pandas functions,
and run_once()'s MT5-touching pieces are mocked the same way
test_mobile_api.py already does it - so this runs anywhere, no terminal
needed.

Run:  PYTHONPATH=. python -m trader.l7_execution.test_momentum_gate
"""

from unittest.mock import patch

import numpy as np
import pandas as pd

from trader.l7_execution import evaluate_momentum_structure_signal, run_once

failures = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"{status} {label}")
    if not condition:
        failures.append(label)


# A df shape close enough to build_live_features() output for
# evaluate_regime_confluence_signal() to run without raising - actual
# values don't matter for THIS file's tests, since
# evaluate_regime_confluence_signal() itself is mocked out below to
# isolate the NEW gate logic (same isolation approach
# test_run_once_forced_signal.py already uses for run_once()'s glue).
N = 30
idx = pd.date_range("2026-01-01", periods=N, freq="4h")
dummy_df = pd.DataFrame({
    "Open": np.full(N, 100.0), "High": np.full(N, 101.0), "Low": np.full(N, 99.0),
    "Close": np.full(N, 100.0), "Volume": np.full(N, 1.0),
    "ma_360": np.full(N, 95.0), "ma_200": np.full(N, 95.0), "ma_89": np.full(N, 98.0),
    "ema_21": np.full(N, 99.0), "ema_8": np.full(N, 100.0),
    "macd_hist": np.full(N, 1.0), "atr_14": np.full(N, 1.0), "adx_14": np.full(N, 30.0),
    "er": np.full(N, 0.9),
}, index=idx)

fake_long_signal = {"signal": "long", "price": 100.0, "sl": 98.0, "tp": 104.0}
fake_short_signal = {"signal": "short", "price": 100.0, "sl": 102.0, "tp": 96.0}
fake_no_signal = {"signal": None}


def make_feats(break_up_pattern, break_down_pattern=None):
    """Fake build_momentum_structure_features() output - only the two
    columns evaluate_momentum_structure_signal() actually reads."""
    up = pd.Series(break_up_pattern, index=idx)
    down = pd.Series(break_down_pattern if break_down_pattern is not None else [False] * N, index=idx)
    return pd.DataFrame({"mom_break_up": up, "mom_break_down": down}, index=idx)


# --- Case A: base signal is long, momentum broke up recently -> gate passes, signal unchanged ---
break_up_recent = [False] * (N - 3) + [True, False, False]  # broke up 3 bars before the last bar
with patch("trader.l7_execution.evaluate_regime_confluence_signal", return_value=fake_long_signal), \
     patch("trader.l7_execution.build_momentum_structure_features", return_value=make_feats(break_up_recent)):
    result_a = evaluate_momentum_structure_signal(dummy_df, mom_lead_bars=10)
check("gate passes when momentum broke up within mom_lead_bars -> signal unchanged", result_a == fake_long_signal)

# --- Case B: base signal is long, momentum NEVER broke up -> gate blocks ---
never_broke_up = [False] * N
with patch("trader.l7_execution.evaluate_regime_confluence_signal", return_value=fake_long_signal), \
     patch("trader.l7_execution.build_momentum_structure_features", return_value=make_feats(never_broke_up)):
    result_b = evaluate_momentum_structure_signal(dummy_df, mom_lead_bars=10)
check("gate blocks when momentum never broke up -> {'signal': None}", result_b == {"signal": None})

# --- Case C: base signal is long, momentum broke up but OUTSIDE the lead window -> gate blocks ---
broke_up_too_early = [True] + [False] * (N - 1)  # only at the very first bar, way outside any reasonable lead window
with patch("trader.l7_execution.evaluate_regime_confluence_signal", return_value=fake_long_signal), \
     patch("trader.l7_execution.build_momentum_structure_features", return_value=make_feats(broke_up_too_early)):
    result_c = evaluate_momentum_structure_signal(dummy_df, mom_lead_bars=5)
check("gate blocks when the only break is outside mom_lead_bars", result_c == {"signal": None})

# --- Case D: base signal is short, checks mom_break_down (not mom_break_up) ---
break_down_recent = [False] * (N - 2) + [True, False]
with patch("trader.l7_execution.evaluate_regime_confluence_signal", return_value=fake_short_signal), \
     patch("trader.l7_execution.build_momentum_structure_features",
           return_value=make_feats([False] * N, break_down_recent)):
    result_d = evaluate_momentum_structure_signal(dummy_df, mom_lead_bars=10)
check("short signal checks mom_break_down and passes when it fired recently", result_d == fake_short_signal)

# --- Case E: base signal is already None -> short-circuits, never even computes momentum features ---
with patch("trader.l7_execution.evaluate_regime_confluence_signal", return_value=fake_no_signal), \
     patch("trader.l7_execution.build_momentum_structure_features") as mock_feats:
    result_e = evaluate_momentum_structure_signal(dummy_df)
check("no base signal -> returns None without computing momentum features", result_e == {"signal": None})
check("build_momentum_structure_features() not called when there's no base signal to gate", not mock_feats.called)

# --- run_once() wiring: use_momentum_gate=True calls the momentum function, not the regime one ---
with patch("trader.l7_execution.connect"), \
     patch("trader.l7_execution.has_open_position", return_value=False), \
     patch("trader.l7_execution.LiveCircuitBreaker") as mock_breaker_cls, \
     patch("trader.l7_execution.build_live_features", return_value=dummy_df), \
     patch("trader.l7_execution.evaluate_regime_confluence_signal", return_value=fake_no_signal) as mock_regime, \
     patch("trader.l7_execution.evaluate_momentum_structure_signal", return_value=fake_long_signal) as mock_momentum, \
     patch("trader.l7_execution.place_trade", return_value={"dry_run": True, "would_send": {}}) as mock_place, \
     patch("trader.l7_execution.SYMBOL_MAP", {"US30": "US30"}):
    mock_breaker_cls.return_value.in_cooldown.return_value = False
    result_f = run_once("US30", {}, magic=999005, dry_run=True, use_momentum_gate=True)
check("run_once(use_momentum_gate=True) calls evaluate_momentum_structure_signal", mock_momentum.called)
check("run_once(use_momentum_gate=True) does NOT call evaluate_regime_confluence_signal", not mock_regime.called)
check("run_once(use_momentum_gate=True) trades on the momentum signal", result_f["action"] == "trade")
check("run_once(use_momentum_gate=True) passes the momentum signal through unchanged", result_f["signal"] == fake_long_signal)

# --- run_once() wiring: use_momentum_gate=False (default) still calls the OLD regime function - no regression ---
with patch("trader.l7_execution.connect"), \
     patch("trader.l7_execution.has_open_position", return_value=False), \
     patch("trader.l7_execution.LiveCircuitBreaker") as mock_breaker_cls2, \
     patch("trader.l7_execution.build_live_features", return_value=dummy_df), \
     patch("trader.l7_execution.evaluate_regime_confluence_signal", return_value=fake_long_signal) as mock_regime2, \
     patch("trader.l7_execution.evaluate_momentum_structure_signal", return_value=fake_no_signal) as mock_momentum2, \
     patch("trader.l7_execution.place_trade", return_value={"dry_run": True, "would_send": {}}), \
     patch("trader.l7_execution.SYMBOL_MAP", {"US30": "US30"}):
    mock_breaker_cls2.return_value.in_cooldown.return_value = False
    result_g = run_once("US30", {}, magic=999006, dry_run=True)  # use_momentum_gate defaults to False
check("run_once() default (use_momentum_gate=False) calls the OLD regime function", mock_regime2.called)
check("run_once() default (use_momentum_gate=False) does NOT call the momentum function", not mock_momentum2.called)
check("run_once() default still trades on the regime signal (unchanged behavior)", result_g["action"] == "trade")

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    raise SystemExit(1)
print("ALL PASSED")
