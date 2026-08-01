"""
run_msa_walkforward.py — full per-fold walk-forward OPTIMIZATION pass for
MomentumStructureConfluenceStrategy vs. its RegimeConfluenceStrategy
baseline, on both US30 and Gold.

This is the test ARCHITECTURE.md's "Momentum structural break gate"
section flagged as NOT run: the sandbox that built the feature couldn't
finish even a single fold with this grid inside its 45-second command
ceiling. Your machine has no such ceiling, so this runs the real thing -
same rigor as the SMC zone-gate section (Backtest.optimize() re-fit on
each fold's training slice, applied out-of-sample on the test slice,
compounded across all folds), not the cheap fixed-params spot checks
already in the doc.

Expect this to take a while - the momentum grid is ~3x the size of the
baseline's own REGIME_OPTIMIZE_KWARGS grid (adds 3 mom_structure_lookback
x 3 mom_lead_bars combinations), and the baseline grid alone was already
documented as "~5-35s per fold, accelerating, ~36 folds for US30 and ~48
for Gold." Could run from several minutes to well over an hour depending
on your machine. It prints progress per fold as it goes so you can see
it's alive, and writes results to reports/ as it finishes each strategy
x instrument combination (not just at the very end), so a Ctrl-C partway
through doesn't lose everything.

Run from the repo root, with your venv active (same one MT5 imports work
in - this only needs pandas/backtesting/numpy, not MetaTrader5 itself,
but running it in the same venv you already have is simplest):

    python run_msa_walkforward.py
"""

import time
import warnings

import pandas as pd

from trader.l2_features import build_bt_df
from trader.backtest_harness import (
    walk_forward, REGIME_OPTIMIZE_KWARGS, MOMENTUM_STRUCTURE_OPTIMIZE_KWARGS,
)
from trader.l4_signal_model import RegimeConfluenceStrategy, MomentumStructureConfluenceStrategy

warnings.filterwarnings("ignore")

RUNS = [
    ("US30", RegimeConfluenceStrategy, REGIME_OPTIMIZE_KWARGS, "reports/walk_forward_US30_regime_baseline_refresh.csv"),
    ("US30", MomentumStructureConfluenceStrategy, MOMENTUM_STRUCTURE_OPTIMIZE_KWARGS, "reports/walk_forward_US30_momentum_structure.csv"),
    ("GOLD", RegimeConfluenceStrategy, REGIME_OPTIMIZE_KWARGS, "reports/walk_forward_GOLD_regime_baseline_refresh.csv"),
    ("GOLD", MomentumStructureConfluenceStrategy, MOMENTUM_STRUCTURE_OPTIMIZE_KWARGS, "reports/walk_forward_GOLD_momentum_structure.csv"),
]


def summarize(folds):
    total_ret = 1.0
    wins = 0
    total_trades = 0
    for f in folds:
        total_ret *= (1 + f["return_pct"] / 100)
        if f["return_pct"] > 0:
            wins += 1
        total_trades += f["num_trades"]
    return {
        "folds": len(folds),
        "compounded_return_pct": round((total_ret - 1) * 100, 2),
        "positive_folds": wins,
        "total_trades": total_trades,
    }


def main():
    summary_rows = []
    for symbol, cls, grid, out_csv in RUNS:
        print(f"\n=== {symbol}  {cls.__name__}  ({len(grid) - 2} tunable params, "
              f"grid size varies per fold's constraint-filtered combos) ===")
        df = build_bt_df(symbol)
        t0 = time.time()
        folds = walk_forward(df, strategy_cls=cls, optimize_kwargs=grid, verbose=True)
        elapsed = time.time() - t0
        pd.DataFrame(folds).to_csv(out_csv, index=False)
        s = summarize(folds)
        s.update(symbol=symbol, strategy=cls.__name__, elapsed_sec=round(elapsed, 1))
        summary_rows.append(s)
        print(f"--- {symbol} {cls.__name__}: {s['compounded_return_pct']}% compounded, "
              f"{s['positive_folds']}/{s['folds']} positive folds, "
              f"{s['total_trades']} trades, {round(elapsed,1)}s. Saved -> {out_csv}")

    print("\n\n=== FINAL COMPARISON ===")
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("reports/msa_walkforward_summary.csv", index=False)
    print("\nSaved summary -> reports/msa_walkforward_summary.csv")


if __name__ == "__main__":
    main()
