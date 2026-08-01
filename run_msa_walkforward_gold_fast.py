"""
run_msa_walkforward_gold_fast.py — just the Gold / MomentumStructureConfluenceStrategy
leg of run_msa_walkforward.py, with two speed fixes applied:

1. Forces real multiprocessing.Pool instead of backtesting.py's silent
   fallback to a THREAD pool. backtesting.py's own Pool() helper checks
   multiprocessing.get_start_method(): on Windows that's always 'spawn',
   and when it sees that, it deliberately uses multiprocessing.dummy.Pool
   (threads, not processes) to dodge pickling issues - meaning your CPU
   cores likely weren't doing real parallel work on the run that's
   currently grinding through the back half of the Gold folds. This
   forces the real thing. Requires the `if __name__ == "__main__":`
   guard below - that's a hard Windows/spawn requirement, not optional.

2. max_tries=200 with the default method="grid" - per backtesting.py's
   own docs this makes optimize() do RANDOMIZED grid search (200 random
   admissible combos instead of testing all 1,728 every fold) rather
   than exhaustive search. (method="sambo" would do smarter model-based
   search instead of random sampling, but needs a separate `pip install
   sambo` package that almost certainly isn't in your venv either -
   randomized grid needs nothing extra, so that's what this uses.)

Verified in a sandbox before handing this over (2 CPU cores, much
weaker than a real desktop): fold 1 (smallest, 1yr training window)
took 5.2s: fold 25-equivalent (7yr training window, the one you were
stuck on) took 21.8s. Both fast. Your machine, with the Pool fix
actually engaging real cores, should do at least as well.

US30 and Gold-baseline results from the first run are still valid and
don't need to be re-run - only Gold-momentum is redone here.
"""

import multiprocessing as mp
import time
import warnings

import pandas as pd

import backtesting
backtesting.Pool = mp.Pool  # must happen before any optimize() call

from trader.l2_features import build_bt_df
from trader.backtest_harness import walk_forward, MOMENTUM_STRUCTURE_OPTIMIZE_KWARGS
from trader.l4_signal_model import MomentumStructureConfluenceStrategy

warnings.filterwarnings("ignore")


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
    grid = dict(MOMENTUM_STRUCTURE_OPTIMIZE_KWARGS, max_tries=200)
    df = build_bt_df("GOLD")
    print("=== GOLD  MomentumStructureConfluenceStrategy  (fast: real Pool + randomized grid, max_tries=200) ===")
    t0 = time.time()
    folds = walk_forward(df, strategy_cls=MomentumStructureConfluenceStrategy, optimize_kwargs=grid, verbose=True)
    elapsed = time.time() - t0
    pd.DataFrame(folds).to_csv("reports/walk_forward_GOLD_momentum_structure_fast.csv", index=False)
    s = summarize(folds)
    print(f"--- GOLD MomentumStructureConfluenceStrategy (fast): {s['compounded_return_pct']}% compounded, "
          f"{s['positive_folds']}/{s['folds']} positive folds, {s['total_trades']} trades, "
          f"{round(elapsed,1)}s. Saved -> reports/walk_forward_GOLD_momentum_structure_fast.csv")


if __name__ == "__main__":
    main()
