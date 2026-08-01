"""
run_msa_walkforward_gold_exhaustive.py — the exhaustive (not randomized)
Gold / MomentumStructureConfluenceStrategy walk-forward pass, closing the
one methodology gap flagged in ARCHITECTURE.md: run_msa_walkforward_gold_fast.py
used max_tries=200 (randomized sampling of the 1,728 admissible combos/fold)
for speed; this tests all 1,728 every fold, same standard of rigor as the
US30 momentum run and both baseline runs.

Estimated ~2h10m based on your machine's own measured throughput from the
randomized run (910.8s for ~200 combos/fold x 48 folds; exhaustive is
1,728/fold x 48 = 8.64x more backtest.run() calls). Real multiprocessing
already confirmed working (you saw the 14 python.exe processes) - this
just runs longer because it's testing more combinations, not because
anything is broken.

Saves to a NEW file (doesn't overwrite the randomized result) so both are
on record. Only run this if you want to fully trust the Gold number
before deciding whether to make MomentumStructureConfluenceStrategy the
production default - the randomized 49.30% is a real result either way,
just not the same rigor standard as the other three numbers in the table.
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
    df = build_bt_df("GOLD")
    print("=== GOLD  MomentumStructureConfluenceStrategy  (exhaustive: real Pool, all 1,728 combos/fold) ===")
    t0 = time.time()
    folds = walk_forward(
        df, strategy_cls=MomentumStructureConfluenceStrategy,
        optimize_kwargs=MOMENTUM_STRUCTURE_OPTIMIZE_KWARGS, verbose=True,
    )
    elapsed = time.time() - t0
    pd.DataFrame(folds).to_csv("reports/walk_forward_GOLD_momentum_structure_exhaustive.csv", index=False)
    s = summarize(folds)
    print(f"--- GOLD MomentumStructureConfluenceStrategy (exhaustive): {s['compounded_return_pct']}% compounded, "
          f"{s['positive_folds']}/{s['folds']} positive folds, {s['total_trades']} trades, "
          f"{round(elapsed,1)}s. Saved -> reports/walk_forward_GOLD_momentum_structure_exhaustive.csv")


if __name__ == "__main__":
    main()
