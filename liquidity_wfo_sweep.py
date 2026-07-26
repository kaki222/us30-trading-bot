"""
One-off script: full walk-forward OPTIMIZATION pass (real per-fold grid
search, not fixed defaults) for LiquiditySweepStrategy and its SMC
zone-gated variant, both instruments - the pass ARCHITECTURE.md's "SMC
zone/session gate" section reports the results of. Not part of the
package - scratch script, run once, read the output. Takes a few
minutes per instrument (grid search re-fit every fold).
"""
from trader.l2_features import build_bt_df, build_liquidity_features
from trader.l4_liquidity_strategy import LiquiditySweepStrategy, SMCZoneLiquiditySweepStrategy
from trader.backtest_harness import walk_forward, LIQUIDITY_OPTIMIZE_KWARGS, SMC_ZONE_LIQUIDITY_OPTIMIZE_KWARGS


def compounded_return(fold_results):
    cash = 100_000.0
    for f in fold_results:
        cash *= (1 + f["return_pct"] / 100)
    return (cash / 100_000.0 - 1) * 100


def summarize(name, fold_results):
    total_trades = sum(f["num_trades"] for f in fold_results)
    positive_folds = sum(1 for f in fold_results if f["return_pct"] > 0)
    n = len(fold_results)
    comp = compounded_return(fold_results)
    print(f"{name}: {n} folds, compounded return {comp:+.2f}%, {total_trades} trades, "
          f"{positive_folds}/{n} folds positive ({100*positive_folds/n:.0f}%)")


print("=== Walk-forward OPTIMIZATION pass: LiquiditySweepStrategy vs SMCZoneLiquiditySweepStrategy ===")
for symbol in ["US30", "GOLD"]:
    df = build_bt_df(symbol, timeframe="H4")
    df = build_liquidity_features(df)

    base_results = walk_forward(df, strategy_cls=LiquiditySweepStrategy,
                                 optimize_kwargs=LIQUIDITY_OPTIMIZE_KWARGS, verbose=False)
    summarize(f"{symbol} LiquiditySweep (optimized)", base_results)

    zone_results = walk_forward(df, strategy_cls=SMCZoneLiquiditySweepStrategy,
                                 optimize_kwargs=SMC_ZONE_LIQUIDITY_OPTIMIZE_KWARGS, verbose=False)
    summarize(f"{symbol} SMCZoneLiquiditySweep (optimized)", zone_results)

print("\nDone. See ARCHITECTURE.md 'SMC zone/session gate' section for the discussion of these numbers.")
