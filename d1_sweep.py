"""
One-off script: walk-forward both strategies on both instruments at D1,
to compare against the existing H4 baselines. Not part of the package -
scratch script, run once, read the output.
"""
import pandas as pd
from trader.l2_features import build_bt_df
from trader.l4_signal_model import RegimeConfluenceStrategy
from trader.l4_liquidity_strategy import LiquiditySweepStrategy
from trader.backtest_harness import walk_forward, REGIME_OPTIMIZE_KWARGS
from trader.l2_features import build_liquidity_features


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


print("=== D1 walk-forward: RegimeConfluenceStrategy (fixed defaults, no per-fold optimize - fast first pass) ===")
for symbol in ["US30", "GOLD"]:
    df = build_bt_df(symbol, timeframe="D1")
    results = walk_forward(df, strategy_cls=RegimeConfluenceStrategy,
                            optimize_kwargs=None, verbose=False)
    summarize(f"{symbol} D1 RegimeConfluence", results)

print("\n=== D1 walk-forward: LiquiditySweepStrategy (fixed defaults, no per-fold optimize) ===")
for symbol in ["US30", "GOLD"]:
    df = build_bt_df(symbol, timeframe="D1")
    df = build_liquidity_features(df)
    results = walk_forward(df, strategy_cls=LiquiditySweepStrategy,
                            optimize_kwargs=None, verbose=False)
    summarize(f"{symbol} D1 LiquiditySweep", results)

print("\nDone.")
