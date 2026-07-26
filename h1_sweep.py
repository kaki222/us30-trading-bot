"""
H1 walk-forward, same small-grid optimize as the D1 pass, for a fair
H1 vs H4 vs D1 comparison.
"""
import sys
from trader.l2_features import build_bt_df
from trader.l4_signal_model import RegimeConfluenceStrategy
from trader.backtest_harness import walk_forward

SMALL_GRID = dict(
    er_threshold=[0.25, 0.45],
    swing_lookback=[10, 30],
    atr_sl_mult=[1.5, 2.5],
    atr_tp_mult=[2.0, 3.5],
    maximize="SQN",
    constraint=lambda p: p.atr_tp_mult > p.atr_sl_mult,
)


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


symbol = sys.argv[1] if len(sys.argv) > 1 else "US30"
df = build_bt_df(symbol, timeframe="H1")
print(f"{symbol} H1 bars: {len(df)}", flush=True)
import time
t0 = time.time()
results = walk_forward(df, strategy_cls=RegimeConfluenceStrategy,
                        optimize_kwargs=None, verbose=False)
print(f"took {time.time()-t0:.1f}s", flush=True)
summarize(f"{symbol} H1 RegimeConfluence (fixed defaults, no optimize)", results)
