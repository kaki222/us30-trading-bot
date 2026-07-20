"""
backtest_harness.py — walk-forward validation harness

Anchored walk-forward: train window grows from a fixed start, test
window rolls forward. Reused across every instrument and every
signal-model version so results stay comparable.
"""

import pandas as pd
from backtesting import Backtest

from .l4_signal_model import ConfluenceStrategy


def run_fold(df, train_start, train_end, test_end, cash=25_000, verbose=True):
    if verbose:
        print(f"Fold: train [{train_start.date()} -> {train_end.date()}]  test [{train_end.date()} -> {test_end.date()}]")

    train_slice = df.loc[train_start:train_end]
    test_slice = df.loc[train_end:test_end]

    bt_train = Backtest(train_slice, ConfluenceStrategy, cash=cash, commission=0.0002, margin=1/30)
    opt_stats = bt_train.optimize(
        adx_threshold=range(15, 31, 5),
        swing_lookback=range(10, 41, 10),
        atr_sl_mult=[1.5, 2.0, 2.5],   # 1.0 dropped - confirmed it whipsaws in real trends
        atr_tp_mult=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        maximize="SQN",
        constraint=lambda p: p.atr_tp_mult > p.atr_sl_mult,
    )
    params = {k: getattr(opt_stats._strategy, k) for k in ["adx_threshold", "swing_lookback", "atr_sl_mult", "atr_tp_mult"]}

    bt_test = Backtest(test_slice, ConfluenceStrategy, cash=cash, commission=0.0002, margin=1/30)
    test_stats = bt_test.run(**params)

    if verbose:
        print(f"  params={params}  Return={test_stats['Return [%]']:.2f}%  WinRate={test_stats['Win Rate [%]']:.1f}%  Trades={test_stats['# Trades']}")

    return params, test_stats


def walk_forward(df, cash=25_000, initial_train_months=12, test_months=3, verbose=True):
    data_start = df.index[0]
    data_end = df.index[-1]
    train_end = data_start + pd.DateOffset(months=initial_train_months)
    fold_results = []
    while train_end + pd.DateOffset(months=test_months) <= data_end:
        test_end = train_end + pd.DateOffset(months=test_months)
        params, test_stats = run_fold(df, data_start, train_end, test_end, cash=cash, verbose=verbose)
        fold_results.append({
            "test_end": test_end,
            "return_pct": test_stats["Return [%]"],
            "win_rate": test_stats["Win Rate [%]"],
            "num_trades": test_stats["# Trades"],
        })
        train_end = test_end
    return fold_results