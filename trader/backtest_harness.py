"""
backtest_harness.py — walk-forward validation harness

Anchored walk-forward: train window grows from a fixed start, test
window rolls forward. Reused across every instrument and every
signal-model version so results stay comparable.
"""

import pandas as pd
from backtesting import Backtest

from .l4_signal_model import ConfluenceStrategy, RegimeConfluenceStrategy

# ADX-gated grid. Kept for explicit opt-in / comparison runs
# (pass strategy_cls=ConfluenceStrategy, optimize_kwargs=ADX_OPTIMIZE_KWARGS).
# No longer the default: walk-forward showed the ER gate beating it on
# both US30 (every metric) and Gold (higher return, worse drawdown) - see
# reports/*_adx_vs_er_comparison.png.
ADX_OPTIMIZE_KWARGS = dict(
    adx_threshold=range(15, 31, 5),
    swing_lookback=range(10, 41, 10),
    atr_sl_mult=[1.5, 2.0, 2.5],
    atr_tp_mult=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    maximize="SQN",
    constraint=lambda p: p.atr_tp_mult > p.atr_sl_mult,
)
DEFAULT_OPTIMIZE_KWARGS = ADX_OPTIMIZE_KWARGS  # back-compat alias, don't use for new code

# Same grid as ADX_OPTIMIZE_KWARGS but with adx_threshold swapped for
# er_threshold, for use with RegimeConfluenceStrategy - this is now the
# default (see run_fold/walk_forward below).
REGIME_OPTIMIZE_KWARGS = dict(
    er_threshold=[0.25, 0.35, 0.45, 0.55],
    swing_lookback=range(10, 41, 10),
    atr_sl_mult=[1.5, 2.0, 2.5],
    atr_tp_mult=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    maximize="SQN",
    constraint=lambda p: p.atr_tp_mult > p.atr_sl_mult,
)


def run_fold(df, train_start, train_end, test_end, cash=25_000, verbose=True,
             strategy_cls=RegimeConfluenceStrategy, optimize_kwargs=REGIME_OPTIMIZE_KWARGS):
    if verbose:
        print(f"Fold: train [{train_start.date()} -> {train_end.date()}]  test [{train_end.date()} -> {test_end.date()}]")

    test_slice = df.loc[train_end:test_end]

    if optimize_kwargs is None:
        # No per-fold search - just the strategy's own fixed defaults, out-of-sample.
        bt_test = Backtest(test_slice, strategy_cls, cash=cash, commission=0.0002, margin=1/30, finalize_trades=True)
        test_stats = bt_test.run()
        params = {}
    else:
        train_slice = df.loc[train_start:train_end]
        bt_train = Backtest(train_slice, strategy_cls, cash=cash, commission=0.0002, margin=1/30, finalize_trades=True)
        opt_stats = bt_train.optimize(**optimize_kwargs)
        param_names = [k for k in optimize_kwargs if k not in ("maximize", "constraint")]
        params = {k: getattr(opt_stats._strategy, k) for k in param_names}

        bt_test = Backtest(test_slice, strategy_cls, cash=cash, commission=0.0002, margin=1/30, finalize_trades=True)
        test_stats = bt_test.run(**params)

    if verbose:
        print(f"  params={params}  Return={test_stats['Return [%]']:.2f}%  WinRate={test_stats['Win Rate [%]']:.1f}%  Trades={test_stats['# Trades']}")

    return params, test_stats


def walk_forward(df, cash=25_000, initial_train_months=12, test_months=3, verbose=True,
                  strategy_cls=RegimeConfluenceStrategy, optimize_kwargs=REGIME_OPTIMIZE_KWARGS):
    data_start = df.index[0]
    data_end = df.index[-1]
    train_end = data_start + pd.DateOffset(months=initial_train_months)
    fold_results = []
    while train_end + pd.DateOffset(months=test_months) <= data_end:
        test_end = train_end + pd.DateOffset(months=test_months)
        params, test_stats = run_fold(
            df, data_start, train_end, test_end, cash=cash, verbose=verbose,
            strategy_cls=strategy_cls, optimize_kwargs=optimize_kwargs,
        )
        fold_results.append({
            "test_end": test_end,
            "return_pct": test_stats["Return [%]"],
            "win_rate": test_stats["Win Rate [%]"],
            "num_trades": test_stats["# Trades"],
        })
        train_end = test_end
    return fold_results