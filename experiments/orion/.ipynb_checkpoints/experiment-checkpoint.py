"""Load orion data, train classifiers with fixed hyperparameters, evaluate on the test split.

Rows where neither take-profit nor stop-loss would trade within the forward window (same rule as
``add_range_target_column`` in ``lib.common.common``) are dropped before the chronological split;
only those ``TP``/``SL``-reachable setups are used for train, validation, and test.

Best params below match a representative grid-search run (MARA, config in ``config.py``).
Update the dict after re-tuning if you want this script to mirror new optima.

Run from project root::

    python -m experiments.orion.experiment
"""

from typing import Any

import numpy as np
import pandas as pd

from lib.common.common import _trade_date_series, evaluate_and_print
from lib.models import (
    train_adaboost,
    train_decision_tree,
    train_forest,
    train_knn,
    train_naive_bayes,
    train_xgboost,
)

from experiments.orion.config import DEFAULT_ORION_CONFIG, OrionExperimentConfig
from experiments.orion.elib import load_orion_training_frames, print_orion_run_preamble, pull_and_clean
from experiments.orion.returns import gross_returns_entry_next_high_tp_sl_horizon

# From grid search console (val F1); keys match each ``train_*`` keyword arguments.
_OPTIMAL: dict[str, dict[str, Any]] = {
    "decision_tree": {"max_depth": 8, "min_samples_leaf": 100},
    "naive_bayes": {"var_smoothing": 1e-9},
    "knn": {"n_neighbors": 5, "p": 2, "weights": "uniform"},
    "random_forest": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 100},
    "adaboost": {"n_estimators": 50, "learning_rate": 0.5},
    "xgboost": {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 150,
        "subsample": 1.0,
    },
}


def _effective_max_bars_after_entry(bars: pd.DataFrame, configured: int | None) -> int:
    if configured is not None:
        return max(1, int(configured))
    trade_date = _trade_date_series(bars)
    symbol = bars.index.get_level_values("symbol")
    longest = 0
    for _, g in bars.groupby([symbol, trade_date], sort=False):
        longest = max(longest, len(g))
    return max(longest, 1)


def print_real_profit(
    *,
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    cfg: OrionExperimentConfig,
    model_name: str = "Model",
    short_symbol: str = "SQQQ",
) -> None:
    """Simulate PnL on test rows: pred 1 -> long primary symbol; pred 0 -> long ``short_symbol``.

    Per row, entry is the **next** bar's high and exit follows TP / SL / horizon low, matching
    ``gross_returns_entry_next_high_tp_sl_horizon`` (and the same bar priority as
    ``create_target_column``: stop before take-profit). Round-trip fees: ``2 * cfg.trade_cost``
    per trade.
    """
    long_bars = pull_and_clean(cfg.symbol, cfg.start_date, cfg.end_date).sort_index()
    max_bars = _effective_max_bars_after_entry(long_bars, cfg.target_max_bars_after_entry)
    short_bars = pull_and_clean(short_symbol, cfg.start_date, cfg.end_date).sort_index()

    gross_long, _ = gross_returns_entry_next_high_tp_sl_horizon(
        long_bars,
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        max_bars_after_entry=max_bars,
    )
    gross_short, _ = gross_returns_entry_next_high_tp_sl_horizon(
        short_bars,
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        max_bars_after_entry=max_bars,
    )

    idx = test_df.index
    ts = idx.get_level_values("timestamp")
    short_idx = pd.MultiIndex.from_arrays(
        [np.full(len(ts), short_symbol, dtype=object), ts],
        names=idx.names,
    )

    aligned_long = gross_long.reindex(idx)
    aligned_short = gross_short.reindex(short_idx)

    pred = np.asarray(y_pred).astype(np.int64, copy=False)
    mask_long = pred == 1
    mask_short = pred == 0
    raw = np.full(len(pred), np.nan, dtype=np.float64)
    raw[mask_long] = aligned_long.to_numpy(dtype=np.float64, copy=False)[mask_long]
    raw[mask_short] = aligned_short.to_numpy(dtype=np.float64, copy=False)[mask_short]

    valid = np.isfinite(raw)
    cost_rt = float(cfg.trade_cost)
    net = raw[valid] - cost_rt

    n_buy_long = int(mask_long.sum())
    n_buy_short = int(mask_short.sum())
    n_valid_long = int((mask_long & valid).sum())
    n_valid_short = int((mask_short & valid).sum())

    print(f"\n--- {model_name} real profit (next-bar-high entry, TP / SL / horizon) ---")
    print(
        f"Bought {cfg.symbol}: {n_buy_long:,} "
        f"({n_valid_long:,} with valid OHLC path) | "
        f"Bought {short_symbol}: {n_buy_short:,} "
        f"({n_valid_short:,} with valid OHLC path)"
    )
    print(f"Test rows: {len(pred):,} | total valid OHLC paths: {int(valid.sum()):,}")
    if not valid.any():
        return
    print(f"Sum gross simple return: {100.0 * float(np.nansum(raw)):.4f}%")
    print(f"Sum net simple return (after {cost_rt:.4f} round-trip cost): {100.0 * float(net.sum()):.4f}%")
    print(f"Mean net per trade (%): {100.0 * float(net.mean()):.6f}")


def main() -> None:
    cfg = DEFAULT_ORION_CONFIG
    trade_cost = cfg.trade_cost

    print_orion_run_preamble(cfg, trade_cost=trade_cost, run_title="Experiment (fixed params)")

    train_df, val_df, test_df = load_orion_training_frames(
        symbol=cfg.symbol,
        reference_symbols=cfg.reference_symbols,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        validation_fraction=cfg.validation_fraction,
        test_fraction=cfg.test_fraction,
        target_max_bars_after_entry=cfg.target_max_bars_after_entry,
        only_rows_hitting_tp_or_sl=True,
    )

    x_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]
    p = _OPTIMAL

    # tree_clf = train_decision_tree(
    #     train_df, val_df, param_grid=None, verbose=False, **p["decision_tree"]
    # )
    # evaluate_and_print(
    #     "Decision Tree",
    #     y_test,
    #     tree_clf.predict(x_test),
    #     take_profit=cfg.take_profit,
    #     stop_loss=cfg.stop_loss,
    #     cost=trade_cost,
    # )

    # nb_clf = train_naive_bayes(train_df, val_df, param_grid=None, verbose=False, **p["naive_bayes"])
    # evaluate_and_print(
    #     "Naive Bayes",
    #     y_test,
    #     nb_clf.predict(x_test),
    #     take_profit=cfg.take_profit,
    #     stop_loss=cfg.stop_loss,
    #     cost=trade_cost,
    # )

    # knn_clf = train_knn(train_df, val_df, param_grid=None, verbose=False, **p["knn"])
    # evaluate_and_print(
    #     "K-Nearest Neighbors",
    #     y_test,
    #     knn_clf.predict(x_test),
    #     take_profit=cfg.take_profit,
    #     stop_loss=cfg.stop_loss,
    #     cost=trade_cost,
    # )

    forest_clf = train_forest(train_df, val_df, param_grid=None, verbose=False, **p["random_forest"])
    evaluate_and_print(
        "Random Forest",
        y_test,
        forest_clf.predict(x_test),
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        cost=trade_cost,
    )
    print_real_profit(
        test_df=test_df,
        y_pred=forest_clf.predict(x_test),
        cfg=cfg,
        model_name="Random Forest",
    )

    # adaboost_clf = train_adaboost(
    #     train_df, val_df, param_grid=None, verbose=False, **p["adaboost"]
    # )
    # evaluate_and_print(
    #     "AdaBoost",
    #     y_test,
    #     adaboost_clf.predict(x_test),
    #     take_profit=cfg.take_profit,
    #     stop_loss=cfg.stop_loss,
    #     cost=trade_cost,
    # )

    # xgb_clf = train_xgboost(train_df, val_df, param_grid=None, verbose=False, **p["xgboost"])
    # evaluate_and_print(
    #     "XGBoost",
    #     y_test,
    #     xgb_clf.predict(x_test),
    #     take_profit=cfg.take_profit,
    #     stop_loss=cfg.stop_loss,
    #     cost=trade_cost,
    # )


if __name__ == "__main__":
    main()
