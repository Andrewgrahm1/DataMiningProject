"""All-in identification — same training setup as :mod:`experiments.orion2.chgeIdnt` with selective PnL.

Uses **TQQQ** minute bars for change-ID features and target (forward absolute move). Training,
validation split, downsampling, z-score, and model grids match ``chgeIdnt``.

**Trading rule:** ``pred == 1`` → long **SQQQ** only; ``pred == 0`` → **no position** (flat).
``trade_gross_SQQQ`` (and stored ``trade_gross_TQQQ`` for reference) use next-bar-low entry and
TP/SL/horizon exit (see :func:`gross_returns_entry_next_low_tp_sl_horizon`).

Run from project root::

    python -m experiments.orion2.allIdnt
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

from experiments.orion2.chgeIdnt import (
    TARGET_COLUMN,
    add_features,
    consensus_unanimous_label,
    downsample_negatives_to_match_positives,
    filter_training_by_session_bounds,
    print_consensus_results,
)
from experiments.orion2.elib.elib import (
    print_training_data_stats,
    pull_and_clean,
    split_training_data,
    zscore_feature_splits,
)
from lib.common.common import _index_position, _trade_date_series
from lib.models import (
    train_decision_tree,
    train_forest,
    train_knn,
    train_naive_bayes,
    train_neural_network,
    train_xgboost,
)

TRADE_GROSS_LONG_SYMBOL_COL = "trade_gross_TQQQ"
TRADE_GROSS_SHORT_LEG_COL = "trade_gross_SQQQ"

COLUMNS_EXCLUDED_FROM_ZSCORE: tuple[str, ...] = (
    TARGET_COLUMN,
    TRADE_GROSS_LONG_SYMBOL_COL,
    TRADE_GROSS_SHORT_LEG_COL,
)
COLUMNS_DROP_FOR_FIT: tuple[str, ...] = (
    TRADE_GROSS_LONG_SYMBOL_COL,
    TRADE_GROSS_SHORT_LEG_COL,
)


def gross_returns_entry_next_low_tp_sl_horizon(
    bars: pd.DataFrame,
    *,
    take_profit: float,
    stop_loss: float,
    max_bars_after_entry: int,
) -> pd.Series:
    """Per signal bar *i*: buy at low of bar *i+1*; scan *i+2*… for stop then TP; else exit at last bar low."""
    if max_bars_after_entry < 1:
        raise ValueError("max_bars_after_entry must be >= 1")

    out = np.full(len(bars), np.nan, dtype=np.float64)
    trade_date = _trade_date_series(bars)
    symbol = bars.index.get_level_values("symbol")
    tp = float(take_profit)
    sl = float(stop_loss)

    for (_sym, _date), group in bars.groupby([symbol, trade_date], sort=False):
        group = group.sort_index(level="timestamp")
        locs = group.index
        n = len(group)
        base = _index_position(bars, locs[0])
        highs = group["high"].to_numpy(dtype=np.float64, copy=False)
        lows = group["low"].to_numpy(dtype=np.float64, copy=False)

        num_signals = max(0, n - 3)
        for i in range(num_signals):
            entry_price = lows[i + 1]
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

            sl_price = entry_price * (1.0 - sl)
            tp_price = entry_price * (1.0 + tp)
            end_idx = min(n, i + 2 + max_bars_after_entry)

            exit_price: float | None = None
            for j in range(i + 2, end_idx):
                lo = lows[j]
                hi = highs[j]
                if lo <= sl_price:
                    exit_price = sl_price
                    break
                if hi >= tp_price:
                    exit_price = tp_price
                    break
            if exit_price is None:
                if end_idx - 1 < i + 2:
                    continue
                exit_price = float(lows[end_idx - 1])

            out[base + i] = exit_price / entry_price - 1.0

    return pd.Series(out, index=bars.index, name="gross_ret_tp_sl_next_low")


def _effective_max_bars_after_entry(bars: pd.DataFrame, configured: int | None) -> int:
    if configured is not None:
        return max(1, int(configured))
    trade_date = _trade_date_series(bars)
    symbol = bars.index.get_level_values("symbol")
    longest = 0
    for _, g in bars.groupby([symbol, trade_date], sort=False):
        longest = max(longest, len(g))
    return max(longest, 1)


def add_trade_gross_columns(
    training_df: pd.DataFrame,
    *,
    long_bars: pd.DataFrame,
    short_bars: pd.DataFrame,
    short_symbol: str,
    take_profit: float,
    stop_loss: float,
    max_bars_after_entry: int,
) -> pd.DataFrame:
    """Attach TQQQ and SQQQ gross-return series aligned to ``training_df`` index (TQQQ timestamps)."""
    long_bars = long_bars.sort_index()
    short_bars = short_bars.sort_index()
    gross_long = gross_returns_entry_next_low_tp_sl_horizon(
        long_bars,
        take_profit=take_profit,
        stop_loss=stop_loss,
        max_bars_after_entry=max_bars_after_entry,
    )
    gross_short = gross_returns_entry_next_low_tp_sl_horizon(
        short_bars,
        take_profit=take_profit,
        stop_loss=stop_loss,
        max_bars_after_entry=max_bars_after_entry,
    )

    idx = training_df.index
    ts = idx.get_level_values("timestamp")
    short_idx = pd.MultiIndex.from_arrays(
        [np.full(len(ts), short_symbol, dtype=object), ts],
        names=idx.names,
    )

    out = training_df.copy()
    out[TRADE_GROSS_LONG_SYMBOL_COL] = gross_long.reindex(idx).to_numpy(dtype=np.float64, copy=False)
    out[TRADE_GROSS_SHORT_LEG_COL] = gross_short.reindex(short_idx).to_numpy(
        dtype=np.float64, copy=False
    )
    return out


def print_preamble(
    *,
    training_df: pd.DataFrame,
    feature_symbol: str,
    short_symbol: str,
    move_pct_threshold: float,
    max_forward_bars: int,
    take_profit: float,
    stop_loss: float,
    trade_cost: float,
) -> None:
    from lib.common.common import calculate_min_win_rate

    print("================== All identification (change-ID + TQQQ/SQQQ legs) ===============")
    print(f"Features / target symbol: {feature_symbol}")
    print(f"pred 1 → long {short_symbol} | pred 0 → no trade")
    ts_idx = training_df.index.get_level_values("timestamp")
    print(f"Date range: {ts_idx.min().date()} to {ts_idx.max().date()}")
    print(f"Rows: {len(training_df):,}")
    print(f"Move threshold: {move_pct_threshold * 100:.4f}% | Forward bars: {max_forward_bars}")
    print(f"PnL TP/SL (per-leg sim): {take_profit * 100:.4f}% / {stop_loss * 100:.4f}%")
    print(
        f"Trade cost per side: {trade_cost * 100:.4f}% "
        f"(PnL deducts {2 * trade_cost * 100:.4f}% round-trip per position)"
    )
    print("========================================================")


def _print_predicted_leg_net_return_pct(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    *,
    trade_cost: float,
) -> None:
    """Only ``pred == 1`` opens SQQQ; other predictions are flat (no return, no fees)."""
    gs = test_df[TRADE_GROSS_SHORT_LEG_COL].to_numpy(dtype=np.float64, copy=False)
    pred = np.asarray(y_pred).astype(np.int64, copy=False)
    raw = np.full(len(pred), np.nan, dtype=np.float64)
    raw[pred == 1] = gs[pred == 1]

    valid = np.isfinite(raw)
    n_positions = int(valid.sum())
    if n_positions == 0:
        print("Return: n/a")
        return
    gross_sum = float(np.nansum(raw[valid]))
    round_trip_per_position = 2.0 * float(trade_cost)
    total_pct = 100.0 * (gross_sum - round_trip_per_position * n_positions)
    print(f"Return: {total_pct:.4f}%")


def print_test_results(
    name: str,
    y_test: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    *,
    test_df: pd.DataFrame,
    trade_cost: float,
) -> None:
    y_true = np.asarray(y_test).ravel()
    y_hat = np.asarray(y_pred).ravel()
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_hat, labels=labels)
    acc = float(accuracy_score(y_true, y_hat))
    recall = float(recall_score(y_true, y_hat, zero_division=0))
    precision = float(precision_score(y_true, y_hat, zero_division=0))

    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])
    n = tn + fp + fn + tp
    denom_f1 = 2 * tp + fp + fn
    f1 = (2 * tp / denom_f1) if denom_f1 else 0.0

    print(f"\n--- {name} ---")
    print(f"Accuracy: {(tn + tp):,} / {n:,} ({100.0 * acc:.4f}%)")
    print(f"Recall: {tp:,} / {tp + fn:,} ({100.0 * recall:.4f}%)")
    print(f"Precision: {tp:,} / {tp + fp:,} ({100.0 * precision:.4f}%)")
    print(f"F1 score: {f1:.6f}")
    try:
        print(f"ROC AUC: {roc_auc_score(y_true, y_hat):.6f}")
    except ValueError:
        print("ROC AUC: n/a (constant predictions)")
    _print_predicted_leg_net_return_pct(test_df, y_pred, trade_cost=trade_cost)


if __name__ == "__main__":
    LONG_SYMBOL = "TQQQ"
    SHORT_SYMBOL = "SQQQ"
    START_DATE = datetime(2022, 1, 1)
    END_DATE = datetime(2025, 12, 31)
    MOVE_PCT_THRESHOLD = 0.02
    MAX_FORWARD_BARS = 90
    TAKE_PROFIT = MOVE_PCT_THRESHOLD
    STOP_LOSS = MOVE_PCT_THRESHOLD
    TRADE_COST = 0.002
    VALIDATION_FRACTION = 0.15
    TEST_FRACTION = 0.2
    MIN_BARS_SINCE_OPEN = 30
    MAX_BARS_UNTIL_CLOSE = 60

    long_bars = pull_and_clean(LONG_SYMBOL, START_DATE, END_DATE).sort_index()
    short_bars = pull_and_clean(SHORT_SYMBOL, START_DATE, END_DATE).sort_index()

    training_table = add_features(
        long_bars,
        move_pct_threshold=MOVE_PCT_THRESHOLD,
        max_forward_bars=MAX_FORWARD_BARS,
    )
    training_table = filter_training_by_session_bounds(
        training_table,
        min_bars_since_open=MIN_BARS_SINCE_OPEN,
        max_bars_until_close=MAX_BARS_UNTIL_CLOSE,
    )

    max_bars = _effective_max_bars_after_entry(long_bars, MAX_FORWARD_BARS)
    training_table = add_trade_gross_columns(
        training_table,
        long_bars=long_bars,
        short_bars=short_bars,
        short_symbol=SHORT_SYMBOL,
        take_profit=TAKE_PROFIT,
        stop_loss=STOP_LOSS,
        max_bars_after_entry=max_bars,
    )

    print_preamble(
        training_df=training_table,
        feature_symbol=LONG_SYMBOL,
        short_symbol=SHORT_SYMBOL,
        move_pct_threshold=MOVE_PCT_THRESHOLD,
        max_forward_bars=MAX_FORWARD_BARS,
        take_profit=TAKE_PROFIT,
        stop_loss=STOP_LOSS,
        trade_cost=TRADE_COST,
    )

    train_df, val_df, test_df = split_training_data(
        training_table,
        validation_fraction=VALIDATION_FRACTION,
        test_fraction=TEST_FRACTION,
    )
    train_df = downsample_negatives_to_match_positives(train_df, target_column=TARGET_COLUMN)
    val_df = downsample_negatives_to_match_positives(
        val_df, target_column=TARGET_COLUMN, random_state=43
    )
    train_df, val_df, test_df = zscore_feature_splits(
        train_df,
        val_df,
        test_df,
        non_feature_columns=COLUMNS_EXCLUDED_FROM_ZSCORE,
    )

    print_training_data_stats(train_df, val_df, test_df, target_column=TARGET_COLUMN)

    train_fit = train_df.drop(columns=list(COLUMNS_DROP_FOR_FIT))
    val_fit = val_df.drop(columns=list(COLUMNS_DROP_FOR_FIT))
    x_test = test_df.drop(columns=list(COLUMNS_EXCLUDED_FROM_ZSCORE))
    y_test = test_df[TARGET_COLUMN]

    print("\n====================== All ID — models (val grid search) ======================")

    print("Training Decision Tree...")
    tree_clf = train_decision_tree(
        train_fit,
        val_fit,
        target_column=TARGET_COLUMN,
        param_grid={
            "max_depth": [8, 16, 24, None],
            "min_samples_leaf": [100, 500],
        },
        scoring="f1",
        verbose=True,
        grid_n_jobs=4,
    )
    print_test_results(
        "Decision tree",
        y_test,
        tree_clf.predict(x_test),
        test_df=test_df,
        trade_cost=TRADE_COST,
    )

    print("Training Naive Bayes...")
    nb_clf = train_naive_bayes(
        train_fit,
        val_fit,
        target_column=TARGET_COLUMN,
        param_grid={"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
        scoring="f1",
        verbose=True,
        grid_n_jobs=4,
    )
    print_test_results(
        "Naive Bayes",
        y_test,
        nb_clf.predict(x_test),
        test_df=test_df,
        trade_cost=TRADE_COST,
    )

    knn_clf = train_knn(
        train_fit,
        val_fit,
        target_column=TARGET_COLUMN,
        param_grid={
            "n_neighbors": [5, 15, 31],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
        scoring="f1",
        verbose=True,
        grid_n_jobs=1,
    )
    print_test_results(
        "K-nearest neighbors",
        y_test,
        knn_clf.predict(x_test),
        test_df=test_df,
        trade_cost=TRADE_COST,
    )

    forest_clf = train_forest(
        train_fit,
        val_fit,
        target_column=TARGET_COLUMN,
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [12, 20, None],
            "min_samples_leaf": [100, 500],
        },
        scoring="f1",
        verbose=True,
        grid_n_jobs=4,
    )
    print_test_results(
        "Random forest",
        y_test,
        forest_clf.predict(x_test),
        test_df=test_df,
        trade_cost=TRADE_COST,
    )

    print("Training XGBoost...")
    xgb_clf = train_xgboost(
        train_fit,
        val_fit,
        target_column=TARGET_COLUMN,
        param_grid={
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [150, 300],
            "subsample": [0.8, 1.0],
        },
        scoring="f1",
        verbose=True,
        grid_n_jobs=4,
    )
    print_test_results(
        "XGBoost",
        y_test,
        xgb_clf.predict(x_test),
        test_df=test_df,
        trade_cost=TRADE_COST,
    )

    print("Training neural network (MLP)...")
    mlp_clf = train_neural_network(
        train_fit,
        val_fit,
        target_column=TARGET_COLUMN,
        param_grid={
            "hidden_layer_sizes": [(64,), (128, 64)],
            "alpha": [1e-4, 1e-3],
            "learning_rate_init": [0.001],
        },
        scoring="f1",
        verbose=True,
        grid_n_jobs=1,
    )
    print_test_results(
        "Neural network (MLP)",
        y_test,
        mlp_clf.predict(x_test),
        test_df=test_df,
        trade_cost=TRADE_COST,
    )

    models: list[tuple[str, Any]] = [
        ("K-nearest neighbors", knn_clf),
        ("Random forest", forest_clf),
        ("XGBoost", xgb_clf),
        ("Neural network (MLP)", mlp_clf),
    ]
    pred_matrix = np.vstack([clf.predict(x_test).astype(np.int64) for _, clf in models])
    y_consensus = consensus_unanimous_label(pred_matrix)
    print("\n====================== Unanimous ensemble (all 1 → 1, all 0 → 0, else -1) ======================")
    print_consensus_results("Unanimous ensemble", y_test, y_consensus)
    _print_predicted_leg_net_return_pct(test_df, y_consensus, trade_cost=TRADE_COST)
