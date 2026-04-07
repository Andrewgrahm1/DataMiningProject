"""Direction identification — predict whether the next ≥X% move within a forward window is up or down.

Uses the same minute-bar setup as change identification, but the training set is **only** bars where
a move of at least ``MOVE_PCT_THRESHOLD`` occurs within the next ``MAX_FORWARD_BARS`` minutes of the
same session (same eligibility rule as :func:`~experiments.orion2.elib.elib.add_forward_absolute_move_target`).
The label is ``1`` if the **up** threshold is hit first in time (chronological forward scan), ``0``
if **down** is hit first; if both first occur on the same bar, the larger proportional excursion wins
(tie → ``1``). See :func:`~experiments.orion2.elib.elib.add_forward_move_eligibility_and_direction`.

All run parameters live in ``main``; this module is self-contained and does not import ``chgeIdnt``
or a separate config file.

Run from project root::

    python -m experiments.orion2.drctIdnt
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

from experiments.orion2.elib.elib import (
    add_feature_adx_di,
    add_feature_atr,
    add_feature_bars_since_open,
    add_feature_bars_until_close,
    add_feature_close_bollinger_pct_diff,
    add_feature_close_ema_pct_diff,
    add_feature_close_lag_pct_diff,
    add_feature_close_sma_pct_diff,
    add_feature_close_vwap_pct_diff,
    add_feature_rsi,
    add_forward_move_eligibility_and_direction,
    bollinger_std_column_tag,
    add_volume_roll_mean_by_day,
    print_training_data_stats,
    pull_and_clean,
    split_training_data,
    zscore_feature_splits,
)
from lib.models import (
    train_decision_tree,
    train_forest,
    train_knn,
    train_naive_bayes,
    train_neural_network,
    train_xgboost,
)

TARGET_COLUMN = "direction_target"
MOVE_ELIGIBLE_COLUMN = "move_eligible"
VOLUME_ROLL_WINDOWS = (1, 2, 5, 10, 20, 30, 60, 90, 180)
ATR_PERIODS = (14, 21, 60, 90, 180)
ADX_DI_PERIODS = (14, 21)
RSI_PERIODS = (7, 14, 21)
VWAP_WINDOWS = (1, 2, 5, 10, 20, 30, 60, 90, 180)
CLOSE_SMA_PERIODS = (1, 2, 5, 10, 20, 30, 60, 90, 180)
CLOSE_EMA_SPANS = CLOSE_SMA_PERIODS
CLOSE_LAG_PCT_DIFF_BARS = CLOSE_SMA_PERIODS
BOLLINGER_PERIODS = CLOSE_SMA_PERIODS
BOLLINGER_STD_MULTIPLES = (2.0,)
COLUMNS_EXCLUDED_FROM_ZSCORE: tuple[str, ...] = (TARGET_COLUMN,)


def training_column_names() -> list[str]:
    """Target first, then features."""
    return [
        TARGET_COLUMN,
        "bars_until_close",
        "bars_since_open",
        *[f"atr_{p}" for p in ATR_PERIODS],
        # *[f"adx_{p}" for p in ADX_DI_PERIODS],
        # *[f"plus_di_{p}" for p in ADX_DI_PERIODS],
        # *[f"minus_di_{p}" for p in ADX_DI_PERIODS],
        *[f"rsi_{p}" for p in RSI_PERIODS],
        # *[f"volume_roll_mean_{w}" for w in VOLUME_ROLL_WINDOWS],
        # *[f"typical_vwap_{w}_pct_diff" for w in VWAP_WINDOWS],
        *[f"close_sma_{p}_pct_diff" for p in CLOSE_SMA_PERIODS],
        # *[f"close_ema_{p}_pct_diff" for p in CLOSE_EMA_SPANS],
        *[f"close_lag_{n}_pct_diff" for n in CLOSE_LAG_PCT_DIFF_BARS],
        # *[
        #     name
        #     for p in BOLLINGER_PERIODS
        #     for k in BOLLINGER_STD_MULTIPLES
        #     for name in (
        #         f"close_bb_upper_{p}_{bollinger_std_column_tag(k)}_pct_diff",
        #         f"close_bb_lower_{p}_{bollinger_std_column_tag(k)}_pct_diff",
        #     )
        # ],
    ]


def add_features(
    bars: pd.DataFrame,
    *,
    move_pct_threshold: float,
    max_forward_bars: int,
) -> pd.DataFrame:
    """Eligibility + direction labels and features; keep only rows with a qualifying forward move."""
    out = bars.copy()
    add_forward_move_eligibility_and_direction(
        out,
        move_pct_threshold,
        max_forward_bars,
        move_eligible_column=MOVE_ELIGIBLE_COLUMN,
        direction_column=TARGET_COLUMN,
    )
    add_feature_bars_until_close(out)
    add_feature_bars_since_open(out)
    out = add_feature_atr(out, list(ATR_PERIODS))
    # out = add_feature_adx_di(out, list(ADX_DI_PERIODS))
    out = add_feature_rsi(out, list(RSI_PERIODS))
    # out = add_volume_roll_mean_by_day(out, list(VOLUME_ROLL_WINDOWS))
    # out = add_feature_close_vwap_pct_diff(out, list(VWAP_WINDOWS))
    out = add_feature_close_sma_pct_diff(out, list(CLOSE_SMA_PERIODS))
    out = add_feature_close_ema_pct_diff(out, list(CLOSE_EMA_SPANS))
    out = add_feature_close_lag_pct_diff(out, list(CLOSE_LAG_PCT_DIFF_BARS))
    # out = add_feature_close_bollinger_pct_diff(
    #     out,
    #     list(BOLLINGER_PERIODS),
    #     std_multiples=BOLLINGER_STD_MULTIPLES,
    # )

    out = out.loc[out[MOVE_ELIGIBLE_COLUMN] == 1].copy()
    out = out.drop(columns=[MOVE_ELIGIBLE_COLUMN])
    out[TARGET_COLUMN] = out[TARGET_COLUMN].astype(np.int64)

    cols = training_column_names()
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise ValueError(f"add_features: expected columns missing: {missing}")
    return out[cols].copy()


def filter_training_by_session_bounds(
    df: pd.DataFrame,
    *,
    min_bars_since_open: int,
    max_bars_until_close: int,
) -> pd.DataFrame:
    """Keep rows with ``bars_since_open >= min_bars_since_open`` and
    ``bars_until_close <= max_bars_until_close``."""
    mask = (df["bars_since_open"] >= min_bars_since_open) & (
        df["bars_until_close"] <= max_bars_until_close
    )
    return df.loc[mask].copy()


def downsample_majority_class_to_match_minority(
    df: pd.DataFrame,
    *,
    target_column: str,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Randomly drop rows of the more frequent class until class counts match.

    The minority class is kept entirely. If already balanced or empty minority, returns ``df`` unchanged.
    """
    y = df[target_column]
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    if n0 == 0 or n1 == 0:
        return df
    if n0 == n1:
        return df
    if n0 < n1:
        minority_val, majority_val = 0, 1
        n_minor = n0
    else:
        minority_val, majority_val = 1, 0
        n_minor = n1
    minor_df = df.loc[y == minority_val]
    maj_df = df.loc[y == majority_val]
    maj_keep = maj_df.sample(n=n_minor, random_state=random_state)
    out = pd.concat([minor_df, maj_keep])
    return out.sort_index()


def consensus_unanimous_label(pred_matrix: np.ndarray) -> np.ndarray:
    """``pred_matrix`` shape ``(n_models, n_samples)`` with values 0 or 1.

    Returns length-``n_samples`` vector: ``1`` if every model predicts 1, ``0`` if every model
    predicts 0, otherwise ``-1``.
    """
    if pred_matrix.ndim != 2:
        raise ValueError("pred_matrix must be 2-D (n_models, n_samples)")
    all_ones = (pred_matrix == 1).all(axis=0)
    all_zeros = (pred_matrix == 0).all(axis=0)
    out = np.full(pred_matrix.shape[1], -1, dtype=np.int64)
    out[all_ones] = 1
    out[all_zeros] = 0
    return out


def print_preamble(df: pd.DataFrame) -> None:
    print("================== Direction Identification ===============")
    print(f"Symbol: {df.index.levels[0][0]}")
    print(f"Date Range: {df.index.levels[1][0].date()} to {df.index.levels[1][-1].date()}")
    print(f"Eligible-move bars: {len(df):,}")
    print("===========================================================")


def print_test_results(name: str, y_test: pd.Series | np.ndarray, y_pred: np.ndarray) -> None:
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
    print(f"ROC AUC: {roc_auc_score(y_true, y_hat):.6f}")
    print()


def print_consensus_results(
    name: str,
    y_test: pd.Series | np.ndarray,
    y_consensus: np.ndarray,
) -> None:
    y_true = np.asarray(y_test).ravel().astype(np.int64)
    y_hat = np.asarray(y_consensus).ravel().astype(np.int64)
    n = len(y_true)
    n_all_1 = int((y_hat == 1).sum())
    n_all_0 = int((y_hat == 0).sum())
    n_split = int((y_hat == -1).sum())

    print(f"\n--- {name} ---")
    print(f"Unanimous 1: {n_all_1:,} ({100.0 * n_all_1 / n:.4f}%)")
    print(f"Unanimous 0: {n_all_0:,} ({100.0 * n_all_0 / n:.4f}%)")
    print(f"Disagree (label -1): {n_split:,} ({100.0 * n_split / n:.4f}%)")

    decided = y_hat >= 0
    if not np.any(decided):
        print("No unanimous test rows; skip accuracy / confusion matrix on decisions.")
        return

    yt = y_true[decided]
    yh = y_hat[decided]
    labels = [0, 1]
    cm = confusion_matrix(yt, yh, labels=labels)
    acc = float(accuracy_score(yt, yh))
    recall = float(recall_score(yt, yh, zero_division=0))
    precision = float(precision_score(yt, yh, zero_division=0))
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])
    denom_f1 = 2 * tp + fp + fn
    f1 = (2 * tp / denom_f1) if denom_f1 else 0.0

    print(f"On unanimous rows only ({decided.sum():,} / {n:,}):")
    print(f"Accuracy: {(tn + tp):,} / {len(yt):,} ({100.0 * acc:.4f}%)")
    print(f"Recall: {tp:,} / {tp + fn:,} ({100.0 * recall:.4f}%)")
    print(f"Precision: {tp:,} / {tp + fp:,} ({100.0 * precision:.4f}%)")
    print(f"F1 score: {f1:.6f}")
    print(f"ROC AUC: {roc_auc_score(yt, yh):.6f}")
    print()


if __name__ == "__main__":
    SYMBOL = "TQQQ"
    START_DATE = datetime(2022, 1, 1)
    END_DATE = datetime(2025, 12, 31)
    MOVE_PCT_THRESHOLD = 0.01
    MAX_FORWARD_BARS = 90
    VALIDATION_FRACTION = 0.15
    TEST_FRACTION = 0.2
    MIN_BARS_SINCE_OPEN = 30
    MAX_BARS_UNTIL_CLOSE = 60

    bars = pull_and_clean(SYMBOL, START_DATE, END_DATE)
    bars = bars.sort_index()

    training_table = add_features(
        bars,
        move_pct_threshold=MOVE_PCT_THRESHOLD,
        max_forward_bars=MAX_FORWARD_BARS,
    )
    training_table = filter_training_by_session_bounds(
        training_table,
        min_bars_since_open=MIN_BARS_SINCE_OPEN,
        max_bars_until_close=MAX_BARS_UNTIL_CLOSE,
    )

    print_preamble(training_table)

    train_df, val_df, test_df = split_training_data(
        training_table,
        validation_fraction=VALIDATION_FRACTION,
        test_fraction=TEST_FRACTION,
    )
    # train_df = downsample_majority_class_to_match_minority(train_df, target_column=TARGET_COLUMN)
    # val_df = downsample_majority_class_to_match_minority(
    #     val_df, target_column=TARGET_COLUMN, random_state=43
    # )
    train_df, val_df, test_df = zscore_feature_splits(
        train_df,
        val_df,
        test_df,
        non_feature_columns=COLUMNS_EXCLUDED_FROM_ZSCORE,
    )

    print_training_data_stats(train_df, val_df, test_df, target_column=TARGET_COLUMN)

    x_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    print("\n====================== Direction ID — models (val grid search) ======================")

    print("Training Decision Tree...")
    tree_clf = train_decision_tree(
        train_df,
        val_df,
        target_column=TARGET_COLUMN,
        param_grid={
            "max_depth": [8, 16, 24, None],
            "min_samples_leaf": [100, 500],
        },
        scoring="f1",
        verbose=True,
        grid_n_jobs=4,
    )
    print_test_results("Decision tree", y_test, tree_clf.predict(x_test))

    print("Training Naive Bayes...")
    nb_clf = train_naive_bayes(
        train_df,
        val_df,
        target_column=TARGET_COLUMN,
        param_grid={"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
        scoring="f1",
        verbose=True,
        grid_n_jobs=4,
    )
    print_test_results("Naive Bayes", y_test, nb_clf.predict(x_test))

    knn_clf = train_knn(
        train_df,
        val_df,
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
    print_test_results("K-nearest neighbors", y_test, knn_clf.predict(x_test))

    # forest_clf = train_forest(
    #     train_df,
    #     val_df,
    #     target_column=TARGET_COLUMN,
    #     param_grid={
    #         "n_estimators": [100, 200],
    #         "max_depth": [12, 20, None],
    #         "min_samples_leaf": [100, 500],
    #     },
    #     scoring="f1",
    #     verbose=True,
    #     grid_n_jobs=4,
    # )
    # print_test_results("Random forest", y_test, forest_clf.predict(x_test))

    print("Training XGBoost...")
    xgb_clf = train_xgboost(
        train_df,
        val_df,
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
    print_test_results("XGBoost", y_test, xgb_clf.predict(x_test))

    print("Training neural network (MLP)...")
    mlp_clf = train_neural_network(
        train_df,
        val_df,
        target_column=TARGET_COLUMN,
        param_grid={
            "hidden_layer_sizes": [(64,), (128,), (128, 64),],
            "alpha": [1e-4, 1e-3],
            "learning_rate_init": [0.001],
        },
        scoring="f1",
        verbose=True,
        grid_n_jobs=1,
    )
    print_test_results("Neural network (MLP)", y_test, mlp_clf.predict(x_test))

    models: list[tuple[str, Any]] = [
        # ("Decision tree", tree_clf),
        # ("Naive Bayes", nb_clf),
        # ("K-nearest neighbors", knn_clf),
        # ("Random forest", forest_clf),
        ("XGBoost", xgb_clf),
        ("Neural network (MLP)", mlp_clf),
    ]
    pred_matrix = np.vstack([clf.predict(x_test).astype(np.int64) for _, clf in models])
    y_consensus = consensus_unanimous_label(pred_matrix)
    print("\n====================== Unanimous ensemble (all 1 → 1, all 0 → 0, else -1) ======================")
    print_consensus_results("Unanimous ensemble", y_test, y_consensus)
