"""Direction identification with a neural network only — same label as :mod:`experiments.orion2.drctIdnt`.

Uses the same minute-bar setup, eligibility rule, and ``direction_target`` (up-first vs down-first
within the forward window). Feature engineering matches :func:`experiments.orion2.drctIdnt.add_features`.

This module trains only an ``MLPClassifier`` via :func:`lib.models.train_neural_network`. All run
parameters live in ``main``; the file is self-contained and does not import ``drctIdnt`` or a
separate config.

Run from project root::

    python -m experiments.orion2.nnDrctIdnt
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

from experiments.orion2.elib.elib import (
    add_feature_bars_since_open,
    add_feature_bars_until_close,
    add_feature_close_lag_pct_diff,
    add_feature_volume_lag_pct_diff,
    add_feature_close_minus_high_lag,
    add_feature_close_minus_low_lag,
    add_feature_high_minus_close_lag,
    add_forward_move_eligibility_and_direction,
    print_training_data_stats,
    pull_and_clean,
    split_training_data,
    zscore_feature_splits,
)
from lib.models import train_neural_network

TARGET_COLUMN = "direction_target"
MOVE_ELIGIBLE_COLUMN = "move_eligible"
CLOSE_LAG_PCT_DIFF_BARS = [i for i in range(1, 180)]
VOLUME_LAG_PCT_DIFF_BARS = CLOSE_LAG_PCT_DIFF_BARS
# Per-bar OHLC offsets vs close: lag 0 = current bar; same offsets as pct-diff lags.
CLOSE_MINUS_HIGH_LAG_BARS = [0, *CLOSE_LAG_PCT_DIFF_BARS]
HIGH_MINUS_CLOSE_LAG_BARS = CLOSE_MINUS_HIGH_LAG_BARS
CLOSE_MINUS_LOW_LAG_BARS = CLOSE_MINUS_HIGH_LAG_BARS
COLUMNS_EXCLUDED_FROM_ZSCORE: tuple[str, ...] = (TARGET_COLUMN,)


def training_column_names() -> list[str]:
    """Target first, then features (aligned with ``drctIdnt``)."""
    return [
        TARGET_COLUMN,
        "bars_until_close",
        "bars_since_open",
        *[f"close_minus_high_lag_{n}" for n in CLOSE_MINUS_HIGH_LAG_BARS],
        *[f"high_minus_close_lag_{n}" for n in HIGH_MINUS_CLOSE_LAG_BARS],
        *[f"close_minus_low_lag_{n}" for n in CLOSE_MINUS_LOW_LAG_BARS],
        *[f"close_lag_{n}_pct_diff" for n in CLOSE_LAG_PCT_DIFF_BARS],
        *[f"volume_lag_{n}_pct_diff" for n in VOLUME_LAG_PCT_DIFF_BARS],
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
    out = add_feature_close_minus_high_lag(out, list(CLOSE_MINUS_HIGH_LAG_BARS))
    out = add_feature_high_minus_close_lag(out, list(HIGH_MINUS_CLOSE_LAG_BARS))
    out = add_feature_close_minus_low_lag(out, list(CLOSE_MINUS_LOW_LAG_BARS))
    out = add_feature_close_lag_pct_diff(out, list(CLOSE_LAG_PCT_DIFF_BARS))
    out = add_feature_volume_lag_pct_diff(out, list(VOLUME_LAG_PCT_DIFF_BARS))

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
    """Randomly drop rows of the more frequent class until class counts match."""
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


def print_preamble(df: pd.DataFrame) -> None:
    print("================== Direction Identification (NN only) ===============")
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

    print("\n====================== Direction ID — neural network (val grid search) ======================")
    mlp_clf = train_neural_network(
        train_df,
        val_df,
        target_column=TARGET_COLUMN,
        param_grid={
            "hidden_layer_sizes": [(64,), (128,), (256,)],
            "alpha": [1e-4, 1e-3],
            "learning_rate_init": [0.001],
            "activation": ["relu", "tanh"],
            "max_iter": [500, 1000],
            # "n_iter_no_change": [10, 20],
        },
        scoring="f1",
        verbose=True,
        grid_n_jobs=4,
    )
    print_test_results("Neural network (MLP)", y_test, mlp_clf.predict(x_test))
