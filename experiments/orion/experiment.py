from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from alpaca.data.timeframe import TimeFrame

from lib.stock.data_cleaner import StockDataCleaner
from lib.stock.data_fetcher import StockDataFetcher

from lib.common.common import (
    add_feature_bars_since_open,
    add_feature_bars_until_close,
    add_feature_pct_change_batch,
    calculate_min_win_rate,
    create_target_column,
    evaluate_and_print,
)

from lib.models import (
    train_adaboost,
    train_forest,
    train_knn,
    train_decision_tree,
    train_naive_bayes,
    train_xgboost,
)

from experiments.orion.elib import (
    add_feature_atr,
    add_feature_close_sma_pct_diff,
    add_feature_close_vwap_pct_diff,
    add_feature_close_vs_reference_bars_pct_diff,
    add_feature_day_of_week,
    add_feature_rsi,
    add_feature_rsi_reference_bars,
    add_feature_volume_and_trade_count_zscore,
    symbols_in_reference_bars,
)

# Cache dir: etc/data under project root (experiment is in experiments/orion/)
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "etc" / "data"


def _cache_path(symbol: str, start_date: datetime, end_date: datetime) -> Path:
    """Path for cached cleaned data: etc/data/orion_{symbol}_{start}_{end}_clean.csv"""
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    return _CACHE_DIR / f"orion_{symbol}_{start_str}_{end_str}_clean.csv"


def pull_and_clean(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch OHLCV bars for symbol in [start, end], restrict to RTH, forward-fill missing bars.
    Uses cached CSV in etc/data when present (prefix decision-tree). Saves cleaned data to
    cache when pulling fresh."""
    cache_path = _cache_path(symbol, start, end)

    if cache_path.exists():
        print(f"Loading cached data: {cache_path.name}")
        data = pd.read_csv(cache_path, index_col=[0, 1], parse_dates=[1])
        if data.index.levels[1].tz is None:
            data.index = data.index.set_levels(
                data.index.levels[1].tz_localize("UTC"), level=1
            )
        return data

    fetcher = StockDataFetcher()
    data = fetcher.get_historical_bars(
        symbol=symbol,
        start_date=start,
        end_date=end,
        timeframe=TimeFrame.Minute,
    )
    cleaner = StockDataCleaner()
    data = cleaner.remove_closed_market_rows(data)
    data = cleaner.forward_propagate(
        data,
        TimeFrame.Minute,
        only_when_market_open=True,
        mark_imputed_rows=False,
    )
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(cache_path)
    print(f"Cached cleaned data to {cache_path.name}")
    return data


def create_training_data(
    data: pd.DataFrame,
    take_profit: float,
    stop_loss: float,
    *,
    reference_bars: pd.DataFrame | None = None,
) -> pd.DataFrame:
    # Column names added by create_training_data (target + features only)
    col_names = []
    col_names.append("target")
    data = create_target_column(
        data, take_profit=take_profit, stop_loss=stop_loss, column_name=col_names[-1]
    )
    col_names.append("bars_until_close")    
    data = add_feature_bars_until_close(data, column_name=col_names[-1])
    col_names.append("bars_since_open")
    data = add_feature_bars_since_open(data, column_name=col_names[-1])
    col_names.append("close_vwap_pct_diff")
    data = add_feature_close_vwap_pct_diff(data, column_name=col_names[-1])
    rolling_windows = [2, 5, 10, 20, 30, 60, 90, 120, 180]
    rsi_rolling_windows = [20, 40, 60, 90, 120, 180]
    if reference_bars is not None:
        ref_syms = symbols_in_reference_bars(reference_bars)
        if ref_syms:
            col_names.extend(f"close_vs_{s}_pct_diff" for s in ref_syms)
            data = add_feature_close_vs_reference_bars_pct_diff(data, reference_bars)
            col_names.extend(f"rsi_{s}_{b}" for s in ref_syms for b in rsi_rolling_windows)
            data = add_feature_rsi_reference_bars(data, reference_bars, rsi_rolling_windows)
    col_names.extend(f"atr_{b}" for b in rolling_windows)
    data = add_feature_atr(data, rolling_windows)
    col_names.append("day_of_week")
    data = add_feature_day_of_week(data, column_name=col_names[-1])
    col_names.extend(f"volume_zscore_{w}" for w in rolling_windows)
    col_names.extend(f"trade_count_zscore_{w}" for w in rolling_windows)
    data = add_feature_volume_and_trade_count_zscore(data, rolling_windows)
    col_names.extend(f"pct_change_{b}" for b in rolling_windows)
    data = add_feature_pct_change_batch(data, rolling_windows)
    col_names.extend(f"close_sma_{b}_pct_diff" for b in rolling_windows)
    data = add_feature_close_sma_pct_diff(data, rolling_windows)
    col_names.extend(f"rsi_{b}" for b in rsi_rolling_windows)
    data = add_feature_rsi(data, rsi_rolling_windows)
    return data[col_names].copy()


def split_training_data(
    data: pd.DataFrame,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically into train, validation, then test (three contiguous time blocks)."""
    data = data.sort_index(level="timestamp")
    n = len(data)
    n_test = int(n * test_fraction)
    n_val = int(n * validation_fraction)
    n_train = n - n_val - n_test
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(
            f"split_training_data: need positive train/val/test sizes; got n={n}, "
            f"train={n_train}, val={n_val}, test={n_test}. Lower validation_fraction or "
            "test_fraction."
        )
    train_df = data.iloc[:n_train]
    val_df = data.iloc[n_train : n_train + n_val]
    test_df = data.iloc[n_train + n_val :]
    return train_df, val_df, test_df


def _print_split_stats(name: str, df: pd.DataFrame, *, target_column: str = "target") -> None:
    """One line: rows, positive class count/rate, timestamp range and span in years."""
    n = len(df)
    if n == 0:
        print(f"{name}: 0 rows")
        return
    ts = df.index.get_level_values("timestamp")
    start, end = ts.min(), ts.max()
    span_years = (end - start).total_seconds() / (365.25 * 24 * 60 * 60)
    pos = int((df[target_column] == 1).sum())
    pos_pct = 100.0 * pos / n
    print(
        f"{name}: {n:,} rows | {pos:,} positive ({pos_pct:.2f}%) | "
        f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')} "
        f"({span_years:.2f} years)"
    )


def print_training_data_stats(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_column: str = "target",
) -> None:
    """Print row counts, target balance, and calendar span for train, validation, and test."""
    print("\n--- Train / validation / test split stats ---")
    _print_split_stats("Train", train_df, target_column=target_column)
    _print_split_stats("Validation", validation_df, target_column=target_column)
    _print_split_stats("Test", test_df, target_column=target_column)


def zscore_feature_columns(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str = "target",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit z-score (StandardScaler) on train features only; transform train, validation, and test."""
    feature_cols = [c for c in train_df.columns if c != target_column]
    scaler = StandardScaler()
    train_out = train_df.copy()
    val_out = validation_df.copy()
    test_out = test_df.copy()
    train_out[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_out[feature_cols] = scaler.transform(validation_df[feature_cols])
    test_out[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_out, val_out, test_out


if __name__ == "__main__":
    SYMBOL = "MARA"
    # Reference symbol(s) for cross-close features; bars are concatenated below.
    REFERENCE_SYMBOLS = ["SPY", "IBIT", "WGMI", "CLSK", "MSTR"]
    START_DATE = datetime(2022, 1, 1)
    END_DATE = datetime(2025, 12, 31)
    TAKE_PROFIT = 0.03
    STOP_LOSS = 0.03
    TRADE_COST = 0.004

    print("====================== Starting Test ======================")
    print(f"SYMBOL: {SYMBOL}")
    print(f"Reference symbols (cross-close features): {REFERENCE_SYMBOLS}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Take Profit: {TAKE_PROFIT * 100}%")
    print(f"Stop Loss: {STOP_LOSS * 100}%")
    print(f"Trade Cost: {TRADE_COST * 100}%")
    print(f"Break Even Win Rate (Percision): {calculate_min_win_rate(TAKE_PROFIT, STOP_LOSS, TRADE_COST) * 100}%")

    raw_df = pull_and_clean(SYMBOL, START_DATE, END_DATE)
    reference_df = pd.concat(
        [pull_and_clean(s, START_DATE, END_DATE) for s in REFERENCE_SYMBOLS],
        axis=0,
    ).sort_index()
    training_df = create_training_data(
        raw_df,
        take_profit=TAKE_PROFIT,
        stop_loss=STOP_LOSS,
        reference_bars=reference_df,
    )

    train_df, val_df, test_df = split_training_data(
        training_df, validation_fraction=0.15, test_fraction=0.2
    )
    train_df, val_df, test_df = zscore_feature_columns(train_df, val_df, test_df)
    print_training_data_stats(train_df, val_df, test_df)

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # # Pearson correlation of each feature column with target (z-scored train features).
    # feat_target_corr = train_df.corr(numeric_only=True)["target"].drop("target", errors="ignore")
    # feat_target_corr = feat_target_corr.reindex(
    #     feat_target_corr.abs().sort_values(ascending=False).index
    # )
    # print("\n--- Feature vs target correlation (train, Pearson) ---")
    # for feat_name, rho in feat_target_corr.items():
    #     rho_str = f"{rho:.6f}" if np.isfinite(rho) else "nan"
    #     print(f"  {feat_name}: {rho_str}")

    # exit(0)

    print("\n--- Decision Tree (grid search) ---")
    tree_clf = train_decision_tree(
        train_df,
        val_df,
        param_grid={
            "max_depth": [8, 16, 24, None],
            "min_samples_leaf": [100, 500],
        },
        verbose=True,
        grid_n_jobs=6,
    )
    tree_pred = tree_clf.predict(X_test)
    evaluate_and_print(
        "Decision Tree", y_test, tree_pred,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST
    )

    print("\n--- Naive Bayes (grid search) ---")
    nb_clf = train_naive_bayes(
        train_df,
        val_df,
        param_grid={"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
        verbose=True,
        grid_n_jobs=1,  # Naive Bayes is very fast, so no need to parallelize
    )
    nb_pred = nb_clf.predict(X_test)
    evaluate_and_print(
        "Naive Bayes", y_test, nb_pred,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    print("\n--- K-Nearest Neighbors (grid search) ---")
    knn_clf = train_knn(
        train_df,
        val_df,
        param_grid={
            "n_neighbors": [5, 15, 31],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
        verbose=True,
        grid_n_jobs=4,
    )
    knn_pred = knn_clf.predict(X_test)
    evaluate_and_print(
        "K-Nearest Neighbors", y_test, knn_pred,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    print("\n--- Random Forest (grid search) ---")
    forest_clf = train_forest(
        train_df,
        val_df,
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [12, 20, None],
            "min_samples_leaf": [100, 500],
        },
        verbose=True,
        grid_n_jobs=4,
    )
    forest_pred = forest_clf.predict(X_test)
    evaluate_and_print(
        "Random Forest", y_test, forest_pred,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    print("\n--- AdaBoost (grid search) ---")
    adaboost_clf = train_adaboost(
        train_df,
        val_df,
        param_grid={"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
        verbose=True,
        grid_n_jobs=4,
    )
    adaboost_pred = adaboost_clf.predict(X_test)
    evaluate_and_print(
        "AdaBoost", y_test, adaboost_pred,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    print("\n--- XGBoost (grid search) ---")
    xgb_clf = train_xgboost(
        train_df,
        val_df,
        param_grid={
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [150, 300],
            "subsample": [0.8, 1.0],
        },
        verbose=True,
        grid_n_jobs=4,
    )
    xgb_pred = xgb_clf.predict(X_test)
    evaluate_and_print(
        "XGBoost", y_test, xgb_pred,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )
