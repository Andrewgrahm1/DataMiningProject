from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.naive_bayes import GaussianNB  # type: ignore[import-untyped]
from sklearn.neighbors import KNeighborsClassifier  # type: ignore[import-untyped]
from sklearn.tree import DecisionTreeClassifier  # type: ignore[import-untyped]
from xgboost import XGBClassifier  # type: ignore[import-untyped]

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

# Cache dir: etc/data under project root (experiment is in experiments/orion/)
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "etc" / "data"


def _cache_path(symbol: str, start_date: datetime, end_date: datetime) -> Path:
    """Path for cached cleaned data: etc/data/zhongjie_{symbol}_{start}_{end}_clean.csv"""
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    return _CACHE_DIR / f"zhongjie_{symbol}_{start_str}_{end_str}_clean.csv"


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


def create_training_data(data: pd.DataFrame, take_profit: float, stop_loss: float) -> pd.DataFrame:
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
    pct_bars = [b for b in range(1, 360) if b < 10 or (b < 120 and b % 10 == 0) or b % 30 == 0]
    col_names.extend(f"pct_change_{b}" for b in pct_bars)
    data = add_feature_pct_change_batch(data, pct_bars)
    return data[col_names].copy()


def split_training_data(
    data: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train and test by time (chronological). Last test_fraction of rows is test."""
    data = data.sort_index(level="timestamp")
    n = len(data)
    test_size = int(n * test_fraction)
    if test_size == 0:
        return data, data.iloc[0:0]
    train_df = data.iloc[: -test_size]
    test_df = data.iloc[-test_size:]
    return train_df, test_df


def train_model(
    data: pd.DataFrame,
    target_column: str = "target",
    **kwargs,
) -> DecisionTreeClassifier:
    """Fit a decision tree on features vs target. Returns the fitted classifier."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    if "class_weight" not in kwargs:
        kwargs["class_weight"] = "balanced"
    clf = DecisionTreeClassifier(random_state=42, **kwargs)
    clf.fit(X, y)
    return clf


def train_forest(
    data: pd.DataFrame,
    target_column: str = "target",
    **kwargs,
) -> RandomForestClassifier:
    """Fit a random forest on features vs target. Returns the fitted classifier."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    if "class_weight" not in kwargs:
        kwargs["class_weight"] = "balanced"
    clf = RandomForestClassifier(random_state=42, **kwargs)
    clf.fit(X, y)
    return clf


def train_adaboost(
    data: pd.DataFrame,
    target_column: str = "target",
    **kwargs,
) -> AdaBoostClassifier:
    """Fit an AdaBoost classifier on features vs target. Returns the fitted classifier.

    Uses a shallow tree (max_depth=3) as base estimator with class_weight='balanced'
    so the first weak learner is better than random and the ensemble can fit; a stump
    (max_depth=1) can be worse than random on this task and cause AdaBoost to raise.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    if "estimator" not in kwargs:
        kwargs["estimator"] = DecisionTreeClassifier(
            max_depth=3, random_state=42, class_weight="balanced"
        )
    clf = AdaBoostClassifier(random_state=42, **kwargs)
    clf.fit(X, y)
    return clf


def train_naive_bayes(
    data: pd.DataFrame,
    target_column: str = "target",
    **kwargs,
) -> GaussianNB:
    """Fit a Gaussian Naive Bayes classifier on features vs target. Returns the fitted classifier."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    clf = GaussianNB(**kwargs)
    clf.fit(X, y)
    return clf


def train_knn(
    data: pd.DataFrame,
    target_column: str = "target",
    **kwargs,
) -> KNeighborsClassifier:
    """Fit a K-Nearest Neighbors classifier on features vs target. Returns the fitted classifier."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    clf = KNeighborsClassifier(**kwargs)
    clf.fit(X, y)
    return clf


def train_xgboost(
    data: pd.DataFrame,
    target_column: str = "target",
    **kwargs,
) -> XGBClassifier:
    """Fit an XGBoost classifier on features vs target. Returns the fitted classifier.

    When ``scale_pos_weight`` is omitted, sets it to n_negative / n_positive so
    imbalance handling is analogous to ``class_weight='balanced'`` on sklearn trees.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    if "scale_pos_weight" not in kwargs:
        n_pos = int((y == 1).sum())
        if n_pos > 0:
            kwargs["scale_pos_weight"] = int((y == 0).sum()) / n_pos
    clf = XGBClassifier(random_state=42, **kwargs)
    clf.fit(X, y)
    return clf


if __name__ == "__main__":
    SYMBOL = "MARA"
    START_DATE = datetime(2022, 1, 1)
    END_DATE = datetime(2025, 12, 31)
    TAKE_PROFIT = 0.04
    STOP_LOSS = 0.02
    TRADE_COST = 0.004

    print("====================== Starting Test ======================")
    print(f"SYMBOL: {SYMBOL}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Take Profit: {TAKE_PROFIT * 100}%")
    print(f"Stop Loss: {STOP_LOSS * 100}%")
    print(f"Trade Cost: {TRADE_COST * 100}%")
    print(f"Break Even Win Rate (Percision): {calculate_min_win_rate(TAKE_PROFIT, STOP_LOSS, TRADE_COST) * 100}%")
    print()

    raw_df = pull_and_clean(SYMBOL, START_DATE, END_DATE)
    training_df = create_training_data(raw_df, take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS)
    train_df, test_df = split_training_data(training_df, test_fraction=0.2)
    print(f"Train rows: {len(train_df):,}, test rows: {len(test_df):,}")

    X_train = train_df.drop(columns=["target"])
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # Decision tree
    tree_clf = train_model(train_df)
    tree_pred = tree_clf.predict(X_test)
    evaluate_and_print("Decision Tree", y_test, tree_pred)

    # Naive Bayes (Gaussian; features are continuous)
    nb_clf = train_naive_bayes(train_df)
    nb_pred = nb_clf.predict(X_test)
    evaluate_and_print(
        "Naive Bayes", y_test, nb_pred,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    # K-Nearest Neighbors
    knn_clf = train_knn(train_df)
    knn_pred = knn_clf.predict(X_test)
    evaluate_and_print(
        "K-Nearest Neighbors", y_test, knn_pred,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    # Random forest
    forest_clf = train_forest(train_df)
    forest_pred = forest_clf.predict(X_test)
    evaluate_and_print(
        "Random Forest", y_test, forest_pred,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    # AdaBoost
    adaboost_clf = train_adaboost(train_df)
    adaboost_pred = adaboost_clf.predict(X_test)
    evaluate_and_print(
        "AdaBoost", y_test, adaboost_pred,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    # XGBoost
    xgb_clf = train_xgboost(train_df)
    xgb_pred = xgb_clf.predict(X_test)
    evaluate_and_print(
        "XGBoost", y_test, xgb_pred,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )
