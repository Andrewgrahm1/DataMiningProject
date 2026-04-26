from datetime import datetime
import time
from pathlib import Path
import numpy as np
import pandas as pd
from alpaca.data.timeframe import TimeFrame
from lib.stock.data_cleaner import StockDataCleaner
from lib.stock.data_fetcher import StockDataFetcher
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from lib.common.common import (
    add_feature_bars_since_open,
    add_feature_bars_until_close,
    add_feature_pct_change_batch,
    calculate_min_win_rate,
    create_target_column,
    evaluate_and_print,
)

# Cache dir: etc/data under project root (experiment is in experiments/andrew/)
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "etc" / "data"

def _cache_path(symbol: str, start_date: datetime, end_date: datetime) -> Path:
    """Path for cached cleaned data: etc/data/andrew_{symbol}_{start}_{end}_clean.csv"""
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    return _CACHE_DIR / f"andrew_{symbol}_{start_str}_{end_str}_clean.csv"

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

def train_GNB(
    trainXData: pd.DataFrame,
    trainYData: pd.DataFrame,
    **kwargs,
) -> GaussianNB:
    """Fit a Naive Bayes Model on features vs target. Returns the fitted classifier."""
    clf = GaussianNB(**kwargs)
    clf.fit(trainXData, trainYData)
    return clf

def train_LR(
    trainXData: pd.DataFrame,
    trainYData: pd.DataFrame,
    **kwargs,
) -> LogisticRegression:
    """Fit a Logistic Regression Model on features vs target. Returns the fitted classifier."""
    if "max_iter" not in kwargs:
        kwargs["max_iter"] = 10000
    if "class_weight" not in kwargs:
        kwargs["class_weight"] = "balanced"
    clf = LogisticRegression(random_state=42, **kwargs)
    clf.fit(trainXData, trainYData)
    return clf

def train_KMeans(
    trainData: pd.DataFrame
) -> KMeans:
    """Fit a K-Means Model on features"""
    Kmeans = KMeans(n_clusters=8, random_state=42, n_init="auto")
    Kmeans.fit(trainData)
    return Kmeans

# # Definition changes run time from 2~ minutes to 10 minutes. Can add back in if necessary.
# def selectingKForGMM(
#         features: pd.DataFrame
# ) -> dict:
#     """Find the best parameters for the Gaussian Mixture Model"""
#     bestBIC = np.inf
#     param = {}
#     for type in ['full', 'diag', 'spherical']:
#         for k in range(2, 11):
#             gaussMix = GaussianMixture(n_components=k, covariance_type=type, random_state=42)
#             gaussMix.fit(features)
#             bic = gaussMix.bic(features)
#             if bic < bestBIC:
#                 bestBIC = bic
#                 param = {
#                     'n_components': k,
#                     'covariance_type': type
#                 }
#     return param

def train_GMM(
    trainData: pd.DataFrame
) -> GaussianMixture:
    """Fit a Gaussian Mixture Model on features"""
    #param = selectingKForGMM(trainData)
    gaussMix = GaussianMixture(n_components=8, covariance_type='spherical', random_state=42)
    gaussMix.fit(trainData)
    return gaussMix

def train_tree(
    trainXData: pd.DataFrame,
    trainYData: pd.DataFrame,
    **kwargs,
) -> DecisionTreeClassifier:
    """Fit a decision tree on features vs target. Returns the fitted classifier."""
    if "class_weight" not in kwargs:
        kwargs["class_weight"] = "balanced"
    clf = DecisionTreeClassifier(random_state=42, **kwargs)
    clf.fit(trainXData, trainYData)
    return clf

def train_forest(
    trainXData: pd.DataFrame,
    trainYData: pd.DataFrame,
    **kwargs,
) -> RandomForestClassifier:
    """Fit a random forest on features vs target. Returns the fitted classifier."""
    if "class_weight" not in kwargs:
        kwargs["class_weight"] = "balanced"
    clf = RandomForestClassifier(random_state=42, **kwargs)
    clf.fit(trainXData, trainYData)
    return clf

def train_adaboost(
    trainXData: pd.DataFrame,
    trainYData: pd.DataFrame,
    **kwargs,
) -> AdaBoostClassifier:
    """Fit an AdaBoost classifier on features vs target. Returns the fitted classifier.

    Uses a shallow tree (max_depth=3) as base estimator with class_weight='balanced'
    so the first weak learner is better than random and the ensemble can fit; a stump
    (max_depth=1) can be worse than random on this task and cause AdaBoost to raise.
    """
    if "estimator" not in kwargs:
        kwargs["estimator"] = DecisionTreeClassifier(
            max_depth=3, random_state=42, class_weight="balanced"
        )
    clf = AdaBoostClassifier(random_state=42, **kwargs)
    clf.fit(trainXData, trainYData)
    return clf

def train_knn(
    trainXData: pd.DataFrame,
    trainYData: pd.DataFrame,
    **kwargs,
) -> KNeighborsClassifier:
    """Fit a K-Nearest Neighbors classifier on features vs target. Returns the fitted classifier."""
    clf = KNeighborsClassifier(**kwargs)
    clf.fit(trainXData, trainYData)
    return clf

def train_xgboost(
    trainXData: pd.DataFrame,
    trainYData: pd.DataFrame,
    **kwargs,
) -> XGBClassifier:
    """Fit an XGBoost classifier on features vs target. Returns the fitted classifier.

    When ``scale_pos_weight`` is omitted, sets it to n_negative / n_positive so
    imbalance handling is analogous to ``class_weight='balanced'`` on sklearn trees.
    """
    if "scale_pos_weight" not in kwargs:
        n_pos = int((trainYData == 1).sum())
        if n_pos > 0:
            kwargs["scale_pos_weight"] = int((trainYData == 0).sum()) / n_pos
    clf = XGBClassifier(random_state=42, **kwargs)
    clf.fit(trainXData, trainYData)
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
    print(f"Break Even Win Rate (Percision): {calculate_min_win_rate(TAKE_PROFIT, STOP_LOSS, TRADE_COST) * 100}%\n")

    raw_df = pull_and_clean(SYMBOL, START_DATE, END_DATE)
    training_df = create_training_data(raw_df, take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS)
    train_df, test_df = split_training_data(training_df, test_fraction=0.2)
    print(f"Train rows: {len(train_df):,}, test rows: {len(test_df):,}")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    scaler = StandardScaler()
    XtrainScale = scaler.fit_transform(train_df.drop(columns=['target']))
    XtestScale = scaler.transform(test_df.drop(columns=['target']))

    # Gaussian Mixture Model
    gaussMix = train_GMM(XtrainScale)
    
    trainProbs = gaussMix.predict_proba(XtrainScale)
    testProbs = gaussMix.predict_proba(XtestScale)

    GGMXtrain = X_train.assign(**{f"prob_{i}": trainProbs[:, i] for i in range(trainProbs.shape[1])}).copy()
    GMMXtest = X_test.assign(**{f"prob_{i}": testProbs[:, i] for i in range(testProbs.shape[1])}).copy()

    GGMLRModel = train_LR(GGMXtrain, y_train)
    GGMLRPreds = GGMLRModel.predict(GMMXtest)
    evaluate_and_print(
        "Gaussian Mixture w/ Logistic Regression", y_test, GGMLRPreds,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    # Kmeans Clustering Model
    kmeans = train_KMeans(XtrainScale)

    trainDist = kmeans.transform(XtrainScale)
    testDist = kmeans.transform(XtestScale)

    KMEANSXtrain = X_train.assign(**{f"distince_{i}": trainDist[:, i] for i in range(trainDist.shape[1])}).copy()
    KMEANSXtest = X_test.assign(**{f"distince_{i}": testDist[:, i] for i in range(testDist.shape[1])}).copy()

    KMEANSLRModel = train_LR(KMEANSXtrain, y_train)
    KMEANSLRPreds = KMEANSLRModel.predict(KMEANSXtest)
    evaluate_and_print(
        "K-Means w/ Logistic Regression", y_test, KMEANSLRPreds,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    # Gaussian Mixture Model
    GGMGNBModel = train_GNB(GGMXtrain, y_train)
    GGMGNBPreds = GGMGNBModel.predict(GMMXtest)
    evaluate_and_print(
        "Gaussian Mixture w/ Gaussian Native Bayes", y_test, GGMGNBPreds,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    # Kmeans Clustering Model
    KMEANSGNBModel = train_GNB(KMEANSXtrain, y_train)
    KMEANSGNBPreds = KMEANSGNBModel.predict(KMEANSXtest)
    evaluate_and_print(
        "K-Means w/ Gaussian Naive Bayes", y_test, KMEANSGNBPreds,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    # Decision Tree
    DTGMMModel = train_tree(GGMXtrain, y_train)
    KNNDTPreds = DTGMMModel.predict(GMMXtest)
    evaluate_and_print(
        "Gaussian Mixture w/ Decision Tree", y_test, KNNDTPreds,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    # Decision Tree
    DTKMeansModel = train_tree(KMEANSXtrain, y_train)
    DTKMeansPreds = DTKMeansModel.predict(KMEANSXtest)
    evaluate_and_print(
        "K-Means w/ Decision Tree", y_test, DTKMeansPreds,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    # K-Nearest Neighbors
    KNNGMMModel = train_knn(GGMXtrain, y_train)
    KNNGMMPreds = KNNGMMModel.predict(GMMXtest)
    evaluate_and_print(
        "Gaussian Mixture w/ K-Nearest Neighbors", y_test, KNNGMMPreds,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    # K-Nearest Neighbors
    KNNKMeansModel = train_knn(KMEANSXtrain, y_train)
    KNNKMeansPreds = KNNKMeansModel.predict(KMEANSXtest)
    evaluate_and_print(
        "K-Means w/ K-Nearest Neighbors", y_test, KNNKMeansPreds,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    # Random forest
    RFGMMModel = train_forest(GGMXtrain, y_train)
    RFGMMPreds = RFGMMModel.predict(GMMXtest)
    evaluate_and_print(
        "Gaussian Mixture w/ Random forest", y_test, RFGMMPreds,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    # Random forest
    RFKMeansModel = train_forest(KMEANSXtrain, y_train)
    RFKMeansPreds = RFKMeansModel.predict(KMEANSXtest)
    evaluate_and_print(
        "K-Means w/ Random forest", y_test, RFKMeansPreds,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    # AdaBoost
    ADAGMMModel = train_adaboost(GGMXtrain, y_train)
    ADAGMMPreds = ADAGMMModel.predict(GMMXtest)
    evaluate_and_print(
        "Gaussian Mixture w/ AdaBoost", y_test, ADAGMMPreds,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    # AdaBoost
    ADAKMeansModel = train_adaboost(KMEANSXtrain, y_train)
    ADAKMeansPreds = ADAKMeansModel.predict(KMEANSXtest)
    evaluate_and_print(
        "K-Means w/ AdaBoost", y_test, ADAKMeansPreds,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    # XGBoost
    XGGMMModel = train_xgboost(GGMXtrain, y_train)
    XGGMMPreds = XGGMMModel.predict(GMMXtest)
    evaluate_and_print(
        "Gaussian Mixture w/ XGBoost", y_test, XGGMMPreds,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )

    # XGBoost
    XGKMeansModel = train_xgboost(KMEANSXtrain, y_train)
    XGKMeansPreds = XGKMeansModel.predict(KMEANSXtest)
    evaluate_and_print(
        "K-Means w/ XGBoost", y_test, XGKMeansPreds,
        take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS, cost=TRADE_COST,
    )