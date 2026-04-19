"""Run all models with hyperparameter grid search and evaluate on the held-out test set."""

from lib.common.common import evaluate_and_print
from lib.models import (
    train_adaboost,
    train_decision_tree,
    train_forest,
    train_knn,
    train_naive_bayes,
    train_xgboost,
)

from experiments.orion.config import DEFAULT_ORION_CONFIG
from experiments.orion.elib import load_orion_training_frames, print_orion_run_preamble


def main() -> None:
    cfg = DEFAULT_ORION_CONFIG
    trade_cost = cfg.trade_cost

    print_orion_run_preamble(cfg, trade_cost=trade_cost, run_title="Grid search")

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
    )

    x_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

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
    evaluate_and_print(
        "Decision Tree",
        y_test,
        tree_clf.predict(x_test),
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        cost=trade_cost,
    )

    print("\n--- Naive Bayes (grid search) ---")
    nb_clf = train_naive_bayes(
        train_df,
        val_df,
        param_grid={"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
        verbose=True,
        grid_n_jobs=1,
    )
    evaluate_and_print(
        "Naive Bayes",
        y_test,
        nb_clf.predict(x_test),
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        cost=trade_cost,
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
    evaluate_and_print(
        "K-Nearest Neighbors",
        y_test,
        knn_clf.predict(x_test),
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        cost=trade_cost,
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
    evaluate_and_print(
        "Random Forest",
        y_test,
        forest_clf.predict(x_test),
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        cost=trade_cost,
    )

    print("\n--- AdaBoost (grid search) ---")
    adaboost_clf = train_adaboost(
        train_df,
        val_df,
        param_grid={"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
        verbose=True,
        grid_n_jobs=4,
    )
    evaluate_and_print(
        "AdaBoost",
        y_test,
        adaboost_clf.predict(x_test),
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        cost=trade_cost,
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
    evaluate_and_print(
        "XGBoost",
        y_test,
        xgb_clf.predict(x_test),
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        cost=trade_cost,
    )


if __name__ == "__main__":
    main()
