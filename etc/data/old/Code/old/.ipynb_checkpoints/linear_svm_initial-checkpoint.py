from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from experiments.orion.config import DEFAULT_ORION_CONFIG
from experiments.zhongjie.elib import load_zhongjie_training_frames
from experiments.zhongjie.elib import evaluate_and_print_trade_zhongjie
from lib.common.common import evaluate_and_print, calculate_min_win_rate


def trade_profit_from_preds(
    y_true,
    y_pred,
    *,
    take_profit: float,
    stop_loss: float,
    cost: float,
) -> float:
    tp_count = int(((y_true == 1) & (y_pred == 1)).sum())
    fp_count = int(((y_true == 0) & (y_pred == 1)).sum())
    win_per = take_profit - cost
    loss_per = stop_loss + cost
    return tp_count * win_per - fp_count * loss_per


def main() -> None:
    cfg = DEFAULT_ORION_CONFIG

    train_df, val_df, test_df = load_zhongjie_training_frames(
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

    x_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    x_val = val_df.drop(columns=["target"])
    y_val = val_df["target"]

    x_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    c_values = [0.01, 0.1, 1.0, 5.0]
    class_weights = [None, "balanced"]
    thresholds = [-0.5, 0.0, 0.2, 0.5, 1.0]

    best_profit = float("-inf")
    best_clf = None
    best_threshold = 0.0
    best_params = None

    for c in c_values:
        for cw in class_weights:
            clf = LinearSVC(
                C=c,
                class_weight=cw,
                random_state=42,
                max_iter=5000,
            )

            print(f"Training LinearSVC with C={c}, class_weight={cw}")
            clf.fit(x_train, y_train)

            val_scores = clf.decision_function(x_val)

            for thr in thresholds:
                val_pred = (val_scores > thr).astype(int)
                profit = trade_profit_from_preds(
                    y_val,
                    val_pred,
                    take_profit=cfg.take_profit,
                    stop_loss=cfg.stop_loss,
                    cost=cfg.trade_cost,
                )

                if profit > best_profit:
                    best_profit = profit
                    best_clf = clf
                    best_threshold = thr
                    best_params = {"C": c, "class_weight": cw}

    print("\nBest validation setup:")
    print(best_params)
    print(f"Best threshold: {best_threshold}")
    print(f"Best validation profit: {100 * best_profit:.2f}%")

    test_scores = best_clf.decision_function(x_test)
    y_pred = (test_scores > best_threshold).astype(int)

    evaluate_and_print_trade_zhongjie(
        "Linear_SVM",
        y_test,
        y_pred,
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        cost=cfg.trade_cost,
    )

if __name__ == "__main__":
    main()