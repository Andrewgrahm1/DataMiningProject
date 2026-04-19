from sklearn.svm import LinearSVC
import numpy as np

from experiments.orion.config import DEFAULT_ORION_CONFIG
from experiments.zhongjie.elib import (
    load_zhongjie_training_frames,
    evaluate_and_print_trade_zhongjie,
)


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

    c_values = [0.01, 0.02, 0.05]
    class_weights = [None]
    thresholds = [0.45, 0.48, 0.50, 0.52, 0.55]

    min_validation_trades = 20

    best_profit = float("-inf")
    best_clf = None
    best_threshold = 0.5
    best_params = None
    best_trade_count = 0

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
            print(
                f"  score range: min={val_scores.min():.4f}, "
                f"median={np.median(val_scores):.4f}, max={val_scores.max():.4f}"
            )

            for thr in thresholds:
                val_pred = (val_scores > thr).astype(int)
                trade_count = int(val_pred.sum())

                if trade_count < min_validation_trades:
                    print(
                        f"  threshold={thr}, trades={trade_count}, skipped "
                        f"(below min_validation_trades={min_validation_trades})"
                    )
                    continue

                profit = trade_profit_from_preds(
                    y_val,
                    val_pred,
                    take_profit=cfg.take_profit,
                    stop_loss=cfg.stop_loss,
                    cost=cfg.trade_cost,
                )

                print(
                    f"  threshold={thr}, trades={trade_count}, "
                    f"validation_profit={100 * profit:.2f}%"
                )

                if profit > best_profit:
                    best_profit = profit
                    best_clf = clf
                    best_threshold = thr
                    best_params = {"C": c, "class_weight": cw}
                    best_trade_count = trade_count

    if best_clf is None:
        raise RuntimeError(
            "No valid LinearSVC setup found. Try lowering min_validation_trades "
            "or widening the threshold search."
        )

    print("\nBest validation setup:")
    print(best_params)
    print(f"Best threshold: {best_threshold}")
    print(f"Best validation trades: {best_trade_count}")
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