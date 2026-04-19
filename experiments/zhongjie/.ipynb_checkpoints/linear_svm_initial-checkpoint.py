from sklearn.svm import LinearSVC
import pandas as pd

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


def basic_metrics_from_preds(y_true, y_pred) -> dict:
    tp_count = int(((y_true == 1) & (y_pred == 1)).sum())
    fp_count = int(((y_true == 0) & (y_pred == 1)).sum())
    fn_count = int(((y_true == 1) & (y_pred == 0)).sum())
    tn_count = int(((y_true == 0) & (y_pred == 0)).sum())

    trades_taken = tp_count + fp_count
    precision = tp_count / trades_taken if trades_taken else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) else 0.0
    accuracy = (tp_count + tn_count) / len(y_true) if len(y_true) else 0.0

    return {
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "tn": tn_count,
        "trades_taken": trades_taken,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


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

    records = []

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
            test_scores = clf.decision_function(x_test)

            for thr in thresholds:
                val_pred = (val_scores > thr).astype(int)
                test_pred = (test_scores > thr).astype(int)

                val_profit = trade_profit_from_preds(
                    y_val,
                    val_pred,
                    take_profit=cfg.take_profit,
                    stop_loss=cfg.stop_loss,
                    cost=cfg.trade_cost,
                )
                test_profit = trade_profit_from_preds(
                    y_test,
                    test_pred,
                    take_profit=cfg.take_profit,
                    stop_loss=cfg.stop_loss,
                    cost=cfg.trade_cost,
                )

                val_metrics = basic_metrics_from_preds(y_val, val_pred)
                test_metrics = basic_metrics_from_preds(y_test, test_pred)

                records.append({
                    "C": c,
                    "class_weight": str(cw),
                    "threshold": thr,

                    "validation_profit_pct": 100 * val_profit,
                    "validation_trades": val_metrics["trades_taken"],
                    "validation_precision_pct": 100 * val_metrics["precision"],
                    "validation_recall_pct": 100 * val_metrics["recall"],
                    "validation_accuracy_pct": 100 * val_metrics["accuracy"],
                    "validation_tp": val_metrics["tp"],
                    "validation_fp": val_metrics["fp"],
                    "validation_fn": val_metrics["fn"],
                    "validation_tn": val_metrics["tn"],

                    "test_profit_pct": 100 * test_profit,
                    "test_trades": test_metrics["trades_taken"],
                    "test_precision_pct": 100 * test_metrics["precision"],
                    "test_recall_pct": 100 * test_metrics["recall"],
                    "test_accuracy_pct": 100 * test_metrics["accuracy"],
                    "test_tp": test_metrics["tp"],
                    "test_fp": test_metrics["fp"],
                    "test_fn": test_metrics["fn"],
                    "test_tn": test_metrics["tn"],
                })

                if val_profit > best_profit:
                    best_profit = val_profit
                    best_clf = clf
                    best_threshold = thr
                    best_params = {"C": c, "class_weight": cw}

    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values(
        by=["validation_profit_pct", "validation_precision_pct", "validation_trades"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    print("\n=== Top validation results (sorted) ===")
    print(results_df.head(15).to_string(index=False))

    output_path = "././docs/Zhongjie_Results/linear_svm_validation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved full validation + test table to {output_path}")

    print("\nBest validation setup:")
    print(best_params)
    print(f"Best threshold: {best_threshold}")
    print(f"Best validation profit: {100 * best_profit:.2f}%")

    test_scores = best_clf.decision_function(x_test)
    y_pred = (test_scores > best_threshold).astype(int)

    print("\n=== Final test result ===")
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