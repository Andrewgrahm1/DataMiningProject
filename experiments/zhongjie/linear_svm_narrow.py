from sklearn.svm import LinearSVC
import numpy as np
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

    c_values = [0.03, 0.05, 0.07, 0.1]
    thresholds = [0.55, 0.56, 0.57, 0.58, 0.60]

    min_validation_trades = 20

    best_score = None
    best_profit = None
    best_clf = None
    best_threshold = None
    best_params = None
    best_trade_count = None

    records = []

    for c in c_values:
        clf = LinearSVC(
            C=c,
            class_weight=None,
            random_state=42,
            max_iter=5000,
        )

        print(f"Training LinearSVC with C={c}")
        clf.fit(x_train, y_train)

        val_scores = clf.decision_function(x_val)
        test_scores = clf.decision_function(x_test)

        score_min = float(val_scores.min())
        score_median = float(np.median(val_scores))
        score_max = float(val_scores.max())

        print(
            f"  score range: min={score_min:.4f}, "
            f"median={score_median:.4f}, max={score_max:.4f}"
        )

        for thr in thresholds:
            val_pred = (val_scores > thr).astype(int)
            test_pred = (test_scores > thr).astype(int)

            trade_count = int(val_pred.sum())
            skipped = trade_count < min_validation_trades

            if skipped:
                print(
                    f"  threshold={thr}, trades={trade_count}, skipped "
                    f"(below min_validation_trades={min_validation_trades})"
                )

                val_profit = None
                val_metrics = {
                    "tp": None,
                    "fp": None,
                    "fn": None,
                    "tn": None,
                    "trades_taken": trade_count,
                    "precision": None,
                    "recall": None,
                    "accuracy": None,
                }

                test_profit = trade_profit_from_preds(
                    y_test,
                    test_pred,
                    take_profit=cfg.take_profit,
                    stop_loss=cfg.stop_loss,
                    cost=cfg.trade_cost,
                )
                test_metrics = basic_metrics_from_preds(y_test, test_pred)

            else:
                val_profit = trade_profit_from_preds(
                    y_val,
                    val_pred,
                    take_profit=cfg.take_profit,
                    stop_loss=cfg.stop_loss,
                    cost=cfg.trade_cost,
                )
                val_metrics = basic_metrics_from_preds(y_val, val_pred)

                test_profit = trade_profit_from_preds(
                    y_test,
                    test_pred,
                    take_profit=cfg.take_profit,
                    stop_loss=cfg.stop_loss,
                    cost=cfg.trade_cost,
                )
                test_metrics = basic_metrics_from_preds(y_test, test_pred)

                print(
                    f"  threshold={thr}, trades={trade_count}, "
                    f"validation_profit={100 * val_profit:.2f}%"
                )

                candidate_score = (
                    val_metrics["precision"],
                    val_profit,
                    -trade_count,
                )

                if best_score is None or candidate_score > best_score:
                    best_score = candidate_score
                    best_profit = val_profit
                    best_clf = clf
                    best_threshold = thr
                    best_params = {"C": c}
                    best_trade_count = trade_count

            records.append({
                "C": c,
                "threshold": thr,
                "score_min": score_min,
                "score_median": score_median,
                "score_max": score_max,
                "skipped": skipped,

                "validation_trades": trade_count,
                "validation_profit_pct": None if val_profit is None else 100 * val_profit,
                "validation_precision_pct": None if val_metrics["precision"] is None else 100 * val_metrics["precision"],
                "validation_recall_pct": None if val_metrics["recall"] is None else 100 * val_metrics["recall"],
                "validation_accuracy_pct": None if val_metrics["accuracy"] is None else 100 * val_metrics["accuracy"],
                "validation_tp": val_metrics["tp"],
                "validation_fp": val_metrics["fp"],
                "validation_fn": val_metrics["fn"],
                "validation_tn": val_metrics["tn"],

                "test_trades": test_metrics["trades_taken"],
                "test_profit_pct": 100 * test_profit,
                "test_precision_pct": 100 * test_metrics["precision"],
                "test_recall_pct": 100 * test_metrics["recall"],
                "test_accuracy_pct": 100 * test_metrics["accuracy"],
                "test_tp": test_metrics["tp"],
                "test_fp": test_metrics["fp"],
                "test_fn": test_metrics["fn"],
                "test_tn": test_metrics["tn"],
            })

    if best_clf is None:
        raise RuntimeError(
            "No valid LinearSVC setup found. Try lowering min_validation_trades "
            "or widening the threshold search."
        )

    results_df = pd.DataFrame(records)

    ranked_df = results_df[results_df["skipped"] == False].copy()
    ranked_df = ranked_df.sort_values(
        by=["validation_precision_pct", "validation_profit_pct", "validation_trades"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    test_best_idx = results_df["test_profit_pct"].idxmax()
    test_best_row = results_df.loc[test_best_idx]

    print("\n=== Top validation results (non-skipped) ===")
    print(ranked_df.head(15).to_string(index=False))

    print("\n=== Best observed test row ===")
    print(test_best_row.to_frame().T.to_string(index=False))

    print("\n=== All searched combinations ===")
    print(results_df.to_string(index=False))

    output_path = "././docs/Zhongjie_Results/linear_svm_final_narrow_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved full validation + test table to {output_path}")

    print("\nBest validation-selected setup:")
    print(best_params)
    print(f"Best threshold: {best_threshold}")
    print(f"Best validation trades: {best_trade_count}")
    print(f"Best validation profit: {100 * best_profit:.2f}%")

    test_scores = best_clf.decision_function(x_test)
    y_pred = (test_scores > best_threshold).astype(int)

    print("\n=== Final test result for validation-selected setup ===")
    evaluate_and_print_trade_zhongjie(
        "Linear_SVM_final_narrow",
        y_test,
        y_pred,
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        cost=cfg.trade_cost,
    )


if __name__ == "__main__":
    main()