"""Print Pearson correlation of each feature with the target (z-scored train split)."""

from experiments.orion.config import DEFAULT_ORION_CONFIG
from experiments.orion.elib import (
    load_orion_training_frames,
    print_feature_target_correlations,
    print_orion_run_preamble,
)


def main() -> None:
    cfg = DEFAULT_ORION_CONFIG
    print_orion_run_preamble(cfg, trade_cost=cfg.trade_cost, run_title="Feature correlation")

    train_df, _val_df, _test_df = load_orion_training_frames(
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

    print_feature_target_correlations(train_df, target_column="target", top_n=None)


if __name__ == "__main__":
    main()
