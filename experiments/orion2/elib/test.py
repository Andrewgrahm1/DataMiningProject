"""Unit tests for ``experiments.orion2.elib.elib`` (split, z-score, generic features, move target)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.orion2.elib.elib import (
    add_feature_close_ema_pct_diff,
    add_feature_close_sma_pct_diff,
    add_forward_absolute_move_target,
    add_forward_move_eligibility_and_direction,
    add_volume_roll_mean_by_day,
    split_training_data,
    zscore_feature_splits,
)


def _ohlc(symbol: str, rows):
    """
    Build DataFrame with MultiIndex (symbol, timestamp) and OHLC columns.

    rows: sequence of (timestamp, open, high, low, close) per bar.
    """
    if not rows:
        index = pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"])
        return pd.DataFrame(columns=["open", "high", "low", "close"], index=index)
    timestamps = pd.to_datetime([r[0] for r in rows])
    opens = [r[1] for r in rows]
    highs = [r[2] for r in rows]
    lows = [r[3] for r in rows]
    closes = [r[4] for r in rows]
    index = pd.MultiIndex.from_arrays(
        [[symbol] * len(rows), timestamps],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=index,
    )


def _ohlcv(symbol: str, rows):
    """
    Build DataFrame with MultiIndex (symbol, timestamp), OHLC, and volume.

    rows: sequence of (timestamp, open, high, low, close, volume) per bar.
    """
    if not rows:
        index = pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"])
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=index,
        )
    timestamps = pd.to_datetime([r[0] for r in rows])
    opens = [r[1] for r in rows]
    highs = [r[2] for r in rows]
    lows = [r[3] for r in rows]
    closes = [r[4] for r in rows]
    vols = [r[5] for r in rows]
    index = pd.MultiIndex.from_arrays(
        [[symbol] * len(rows), timestamps],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=index,
    )


class TestAddFeatureCloseSmaPctDiff(unittest.TestCase):
    """Tests for add_feature_close_sma_pct_diff (same semantics as ``experiments.orion`` elib)."""

    def test_period_two_hand_computed(self):
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 101, 99, 100),
                ("2025-01-02 09:33", 100, 102, 99, 110),
            ],
        )
        result = add_feature_close_sma_pct_diff(df, [2])
        self.assertIn("close_sma_2_pct_diff", result.columns)
        self.assertEqual(result["close_sma_2_pct_diff"].iloc[0], 0.0)
        self.assertEqual(result["close_sma_2_pct_diff"].iloc[1], 0.0)
        self.assertAlmostEqual(
            float(result["close_sma_2_pct_diff"].iloc[2]), (110.0 - 105.0) / 105.0
        )

    def test_period_one_is_always_zero(self):
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 101, 99, 105),
            ],
        )
        result = add_feature_close_sma_pct_diff(df, [1])
        self.assertEqual(float(result["close_sma_1_pct_diff"].iloc[0]), 0.0)
        self.assertEqual(float(result["close_sma_1_pct_diff"].iloc[1]), 0.0)

    def test_multiple_symbols_independent(self):
        df_aapl = _ohlc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 101, 99, 110),
            ],
        )
        df_googl = _ohlc(
            "GOOGL",
            [
                ("2025-01-02 09:31", 50, 51, 49, 50),
                ("2025-01-02 09:32", 50, 51, 49, 60),
            ],
        )
        df = pd.concat([df_aapl, df_googl])
        result = add_feature_close_sma_pct_diff(df, [2])
        self.assertAlmostEqual(float(result["close_sma_2_pct_diff"].iloc[1]), (110 - 105) / 105)
        self.assertAlmostEqual(float(result["close_sma_2_pct_diff"].iloc[3]), (60 - 55) / 55)

    def test_empty_period_list_returns_unchanged(self):
        df = _ohlc(
            "AAPL",
            [("2025-01-02 09:31", 100, 101, 99, 100)],
        )
        result = add_feature_close_sma_pct_diff(df, [])
        self.assertEqual(list(result.columns), list(df.columns))

    def test_period_zero_raises(self):
        df = _ohlc("AAPL", [("2025-01-02 09:31", 100, 101, 99, 100)])
        with self.assertRaises(ValueError):
            add_feature_close_sma_pct_diff(df, [0])

    def test_custom_column_name_fn(self):
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 101, 99, 100),
            ],
        )
        result = add_feature_close_sma_pct_diff(
            df, [2], column_name_fn=lambda p: f"sma_gap_{p}"
        )
        self.assertIn("sma_gap_2", result.columns)
        self.assertNotIn("close_sma_2_pct_diff", result.columns)

    def test_multiple_periods_match_sequential_calls(self):
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 101, 99, 101),
                ("2025-01-02 09:33", 100, 102, 99, 102),
            ],
        )
        single = add_feature_close_sma_pct_diff(df.copy(), [2])
        single = add_feature_close_sma_pct_diff(single, [3])
        batch = add_feature_close_sma_pct_diff(df.copy(), [2, 3])
        pd.testing.assert_series_equal(
            single["close_sma_2_pct_diff"], batch["close_sma_2_pct_diff"]
        )
        pd.testing.assert_series_equal(
            single["close_sma_3_pct_diff"], batch["close_sma_3_pct_diff"]
        )


class TestSplitTrainingData(unittest.TestCase):
    """Tests for chronological ``split_training_data``."""

    def _frame(self, n: int) -> pd.DataFrame:
        idx = pd.MultiIndex.from_arrays(
            [["X"] * n, pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")],
            names=["symbol", "timestamp"],
        )
        return pd.DataFrame({"a": np.arange(n, dtype=float), "target": np.zeros(n)}, index=idx)

    def test_three_contiguous_blocks_ordered_by_time(self):
        df = self._frame(100)
        tr, va, te = split_training_data(df, validation_fraction=0.15, test_fraction=0.2)
        self.assertEqual(len(tr), 65)
        self.assertEqual(len(va), 15)
        self.assertEqual(len(te), 20)
        t_tr = tr.index.get_level_values("timestamp")
        t_va = va.index.get_level_values("timestamp")
        t_te = te.index.get_level_values("timestamp")
        self.assertTrue(t_tr.max() < t_va.min())
        self.assertTrue(t_va.max() < t_te.min())

    def test_too_small_raises(self):
        # n=10, val=5, test=5 -> train=0 must raise.
        df = self._frame(10)
        with self.assertRaises(ValueError):
            split_training_data(df, validation_fraction=0.5, test_fraction=0.5)


class TestZscoreFeatureSplits(unittest.TestCase):
    """Tests for ``zscore_feature_splits`` (fit on train only)."""

    def test_excluded_columns_untouched(self):
        idx = pd.MultiIndex.from_arrays(
            [["S"] * 3, pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")],
            names=["symbol", "timestamp"],
        )
        train = pd.DataFrame({"f": [0.0, 2.0, 4.0], "target": [0, 1, 0]}, index=idx)
        val = pd.DataFrame({"f": [1.0], "target": [0]}, index=idx[:1])
        test = pd.DataFrame({"f": [3.0], "target": [1]}, index=idx[1:2])
        tr2, va2, te2 = zscore_feature_splits(
            train, val, test, non_feature_columns=("target",)
        )
        pd.testing.assert_series_equal(tr2["target"], train["target"])
        self.assertAlmostEqual(float(tr2["f"].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(tr2["f"].std(ddof=0)), 1.0, places=6)


class TestAddVolumeRollMeanByDay(unittest.TestCase):
    """Tests for per-session rolling mean volume (orion2 elib)."""

    def test_window_two_within_day(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100, 10),
                ("2025-01-02 09:32", 100, 101, 99, 100, 30),
                ("2025-01-02 09:33", 100, 101, 99, 100, 50),
            ],
        )
        result = add_volume_roll_mean_by_day(df, [2])
        self.assertEqual(float(result["volume_roll_mean_2"].iloc[0]), 0.0)
        self.assertAlmostEqual(float(result["volume_roll_mean_2"].iloc[1]), 20.0)
        self.assertAlmostEqual(float(result["volume_roll_mean_2"].iloc[2]), 40.0)

    def test_new_day_resets_window(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 16:00", 100, 101, 99, 100, 100),
                ("2025-01-03 09:31", 100, 101, 99, 100, 10),
                ("2025-01-03 09:32", 100, 101, 99, 100, 20),
            ],
        )
        result = add_volume_roll_mean_by_day(df, [2])
        self.assertEqual(float(result["volume_roll_mean_2"].iloc[1]), 0.0)
        self.assertAlmostEqual(float(result["volume_roll_mean_2"].iloc[2]), 15.0)

    def test_empty_window_list_returns_unchanged(self):
        df = _ohlcv("AAPL", [("2025-01-02 09:31", 100, 101, 99, 100, 1)])
        result = add_volume_roll_mean_by_day(df, [])
        self.assertEqual(list(result.columns), list(df.columns))

    def test_window_zero_raises(self):
        df = _ohlcv("AAPL", [("2025-01-02 09:31", 100, 101, 99, 100, 1)])
        with self.assertRaises(ValueError):
            add_volume_roll_mean_by_day(df, [0])


class TestAddFeatureCloseEmaPctDiff(unittest.TestCase):
    """Tests for ``add_feature_close_ema_pct_diff``."""

    def test_span_one_matches_close_zero_diff(self):
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 101, 99, 110),
            ],
        )
        result = add_feature_close_ema_pct_diff(df, [1])
        self.assertAlmostEqual(float(result["close_ema_1_pct_diff"].iloc[0]), 0.0)
        self.assertAlmostEqual(float(result["close_ema_1_pct_diff"].iloc[1]), 0.0)

    def test_empty_span_list_returns_unchanged(self):
        df = _ohlc("AAPL", [("2025-01-02 09:31", 100, 101, 99, 100)])
        result = add_feature_close_ema_pct_diff(df, [])
        self.assertEqual(list(result.columns), list(df.columns))

    def test_span_zero_raises(self):
        df = _ohlc("AAPL", [("2025-01-02 09:31", 100, 101, 99, 100)])
        with self.assertRaises(ValueError):
            add_feature_close_ema_pct_diff(df, [0])


class TestAddForwardAbsoluteMoveTarget(unittest.TestCase):
    """Tests for ``add_forward_absolute_move_target``."""

    def test_up_move_triggers(self):
        # close 100; next bar high 104 -> 4% up >= 3%
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 105, 99, 102),
            ],
        )
        out = add_forward_absolute_move_target(df.copy(), 0.03, 5, column_name="t")
        self.assertEqual(int(out["t"].iloc[0]), 1)
        self.assertEqual(int(out["t"].iloc[1]), 0)

    def test_down_move_triggers(self):
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 101, 94, 98),
            ],
        )
        out = add_forward_absolute_move_target(df.copy(), 0.03, 5, column_name="t")
        self.assertEqual(int(out["t"].iloc[0]), 1)

    def test_no_forward_bars_is_zero(self):
        df = _ohlc(
            "AAPL",
            [("2025-01-02 09:31", 100, 101, 99, 100)],
        )
        out = add_forward_absolute_move_target(df.copy(), 0.01, 3, column_name="t")
        self.assertEqual(int(out["t"].iloc[0]), 0)

    def test_session_boundary_no_next_day(self):
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 16:00", 100, 101, 99, 100),
                ("2025-01-03 09:31", 100, 110, 99, 105),
            ],
        )
        out = add_forward_absolute_move_target(df.copy(), 0.05, 10, column_name="t")
        self.assertEqual(int(out["t"].iloc[0]), 0)

    def test_negative_threshold_raises(self):
        df = _ohlc("AAPL", [("2025-01-02 09:31", 100, 101, 99, 100)])
        with self.assertRaises(ValueError):
            add_forward_absolute_move_target(df, -0.01, 2)

    def test_zero_forward_bars_raises(self):
        df = _ohlc("AAPL", [("2025-01-02 09:31", 100, 101, 99, 100)])
        with self.assertRaises(ValueError):
            add_forward_absolute_move_target(df, 0.01, 0)


class TestAddForwardMoveEligibilityAndDirection(unittest.TestCase):
    """Tests for ``add_forward_move_eligibility_and_direction``."""

    def test_up_only_eligible_direction_one(self):
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 105, 99, 102),
            ],
        )
        out = add_forward_move_eligibility_and_direction(df.copy(), 0.03, 5)
        self.assertEqual(int(out["move_eligible"].iloc[0]), 1)
        self.assertEqual(int(out["direction_target"].iloc[0]), 1)

    def test_down_only_eligible_direction_zero(self):
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 101, 94, 98),
            ],
        )
        out = add_forward_move_eligibility_and_direction(df.copy(), 0.03, 5)
        self.assertEqual(int(out["move_eligible"].iloc[0]), 1)
        self.assertEqual(int(out["direction_target"].iloc[0]), 0)

    def test_no_move_ineligible(self):
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 101, 99, 100),
            ],
        )
        out = add_forward_move_eligibility_and_direction(df.copy(), 0.05, 5)
        self.assertEqual(int(out["move_eligible"].iloc[0]), 0)
        self.assertTrue(np.isnan(out["direction_target"].iloc[0]))

    def test_up_hits_before_down(self):
        # Bar1: up 10% >= 3% at k=0; bar2: running min reaches 12% down first at k=1 → tu=0 < td=1 → up wins
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 110, 99, 105),
                ("2025-01-02 09:33", 100, 102, 88, 95),
            ],
        )
        out = add_forward_move_eligibility_and_direction(df.copy(), 0.03, 5)
        self.assertEqual(int(out["move_eligible"].iloc[0]), 1)
        self.assertEqual(int(out["direction_target"].iloc[0]), 1)

    def test_matches_absolute_move_eligibility(self):
        df = _ohlc(
            "AAPL",
            [
                ("2025-01-02 16:00", 100, 101, 99, 100),
                ("2025-01-03 09:31", 100, 110, 99, 105),
            ],
        )
        abs_t = add_forward_absolute_move_target(df.copy(), 0.05, 10, column_name="t")
        dir_t = add_forward_move_eligibility_and_direction(df.copy(), 0.05, 10)
        self.assertEqual(int(dir_t["move_eligible"].iloc[0]), int(abs_t["t"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
