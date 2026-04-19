"""Unit tests for common dataframe feature-engineering helpers."""

import unittest

import pandas as pd

from lib.common.common import (
    add_feature_bars_since_open,
    add_feature_bars_until_close,
    add_feature_pct_change,
    add_feature_pct_change_batch,
    calculate_min_win_rate,
    create_target_column,
)


def _ohlcv(symbol: str, rows):
    """
    Build DataFrame with MultiIndex (symbol, timestamp) and OHLC columns.

    rows: sequence of (timestamp, open, high, low, close) per bar.
    timestamp can be a string (e.g. "2025-01-02 09:31") or datetime-like.
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


class TestCreateTargetColumn(unittest.TestCase):
    """Tests for create_target_column."""

    def test_take_profit_hit_before_stop_loss(self):
        # Bar 0: entry = bar 1's high = 102 -> TP 103.02, SL 100.98. Bar 2: high 103.1 hits TP.
        # Need 4+ bars so bar 0 is not in last 3.
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100.5, 99.5, 100),  # bar 0
                ("2025-01-02 09:32", 101, 102, 99.5, 101),  # bar 1: entry high=102
                ("2025-01-02 09:33", 102, 103.1, 101, 102),  # bar 2: hits TP 103.02
                ("2025-01-02 09:34", 102, 103, 102, 102),  # bar 3
            ],
        )
        result = create_target_column(df, take_profit=0.01, stop_loss=0.01)
        self.assertEqual(result["target"].iloc[0], 1)
        # Last 3 bars of day are always 0
        self.assertEqual(result["target"].iloc[1], 0)
        self.assertEqual(result["target"].iloc[2], 0)
        self.assertEqual(result["target"].iloc[3], 0)

    def test_stop_loss_hit_before_take_profit(self):
        # Bar 0: entry = bar 1's high = 100 -> TP 101, SL 99. Bar 2: low 98 hits SL.
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100.5, 99.5, 100),  # bar 0
                ("2025-01-02 09:32", 100, 100, 99.5, 100),  # bar 1: entry high=100
                ("2025-01-02 09:33", 100, 100.5, 98, 99),  # bar 2: low 98 hits SL 99
                ("2025-01-02 09:34", 100, 100.5, 99, 100),  # bar 3
            ],
        )
        result = create_target_column(df, take_profit=0.01, stop_loss=0.01)
        self.assertEqual(result["target"].iloc[0], 0)

    def test_eod_without_hit(self):
        # Bar 0: entry = bar 1's high = 100.5. Bars 2,3 stay in range, no TP 101.505 or SL 99.495.
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100.5, 99.5, 100),  # bar 0
                ("2025-01-02 09:32", 100, 100.5, 99.5, 100),  # bar 1: entry high=100.5
                ("2025-01-02 09:33", 100, 100.5, 99.5, 100),  # bar 2, 3: no TP or SL
                ("2025-01-02 09:34", 100, 100.5, 99.5, 100),
            ],
        )
        result = create_target_column(df, take_profit=0.01, stop_loss=0.01)
        self.assertEqual(result["target"].iloc[0], 0)

    def test_same_bar_both_tp_and_sl_treated_as_stop_loss(self):
        # Bar 0: entry = bar 1's high = 102 -> TP 103.02, SL 100.98. Bar 2: high>=TP and low<=SL -> SL.
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100.5, 99.5, 100),  # bar 0
                ("2025-01-02 09:32", 100, 102, 99, 101),  # bar 1: entry high=102
                ("2025-01-02 09:33", 101, 103.1, 100, 101),  # bar 2: both TP and SL in range -> SL
                ("2025-01-02 09:34", 101, 103, 101, 102),  # bar 3
            ],
        )
        result = create_target_column(df, take_profit=0.01, stop_loss=0.01)
        self.assertEqual(result["target"].iloc[0], 0)

    def test_take_profit_hit_on_later_bar(self):
        # Bar 0: entry = bar 1's high = 100.5 -> TP 101.505. Bar 2 no hit, bar 3 high=102.02 hits TP.
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100.5, 99.5, 100),  # bar 0
                ("2025-01-02 09:32", 100, 100.5, 99.5, 100),  # bar 1: entry high=100.5
                ("2025-01-02 09:33", 100.5, 101, 100, 100.5),  # bar 2
                ("2025-01-02 09:34", 101, 101.5, 100.5, 101),  # bar 3: high 102.02 hits TP
                ("2025-01-02 09:35", 101, 102.02, 101, 101),  # bar 4
            ],
        )
        result = create_target_column(df, take_profit=0.01, stop_loss=0.01)
        self.assertEqual(result["target"].iloc[0], 1)  # entry=100.5, bar 3 high 102.02 >= 101.505
        self.assertEqual(result["target"].iloc[1], 1)  # bar 1 entry=101, bar 3 high 102.02 >= 102.01
        # Last 3 bars always 0
        self.assertEqual(result["target"].iloc[2], 0)
        self.assertEqual(result["target"].iloc[3], 0)
        self.assertEqual(result["target"].iloc[4], 0)

    def test_single_bar_in_day_has_no_future_bars(self):
        df = _ohlcv("AAPL", [("2025-01-02 09:31", 100, 101, 99, 100)])
        result = create_target_column(df, take_profit=0.01, stop_loss=0.01)
        self.assertEqual(result["target"].iloc[0], 0)

    def test_last_bar_of_day_target_zero(self):
        # With 2 bars, both are in last 3 -> target 0
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 101, 99, 100),
                ("2025-01-02 09:32", 100, 101, 99, 100),
            ],
        )
        result = create_target_column(df, take_profit=0.01, stop_loss=0.01)
        self.assertEqual(result["target"].iloc[0], 0)
        self.assertEqual(result["target"].iloc[1], 0)

    def test_entry_zero_target_zero(self):
        # Bar 0: entry = bar 1's high = 0 -> target 0. Need 4 bars so bar 0 is not in last 3.
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 1, 1, 0, 1),
                ("2025-01-02 09:32", 0, 0, 0, 0),  # bar 1: entry high=0
                ("2025-01-02 09:33", 1, 1, 0, 1),
                ("2025-01-02 09:34", 1, 1, 0, 1),
            ],
        )
        result = create_target_column(df, take_profit=0.01, stop_loss=0.01)
        self.assertEqual(result["target"].iloc[0], 0)

    def test_does_not_look_past_end_of_day(self):
        # Day 1: only 1 bar -> target 0. Day 2: 4 bars; bar 0 entry = bar 1's high, bar 2 hits TP.
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 15:59", 100, 100.5, 99.5, 100),  # day 1: only bar
                ("2025-01-03 09:31", 100, 102, 99, 101),  # day 2 bar 0
                ("2025-01-03 09:32", 101, 102, 101, 101),  # day 2 bar 1: entry high=102
                ("2025-01-03 09:33", 102, 103.02, 102, 102),  # day 2 bar 2: hits TP
                ("2025-01-03 09:34", 102, 103, 102, 102),  # day 2 bar 3
            ],
        )
        result = create_target_column(df, take_profit=0.01, stop_loss=0.01)
        self.assertEqual(result["target"].iloc[0], 0)  # day 1, only bar
        self.assertEqual(result["target"].iloc[1], 1)  # day 2 bar 0: entry=102, bar 2 hits TP
        self.assertEqual(result["target"].iloc[2], 0)  # last 3 of day 2
        self.assertEqual(result["target"].iloc[3], 0)
        self.assertEqual(result["target"].iloc[4], 0)

    def test_multiple_symbols_independent(self):
        # AAPL: 4 bars, bar 0 entry=high[1], bar 2 hits TP. GOOGL: 4 bars, bar 0 entry=high[1], bar 2 hits SL.
        df_aapl = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100.5, 99.5, 100),  # bar 0
                ("2025-01-02 09:32", 101, 102, 99.5, 101),  # bar 1: entry high=102
                ("2025-01-02 09:33", 102, 103.02, 101, 102),  # bar 2: hits TP
                ("2025-01-02 09:34", 102, 103, 102, 102),  # bar 3
            ],
        )
        df_googl = _ohlcv(
            "GOOGL",
            [
                ("2025-01-02 09:31", 200, 200.5, 199.5, 200),  # bar 0
                ("2025-01-02 09:32", 200, 200, 199, 200),  # bar 1: entry high=200
                ("2025-01-02 09:33", 199, 199.5, 197, 198),  # bar 2: low 197 hits SL 198
                ("2025-01-02 09:34", 198, 198, 197, 198),  # bar 3
            ],
        )
        df = pd.concat([df_aapl, df_googl])
        result = create_target_column(df, take_profit=0.01, stop_loss=0.01)
        self.assertEqual(result["target"].iloc[0], 1)  # AAPL bar 0 -> TP at bar 2
        self.assertEqual(result["target"].iloc[4], 0)  # GOOGL bar 0 -> SL at bar 2

    def test_last_three_bars_of_day_always_zero(self):
        # 5 bars: last 3 by index are 2, 3, 4 -> always target 0. Bar 0 hits TP at bar 2. Bar 1 no TP.
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100.5, 99.5, 100),  # bar 0
                ("2025-01-02 09:32", 101, 102, 99.5, 101),  # bar 1: entry high=102
                ("2025-01-02 09:33", 102, 103.02, 102, 102),  # bar 2: hits TP for bar 0
                ("2025-01-02 09:34", 103, 104, 103, 103),  # bar 3: high 104 < 104.05
                ("2025-01-02 09:35", 104, 104, 104, 104),  # bar 4
            ],
        )
        result = create_target_column(df, take_profit=0.01, stop_loss=0.01)
        self.assertEqual(result["target"].iloc[0], 1)
        self.assertEqual(result["target"].iloc[1], 0)
        self.assertEqual(result["target"].iloc[2], 0, "bar 2 is in last 3")
        self.assertEqual(result["target"].iloc[3], 0, "bar 3 is in last 3")
        self.assertEqual(result["target"].iloc[4], 0, "bar 4 is in last 3")

    def test_max_bars_after_entry_limits_when_tp_later(self):
        # Same path as test_take_profit_hit_on_later_bar: TP first at bar 4 (index 4).
        # Exit window starts at bar 2; need bars 2,3,4 -> max_bars_after_entry must be >= 3.
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100.5, 99.5, 100),  # bar 0
                ("2025-01-02 09:32", 100, 100.5, 99.5, 100),  # bar 1: entry high=100.5
                ("2025-01-02 09:33", 100.5, 101, 100, 100.5),  # bar 2
                ("2025-01-02 09:34", 101, 101.5, 100.5, 101),  # bar 3
                ("2025-01-02 09:35", 101, 102.02, 101, 101),  # bar 4: TP
            ],
        )
        full = create_target_column(
            df.copy(), take_profit=0.01, stop_loss=0.01, max_bars_after_entry=None
        )
        capped = create_target_column(
            df.copy(), take_profit=0.01, stop_loss=0.01, max_bars_after_entry=2
        )
        self.assertEqual(full["target"].iloc[0], 1)
        self.assertEqual(capped["target"].iloc[0], 0)


class TestAddFeatureBarsUntilClose(unittest.TestCase):
    """Tests for add_feature_bars_until_close."""

    def test_single_day_four_bars(self):
        # 4 bars: first has 3 left, then 2, 1, last has 0.
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 100, 100, 99, 100),
                ("2025-01-02 09:33", 100, 100, 99, 100),
                ("2025-01-02 09:34", 100, 100, 99, 100),
            ],
        )
        result = add_feature_bars_until_close(df)
        self.assertEqual(result["bars_until_close"].iloc[0], 3)
        self.assertEqual(result["bars_until_close"].iloc[1], 2)
        self.assertEqual(result["bars_until_close"].iloc[2], 1)
        self.assertEqual(result["bars_until_close"].iloc[3], 0)

    def test_single_bar_in_day_is_zero(self):
        df = _ohlcv("AAPL", [("2025-01-02 09:31", 100, 100, 99, 100)])
        result = add_feature_bars_until_close(df)
        self.assertEqual(result["bars_until_close"].iloc[0], 0)

    def test_last_bar_of_day_is_zero(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 100, 100, 99, 100),
            ],
        )
        result = add_feature_bars_until_close(df)
        self.assertEqual(result["bars_until_close"].iloc[0], 1)
        self.assertEqual(result["bars_until_close"].iloc[1], 0)

    def test_multiple_days_counted_per_day(self):
        # Day 1: 1 bar -> 0. Day 2: 3 bars -> 2, 1, 0.
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 15:59", 100, 100, 99, 100),  # day 1 last bar
                ("2025-01-03 09:31", 100, 100, 99, 100),  # day 2 bar 0
                ("2025-01-03 09:32", 100, 100, 99, 100),  # day 2 bar 1
                ("2025-01-03 09:33", 100, 100, 99, 100),  # day 2 bar 2
            ],
        )
        result = add_feature_bars_until_close(df)
        self.assertEqual(result["bars_until_close"].iloc[0], 0)  # only bar in day 1
        self.assertEqual(result["bars_until_close"].iloc[1], 2)  # 2 bars left in day 2
        self.assertEqual(result["bars_until_close"].iloc[2], 1)
        self.assertEqual(result["bars_until_close"].iloc[3], 0)

    def test_multiple_symbols_independent(self):
        # AAPL 2 bars -> 1, 0. GOOGL 2 bars -> 1, 0.
        df_aapl = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 100, 100, 99, 100),
            ],
        )
        df_googl = _ohlcv(
            "GOOGL",
            [
                ("2025-01-02 09:31", 200, 200, 199, 200),
                ("2025-01-02 09:32", 200, 200, 199, 200),
            ],
        )
        df = pd.concat([df_aapl, df_googl])
        result = add_feature_bars_until_close(df)
        self.assertEqual(result["bars_until_close"].iloc[0], 1)
        self.assertEqual(result["bars_until_close"].iloc[1], 0)
        self.assertEqual(result["bars_until_close"].iloc[2], 1)
        self.assertEqual(result["bars_until_close"].iloc[3], 0)


class TestAddFeatureBarsSinceOpen(unittest.TestCase):
    """Tests for add_feature_bars_since_open."""

    def test_single_day_four_bars(self):
        # 4 bars: first has 0 before, then 1, 2, 3.
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 100, 100, 99, 100),
                ("2025-01-02 09:33", 100, 100, 99, 100),
                ("2025-01-02 09:34", 100, 100, 99, 100),
            ],
        )
        result = add_feature_bars_since_open(df)
        self.assertEqual(result["bars_since_open"].iloc[0], 0)
        self.assertEqual(result["bars_since_open"].iloc[1], 1)
        self.assertEqual(result["bars_since_open"].iloc[2], 2)
        self.assertEqual(result["bars_since_open"].iloc[3], 3)

    def test_first_bar_of_day_is_zero(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 100, 100, 99, 100),
            ],
        )
        result = add_feature_bars_since_open(df)
        self.assertEqual(result["bars_since_open"].iloc[0], 0)
        self.assertEqual(result["bars_since_open"].iloc[1], 1)

    def test_single_bar_in_day_is_zero(self):
        df = _ohlcv("AAPL", [("2025-01-02 09:31", 100, 100, 99, 100)])
        result = add_feature_bars_since_open(df)
        self.assertEqual(result["bars_since_open"].iloc[0], 0)

    def test_multiple_days_counted_per_day(self):
        # Day 1: 1 bar -> 0. Day 2: 3 bars -> 0, 1, 2.
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 15:59", 100, 100, 99, 100),  # day 1 only bar
                ("2025-01-03 09:31", 100, 100, 99, 100),  # day 2 bar 0
                ("2025-01-03 09:32", 100, 100, 99, 100),  # day 2 bar 1
                ("2025-01-03 09:33", 100, 100, 99, 100),  # day 2 bar 2
            ],
        )
        result = add_feature_bars_since_open(df)
        self.assertEqual(result["bars_since_open"].iloc[0], 0)
        self.assertEqual(result["bars_since_open"].iloc[1], 0)  # first bar of day 2
        self.assertEqual(result["bars_since_open"].iloc[2], 1)
        self.assertEqual(result["bars_since_open"].iloc[3], 2)

    def test_multiple_symbols_independent(self):
        df_aapl = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 100, 100, 99, 100),
            ],
        )
        df_googl = _ohlcv(
            "GOOGL",
            [
                ("2025-01-02 09:31", 200, 200, 199, 200),
                ("2025-01-02 09:32", 200, 200, 199, 200),
            ],
        )
        df = pd.concat([df_aapl, df_googl])
        result = add_feature_bars_since_open(df)
        self.assertEqual(result["bars_since_open"].iloc[0], 0)
        self.assertEqual(result["bars_since_open"].iloc[1], 1)
        self.assertEqual(result["bars_since_open"].iloc[2], 0)
        self.assertEqual(result["bars_since_open"].iloc[3], 1)


class TestAddFeaturePctChange(unittest.TestCase):
    """Tests for add_feature_pct_change."""

    def test_one_bar_back(self):
        # Close: 100, 102, 101. pct_change_1: 0, (102-100)/100=0.02, (101-102)/102≈-0.0098
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 101, 102, 100, 102),
                ("2025-01-02 09:33", 102, 102, 101, 101),
            ],
        )
        result = add_feature_pct_change(df, bars_back=1)
        self.assertIn("pct_change_1", result.columns)
        self.assertEqual(result["pct_change_1"].iloc[0], 0)
        self.assertEqual(result["pct_change_1"].iloc[1], 0.02)
        self.assertAlmostEqual(result["pct_change_1"].iloc[2], (101 - 102) / 102)

    def test_not_enough_bars_back_encoded_zero(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 100, 102, 99, 102),
            ],
        )
        result = add_feature_pct_change(df, bars_back=2)
        self.assertEqual(result["pct_change_2"].iloc[0], 0)
        self.assertEqual(result["pct_change_2"].iloc[1], 0)

    def test_two_bars_back(self):
        # 3 bars: closes 100, 102, 104. Bar 2 has pct_change_2 = (104-100)/100 = 0.04
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 101, 102, 100, 102),
                ("2025-01-02 09:33", 102, 104, 102, 104),
            ],
        )
        result = add_feature_pct_change(df, bars_back=2)
        self.assertEqual(result["pct_change_2"].iloc[0], 0)
        self.assertEqual(result["pct_change_2"].iloc[1], 0)
        self.assertEqual(result["pct_change_2"].iloc[2], 0.04)

    def test_multiple_days_no_lookback_across_days(self):
        # Day 1: close 100. Day 2: close 200, 202. pct_change_1 for day 2 bar 1 = (202-200)/200 = 0.01
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 15:59", 100, 100, 99, 100),
                ("2025-01-03 09:31", 200, 200, 199, 200),
                ("2025-01-03 09:32", 200, 202, 200, 202),
            ],
        )
        result = add_feature_pct_change(df, bars_back=1)
        self.assertEqual(result["pct_change_1"].iloc[0], 0)
        self.assertEqual(result["pct_change_1"].iloc[1], 0)
        self.assertEqual(result["pct_change_1"].iloc[2], 0.01)

    def test_multiple_symbols_independent(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 100, 100, 99, 101),
            ],
        )
        df2 = _ohlcv(
            "GOOGL",
            [
                ("2025-01-02 09:31", 50, 50, 49, 50),
                ("2025-01-02 09:32", 50, 50, 49, 55),
            ],
        )
        df = pd.concat([df, df2])
        result = add_feature_pct_change(df, bars_back=1)
        self.assertEqual(result["pct_change_1"].iloc[0], 0)
        self.assertEqual(result["pct_change_1"].iloc[1], 0.01)  # (101-100)/100
        self.assertEqual(result["pct_change_1"].iloc[2], 0)  # GOOGL first bar of day
        self.assertEqual(result["pct_change_1"].iloc[3], 0.1)  # (55-50)/50


class TestAddFeaturePctChangeBatch(unittest.TestCase):
    """Tests for add_feature_pct_change_batch."""

    def test_empty_bars_back_list_returns_unchanged(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 100, 102, 99, 102),
            ],
        )
        result = add_feature_pct_change_batch(df, [])
        self.assertEqual(list(result.columns), list(df.columns))
        pd.testing.assert_frame_equal(result[df.columns], df)

    def test_single_bars_back_matches_add_feature_pct_change(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 101, 102, 100, 102),
                ("2025-01-02 09:33", 102, 102, 101, 101),
            ],
        )
        single = add_feature_pct_change(df.copy(), bars_back=1)
        batch = add_feature_pct_change_batch(df.copy(), [1])
        self.assertIn("pct_change_1", batch.columns)
        pd.testing.assert_series_equal(single["pct_change_1"], batch["pct_change_1"])

    def test_multiple_bars_back_match_single_calls(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 101, 102, 100, 102),
                ("2025-01-02 09:33", 102, 104, 102, 104),
            ],
        )
        single = df.copy()
        single = add_feature_pct_change(single, bars_back=1)
        single = add_feature_pct_change(single, bars_back=2)
        batch = add_feature_pct_change_batch(df.copy(), [1, 2])
        pd.testing.assert_series_equal(single["pct_change_1"], batch["pct_change_1"])
        pd.testing.assert_series_equal(single["pct_change_2"], batch["pct_change_2"])

    def test_one_bar_back_values(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 101, 102, 100, 102),
                ("2025-01-02 09:33", 102, 102, 101, 101),
            ],
        )
        result = add_feature_pct_change_batch(df, [1])
        self.assertEqual(result["pct_change_1"].iloc[0], 0)
        self.assertEqual(result["pct_change_1"].iloc[1], 0.02)
        self.assertAlmostEqual(result["pct_change_1"].iloc[2], (101 - 102) / 102)

    def test_two_bars_back_values(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 101, 102, 100, 102),
                ("2025-01-02 09:33", 102, 104, 102, 104),
            ],
        )
        result = add_feature_pct_change_batch(df, [2])
        self.assertEqual(result["pct_change_2"].iloc[0], 0)
        self.assertEqual(result["pct_change_2"].iloc[1], 0)
        self.assertEqual(result["pct_change_2"].iloc[2], 0.04)

    def test_not_enough_bars_back_encoded_zero(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 100, 102, 99, 102),
            ],
        )
        result = add_feature_pct_change_batch(df, [2])
        self.assertEqual(result["pct_change_2"].iloc[0], 0)
        self.assertEqual(result["pct_change_2"].iloc[1], 0)

    def test_multiple_days_no_lookback_across_days(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 15:59", 100, 100, 99, 100),
                ("2025-01-03 09:31", 200, 200, 199, 200),
                ("2025-01-03 09:32", 200, 202, 200, 202),
            ],
        )
        result = add_feature_pct_change_batch(df, [1])
        self.assertEqual(result["pct_change_1"].iloc[0], 0)
        self.assertEqual(result["pct_change_1"].iloc[1], 0)
        self.assertEqual(result["pct_change_1"].iloc[2], 0.01)

    def test_multiple_symbols_independent(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 100, 100, 99, 101),
            ],
        )
        df2 = _ohlcv(
            "GOOGL",
            [
                ("2025-01-02 09:31", 50, 50, 49, 50),
                ("2025-01-02 09:32", 50, 50, 49, 55),
            ],
        )
        df = pd.concat([df, df2])
        result = add_feature_pct_change_batch(df, [1])
        self.assertEqual(result["pct_change_1"].iloc[0], 0)
        self.assertEqual(result["pct_change_1"].iloc[1], 0.01)
        self.assertEqual(result["pct_change_1"].iloc[2], 0)
        self.assertEqual(result["pct_change_1"].iloc[3], 0.1)

    def test_custom_column_name_fn(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 101, 102, 100, 102),
            ],
        )
        result = add_feature_pct_change_batch(df, [1], column_name_fn=lambda b: f"ret_{b}b")
        self.assertIn("ret_1b", result.columns)
        self.assertNotIn("pct_change_1", result.columns)
        self.assertEqual(result["ret_1b"].iloc[0], 0)
        self.assertEqual(result["ret_1b"].iloc[1], 0.02)

    def test_all_columns_present_and_index_unchanged(self):
        df = _ohlcv(
            "AAPL",
            [
                ("2025-01-02 09:31", 100, 100, 99, 100),
                ("2025-01-02 09:32", 100, 102, 99, 101),
                ("2025-01-02 09:33", 100, 102, 99, 103),
            ],
        )
        result = add_feature_pct_change_batch(df, [1, 2])
        self.assertIn("pct_change_1", result.columns)
        self.assertIn("pct_change_2", result.columns)
        self.assertEqual(list(result.index.names), ["symbol", "timestamp"])
        pd.testing.assert_index_equal(result.index, df.index)


class TestCalculateMinWinRate(unittest.TestCase):
    """Tests for calculate_min_win_rate."""

    def test_calculate_min_win_rate(self):
        self.assertEqual(calculate_min_win_rate(0.04, 0.02, 0.004), 0.4)
        self.assertEqual(calculate_min_win_rate(0.03, 0.02, 0.004), 0.48)

if __name__ == "__main__":
    unittest.main()
