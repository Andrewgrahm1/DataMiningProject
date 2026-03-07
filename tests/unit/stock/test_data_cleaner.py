"""
Unit tests for StockDataCleaner.
"""

import unittest
import pandas as pd

from alpaca.data.timeframe import TimeFrame

from lib.stock.data_cleaner import StockDataCleaner


class TestStockDataCleaner(unittest.TestCase):
    """Test StockDataCleaner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cleaner = StockDataCleaner()

    def _create_stock_dataframe(self, symbols=None, timestamps=None, **column_overrides):
        """
        Helper to create a DataFrame with MultiIndex (symbol, timestamp) and OHLCV columns.

        Args:
            symbols: List of symbol strings (default: ['AAPL', 'AAPL', 'AAPL'])
            timestamps: DatetimeIndex or list, UTC (default: 3 UTC datetimes)
            **column_overrides: Column names -> values to override

        Returns:
            DataFrame with MultiIndex and columns open, high, low, close, volume, trade_count, vwap
        """
        if symbols is None:
            symbols = ['AAPL', 'AAPL', 'AAPL']
        if timestamps is None:
            timestamps = pd.to_datetime(
                ['2024-01-02 14:30:00', '2024-01-02 15:00:00', '2024-01-02 20:00:00'],
                utc=True
            )
        timestamps = pd.to_datetime(timestamps, utc=True)
        n = min(len(symbols), len(timestamps))
        symbols, timestamps = symbols[:n], timestamps[:n]
        defaults = {
            'open': [150.0, 200.0, 250.0][:n],
            'high': [155.0, 205.0, 255.0][:n],
            'low': [148.0, 198.0, 248.0][:n],
            'close': [152.0, 202.0, 252.0][:n],
            'volume': [1000000, 2000000, 3000000][:n],
            'trade_count': [5000, 6000, 7000][:n],
            'vwap': [151.0, 201.0, 251.0][:n]
        }
        defaults.update(column_overrides)
        index = pd.MultiIndex.from_arrays([symbols, timestamps], names=['symbol', 'timestamp'])
        return pd.DataFrame(defaults, index=index)

    def test_remove_closed_market_rows_empty_returns_empty_copy(self):
        """Test remove_closed_market_rows on empty DataFrame returns empty copy."""
        data = pd.DataFrame()
        result = self.cleaner.remove_closed_market_rows(data)
        self.assertIsNot(result, data)
        self.assertTrue(result.empty)

    def test_remove_closed_market_rows_non_multiindex_returns_copy(self):
        """Test remove_closed_market_rows on non-MultiIndex DataFrame returns copy unchanged."""
        data = pd.DataFrame({'open': [100.0], 'high': [101.0], 'low': [99.0], 'close': [100.0]})
        result = self.cleaner.remove_closed_market_rows(data)
        self.assertIsNot(result, data)
        self.assertEqual(len(result), len(data))

    def test_remove_closed_market_rows_keeps_regular_session_only(self):
        """Test remove_closed_market_rows keeps only rows during NYSE regular hours (9:30-16:00 Eastern)."""
        data = self._create_stock_dataframe(
            symbols=['AAPL', 'AAPL', 'AAPL'],
            timestamps=pd.to_datetime(
                ['2024-01-02 14:30:00', '2024-01-02 22:00:00', '2024-01-06 14:30:00'],
                utc=True
            )
        )
        result = self.cleaner.remove_closed_market_rows(data)
        self.assertEqual(len(result), 1)
        kept_ts = result.index.get_level_values('timestamp')[0]
        self.assertEqual(kept_ts, pd.Timestamp('2024-01-02 14:30:00', tz='UTC'))

    def test_remove_closed_market_rows_all_during_session_keeps_all(self):
        """Test remove_closed_market_rows when all timestamps are during session keeps all rows."""
        data = self._create_stock_dataframe(
            symbols=['AAPL', 'AAPL', 'AAPL'],
            timestamps=pd.to_datetime(
                ['2024-01-02 14:30:00', '2024-01-02 15:00:00', '2024-01-02 20:00:00'],
                utc=True
            )
        )
        result = self.cleaner.remove_closed_market_rows(data)
        self.assertEqual(len(result), 3)

    def test_remove_closed_market_rows_weekend_removed(self):
        """Test remove_closed_market_rows removes rows on weekend."""
        data = self._create_stock_dataframe(
            symbols=['AAPL', 'AAPL'],
            timestamps=pd.to_datetime(['2024-01-06 14:30:00', '2024-01-07 14:30:00'], utc=True)
        )
        result = self.cleaner.remove_closed_market_rows(data)
        self.assertEqual(len(result), 0)

    def test_remove_closed_market_rows_after_hours_removed(self):
        """Test remove_closed_market_rows removes after-hours timestamp."""
        data = self._create_stock_dataframe(
            symbols=['AAPL'],
            timestamps=pd.to_datetime(['2024-01-02 22:00:00'], utc=True)
        )
        result = self.cleaner.remove_closed_market_rows(data)
        self.assertEqual(len(result), 0)

    def test_remove_closed_market_rows_does_not_mutate_input(self):
        """Test remove_closed_market_rows returns new DataFrame and does not mutate input."""
        data = self._create_stock_dataframe(
            symbols=['AAPL', 'AAPL'],
            timestamps=pd.to_datetime(['2024-01-02 14:30:00', '2024-01-02 22:00:00'], utc=True)
        )
        original_len = len(data)
        result = self.cleaner.remove_closed_market_rows(data)
        self.assertIsNot(result, data)
        self.assertEqual(len(data), original_len)
        self.assertEqual(len(result), 1)

    def test_remove_closed_market_rows_multiple_symbols_same_timestamp(self):
        """Test remove_closed_market_rows keeps all symbols for a timestamp that is open."""
        data = self._create_stock_dataframe(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            timestamps=pd.to_datetime(
                ['2024-01-02 14:30:00', '2024-01-02 14:30:00', '2024-01-02 14:30:00'],
                utc=True
            )
        )
        result = self.cleaner.remove_closed_market_rows(data)
        self.assertEqual(len(result), 3)
        self.assertEqual(set(result.index.get_level_values('symbol')), {'AAPL', 'MSFT', 'GOOGL'})

    def test_remove_closed_market_rows_naive_timestamps_localized_to_utc(self):
        """Test remove_closed_market_rows with timezone-naive timestamps localizes to UTC then filters."""
        naive_ts = pd.to_datetime(['2024-01-02 14:30:00'])
        self.assertIsNone(naive_ts.tz)
        index = pd.MultiIndex.from_arrays(
            [['AAPL'], naive_ts],
            names=['symbol', 'timestamp']
        )
        data = pd.DataFrame(
            {'open': [150.0], 'high': [155.0], 'low': [148.0], 'close': [152.0],
             'volume': [1000000], 'trade_count': [5000], 'vwap': [151.0]},
            index=index
        )
        result = self.cleaner.remove_closed_market_rows(data)
        self.assertEqual(len(result), 1)

    def test_forward_propagate_empty_returns_copy(self):
        """Test forward_propagate on empty DataFrame returns empty copy."""
        data = pd.DataFrame()
        result = self.cleaner.forward_propagate(
            data, TimeFrame.Minute, only_when_market_open=False
        )
        self.assertIsNot(result, data)
        self.assertTrue(result.empty)

    def test_forward_propagate_fills_missing_minute_bar(self):
        """Test forward_propagate inserts missing 1-min bar and fills with previous close."""
        data = self._create_stock_dataframe(
            symbols=['AAPL', 'AAPL'],
            timestamps=pd.to_datetime(
                ['2024-01-02 14:30:00', '2024-01-02 14:32:00'],
                utc=True
            ),
            close=[100.0, 102.0],
        )
        result = self.cleaner.forward_propagate(
            data, TimeFrame.Minute, only_when_market_open=False
        )
        self.assertEqual(len(result), 3)
        mid = result.loc[('AAPL', pd.Timestamp('2024-01-02 14:31:00', tz='UTC'))]
        self.assertEqual(mid['close'], 100.0)
        self.assertEqual(mid['volume'], 0)
        self.assertEqual(mid['trade_count'], 0)

    def test_forward_propagate_only_when_market_open_imputes_rth_only(self):
        """Test forward_propagate with only_when_market_open runs and returns RTH bars only."""
        data = self._create_stock_dataframe(
            symbols=['AAPL', 'AAPL'],
            timestamps=pd.to_datetime(
                ['2024-01-02 14:30:00', '2024-01-02 14:32:00'],
                utc=True
            ),
            close=[100.0, 102.0],
        )
        result = self.cleaner.forward_propagate(
            data, TimeFrame.Minute, only_when_market_open=True
        )
        self.assertEqual(len(result), 3)
        mid = result.loc[('AAPL', pd.Timestamp('2024-01-02 14:31:00', tz='UTC'))]
        self.assertEqual(mid['close'], 100.0)
        self.assertEqual(mid['volume'], 0)

    def test_forward_propagate_mark_imputed_rows_adds_imputed_column(self):
        """Test forward_propagate with mark_imputed_rows=True adds 'imputed' column."""
        data = self._create_stock_dataframe(
            symbols=['AAPL', 'AAPL'],
            timestamps=pd.to_datetime(
                ['2024-01-02 14:30:00', '2024-01-02 14:32:00'],
                utc=True
            ),
            close=[100.0, 102.0],
        )
        result = self.cleaner.forward_propagate(
            data, TimeFrame.Minute, only_when_market_open=False, mark_imputed_rows=True
        )
        self.assertIn('imputed', result.columns)
        ts_14_30 = pd.Timestamp('2024-01-02 14:30:00', tz='UTC')
        ts_14_31 = pd.Timestamp('2024-01-02 14:31:00', tz='UTC')
        ts_14_32 = pd.Timestamp('2024-01-02 14:32:00', tz='UTC')
        self.assertFalse(result.loc[('AAPL', ts_14_30), 'imputed'])
        self.assertTrue(result.loc[('AAPL', ts_14_31), 'imputed'])
        self.assertFalse(result.loc[('AAPL', ts_14_32), 'imputed'])

    def test_forward_propagate_mark_imputed_rows_false_no_imputed_column(self):
        """Test forward_propagate with mark_imputed_rows=False does not add 'imputed' column."""
        data = self._create_stock_dataframe(
            symbols=['AAPL', 'AAPL'],
            timestamps=pd.to_datetime(
                ['2024-01-02 14:30:00', '2024-01-02 14:32:00'],
                utc=True
            ),
        )
        result = self.cleaner.forward_propagate(
            data, TimeFrame.Minute, only_when_market_open=False, mark_imputed_rows=False
        )
        self.assertNotIn('imputed', result.columns)


if __name__ == '__main__':
    unittest.main()
