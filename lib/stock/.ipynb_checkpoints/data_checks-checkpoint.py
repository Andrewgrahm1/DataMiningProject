"""
Data validation utilities for stock data.
"""

from datetime import timezone
from typing import Optional

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from alpaca.data.timeframe import TimeFrame

from lib.utils.conversions import timeframe_to_timedelta
from lib.utils.rth import rth_timestamps_from_schedule


class InvalidDataException(Exception):
    """Exception raised when data validation fails."""


class StockDataChecker:
    """Checker for stock data quality."""

    @staticmethod
    def _assert_correct_columns(data: pd.DataFrame) -> None:
        """
        Check that the data has the correct multi-index and columns.

        Expected from get_historical_bars (multi-index format):
        - Index: MultiIndex with levels 'symbol' (string), 'timestamp' (datetime GMT)
        - Columns: open, high, low, close, volume, trade_count, vwap

        Args:
            data: DataFrame to check

        Raises:
            InvalidDataException: If index/columns are wrong or data types are incorrect
        """
        if data.empty:
            return

        # Require MultiIndex with symbol and timestamp
        if not isinstance(data.index, pd.MultiIndex):
            raise InvalidDataException(
                "Data must have a MultiIndex with 'symbol' and 'timestamp' levels"
            )
        names = [n for n in data.index.names if n is not None]
        if set(['symbol', 'timestamp']) > set(names):
            raise InvalidDataException(
                "Data index must have 'symbol' and 'timestamp' levels"
            )

        # Required columns (no symbol/timestamp; they are in the index)
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise InvalidDataException(
                f"Data is missing required columns: {sorted(missing_columns)}"
            )

        # Check index level types
        type_errors = []
        symbol_level = data.index.get_level_values('symbol')
        ts_level = data.index.get_level_values('timestamp')
        is_symbol_ok = (
            pd.api.types.is_string_dtype(symbol_level)
            or pd.api.types.is_object_dtype(symbol_level)
        )
        if not is_symbol_ok:
            type_errors.append("Index level 'symbol' should be string/object type")
        if not pd.api.types.is_datetime64_any_dtype(ts_level):
            type_errors.append("Index level 'timestamp' should be datetime type")

        # Check column types
        float_columns = ['open', 'high', 'low', 'close', 'vwap']
        for col in float_columns:
            if not pd.api.types.is_float_dtype(data[col]):
                type_errors.append(f"Column '{col}' should be float type")
        int_columns = ['volume', 'trade_count']
        for col in int_columns:
            is_int_ok = (
                pd.api.types.is_integer_dtype(data[col])
                or pd.api.types.is_float_dtype(data[col])
            )
            if not is_int_ok:
                type_errors.append(f"Column '{col}' should be integer or float type")

        if type_errors:
            raise InvalidDataException("Data type errors: " + "; ".join(type_errors))

    @staticmethod
    def _assert_no_missing_values(data: pd.DataFrame) -> None:
        """
        Check that no columns contain NaN values.

        Args:
            data: DataFrame to check

        Raises:
            InvalidDataException: If any columns contain NaN values
        """
        if data.empty:
            return

        # Check for NaN values in any column
        missing_values = data.isna().any()

        if missing_values.any():
            columns_with_nans = missing_values[missing_values].index.tolist()
            raise InvalidDataException(
                f"Data contains missing values (NaN) in columns: {columns_with_nans}"
            )

    @staticmethod
    def _assert_dates_are_in_gmt(data: pd.DataFrame) -> None:
        """
        Check that the timestamp index level is in GMT.

        Args:
            data: DataFrame with MultiIndex including 'timestamp' level

        Raises:
            InvalidDataException: If the dates are not in GMT
        """
        if data.empty:
            return

        ts = data.index.get_level_values('timestamp')
        if not pd.api.types.is_datetime64_any_dtype(ts):
            raise InvalidDataException("Index level 'timestamp' is not datetime type")

        # Naive (no timezone) is not GMT
        if ts.tz is None:
            raise InvalidDataException("Data contains dates that are not in GMT")

        # GMT is equivalent to UTC; compare to UTC
        if ts.tz != timezone.utc:
            raise InvalidDataException("Data contains dates that are not in GMT")

    @staticmethod
    def _assert_ohlc_valid(data: pd.DataFrame) -> None:
        """
        Check OHLC consistency: low <= high, low <= open <= high, low <= close <= high.

        Args:
            data: DataFrame with open, high, low, close columns

        Raises:
            InvalidDataException: If any OHLC relationship is violated
        """
        if data.empty:
            return

        errors = []
        if (data['low'] > data['high']).any():
            errors.append("low must be <= high")
        if (data['open'] < data['low']).any():
            errors.append("open must be >= low")
        if (data['open'] > data['high']).any():
            errors.append("open must be <= high")
        if (data['close'] < data['low']).any():
            errors.append("close must be >= low")
        if (data['close'] > data['high']).any():
            errors.append("close must be <= high")

        if errors:
            raise InvalidDataException("OHLC validity: " + "; ".join(errors))

    @staticmethod
    def _timestamps_to_check(
        timestamps: pd.DatetimeIndex,
        contains_closed_market_data: bool,
        nyse,
    ) -> Optional[np.ndarray]:
        """
        Return sorted unique timestamps to validate for completeness.
        If session-only, filter to NYSE RTH; otherwise return all unique timestamps.
        Returns None if empty or fewer than 2 timestamps.
        """
        if timestamps.tz is not None:
            timestamps = timestamps.tz_convert(timezone.utc)
        else:
            timestamps = timestamps.tz_localize('UTC')

        if not contains_closed_market_data:
            timestamps = StockDataChecker._filter_to_session_only(timestamps, nyse)

        timestamps = pd.Series(timestamps.unique()).sort_values().values
        if len(timestamps) < 2:
            return None
        return timestamps

    @staticmethod
    def _filter_to_session_only(timestamps: pd.DatetimeIndex, nyse) -> pd.DatetimeIndex:
        """Keep only timestamps that fall during NYSE regular trading hours."""
        schedule = nyse.schedule(
            start_date=timestamps.min(),
            end_date=timestamps.max(),
        )
        valid_rth = rth_timestamps_from_schedule(
            schedule, pd.Timedelta(minutes=1)
        )
        if valid_rth.empty:
            return pd.DatetimeIndex([])
        return timestamps[timestamps.isin(valid_rth)]

    @staticmethod
    def _raise_if_gaps_invalid(
        timestamps: np.ndarray,
        symbol: str,
        delta_sec: float,
        tolerance_sec: float,
        timeframe: TimeFrame,
    ) -> None:
        """Raise InvalidDataException if consecutive timestamps are not exactly delta apart."""
        for i in range(len(timestamps) - 1):
            diff = timestamps[i + 1] - timestamps[i]
            diff_sec = pd.Timedelta(diff).total_seconds()
            if diff_sec < delta_sec - tolerance_sec:
                raise InvalidDataException(
                    f"Consecutive bars for {symbol} are closer than {timeframe.value}: "
                    f"gap {diff} at {timestamps[i]}"
                )
            if diff_sec > delta_sec + tolerance_sec:
                raise InvalidDataException(
                    f"Missing bar(s) for {symbol}: gap {diff} is more than one "
                    f"{timeframe.value} at {timestamps[i]}"
                )

    @staticmethod
    def _split_into_sessions(
        timestamps: np.ndarray,
        tz: str = 'America/New_York',
    ) -> list:
        """
        Split RTH timestamps into one chunk per trading session (same Eastern date).
        Gaps across sessions (overnight, weekend) are not checked; gaps within a session are.
        """
        if len(timestamps) == 0:
            return []
        if len(timestamps) == 1:
            return [timestamps]
        # Group by calendar date in exchange timezone (vectorized)
        ts_index = pd.DatetimeIndex(timestamps)
        if ts_index.tz is None:
            ts_index = ts_index.tz_localize('UTC')
        else:
            ts_index = ts_index.tz_convert('UTC')
        eastern_dates = ts_index.tz_convert(tz).date
        order = np.argsort(timestamps)
        sorted_ts = timestamps[order]
        sorted_dates = eastern_dates[order]
        chunks = []
        for _, group in pd.Series(sorted_ts).groupby(sorted_dates, sort=True):
            chunks.append(group.values)
        return chunks

    @staticmethod
    def _assert_complete_timeframe(
        data: pd.DataFrame,
        timeframe: TimeFrame,
        contains_closed_market_data: bool = True,
    ) -> None:
        """
        Check that bars are present at the expected time resolution (no missing bars).

        For each symbol, consecutive timestamps must be exactly one period apart:
        no gaps (missing bars) and no overlapping bars.

        When contains_closed_market_data is False, only timestamps during NYSE regular
        trading hours (RTH) are considered; gaps over closed periods (nights, weekends,
        holidays) are ignored so that session-only data is validated correctly.

        Args:
            data: DataFrame with MultiIndex (symbol, timestamp)
            timeframe: TimeFrame used to fetch the data (e.g. TimeFrame.Day, TimeFrame.Minute)
            contains_closed_market_data: If True (default), check all consecutive bars.
                If False, only check bars during market hours; skip closed-market timestamps.

        Raises:
            InvalidDataException: If consecutive bars are not exactly one period apart
        """
        if data.empty:
            return
        if not isinstance(data.index, pd.MultiIndex):
            raise InvalidDataException(
                "Data must have a MultiIndex with 'symbol' and 'timestamp' for timeframe check"
            )

        delta_sec = timeframe_to_timedelta(timeframe).total_seconds()
        tolerance_sec = 1.0
        nyse = mcal.get_calendar('NYSE') if not contains_closed_market_data else None

        for symbol in data.index.get_level_values('symbol').unique():
            sym_df = data.xs(symbol, level='symbol')
            ts_level = sym_df.index.get_level_values('timestamp').sort_values()
            to_check = StockDataChecker._timestamps_to_check(
                ts_level, contains_closed_market_data, nyse
            )
            if to_check is None:
                continue
            if contains_closed_market_data:
                StockDataChecker._raise_if_gaps_invalid(
                    to_check, symbol, delta_sec, tolerance_sec, timeframe
                )
            else:
                # Only require no missing bars within each session; allow gap across sessions
                for chunk in StockDataChecker._split_into_sessions(to_check):
                    if len(chunk) >= 2:
                        StockDataChecker._raise_if_gaps_invalid(
                            chunk, symbol, delta_sec, tolerance_sec, timeframe
                        )

    def assert_data_clean(
        self,
        data: pd.DataFrame,
        timeframe: Optional[TimeFrame] = None,
        contains_closed_market_data: bool = True,
    ) -> None:
        """
        Assert that the data passes all validation checks.

        Expects DataFrames from get_historical_bars (multi-index with symbol
        and timestamp in the index).

        Args:
            data: DataFrame to validate
            timeframe: If provided, assert bars are complete at this resolution (no missing bars).
            contains_closed_market_data: If False, timeframe completeness is checked only
                for bars during market hours (closed periods are skipped). Default True.

        Raises:
            InvalidDataException: If data validation fails
        """
        self._assert_correct_columns(data)
        self._assert_no_missing_values(data)
        self._assert_ohlc_valid(data)
        self._assert_dates_are_in_gmt(data)
        if timeframe is not None:
            self._assert_complete_timeframe(
                data, timeframe, contains_closed_market_data=contains_closed_market_data
            )

    def check_data(
        self,
        data: pd.DataFrame,
        timeframe: Optional[TimeFrame] = None,
        contains_closed_market_data: bool = True,
    ) -> bool:
        """
        Check if data passes all validation checks.

        Args:
            data: DataFrame to validate
            timeframe: If provided, check bars are complete at this resolution (no missing bars).
            contains_closed_market_data: If False, timeframe completeness is checked only
                for bars during market hours. Default True.

        Returns:
            True if data is clean, False otherwise
        """
        try:
            self.assert_data_clean(
                data,
                timeframe=timeframe,
                contains_closed_market_data=contains_closed_market_data,
            )
            return True
        except InvalidDataException:
            return False
