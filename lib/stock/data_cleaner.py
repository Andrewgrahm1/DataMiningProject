"""
Stock data cleaning utilities.
"""

import pandas as pd
import pandas_market_calendars as mcal

from alpaca.data.timeframe import TimeFrame

from lib.utils.conversions import timeframe_to_timedelta
from lib.utils.rth import rth_timestamps_from_schedule, to_utc


class StockDataCleaner:
    """
    Cleans stock DataFrames (MultiIndex symbol, timestamp).
    """

    def __init__(self):
        self._nyse = mcal.get_calendar('NYSE')

    def remove_closed_market_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows when the NYSE is closed (after hours, weekends, holidays).

        Keeps only rows whose timestamp falls within NYSE regular trading hours
        (9:30 AM - 4:00 PM Eastern) on a trading day. Drops weekends, holidays,
        and before/after market times.

        Args:
            data: DataFrame with MultiIndex (symbol, timestamp). Timestamps
                  should be timezone-aware (e.g. UTC).

        Returns:
            New DataFrame with only rows during NYSE regular session. Index
            and column structure unchanged.
        """
        if data.empty:
            return data.copy()

        if not isinstance(data.index, pd.MultiIndex):
            return data.copy()

        timestamps = data.index.get_level_values('timestamp')
        if timestamps.tz is None:
            timestamps = timestamps.tz_localize('UTC')
        else:
            timestamps = timestamps.tz_convert('UTC')

        start_ts = timestamps.min()
        end_ts = timestamps.max()
        schedule = self._nyse.schedule(
            start_date=start_ts,
            end_date=end_ts
        )
        if schedule.empty:
            return data.iloc[0:0].copy()

        valid_rth = rth_timestamps_from_schedule(
            schedule, pd.Timedelta(minutes=1)
        )
        if valid_rth.empty:
            return data.iloc[0:0].copy()
        keep_mask = timestamps.isin(valid_rth)
        return data.loc[keep_mask]

    def _expected_timestamps(
        self,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        timeframe: TimeFrame,
        only_when_market_open: bool,
    ) -> pd.DatetimeIndex:
        """Return expected timestamps at the given timeframe from start to end."""
        start_ts = to_utc(start_ts)
        end_ts = to_utc(end_ts)

        delta = timeframe_to_timedelta(timeframe)
        freq = pd.Timedelta(seconds=delta.total_seconds())

        if only_when_market_open:
            schedule = self._nyse.schedule(start_date=start_ts, end_date=end_ts)
            if schedule.empty:
                return pd.DatetimeIndex([])
            return rth_timestamps_from_schedule(
                schedule, freq, start_ts=start_ts, end_ts=end_ts
            )
        else:
            out = pd.date_range(
                start=start_ts,
                end=end_ts,
                freq=freq,
                inclusive='both',
            )
            return out

    def forward_propagate(
        self,
        data: pd.DataFrame,
        timeframe: TimeFrame,
        *,
        only_when_market_open: bool = False,
        mark_imputed_rows: bool = False,
    ) -> pd.DataFrame:
        """
        Forward-propagate price data by filling missing bars with the previous bar's values.

        Inserts missing timestamps at the given timeframe and fills OHLC/vwap from the
        last known bar; volume and trade_count are set to 0 for inserted bars.

        Args:
            data: DataFrame with MultiIndex (symbol, timestamp) and columns open, high,
                  low, close, volume, trade_count, vwap. Timestamps should be
                  timezone-aware (e.g. UTC).
            timeframe: Bar resolution (e.g. TimeFrame.Minute, TimeFrame.Day) used to
                      determine expected timestamps.
            only_when_market_open: If True, only impute bars that fall during NYSE
                                  regular trading hours. If False, impute every
                                  timeframe step from min to max timestamp.
            mark_imputed_rows: If True, add an "imputed" column: True for rows that
                              were filled by propagation, False for original rows.

        Returns:
            New DataFrame with the same columns and a complete index (no missing bars
            in the expected sequence). Price columns are forward-filled; volume and
            trade_count are 0 where bars were inserted. If mark_imputed_rows is True,
            includes an "imputed" boolean column.
        """
        if data.empty:
            return data.copy()
        if not isinstance(data.index, pd.MultiIndex):
            return data.copy()

        timestamps = data.index.get_level_values('timestamp')
        if timestamps.tz is None:
            data = data.copy()
            data.index = data.index.set_levels(
                data.index.levels[1].tz_localize('UTC'),
                level='timestamp',
            )
        else:
            data = data.copy()
            data.index = data.index.set_levels(
                data.index.levels[1].tz_convert('UTC'),
                level='timestamp',
            )

        pieces = []
        for symbol in data.index.get_level_values('symbol').unique():
            sym_df = data.xs(symbol, level='symbol', drop_level=False)
            ts_level = sym_df.index.get_level_values('timestamp')
            start_ts = ts_level.min()
            end_ts = ts_level.max()
            expected_ts = self._expected_timestamps(
                start_ts, end_ts, timeframe, only_when_market_open
            )
            if expected_ts.empty:
                if mark_imputed_rows:
                    sym_df = sym_df.copy()
                    sym_df['imputed'] = False
                pieces.append(sym_df)
                continue

            full_index = pd.MultiIndex.from_product(
                [[symbol], expected_ts],
                names=['symbol', 'timestamp'],
            )
            reindexed = sym_df.reindex(full_index)
            inserted = reindexed['close'].isna()
            reindexed = reindexed.ffill().bfill()
            reindexed.loc[inserted, ['volume', 'trade_count']] = 0
            if mark_imputed_rows:
                reindexed['imputed'] = inserted
            pieces.append(reindexed)

        return pd.concat(pieces, axis=0)
