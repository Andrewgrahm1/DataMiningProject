"""
Data-feed abstraction: get bar(s) at time T (and optionally by symbol).
Backtest implementation returns slice of preloaded DataFrame; paper/live can wrap broker/API.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd


class DataFeed(ABC):
    """Abstract data feed: provides market bars at a given time."""

    @abstractmethod
    def get_bars(
        self,
        current_time: datetime,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return bar(s) at current_time, optionally for a single symbol.
        May return one row per symbol or a window; index should be MultiIndex (symbol, timestamp)
        when multiple symbols, with columns open, high, low, close, volume, etc.
        """
