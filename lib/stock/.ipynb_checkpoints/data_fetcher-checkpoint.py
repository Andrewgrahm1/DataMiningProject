"""
Stock data retrieval using Alpaca API.
"""

import json
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest

from alpaca.data.timeframe import TimeFrame


class StockDataFetcher:
    """
    Fetches stock data using the Alpaca API.

    Automatically loads API credentials from the specified JSON file.
    """

    def __init__(self, key_file_path: Optional[Union[str, Path]] = None):
        """
        Initialize the StockDataFetcher.

        Args:
            key_file_path: Path to the JSON file containing API keys.
                          Defaults to './etc/private/alpaca_key.json'
        """
        if key_file_path is None:
            # Default to project root relative path (lib/stock/data_fetcher.py -> project_root)
            key_file_path = (
                Path(__file__).parent.parent.parent
                / "etc"
                / "private"
                / "alpaca_key.json"
            )
        else:
            key_file_path = Path(key_file_path)

        if not key_file_path.exists():
            raise FileNotFoundError(f"API key file not found: {key_file_path}")

        self.api_key, self.secret_key = self._load_api_keys(key_file_path)
        self.client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )

    def _load_api_keys(self, key_file_path: Path) -> tuple[str, str]:
        """
        Load API keys from JSON file.

        Supports both formats:
        - {"api_key": "...", "secret_key": "..."}
        - {"APCA-API-KEY-ID": "...", "APCA-API-SECRET-KEY": "..."}

        Args:
            key_file_path: Path to the JSON file

        Returns:
            Tuple of (api_key, secret_key)
        """
        with open(key_file_path, 'r', encoding='utf-8') as f:
            keys = json.load(f)

        # Try standard Alpaca format first
        if "APCA-API-KEY-ID" in keys and "APCA-API-SECRET-KEY" in keys:
            return keys["APCA-API-KEY-ID"], keys["APCA-API-SECRET-KEY"]
        # Fall back to api_key/secret_key format
        elif "api_key" in keys and "secret_key" in keys:
            return keys["api_key"], keys["secret_key"]
        else:
            raise ValueError(
                f"Invalid key file format. Expected 'api_key'/'secret_key' or "
                f"'APCA-API-KEY-ID'/'APCA-API-SECRET-KEY' keys in {key_file_path}"
            )

    def get_historical_bars(
        self,
        symbol: Union[str, List[str]],
        start_date: Union[str, datetime, date],
        end_date: Union[str, datetime, date],
        timeframe: TimeFrame = TimeFrame.Minute
    ) -> pd.DataFrame:
        """
        Get historical bar (OHLCV) data for one or more symbols.

        Args:
            symbol: Stock symbol(s) as string or list of strings
            start_date: Start date (string 'YYYY-MM-DD' or datetime/date object)
            end_date: End date (string 'YYYY-MM-DD' or datetime/date object)
            timeframe: TimeFrame enum (Day, Hour, Minute, etc.)

        Returns:
            pandas DataFrame with historical bar data. The resulting DataFrame has
            a MultiIndex with levels 'symbol' and 'timestamp', and columns:
            open, high, low, close, volume, trade_count, vwap.
        """
        # Convert dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        elif isinstance(start_date, date) and not isinstance(start_date, datetime):
            start_date = datetime.combine(start_date, datetime.min.time())

        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        elif isinstance(end_date, date) and not isinstance(end_date, datetime):
            end_date = datetime.combine(end_date, datetime.min.time())

        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date
        )

        bars = self.client.get_stock_bars(request_params)
        return bars.df
