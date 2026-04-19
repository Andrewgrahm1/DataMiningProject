# stock

This subdirectory contains **classes and code** for pulling and cleaning stock data.

Use this module from experiments (e.g. `from lib.stock import ...`) when you need to fetch or clean market/stock data.

---

## Modules and classes

### `data_fetcher` — `StockDataFetcher`

Fetches stock data using the Alpaca API. Loads API credentials from a JSON file (default: `etc/private/alpaca_key.json`).

| Method | Description |
|--------|-------------|
| **`__init__(key_file_path=None)`** | Initialize the fetcher. `key_file_path` defaults to `./etc/private/alpaca_key.json`. Supports JSON with `api_key`/`secret_key` or `APCA-API-KEY-ID`/`APCA-API-SECRET-KEY`. |
| **`get_historical_bars(symbol, start_date, end_date, timeframe=TimeFrame.Minute)`** | Get historical bar (OHLCV) data for one or more symbols. `symbol` can be a string or list of strings; dates can be `'YYYY-MM-DD'` strings or `datetime`/`date`. Returns a DataFrame with MultiIndex `(symbol, timestamp)` and columns: `open`, `high`, `low`, `close`, `volume`, `trade_count`, `vwap`. |

---

### `data_cleaner` — `StockDataCleaner`

Cleans stock DataFrames that use a MultiIndex `(symbol, timestamp)`.

| Method | Description |
|--------|-------------|
| **`remove_closed_market_rows(data)`** | Remove rows when the NYSE is closed (nights, weekends, holidays). Keeps only timestamps during NYSE regular trading hours (9:30 AM–4:00 PM Eastern). Returns a new DataFrame; index and column structure unchanged. |
| **`forward_propagate(data, timeframe, *, only_when_market_open=False)`** | Fill missing bars by inserting expected timestamps at the given `timeframe` and forward-filling OHLC/vwap from the previous bar. Inserted bars have `volume` and `trade_count` set to 0. If `only_when_market_open` is True, only imputes bars during NYSE RTH. Returns a new DataFrame with a complete time sequence. |

---

### `data_checks` — `StockDataChecker` and `InvalidDataException`

Validates stock data quality. Expects DataFrames in the same MultiIndex format as `get_historical_bars`.

| Class / method | Description |
|----------------|-------------|
| **`InvalidDataException`** | Exception raised when validation fails (wrong columns, NaNs, non-GMT dates, invalid OHLC, or missing bars). |
| **`StockDataChecker.assert_data_clean(data, timeframe=None, contains_closed_market_data=True)`** | Run all checks; raises `InvalidDataException` if the data is invalid. If `timeframe` is set, asserts bars are complete (no missing bars) at that resolution. Use `contains_closed_market_data=False` when validating session-only (RTH) data so gaps over nights/weekends are ignored. |
| **`StockDataChecker.check_data(data, timeframe=None, contains_closed_market_data=True)`** | Same checks as `assert_data_clean`, but returns `True` if valid and `False` if invalid (does not raise). |

Checks performed: correct MultiIndex and columns, no missing values, OHLC consistency (low ≤ open/close ≤ high), timestamps in GMT, and optionally complete timeframe (no gaps or overlaps between bars).
