from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, timedelta

from alpaca.data.timeframe import TimeFrame

from lib.stock.data_fetcher import StockDataFetcher
from lib.stock.data_checks import StockDataChecker
from lib.stock.data_cleaner import StockDataCleaner

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "etc" / "data"


def _raw_trade_data_path() -> Path:
    """Path to raw trade data CSV for the given symbol."""
    return DATA_DIR / f"test_raw_trade_data.csv"


def _cleaned_trade_data_path() -> Path:
    """Path to cleaned trade data CSV for the given symbol."""
    return DATA_DIR / f"test_clean_trade_data_cleaned.csv"


def _load_trade_data_from_csv(path: Path) -> pd.DataFrame:
    """Load a MultiIndex (symbol, timestamp) trade DataFrame from CSV."""
    df = pd.read_csv(path, index_col=[0, 1], parse_dates=[1])
    # Ensure timestamp index level is timezone-aware UTC
    ts = df.index.get_level_values("timestamp")
    if ts.tz is None:
        df.index = df.index.set_levels(
            ts.tz_localize("UTC"), level="timestamp"
        )
    else:
        df.index = df.index.set_levels(
            ts.tz_convert("UTC"), level="timestamp"
        )
    return df


def _save_trade_data_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Save raw trade DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def _get_raw_trade_data() -> pd.DataFrame:
    data_path = _raw_trade_data_path()

    if data_path.exists():
        print(f"Loading trade data from {data_path}...")
        trade_data = _load_trade_data_from_csv(data_path)
    else:
        print("No local data found; fetching from API...")
        fetcher = StockDataFetcher()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        trade_data = fetcher.get_historical_bars(
            symbol="SPY",
            start_date=start_date,
            end_date=end_date,
            timeframe=TimeFrame.Minute,
        )
        print(f"Saving raw data to {data_path}...")
        _save_trade_data_to_csv(trade_data, data_path)
        print(f"Saved raw data to {data_path}.")
    return trade_data


def _get_cleaned_trade_data(raw_trade_data: pd.DataFrame) -> pd.DataFrame:
    """Load cleaned data from CSV if present; otherwise get raw data, clean it, and save cleaned."""
    cleaned_path = _cleaned_trade_data_path()

    if cleaned_path.exists():
        print(f"Loading cleaned trade data from {cleaned_path}...")
        cleaned_trade_data = _load_trade_data_from_csv(cleaned_path)
        checker = StockDataChecker()
        try:
            print("Checking data...")
            checker.assert_data_clean(
                cleaned_trade_data,
                timeframe=TimeFrame.Minute,
                contains_closed_market_data=False,
            )
        except Exception as e:
            print(f"Error: {e}")
            raise
    else:
        print("No local cleaned data found; cleaning raw data...")
        cleaner = StockDataCleaner()
        print("Removing closed market rows...")
        cleaned_trade_data = cleaner.remove_closed_market_rows(raw_trade_data)
        print("Forward-propagating data...")
        cleaned_trade_data = cleaner.forward_propagate(
            cleaned_trade_data, TimeFrame.Minute, only_when_market_open=True
        )
        _save_trade_data_to_csv(cleaned_trade_data, cleaned_path)
        print(f"Saved cleaned data to {cleaned_path}.")

        checker = StockDataChecker()
        try:
            print("Checking data...")
            checker.assert_data_clean(
                cleaned_trade_data,
                timeframe=TimeFrame.Minute,
                contains_closed_market_data=False,
            )
        except Exception as e:
            print(f"Error: {e}")
            raise

    return cleaned_trade_data


def _non_rth_intervals_utc(ts_min: pd.Timestamp, ts_max: pd.Timestamp):
    """Yield (start, end) UTC timestamps for each continuous non-RTH interval."""
    ts_min = ts_min.tz_convert("UTC") if ts_min.tz else ts_min.tz_localize("UTC")
    ts_max = ts_max.tz_convert("UTC") if ts_max.tz else ts_max.tz_localize("UTC")
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=ts_min, end_date=ts_max)
    if schedule.empty:
        yield (ts_min, ts_max)
        return
    # Sessions in chronological order (schedule is typically date-indexed)
    sessions = []
    for _, row in schedule.iterrows():
        o = row["market_open"]
        c = row["market_close"]
        o_utc = o.tz_convert("UTC") if o.tz else o.tz_localize("UTC")
        c_utc = c.tz_convert("UTC") if c.tz else c.tz_localize("UTC")
        sessions.append((o_utc, c_utc))
    sessions.sort(key=lambda x: x[0])
    # Non-RTH: before first open, between each close and next open, after last close
    if ts_min < sessions[0][0]:
        yield (ts_min, sessions[0][0])
    for i in range(len(sessions) - 1):
        yield (sessions[i][1], sessions[i + 1][0])
    if sessions[-1][1] < ts_max:
        yield (sessions[-1][1], ts_max)


def _show_differences(raw_trade_data: pd.DataFrame, cleaned_trade_data: pd.DataFrame) -> None:
    """Plot close price over time: raw vs cleaned, to compare and spot errors."""
    if raw_trade_data.empty or cleaned_trade_data.empty:
        print("Cannot plot: raw or cleaned data is empty.")
        return
    if not isinstance(raw_trade_data.index, pd.MultiIndex) or not isinstance(
        cleaned_trade_data.index, pd.MultiIndex
    ):
        print("Cannot plot: expected MultiIndex (symbol, timestamp).")
        return
    symbol = raw_trade_data.index.get_level_values("symbol").unique()[0]
    raw_sym = raw_trade_data.xs(symbol, level="symbol")
    cleaned_sym = cleaned_trade_data.xs(symbol, level="symbol")
    raw_sym = raw_sym.sort_index()
    cleaned_sym = cleaned_sym.sort_index()
    ts_min = raw_sym.index.min()
    ts_max = raw_sym.index.max()

    fig, ax = plt.subplots(figsize=(12, 5))
    # Grey out non-RTH periods (behind the lines)
    for start, end in _non_rth_intervals_utc(ts_min, ts_max):
        ax.axvspan(start, end, facecolor="gray", alpha=0.35, zorder=0)
    ax.set_xlim(ts_min, ts_max)

    ax.plot(
        raw_sym.index,
        raw_sym["close"],
        label="Raw",
        alpha=0.7,
        color="gray",
        linewidth=0.8,
        zorder=2,
    )
    ax.plot(
        cleaned_sym.index,
        cleaned_sym["close"],
        label="Cleaned (RTH + forward-filled)",
        alpha=0.9,
        color="tab:blue",
        linewidth=0.8,
        zorder=2,
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Close")
    ax.set_title(f"Raw vs cleaned close price — {symbol} (grey = outside RTH)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def main():
    """Fetch or load trade data for the symbol, then clean (if needed) and validate."""
    raw_trade_data = _get_raw_trade_data()
    cleaned_trade_data = _get_cleaned_trade_data(raw_trade_data)
    _show_differences(raw_trade_data, cleaned_trade_data)


if __name__ == "__main__":
    main()
