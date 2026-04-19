"""
Scanner: grid search over (take_profit, stop_loss) on OHLCV bar data.

Pulls cleaned data (with optional cache in etc/data), runs exit simulation
per symbol and date, and aggregates profit. Supports parallel evaluation of
(tp, sl) pairs via ProcessPoolExecutor.
"""

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alpaca.data.timeframe import TimeFrame

from lib.stock.data_cleaner import StockDataCleaner
from lib.stock.data_fetcher import StockDataFetcher

# Cache dir for cleaned data: etc/data under project root (scanner is in experiments/)
_CACHE_DIR = Path(__file__).resolve().parent.parent / "etc" / "data"

# Sentinel for "no exit bar found" in vectorized day simulation
_NO_EXIT_INDEX = 999999


class _Tee:
    """Write to multiple file-like objects (e.g. stdout and a file)."""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)

    def flush(self):
        for f in self.files:
            f.flush()


def _data_stats_row(symbol: str, data: pd.DataFrame) -> pd.DataFrame:
    """One-row DataFrame: symbol, start_date, end_date, total_rows, lowest_price, highest_price, average_price, pct_imputed."""
    ts = data.index.get_level_values("timestamp")
    start = ts.min()
    end = ts.max()
    # Normalize to date for display if user prefers; keeping as full timestamp is fine too
    start_date = start.date() if hasattr(start, "date") else start
    end_date = end.date() if hasattr(end, "date") else end
    n = len(data)
    lowest = float(data["low"].min())
    highest = float(data["high"].max())
    average = float(data["close"].mean())
    pct_imputed = 100.0 * data["imputed"].sum() / n if n else 0.0
    row = pd.DataFrame(
        [
            {
                "start_date": start_date,
                "end_date": end_date,
                "total_rows": n,
                "lowest_price": lowest,
                "highest_price": highest,
                "average_price": round(average, 4),
                "pct_imputed": round(pct_imputed, 2),
            }
        ],
        index=[symbol],
    )
    return row


def _print_full_df(df: pd.DataFrame) -> None:
    """Print a DataFrame with all rows/columns visible and no wrap to a second table."""
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", None,
        "display.expand_frame_repr", False,
    ):
        print(df)


def _cache_path(symbol: str, start_date: datetime, end_date: datetime) -> Path:
    """Path for cached cleaned data: etc/data/{stem}_{symbol}_{start}_{end}_clean.csv"""
    stem = Path(__file__).stem
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    return _CACHE_DIR / f"{stem}_{symbol}_{start_str}_{end_str}_clean.csv"


def pull_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """Pull OHLCV bar data for the given symbol, cleaned. Uses cache in etc/data when present."""
    cache_path = _cache_path(symbol, start_date, end_date)

    if cache_path.exists():
        print(f"Loading cached data: {cache_path.name}")
        data = pd.read_csv(cache_path, index_col=[0, 1], parse_dates=[1])
        if data.index.levels[1].tz is None:
            data.index = data.index.set_levels(
                data.index.levels[1].tz_localize("UTC"), level=1
            )
        return data

    fetcher = StockDataFetcher()
    data = fetcher.get_historical_bars(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=TimeFrame.Minute,
    )
    cleaner = StockDataCleaner()
    data = cleaner.remove_closed_market_rows(data)
    data = cleaner.forward_propagate(
        data, TimeFrame.Minute, only_when_market_open=True, mark_imputed_rows=True
    )

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(cache_path)
    print(f"Cached cleaned data to {cache_path.name}")
    return data


def _run_day_vectorized(
    highs: np.ndarray,
    lows: np.ndarray,
    imputed: np.ndarray,
    take_profit: float,
    stop_loss: float,
) -> list[tuple[float, str, bool, int]]:
    """
    For one day, run exit simulation for each valid entry bar using vectorized NumPy.
    One Python loop over entry bars; exit-bar search uses argmax/min instead of inner loop.
    Returns list of (realized_pnl, exit_reason, was_imputed, duration_bars).
    duration_bars: bars from entry (bar i+1) to exit bar; 0 for eod (unused in aggregates).
    """
    n = len(highs)
    num_entries = max(0, n - 3)
    out = []
    for i in range(num_entries):
        entry = highs[i + 1]
        if entry <= 0:
            out.append((0.0, "eod", bool(imputed[i]) if i < len(imputed) else False, 0))
            continue
        sl_price = entry * (1 - stop_loss)
        tp_price = entry * (1 + take_profit)
        low_hit = lows[i + 2 : n] <= sl_price
        high_hit = highs[i + 2 : n] >= tp_price
        first_sl = np.argmax(low_hit) if low_hit.any() else _NO_EXIT_INDEX
        first_tp = np.argmax(high_hit) if high_hit.any() else _NO_EXIT_INDEX
        if first_sl <= first_tp and low_hit.any():
            reason = "stop_loss"
            pnl = -stop_loss
            duration_bars = 1 + first_sl
        elif high_hit.any():
            reason = "take_profit"
            pnl = take_profit
            duration_bars = 1 + first_tp
        else:
            reason = "eod"
            pnl = 0.0
            duration_bars = 0
        was_imputed = bool(imputed[i]) if i < len(imputed) else False
        out.append((pnl, reason, was_imputed, duration_bars))
    return out


def _pregroup_by_day(data: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, date]]:
    """Group data by (symbol, date) once; return list of (highs, lows, imputed, date)."""
    ts = data.index.get_level_values("timestamp")
    if hasattr(ts, "tz") and ts.tz is not None:
        trade_date = pd.Series(ts, index=data.index).dt.tz_convert("America/New_York").dt.date
    else:
        trade_date = pd.Series(ts, index=data.index).dt.date
    symbol = data.index.get_level_values("symbol")
    groups = []
    for (_sym, _date), group in data.groupby([symbol, trade_date]):
        group = group.sort_index(level="timestamp")
        highs = np.asarray(group["high"].values, dtype=np.float64)
        lows = np.asarray(group["low"].values, dtype=np.float64)
        imputed = (
            np.asarray(group["imputed"].values, dtype=bool)
            if "imputed" in group.columns
            else np.zeros(len(group), dtype=bool)
        )
        groups.append((highs, lows, imputed, _date))
    return groups


def _aggregate_trades(
    all_trades: list[tuple[float, str, bool, date, int]],
    cost_per_trade: float,
) -> tuple[float, float, float, float, float, int, float]:
    """
    Aggregate per-trade results. Each element is (pnl, reason, imputed, date, duration_bars).
    Only counts trades that exited at take_profit (strategy: only take trades that hit TP;
    ignore trades that hit stop loss or end of day).
    Returns revenue, cost, profit, percent_trades, percent_days (decimals), imputed_count, avg_trade_duration_bars.
    """
    total_opportunities = len(all_trades)
    total_days = len(set(d for _, _, _, d, _ in all_trades))
    exited = [
        (pnl, reason, imp, d, duration_bars)
        for pnl, reason, imp, d, duration_bars in all_trades
        if reason == "take_profit"
    ]
    if not exited:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0
    pnls, reasons, imputed_flags, dates, durations = zip(*exited)
    num_trades = len(exited)
    percent_trades = num_trades / total_opportunities if total_opportunities else 0.0
    day_count = len(set(dates))
    percent_days = day_count / total_days if total_days else 0.0
    revenue = sum(p for p in pnls if p > 0)
    cost = cost_per_trade * num_trades
    profit = sum(pnls) - cost
    imputed_count = sum(imputed_flags)
    avg_duration_bars = sum(durations) / num_trades if num_trades else 0.0
    return revenue, cost, profit, percent_trades, percent_days, imputed_count, avg_duration_bars


def _eval_one_tp_sl(
    tp: float,
    sl: float,
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, date]],
    cost_per_trade: float,
) -> tuple[float, float, float, float, float, float, float, float, int]:
    """Evaluate one (take_profit, stop_loss) over pre-grouped days. Used for parallel grid."""
    all_trades = []
    for highs, lows, imputed, _date in groups:
        day_trades = _run_day_vectorized(highs, lows, imputed, tp, sl)
        for pnl, reason, imp, duration_bars in day_trades:
            all_trades.append((pnl, reason, imp, _date, duration_bars))
    (
        revenue,
        cost,
        profit,
        percent_trades,
        percent_days,
        imputed_count,
        avg_trade_duration_bars,
    ) = _aggregate_trades(all_trades, cost_per_trade)
    return (
        tp,
        sl,
        revenue,
        cost,
        profit,
        percent_trades,
        percent_days,
        avg_trade_duration_bars,
        imputed_count,
    )


def run_grid_for_symbol(
    data: pd.DataFrame,
    take_profit_values: list[float],
    stop_loss_values: list[float],
    cost_per_trade: float,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Run a grid search over (take_profit, stop_loss) for one symbol's data.
    Pre-groups by day once; for each (tp, sl) iterates over cached groups only.
    Use n_jobs > 1 to evaluate (tp, sl) pairs in parallel.
    """
    groups = _pregroup_by_day(data)

    if n_jobs <= 1:
        results = []
        for tp in take_profit_values:
            for sl in stop_loss_values:
                results.append(_eval_one_tp_sl(tp, sl, groups, cost_per_trade))
    else:
        tasks = [
            (tp, sl, groups, cost_per_trade)
            for tp in take_profit_values
            for sl in stop_loss_values
        ]
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_tp_sl = {
                executor.submit(_eval_one_tp_sl, tp, sl, g, cost): (tp, sl)
                for tp, sl, g, cost in tasks
            }
            for fut in as_completed(future_to_tp_sl):
                tp, sl = future_to_tp_sl[fut]
                results.append(fut.result())
        # Preserve grid order (take_profit, then stop_loss)
        tp_sl_to_row = {(r[0], r[1]): r for r in results}
        results = [tp_sl_to_row[(tp, sl)] for tp in take_profit_values for sl in stop_loss_values]

    return pd.DataFrame(
        results,
        columns=[
            "take_profit",
            "stop_loss",
            "revenue",
            "cost",
            "profit",
            "percent_trades",
            "percent_days",
            "avg_trade_duration_bars",
            "imputed_count",
        ],
    )


# Default grids for grid search (take_profit %, stop_loss %)
DEFAULT_TAKE_PROFIT_VALUES = [0.006, 0.007, 0.008, 0.009] + [
    round(i * 0.01, 2) for i in range(1, 8)
]
DEFAULT_STOP_LOSS_VALUES = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]


def scanner(
    symbols: list[str],
    take_profit_values: list[float] | None = None,
    stop_loss_values: list[float] | None = None,
    n_jobs: int = 1,
    start_date: datetime = datetime(2025, 1, 1),
    end_date: datetime = datetime(2025, 12, 31),
    cost_per_trade: float = 0.004,
) -> None:
    """
    Scan the given symbols with a grid search over (take_profit, stop_loss).
    For each symbol and each (tp, sl), simulates exits at TP or SL and aggregates profit.
    Use n_jobs > 1 to run (tp, sl) grid in parallel.
    """
    if take_profit_values is None:
        take_profit_values = list(DEFAULT_TAKE_PROFIT_VALUES)
    if stop_loss_values is None:
        stop_loss_values = list(DEFAULT_STOP_LOSS_VALUES)
    stem = Path(__file__).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = _CACHE_DIR / f"{stem}_results_{timestamp}.txt"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    old_stdout = sys.stdout
    try:
        with open(results_path, "w", encoding="utf-8") as results_file:
            sys.stdout = _Tee(old_stdout, results_file)
            print("Input parameters:")
            print("  symbols:            [")
            for sym in symbols:
                print(f"    '{sym}',")
            print("  ]")
            print(f"  take_profit_values: [")
            for tp in take_profit_values:
                print(f"    {tp},")
            print("  ]")
            print(f"  stop_loss_values:   [")
            for sl in stop_loss_values:
                print(f"    {sl},")
            print("  ]")
            print(f"  start_date:         {start_date}")
            print(f"  end_date:           {end_date}")
            print(f"  cost_per_trade:     {cost_per_trade}")
            print(f"  n_jobs:             {n_jobs}")
            print()
            top10_per_symbol = []  # list of (symbol, top10 DataFrame)
            for symbol in symbols:
                print(f"Scanning {symbol}...")
                data = pull_data(symbol, start_date=start_date, end_date=end_date)
                stats_row = _data_stats_row(symbol, data)
                print("Data stats:")
                _print_full_df(stats_row)
                print()
                grid_table = run_grid_for_symbol(
                    data, take_profit_values, stop_loss_values, cost_per_trade, n_jobs=n_jobs
                )
                _print_full_df(grid_table)
                print()
                top10 = grid_table.nlargest(10, "profit").copy()
                print("Top 10 (take_profit, stop_loss) rows by profit:")
                _print_full_df(top10)
                top10.index = [symbol] * len(top10)
                top10_per_symbol.append((symbol, top10))
            print()
            print("Best rows (top 10 per symbol, sorted by profit):")
            df_best = pd.concat([df for _, df in top10_per_symbol], axis=0)
            _print_full_df(df_best.sort_values(by="profit", ascending=False))
    finally:
        sys.stdout = old_stdout
    print(f"Results written to {results_path}", file=old_stdout)


if __name__ == "__main__":
    scanner(
        ["SPY", "AAPL", "AMD", "PLTR", "SOFI", "RIVN", "TQQQ", "SOXL", "MARA", "NU", "NOK"],
        n_jobs=4,
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2025, 12, 31),
    )