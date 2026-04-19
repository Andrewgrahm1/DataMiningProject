"""Experiment-specific dataframe feature-engineering helpers (orion).

Also includes the orion data pipeline: Alpaca fetch/cache, training table assembly,
chronological split, z-scoring, and feature–target correlation / mutual-information helpers (used by
``grid_search`` / ``feature_correlation``).
"""

from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from lib.common.common import (
    _index_position,
    _trade_date_series,
    add_feature_bars_since_open,
    add_feature_bars_until_close,
    add_feature_pct_change_batch,
    add_range_target_column,
    create_target_column,
)

from experiments.orion.config import OrionExperimentConfig


def _rolling_zscore_1d(values: np.ndarray, window: int) -> np.ndarray:
    """
    Z-score of each point vs the mean and population std of the last ``window`` values
    ending at that index (inclusive). Leading indices where the window is incomplete are 0.
    If the window's std is 0, the z-score is 0.

    Uses pandas rolling (C-level) instead of a Python loop over bars.
    """
    n = len(values)
    out = np.zeros(n, dtype=np.float64)
    if window < 1 or n == 0:
        return out
    s = pd.Series(values, dtype=np.float64, copy=False)
    roll_mean = s.rolling(window, min_periods=window).mean()
    roll_std = s.rolling(window, min_periods=window).std(ddof=0)
    rm = roll_mean.to_numpy(dtype=np.float64, copy=False)
    rs = roll_std.to_numpy(dtype=np.float64, copy=False)
    valid = np.isfinite(rm) & np.isfinite(rs) & (rs > 0)
    out[valid] = (values - rm)[valid] / rs[valid]
    return out


def _zscore_columns_by_day(
    data: pd.DataFrame,
    values_all: np.ndarray,
    window_list: list[int],
    name_fn: Callable[[int], str],
) -> dict[str, np.ndarray]:
    """Build full-length z-score arrays per window; one pass over (symbol, trade_date) groups."""
    n_rows = len(data)
    trade_date = _trade_date_series(data)
    symbol = data.index.get_level_values("symbol")
    columns = {name_fn(w): np.zeros(n_rows, dtype=np.float64) for w in window_list}

    for (_sym, _date), group in data.groupby([symbol, trade_date], sort=False):
        group = group.sort_index(level="timestamp")
        base = _index_position(data, group.index[0])
        n = len(group)
        seg = values_all[base : base + n]
        for w in window_list:
            columns[name_fn(w)][base : base + n] = _rolling_zscore_1d(seg, w)

    return columns


def add_feature_volume_zscore(
    data: pd.DataFrame,
    window_list: list[int],
    *,
    column_name_fn: Callable[[int], str] | None = None,
) -> pd.DataFrame:
    """
    Add one or more volume z-score columns. For each bar, the z-score is
    ``(volume - mean) / std`` where mean and std are taken over the last ``window`` bars
    in the **same trading day** (including the current bar), in timestamp order.
    Uses America/New_York calendar dates when timestamps are tz-aware (same rule as
    ``add_feature_pct_change`` in ``lib.common.common``).

    Rows before a full window exists within the day are 0. If the window's standard
    deviation is 0, the z-score is 0.

    Assumes data has MultiIndex (symbol, timestamp) and a ``volume`` column.

    Args:
        window_list: Rolling window lengths (e.g. ``[20]`` or ``[10, 30, 60]``).
            If empty, returns ``data`` unchanged.
        column_name_fn: If given, called as ``column_name_fn(window)`` for each column
            name; else uses ``volume_zscore_<window>`` (e.g. ``volume_zscore_20``).

    Returns:
        ``data`` with new columns added (single concat, no fragmentation).
    """
    if not window_list:
        return data
    for w in window_list:
        if w < 1:
            raise ValueError("each window size must be >= 1")
    name_fn = column_name_fn if column_name_fn is not None else (lambda x: f"volume_zscore_{x}")
    vol_all = data["volume"].to_numpy(dtype=np.float64, copy=False)
    columns = _zscore_columns_by_day(data, vol_all, window_list, name_fn)
    new_df = pd.DataFrame(columns, index=data.index)
    return pd.concat([data, new_df], axis=1)


def add_feature_volume_and_trade_count_zscore(
    data: pd.DataFrame,
    window_list: list[int],
    *,
    volume_column_name_fn: Callable[[int], str] | None = None,
    trade_count_column_name_fn: Callable[[int], str] | None = None,
) -> pd.DataFrame:
    """
    Add volume and trade-count z-score columns in **one** pass over calendar days.

    Same semantics as calling ``add_feature_volume_zscore`` and
    ``add_feature_trade_count_zscore`` separately, but avoids a second groupby over
    ``(symbol, trade_date)`` when both are needed.

    Args:
        window_list: Rolling window lengths. If empty, returns ``data`` unchanged.
        volume_column_name_fn: Optional name factory for volume z-score columns.
        trade_count_column_name_fn: Optional name factory for trade-count z-score columns.

    Returns:
        ``data`` with new columns added (single concat, no fragmentation).
    """
    if not window_list:
        return data
    for w in window_list:
        if w < 1:
            raise ValueError("each window size must be >= 1")
    vol_name = (
        volume_column_name_fn
        if volume_column_name_fn is not None
        else (lambda x: f"volume_zscore_{x}")
    )
    tc_name = (
        trade_count_column_name_fn
        if trade_count_column_name_fn is not None
        else (lambda x: f"trade_count_zscore_{x}")
    )
    n_rows = len(data)
    trade_date = _trade_date_series(data)
    symbol = data.index.get_level_values("symbol")
    vol_all = data["volume"].to_numpy(dtype=np.float64, copy=False)
    tc_all = data["trade_count"].to_numpy(dtype=np.float64, copy=False)
    columns: dict[str, np.ndarray] = {}
    for w in window_list:
        columns[vol_name(w)] = np.zeros(n_rows, dtype=np.float64)
        columns[tc_name(w)] = np.zeros(n_rows, dtype=np.float64)

    for (_sym, _date), group in data.groupby([symbol, trade_date], sort=False):
        group = group.sort_index(level="timestamp")
        base = _index_position(data, group.index[0])
        n = len(group)
        vseg = vol_all[base : base + n]
        tseg = tc_all[base : base + n]
        for w in window_list:
            columns[vol_name(w)][base : base + n] = _rolling_zscore_1d(vseg, w)
            columns[tc_name(w)][base : base + n] = _rolling_zscore_1d(tseg, w)

    new_df = pd.DataFrame(columns, index=data.index)
    return pd.concat([data, new_df], axis=1)


def add_feature_trade_count_zscore(
    data: pd.DataFrame,
    window_list: list[int],
    *,
    column_name_fn: Callable[[int], str] | None = None,
) -> pd.DataFrame:
    """
    Add one or more trade-count z-score columns. Same semantics as
    ``add_feature_volume_zscore``, but uses the ``trade_count`` column instead of
    ``volume``.

    For each bar, the z-score is ``(trade_count - mean) / std`` over the last ``window``
    bars in the **same trading day** (including the current bar), in timestamp order.

    Assumes data has MultiIndex (symbol, timestamp) and a ``trade_count`` column.

    Args:
        window_list: Rolling window lengths. If empty, returns ``data`` unchanged.
        column_name_fn: If given, called as ``column_name_fn(window)`` for each column
            name; else uses ``trade_count_zscore_<window>``.

    Returns:
        ``data`` with new columns added (single concat, no fragmentation).
    """
    if not window_list:
        return data
    for w in window_list:
        if w < 1:
            raise ValueError("each window size must be >= 1")
    name_fn = column_name_fn if column_name_fn is not None else (lambda x: f"trade_count_zscore_{x}")
    tc_all = data["trade_count"].to_numpy(dtype=np.float64, copy=False)
    columns = _zscore_columns_by_day(data, tc_all, window_list, name_fn)
    new_df = pd.DataFrame(columns, index=data.index)
    return pd.concat([data, new_df], axis=1)


def add_feature_day_of_week(
    data: pd.DataFrame,
    *,
    column_name: str = "day_of_week",
) -> pd.DataFrame:
    """
    Add a column with the weekday as an integer: Monday=1 through Sunday=7 (ISO-style).

    Modifies data in place and returns it. Assumes data has MultiIndex (symbol, timestamp).
    When timestamps are tz-aware, the weekday is taken in America/New_York (same rule as
    other calendar features in ``lib.common.common``); naive timestamps are not shifted.

    Args:
        column_name: Name of the column to add (default ``day_of_week``).
    """
    ts = data.index.get_level_values("timestamp")
    series = pd.Series(ts, index=data.index)
    if hasattr(ts, "tz") and ts.tz is not None:
        series = series.dt.tz_convert("America/New_York")
    data[column_name] = (series.dt.dayofweek + 1).astype(np.int64)
    return data


def _true_range_np(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True range per bar. First bar uses high - low only (no prior close)."""
    n = len(high)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    if n <= 1:
        return tr
    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - close[:-1])
    lc = np.abs(low[1:] - close[:-1])
    tr[1:] = np.maximum(np.maximum(hl, hc), lc)
    return tr


def _rsi_from_close(closes: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder RSI from close prices, continuous along the array. Indices ``0 .. period-1``
    are 0 (not enough history). First valid RSI at index ``period``.

    When average loss is 0 and average gain > 0, RSI is 100; when both averages are 0,
    RSI is 50.
    """
    n = len(closes)
    out = np.zeros(n, dtype=np.float64)
    if period < 2:
        raise ValueError("RSI period must be >= 2")
    if n == 0:
        return out
    delta = np.zeros(n, dtype=np.float64)
    delta[1:] = closes[1:] - closes[:-1]
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    if n < period + 1:
        return out
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    avg_gain[period] = float(np.mean(gain[1 : period + 1]))
    avg_loss[period] = float(np.mean(loss[1 : period + 1]))
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        ag = avg_gain[i]
        al = avg_loss[i]
        if al == 0.0:
            out[i] = 100.0 if ag > 0 else 50.0
        else:
            rs = ag / al
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def _wilder_atr_from_tr(tr: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder (smoothed) ATR from true range. Rows before the first full ATR are 0.

    For ``period == 1``, ATR equals TR on every bar (no leading zeros).
    """
    n = len(tr)
    out = np.zeros(n, dtype=np.float64)
    if period < 1:
        raise ValueError("period must be >= 1")
    if n == 0:
        return out
    if period == 1:
        out[:] = tr
        return out
    if n < period:
        return out
    out[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def add_feature_atr(
    data: pd.DataFrame,
    period_list: list[int],
    *,
    column_name_fn: Callable[[int], str] | None = None,
) -> pd.DataFrame:
    """
    Add one or more ATR columns using Wilder's smoothing (same semantics per period as a
    single-period ATR). Computes true range once per symbol segment, then smooths for each
    period. Avoids DataFrame fragmentation from many single-column inserts.

    True range per bar is ``max(H-L, |H-prev_close|, |L-prev_close|)``; the first bar
    of each symbol uses ``H - L``. ATR is computed in timestamp order per symbol (continuous
    across trading days). Rows before the first ATR can be formed are 0 (except when
    ``period == 1``, where ATR equals TR from the first bar).

    Assumes data has MultiIndex (symbol, timestamp) and columns ``high``, ``low``, ``close``.

    Args:
        period_list: Wilder periods (e.g. ``[14]`` or ``[7, 14, 21]``). If empty, returns
            ``data`` unchanged.
        column_name_fn: If given, called as ``column_name_fn(period)`` for each column
            name; else uses ``f"atr_{period}"``.

    Returns:
        ``data`` with new columns added (single concat, no fragmentation).
    """
    if not period_list:
        return data
    for p in period_list:
        if p < 1:
            raise ValueError("each period must be >= 1")
    name_fn = column_name_fn if column_name_fn is not None else (lambda n: f"atr_{n}")
    n_rows = len(data)
    symbol = data.index.get_level_values("symbol")
    columns = {name_fn(p): np.zeros(n_rows, dtype=np.float64) for p in period_list}

    for _sym, group in data.groupby(symbol, sort=False):
        group = group.sort_index(level="timestamp")
        locs = group.index
        base = _index_position(data, locs[0])
        n = len(group)
        high = group["high"].to_numpy(dtype=np.float64, copy=False)
        low = group["low"].to_numpy(dtype=np.float64, copy=False)
        close = group["close"].to_numpy(dtype=np.float64, copy=False)
        tr = _true_range_np(high, low, close)
        for p in period_list:
            atr = _wilder_atr_from_tr(tr, p)
            columns[name_fn(p)][base : base + n] = atr

    new_df = pd.DataFrame(columns, index=data.index)
    return pd.concat([data, new_df], axis=1)


def add_feature_rsi(
    data: pd.DataFrame,
    period_list: list[int],
    *,
    column_name_fn: Callable[[int], str] | None = None,
) -> pd.DataFrame:
    """
    Add one or more RSI columns (Wilder smoothing on average gain / average loss).

    Uses close-to-close changes in timestamp order per symbol (continuous across trading
    days). Rows before the first valid RSI for a period are 0.

    Assumes data has MultiIndex (symbol, timestamp) and a ``close`` column.

    Args:
        period_list: RSI lookback periods (e.g. ``[14]``). Each must be >= 2.
            If empty, returns ``data`` unchanged.
        column_name_fn: If given, ``column_name_fn(period)`` for each column; else
            ``rsi_{period}``.

    Returns:
        ``data`` with new columns added (single concat, no fragmentation).
    """
    if not period_list:
        return data
    for p in period_list:
        if p < 2:
            raise ValueError("each RSI period must be >= 2")
    name_fn = column_name_fn if column_name_fn is not None else (lambda n: f"rsi_{n}")
    n_rows = len(data)
    symbol = data.index.get_level_values("symbol")
    columns = {name_fn(p): np.zeros(n_rows, dtype=np.float64) for p in period_list}

    for _sym, group in data.groupby(symbol, sort=False):
        group = group.sort_index(level="timestamp")
        locs = group.index
        base = _index_position(data, locs[0])
        n = len(group)
        close = group["close"].to_numpy(dtype=np.float64, copy=False)
        for p in period_list:
            rsi = _rsi_from_close(close, p)
            columns[name_fn(p)][base : base + n] = rsi

    new_df = pd.DataFrame(columns, index=data.index)
    return pd.concat([data, new_df], axis=1)


def add_feature_rsi_reference_bars(
    data: pd.DataFrame,
    reference_bars: pd.DataFrame,
    period_list: list[int],
    *,
    column_name_fn: Callable[[str, int], str] | None = None,
) -> pd.DataFrame:
    """
    Add Wilder RSI columns computed from each reference symbol's ``close`` (same semantics
    as ``add_feature_rsi`` on that symbol alone), aligned to ``data`` rows by **exact**
    timestamp match.

    ``reference_bars`` uses MultiIndex ``(symbol, timestamp)``. For each reference symbol and
    each period, RSI is computed in timestamp order on that symbol's closes (continuous across
    days), then values are taken at ``data``'s timestamps; missing reference bars yield 0.

    Assumes ``data`` has MultiIndex ``(symbol, timestamp)`` and ``reference_bars`` has
    ``close``.

    Args:
        period_list: RSI periods (each must be >= 2). If empty, returns ``data`` unchanged.
        column_name_fn: If given, ``column_name_fn(symbol, period)`` for each column; else
            ``rsi_{symbol}_{period}`` (e.g. ``rsi_SPY_14``).

    Returns:
        ``data`` with new columns added (single concat, no fragmentation). If
        ``reference_bars`` is empty or has no symbols, returns ``data`` unchanged.
    """
    reference_symbols = symbols_in_reference_bars(reference_bars)
    if not reference_symbols or not period_list:
        return data
    for p in period_list:
        if p < 2:
            raise ValueError("each RSI period must be >= 2")
    name_fn = (
        column_name_fn
        if column_name_fn is not None
        else (lambda sym, period: f"rsi_{sym}_{period}")
    )
    n_rows = len(data)
    ts = data.index.get_level_values("timestamp")
    columns: dict[str, np.ndarray] = {}
    for sym in reference_symbols:
        for p in period_list:
            columns[name_fn(sym, p)] = np.zeros(n_rows, dtype=np.float64)

    for sym in reference_symbols:
        ref_close = reference_bars.xs(sym, level="symbol")["close"].sort_index()
        close_arr = ref_close.to_numpy(dtype=np.float64, copy=False)
        for p in period_list:
            rsi_arr = _rsi_from_close(close_arr, p)
            rsi_series = pd.Series(rsi_arr, index=ref_close.index)
            aligned = rsi_series.reindex(ts)
            col = np.nan_to_num(
                aligned.to_numpy(dtype=np.float64, copy=False),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            columns[name_fn(sym, p)][:] = col

    new_df = pd.DataFrame(columns, index=data.index)
    return pd.concat([data, new_df], axis=1)


def _rolling_sma_np(close: np.ndarray, period: int) -> np.ndarray:
    """
    Simple moving average of ``close`` with window ``period`` (inclusive of current bar).
    Leading indices where fewer than ``period`` samples exist are 0.
    For ``period == 1``, equals ``close`` (every bar).
    """
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
    if period < 1:
        raise ValueError("SMA period must be >= 1")
    if n == 0:
        return out
    if period == 1:
        out[:] = close
        return out
    c = np.concatenate([[0.0], np.cumsum(close, dtype=np.float64)])
    for i in range(period - 1, n):
        out[i] = (c[i + 1] - c[i + 1 - period]) / period
    return out


def add_feature_close_sma_pct_diff(
    data: pd.DataFrame,
    period_list: list[int],
    *,
    column_name_fn: Callable[[int], str] | None = None,
) -> pd.DataFrame:
    """
    Add one column per period: percent distance of ``close`` from its simple moving average,
    ``(close - SMA) / SMA``, in timestamp order per symbol (continuous across trading days).

    Uses the SMA of the last ``period`` closes including the current bar. Rows before a full
    window exists are 0. Where ``SMA`` is 0 or non-finite, the value is 0 (no division).

    For ``period == 1``, SMA equals ``close`` on every bar, so the percent difference is 0.

    Assumes data has MultiIndex ``(symbol, timestamp)`` and a ``close`` column.

    Args:
        period_list: SMA window lengths (e.g. ``[20]`` or ``[10, 50]``). If empty, returns
            ``data`` unchanged.
        column_name_fn: If given, ``column_name_fn(period)`` for each column; else
            ``close_sma_{period}_pct_diff``.

    Returns:
        ``data`` with new columns added (single concat, no fragmentation).
    """
    if not period_list:
        return data
    for p in period_list:
        if p < 1:
            raise ValueError("each SMA period must be >= 1")
    name_fn = (
        column_name_fn
        if column_name_fn is not None
        else (lambda n: f"close_sma_{n}_pct_diff")
    )
    n_rows = len(data)
    symbol = data.index.get_level_values("symbol")
    columns = {name_fn(p): np.zeros(n_rows, dtype=np.float64) for p in period_list}

    for _sym, group in data.groupby(symbol, sort=False):
        group = group.sort_index(level="timestamp")
        locs = group.index
        base = _index_position(data, locs[0])
        n = len(group)
        close = group["close"].to_numpy(dtype=np.float64, copy=False)
        for p in period_list:
            sma = _rolling_sma_np(close, p)
            columns[name_fn(p)][base : base + n] = _pct_diff_vs_aligned_close(close, sma)

    new_df = pd.DataFrame(columns, index=data.index)
    return pd.concat([data, new_df], axis=1)


def _pct_diff_vs_aligned_close(
    close_main: np.ndarray,
    close_other_aligned: np.ndarray,
) -> np.ndarray:
    """``(close_main - close_other) / close_other`` with safe zeros where invalid."""
    out = np.zeros(len(close_main), dtype=np.float64)
    valid = (
        np.isfinite(close_other_aligned)
        & (close_other_aligned > 0)
        & np.isfinite(close_main)
    )
    out[valid] = (
        close_main[valid] - close_other_aligned[valid]
    ) / close_other_aligned[valid]
    return out


def symbols_in_reference_bars(reference_bars: pd.DataFrame) -> list[str]:
    """
    Sorted unique symbols in ``reference_bars`` index level ``symbol``.
    Empty frame returns an empty list.
    """
    if len(reference_bars) == 0:
        return []
    u = pd.Index(reference_bars.index.get_level_values("symbol")).unique().sort_values()
    return [str(x) for x in u]


def add_feature_close_vs_reference_bars_pct_diff(
    data: pd.DataFrame,
    reference_bars: pd.DataFrame,
    *,
    column_name_fn: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """
    Add one column per symbol present in ``reference_bars``: ``(close - close_ref) / close_ref``
    at the **same timestamp**.

    ``data`` is the primary symbol's frame. ``reference_bars`` uses MultiIndex
    ``(symbol, timestamp)``; **all** symbols in that frame get a feature column. Timestamps
    are aligned by exact match; missing reference bars or invalid closes yield 0 for that row.

    Args:
        column_name_fn: If given, ``column_name_fn(symbol)`` for each column; else
            ``close_vs_<symbol>_pct_diff``.

    Returns:
        ``data`` with new columns added (single concat, no fragmentation). If
        ``reference_bars`` is empty or has no symbols, returns ``data`` unchanged.
    """
    reference_symbols = symbols_in_reference_bars(reference_bars)
    if not reference_symbols:
        return data
    name_fn = (
        column_name_fn
        if column_name_fn is not None
        else (lambda sym: f"close_vs_{sym}_pct_diff")
    )

    ts = data.index.get_level_values("timestamp")
    close_main = data["close"].to_numpy(dtype=np.float64, copy=False)
    columns: dict[str, np.ndarray] = {}

    for sym in reference_symbols:
        other_close = reference_bars.xs(sym, level="symbol")["close"].sort_index()
        aligned = other_close.reindex(ts)
        close_other = aligned.to_numpy(dtype=np.float64, copy=False)
        columns[name_fn(sym)] = _pct_diff_vs_aligned_close(close_main, close_other)

    new_df = pd.DataFrame(columns, index=data.index)
    return pd.concat([data, new_df], axis=1)


def add_feature_close_vs_symbol_pct_diff(
    data: pd.DataFrame,
    other_data: pd.DataFrame,
    *,
    other_symbol: str,
    column_name: str = "close_vs_other_pct_diff",
) -> pd.DataFrame:
    """
    Add a single column with ``(close - close_other) / close_other`` vs one reference
    symbol. Uses only rows for ``other_symbol`` in ``other_data`` (other symbols in the
    frame are ignored).
    """
    levels = other_data.index.get_level_values("symbol")
    if other_symbol not in levels.unique():
        raise ValueError(f"other_data has no rows for symbol {other_symbol!r}")
    ref_slice = other_data.loc[[other_symbol]]
    return add_feature_close_vs_reference_bars_pct_diff(
        data, ref_slice, column_name_fn=lambda _s: column_name
    )


def add_feature_close_vwap_pct_diff(
    data: pd.DataFrame,
    *,
    column_name: str = "close_vwap_pct_diff",
) -> pd.DataFrame:
    """
    Add a column with (close - vwap) / vwap for each bar — the percent difference of
    the close from VWAP on the same bar, relative to VWAP.

    Modifies data in place and returns it. Assumes data has MultiIndex (symbol, timestamp)
    and columns ``close`` and ``vwap``. Where ``vwap`` is 0, the value is 0 (no division).

    Args:
        column_name: Name of the column to add (default ``close_vwap_pct_diff``).
    """
    close = data["close"].to_numpy(dtype=np.float64, copy=False)
    vwap = data["vwap"].to_numpy(dtype=np.float64, copy=False)
    out = np.zeros(len(data), dtype=np.float64)
    np.divide(close - vwap, vwap, out=out, where=vwap != 0)
    data[column_name] = out
    return data


# --- Data pipeline (fetch, training table, split, z-score) ---------------------------------

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "etc" / "data"


def print_orion_run_preamble(
    cfg: OrionExperimentConfig,
    *,
    trade_cost: float,
    run_title: str = "Starting",
) -> None:
    """Print symbol, dates, costs, and break-even win rate (shared by orion scripts)."""
    from lib.common.common import calculate_min_win_rate

    print(f"====================== {run_title} ======================")
    print(f"SYMBOL: {cfg.symbol}")
    print(f"Reference symbols (cross-close features): {list(cfg.reference_symbols)}")
    print(f"Date Range: {cfg.start_date} to {cfg.end_date}")
    print(f"Take Profit: {cfg.take_profit * 100}%")
    print(f"Stop Loss: {cfg.stop_loss * 100}%")
    horizon = cfg.target_max_bars_after_entry
    print(
        "Target max bars after entry: "
        f"{horizon if horizon is not None else 'unlimited (session only)'}"
    )
    print(f"Trade Cost: {trade_cost * 100}%")
    print(
        "Break Even Win Rate (Percision): "
        f"{calculate_min_win_rate(cfg.take_profit, cfg.stop_loss, trade_cost) * 100}%"
    )


def orion_cache_path(symbol: str, start_date: datetime, end_date: datetime) -> Path:
    """Path for cached cleaned data: ``etc/data/orion_{symbol}_{start}_{end}_clean.csv``."""
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    return _CACHE_DIR / f"orion_{symbol}_{start_str}_{end_str}_clean.csv"


def pull_and_clean(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch OHLCV for ``symbol`` in [start, end], RTH only, forward-fill gaps; use CSV cache."""
    from alpaca.data.timeframe import TimeFrame

    from lib.stock.data_cleaner import StockDataCleaner
    from lib.stock.data_fetcher import StockDataFetcher

    path = orion_cache_path(symbol, start, end)

    if path.exists():
        print(f"Loading cached data: {path.name}")
        data = pd.read_csv(path, index_col=[0, 1], parse_dates=[1])
        if data.index.levels[1].tz is None:
            data.index = data.index.set_levels(
                data.index.levels[1].tz_localize("UTC"), level=1
            )
        return data

    fetcher = StockDataFetcher()
    data = fetcher.get_historical_bars(
        symbol=symbol,
        start_date=start,
        end_date=end,
        timeframe=TimeFrame.Minute,
    )
    cleaner = StockDataCleaner()
    data = cleaner.remove_closed_market_rows(data)
    data = cleaner.forward_propagate(
        data,
        TimeFrame.Minute,
        only_when_market_open=True,
        mark_imputed_rows=False,
    )
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(path)
    print(f"Cached cleaned data to {path.name}")
    return data


def create_orion_training_data(
    data: pd.DataFrame,
    take_profit: float,
    stop_loss: float,
    *,
    reference_bars: pd.DataFrame | None = None,
    max_bars_after_entry: int | None = None,
    only_rows_hitting_tp_or_sl: bool = False,
) -> pd.DataFrame:
    """Build target + feature columns for the orion experiment schema.

    When ``only_rows_hitting_tp_or_sl`` is True, drop rows where neither take-profit nor
    stop-loss would trade within the forward window (same rule as ``add_range_target_column``).
    Filtering runs after all features are computed so rolling indicators still see full history.
    """
    col_names: list[str] = []
    col_names.append("target")
    data = create_target_column(
        data,
        take_profit=take_profit,
        stop_loss=stop_loss,
        column_name=col_names[-1],
        max_bars_after_entry=max_bars_after_entry,
    )
    col_names.append("bars_until_close")
    data = add_feature_bars_until_close(data, column_name=col_names[-1])
    col_names.append("bars_since_open")
    data = add_feature_bars_since_open(data, column_name=col_names[-1])
    col_names.append("close_vwap_pct_diff")
    data = add_feature_close_vwap_pct_diff(data, column_name=col_names[-1])
    rolling_windows = [1, 2, 5, 10, 20, 30, 60, 120]
    rsi_rolling_windows = [7, 14, 28]
    if reference_bars is not None:
        ref_syms = symbols_in_reference_bars(reference_bars)
        if ref_syms:
            col_names.extend(f"close_vs_{s}_pct_diff" for s in ref_syms)
            data = add_feature_close_vs_reference_bars_pct_diff(data, reference_bars)
            col_names.extend(f"rsi_{s}_{b}" for s in ref_syms for b in rsi_rolling_windows)
            data = add_feature_rsi_reference_bars(data, reference_bars, rsi_rolling_windows)
    col_names.extend(f"atr_{b}" for b in rolling_windows)
    data = add_feature_atr(data, rolling_windows)
    col_names.append("day_of_week")
    data = add_feature_day_of_week(data, column_name=col_names[-1])
    col_names.extend(f"volume_zscore_{w}" for w in rolling_windows)
    data = add_feature_volume_zscore(data, rolling_windows)
    col_names.extend(f"pct_change_{b}" for b in rolling_windows)
    data = add_feature_pct_change_batch(data, rolling_windows)
    sma_rolling_windows = [9, 20, 50, 100]
    col_names.extend(f"close_sma_{b}_pct_diff" for b in sma_rolling_windows)
    data = add_feature_close_sma_pct_diff(data, sma_rolling_windows)
    col_names.extend(f"rsi_{b}" for b in rsi_rolling_windows)
    data = add_feature_rsi(data, rsi_rolling_windows)
    if only_rows_hitting_tp_or_sl:
        tmp_col = "__tp_or_sl_touched"
        add_range_target_column(
            data,
            take_profit,
            stop_loss,
            column_name=tmp_col,
            max_bars_after_entry=max_bars_after_entry,
        )
        data = data.loc[data[tmp_col] == 1].drop(columns=[tmp_col])
    return data[col_names].copy()


def split_training_data(
    data: pd.DataFrame,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically into train, validation, then test (three contiguous blocks)."""
    data = data.sort_index(level="timestamp")
    n = len(data)
    n_test = int(n * test_fraction)
    n_val = int(n * validation_fraction)
    n_train = n - n_val - n_test
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(
            f"split_training_data: need positive train/val/test sizes; got n={n}, "
            f"train={n_train}, val={n_val}, test={n_test}. Lower validation_fraction or "
            "test_fraction."
        )
    train_df = data.iloc[:n_train]
    val_df = data.iloc[n_train : n_train + n_val]
    test_df = data.iloc[n_train + n_val :]
    return train_df, val_df, test_df


def _print_split_stats(name: str, df: pd.DataFrame, *, target_column: str = "target") -> None:
    n = len(df)
    if n == 0:
        print(f"{name}: 0 rows")
        return
    ts = df.index.get_level_values("timestamp")
    start, end = ts.min(), ts.max()
    span_years = (end - start).total_seconds() / (365.25 * 24 * 60 * 60)
    pos = int((df[target_column] == 1).sum())
    pos_pct = 100.0 * pos / n
    print(
        f"{name}: {n:,} rows | {pos:,} positive ({pos_pct:.2f}%) | "
        f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')} "
        f"({span_years:.2f} years)"
    )


def print_training_data_stats(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_column: str = "target",
) -> None:
    """Print row counts, target balance, and calendar span for train, validation, and test."""
    print("\n--- Train / validation / test split stats ---")
    _print_split_stats("Train", train_df, target_column=target_column)
    _print_split_stats("Validation", validation_df, target_column=target_column)
    _print_split_stats("Test", test_df, target_column=target_column)


def zscore_feature_columns(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str = "target",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit StandardScaler on train features only; transform train, validation, and test."""
    feature_cols = [c for c in train_df.columns if c != target_column]
    scaler = StandardScaler()
    train_out = train_df.copy()
    val_out = validation_df.copy()
    test_out = test_df.copy()
    train_out[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_out[feature_cols] = scaler.transform(validation_df[feature_cols])
    test_out[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_out, val_out, test_out


def load_orion_training_frames(
    *,
    symbol: str,
    reference_symbols: tuple[str, ...],
    start_date: datetime,
    end_date: datetime,
    take_profit: float,
    stop_loss: float,
    validation_fraction: float,
    test_fraction: float,
    target_max_bars_after_entry: int | None = None,
    print_split: bool = True,
    only_rows_hitting_tp_or_sl: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch/clean, build features, split, z-score. Returns ``(train_df, val_df, test_df)``."""
    raw_df = pull_and_clean(symbol, start_date, end_date)
    reference_df = pd.concat(
        [pull_and_clean(s, start_date, end_date) for s in reference_symbols],
        axis=0,
    ).sort_index()
    training_df = create_orion_training_data(
        raw_df,
        take_profit=take_profit,
        stop_loss=stop_loss,
        reference_bars=reference_df,
        max_bars_after_entry=target_max_bars_after_entry,
        only_rows_hitting_tp_or_sl=only_rows_hitting_tp_or_sl,
    )

    train_df, val_df, test_df = split_training_data(
        training_df,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    train_df, val_df, test_df = zscore_feature_columns(train_df, val_df, test_df)
    if print_split:
        print_training_data_stats(train_df, val_df, test_df)
    return train_df, val_df, test_df


def feature_target_correlations(
    train_df: pd.DataFrame,
    *,
    target_column: str = "target",
) -> pd.Series:
    """Pearson correlation of each feature with the target (train, z-scored features)."""
    corr = train_df.corr(numeric_only=True)[target_column].drop(target_column, errors="ignore")
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def print_feature_target_correlations(
    train_df: pd.DataFrame,
    *,
    target_column: str = "target",
    top_n: int | None = None,
) -> None:
    """Print features sorted by absolute Pearson correlation with the target."""
    feat_target_corr = feature_target_correlations(train_df, target_column=target_column)
    if top_n is not None:
        feat_target_corr = feat_target_corr.head(top_n)
    print("\n--- Feature vs target correlation (train, Pearson) ---")
    for feat_name, rho in feat_target_corr.items():
        rho_str = f"{rho:.6f}" if np.isfinite(rho) else "nan"
        print(f"  {feat_name}: {rho_str}")
