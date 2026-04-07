"""Shared helpers for orion2 experiments: cache load, chronological split, generic features, labels."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from lib.common.common import (
    _index_position,
    _trade_date_series,
    add_feature_bars_since_open,
    add_feature_bars_until_close,
)

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "etc" / "data"


def orion_cache_path(symbol: str, start_date: datetime, end_date: datetime) -> Path:
    """Path for cached cleaned minute bars (same naming as ``experiments.orion.elib``)."""
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


def _print_split_stats(name: str, df: pd.DataFrame, *, target_column: str) -> None:
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
    target_column: str,
) -> None:
    """Print row counts, target balance, and calendar span for train, validation, and test."""
    print("\n--- Train / validation / test split stats ---")
    _print_split_stats("Train", train_df, target_column=target_column)
    _print_split_stats("Validation", validation_df, target_column=target_column)
    _print_split_stats("Test", test_df, target_column=target_column)


def zscore_feature_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    non_feature_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit ``StandardScaler`` on train feature columns only; transform all three splits."""
    exclude = set(non_feature_columns)
    feature_cols = [c for c in train_df.columns if c not in exclude]
    scaler = StandardScaler()
    train_out = train_df.copy()
    val_out = validation_df.copy()
    test_out = test_df.copy()
    train_out[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_out[feature_cols] = scaler.transform(validation_df[feature_cols])
    test_out[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_out, val_out, test_out


def _rolling_sma_np(close: np.ndarray, period: int) -> np.ndarray:
    """SMA of ``close`` with window ``period``; leading incomplete windows are 0."""
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


def _rolling_std_population_np(close: np.ndarray, period: int) -> np.ndarray:
    """Rolling population standard deviation of ``close``; leading incomplete windows are 0.

    For ``period == 1``, returns zeros (no dispersion in a one-bar window).
    """
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
    if period < 1 or n == 0:
        return out
    if period == 1:
        return out
    s = pd.Series(close, dtype=np.float64, copy=False)
    rolled = s.rolling(period, min_periods=period).std(ddof=0).to_numpy(dtype=np.float64, copy=False)
    valid = np.isfinite(rolled)
    out[valid] = rolled[valid]
    return out


def _bollinger_upper_lower_pct_diff_np(
    close: np.ndarray,
    period: int,
    std_multiple: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(close - upper) / upper`` and ``(close - lower) / lower``; bands use SMA +/- ``std_multiple`` * std."""
    sma = _rolling_sma_np(close, period)
    std = _rolling_std_population_np(close, period)
    upper = sma + std_multiple * std
    lower = sma - std_multiple * std
    upper_pct = _pct_diff_vs_aligned_close(close, upper)
    lower_pct = _pct_diff_vs_aligned_close(close, lower)
    return upper_pct, lower_pct


def bollinger_std_column_tag(std_m: float) -> str:
    """Name fragment for default Bollinger columns, e.g. ``1std``, ``2std``, ``1p5std``."""
    if std_m <= 0:
        raise ValueError("std multiple must be positive")
    if std_m == int(std_m):
        return f"{int(std_m)}std"
    s = f"{std_m:.6g}".replace(".", "p")
    return f"{s}std"


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


def _wilder_smooth_np(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder (RMA) smooth of ``x``. Index ``period - 1`` is the mean of ``x[:period]``; earlier are 0.

    For ``period == 1``, output equals ``x`` (no leading zeros).
    """
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    if period < 1:
        raise ValueError("period must be >= 1")
    if n == 0:
        return out
    if period == 1:
        out[:] = x
        return out
    if n < period:
        return out
    out[period - 1] = float(np.mean(x[:period]))
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + x[i]) / period
    return out


def _wilder_atr_from_tr(tr: np.ndarray, period: int) -> np.ndarray:
    """Wilder (smoothed) ATR from true range. Rows before the first full ATR are 0.

    For ``period == 1``, ATR equals TR on every bar (no leading zeros).
    """
    return _wilder_smooth_np(tr, period)


def _plus_dm_minus_dm_np(high: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Welles Wilder +DM / -DM; index 0 is zero (no prior bar)."""
    n = len(high)
    pdm = np.zeros(n, dtype=np.float64)
    mdm = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return pdm, mdm
    up = high[1:] - high[:-1]
    down = low[:-1] - low[1:]
    plus_bar = (up > down) & (up > 0.0)
    minus_bar = (down > up) & (down > 0.0)
    pdm[1:] = np.where(plus_bar, up, 0.0)
    mdm[1:] = np.where(minus_bar, down, 0.0)
    return pdm, mdm


def _adx_plus_di_minus_di_np(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wilder +DI, -DI (0–100), and ADX from high/low/close."""
    n = len(close)
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    adx = np.zeros(n, dtype=np.float64)
    if period < 1 or n == 0:
        return plus_di, minus_di, adx

    tr = _true_range_np(high, low, close)
    pdm, mdm = _plus_dm_minus_dm_np(high, low)
    tr_s = _wilder_smooth_np(tr, period)
    pdm_s = _wilder_smooth_np(pdm, period)
    mdm_s = _wilder_smooth_np(mdm, period)

    denom_ok = np.isfinite(tr_s) & (tr_s > 0)
    plus_di[denom_ok] = 100.0 * pdm_s[denom_ok] / tr_s[denom_ok]
    minus_di[denom_ok] = 100.0 * mdm_s[denom_ok] / tr_s[denom_ok]

    di_sum = plus_di + minus_di
    dx = np.zeros(n, dtype=np.float64)
    ok_dx = np.isfinite(di_sum) & (di_sum > 0) & np.isfinite(plus_di) & np.isfinite(minus_di)
    dx[ok_dx] = 100.0 * np.abs(plus_di[ok_dx] - minus_di[ok_dx]) / di_sum[ok_dx]

    adx[:] = _wilder_smooth_np(dx, period)
    return plus_di, minus_di, adx


def add_feature_atr(
    data: pd.DataFrame,
    period_list: list[int],
    *,
    column_name_fn: Callable[[int], str] | None = None,
) -> pd.DataFrame:
    """Add Wilder ATR columns per period; true range is continuous in timestamp order per symbol.

    True range is ``max(H-L, |H-prev_close|, |L-prev_close|)``; first bar of each symbol uses
    ``H - L``. Leading bars before the first full ATR are 0 (except ``period == 1``, TR from bar 0).

    Expects MultiIndex ``(symbol, timestamp)`` and columns ``high``, ``low``, ``close``.
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


def add_feature_adx_di(
    data: pd.DataFrame,
    period_list: list[int],
    *,
    adx_column_name_fn: Callable[[int], str] | None = None,
    plus_di_column_name_fn: Callable[[int], str] | None = None,
    minus_di_column_name_fn: Callable[[int], str] | None = None,
) -> pd.DataFrame:
    """Add Wilder ADX, +DI, and -DI per period; timestamp order per symbol (continuous across days).

    +DI / -DI are scaled to 0–100. ADX is a Wilder smooth of DX (directional index) with the same
    ``period``. Leading bars follow the same warmup as other Wilder features in this module.

    Expects MultiIndex ``(symbol, timestamp)`` and columns ``high``, ``low``, ``close``.
    """
    if not period_list:
        return data
    for p in period_list:
        if p < 1:
            raise ValueError("each period must be >= 1")
    adx_fn = adx_column_name_fn if adx_column_name_fn is not None else (lambda n: f"adx_{n}")
    p_fn = plus_di_column_name_fn if plus_di_column_name_fn is not None else (lambda n: f"plus_di_{n}")
    m_fn = minus_di_column_name_fn if minus_di_column_name_fn is not None else (lambda n: f"minus_di_{n}")
    n_rows = len(data)
    symbol = data.index.get_level_values("symbol")
    col_names: list[str] = []
    for p in period_list:
        col_names.extend([adx_fn(p), p_fn(p), m_fn(p)])
    columns = {name: np.zeros(n_rows, dtype=np.float64) for name in col_names}

    for _sym, group in data.groupby(symbol, sort=False):
        group = group.sort_index(level="timestamp")
        locs = group.index
        base = _index_position(data, locs[0])
        n = len(group)
        high = group["high"].to_numpy(dtype=np.float64, copy=False)
        low = group["low"].to_numpy(dtype=np.float64, copy=False)
        close = group["close"].to_numpy(dtype=np.float64, copy=False)
        for p in period_list:
            pdi, mdi, adxv = _adx_plus_di_minus_di_np(high, low, close, p)
            columns[p_fn(p)][base : base + n] = pdi
            columns[m_fn(p)][base : base + n] = mdi
            columns[adx_fn(p)][base : base + n] = adxv

    new_df = pd.DataFrame(columns, index=data.index)
    return pd.concat([data, new_df], axis=1)


def _rsi_wilder_np(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI in [0, 100]; leading bars before the first valid RSI are 0."""
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
    if period < 1 or n <= period:
        return out
    deltas = np.diff(close.astype(np.float64, copy=False))
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    avg_gain = float(np.mean(gains[0:period]))
    avg_loss = float(np.mean(losses[0:period]))
    if avg_loss == 0.0:
        out[period] = 100.0 if avg_gain > 0.0 else 50.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - (100.0 / (1.0 + rs))
    for j in range(period, len(gains)):
        g = float(gains[j])
        l = float(losses[j])
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        idx = j + 1
        if avg_loss == 0.0:
            out[idx] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[idx] = 100.0 - (100.0 / (1.0 + rs))
    return out


def add_feature_rsi(
    data: pd.DataFrame,
    period_list: list[int],
    *,
    column_name_fn: Callable[[int], str] | None = None,
) -> pd.DataFrame:
    """Add Wilder RSI columns per period; computed in timestamp order per symbol (continuous across days).

    Expects MultiIndex ``(symbol, timestamp)`` and column ``close``.
    """
    if not period_list:
        return data
    for p in period_list:
        if p < 1:
            raise ValueError("each RSI period must be >= 1")
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
            rsi = _rsi_wilder_np(close, p)
            columns[name_fn(p)][base : base + n] = rsi

    new_df = pd.DataFrame(columns, index=data.index)
    return pd.concat([data, new_df], axis=1)


def add_feature_close_sma_pct_diff(
    data: pd.DataFrame,
    period_list: list[int],
    *,
    column_name_fn: Callable[[int], str] | None = None,
) -> pd.DataFrame:
    """Add ``(close - SMA) / SMA`` per period, timestamp order per symbol (continuous across days)."""
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


def add_feature_close_bollinger_pct_diff(
    data: pd.DataFrame,
    period_list: list[int],
    *,
    std_multiples: Sequence[float] = (1.0, 2.0),
    upper_column_name_fn: Callable[[int, float], str] | None = None,
    lower_column_name_fn: Callable[[int, float], str] | None = None,
) -> pd.DataFrame:
    """Add Bollinger band percent distance from ``close`` per period and per std width.

    Middle band is SMA(close, period); upper / lower are ``middle +/- k * std`` for each
    ``k`` in ``std_multiples``, where ``std`` is the rolling **population** standard deviation
    of ``close`` over the same window (incomplete leading windows yield 0, matching
    ``_rolling_sma_np``). Columns are ``(close - upper) / upper`` and ``(close - lower) / lower``.

    Default column names use :func:`bollinger_std_column_tag`, e.g.
    ``close_bb_upper_20_1std_pct_diff``.

    Expects MultiIndex ``(symbol, timestamp)`` and column ``close``.
    """
    if not period_list:
        return data
    mults = tuple(std_multiples)
    if not mults:
        return data
    for k in mults:
        if k <= 0:
            raise ValueError("each std multiple must be positive")
    for p in period_list:
        if p < 1:
            raise ValueError("each Bollinger period must be >= 1")
    upper_fn = (
        upper_column_name_fn
        if upper_column_name_fn is not None
        else (lambda n, k: f"close_bb_upper_{n}_{bollinger_std_column_tag(k)}_pct_diff")
    )
    lower_fn = (
        lower_column_name_fn
        if lower_column_name_fn is not None
        else (lambda n, k: f"close_bb_lower_{n}_{bollinger_std_column_tag(k)}_pct_diff")
    )
    n_rows = len(data)
    symbol = data.index.get_level_values("symbol")
    col_names: list[str] = []
    for p in period_list:
        for k in mults:
            col_names.append(upper_fn(p, k))
            col_names.append(lower_fn(p, k))
    columns = {name: np.zeros(n_rows, dtype=np.float64) for name in col_names}

    for _sym, group in data.groupby(symbol, sort=False):
        group = group.sort_index(level="timestamp")
        locs = group.index
        base = _index_position(data, locs[0])
        n = len(group)
        close = group["close"].to_numpy(dtype=np.float64, copy=False)
        for p in period_list:
            for k in mults:
                up_pct, lo_pct = _bollinger_upper_lower_pct_diff_np(close, p, k)
                columns[upper_fn(p, k)][base : base + n] = up_pct
                columns[lower_fn(p, k)][base : base + n] = lo_pct

    new_df = pd.DataFrame(columns, index=data.index)
    return pd.concat([data, new_df], axis=1)


def _rolling_volume_mean_1d(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean of ``values`` over ``window`` bars; leading incomplete windows are 0."""
    n = len(values)
    out = np.zeros(n, dtype=np.float64)
    if window < 1 or n == 0:
        return out
    s = pd.Series(values, dtype=np.float64, copy=False)
    rolled = s.rolling(window, min_periods=window).mean().to_numpy(dtype=np.float64, copy=False)
    valid = np.isfinite(rolled)
    out[valid] = rolled[valid]
    return out


def add_volume_roll_mean_by_day(
    data: pd.DataFrame,
    window_list: list[int],
    *,
    column_name_fn: Callable[[int], str] | None = None,
) -> pd.DataFrame:
    """Per (symbol, trade_date) session rolling mean of ``volume``; column ``volume_roll_mean_w``."""
    if not window_list:
        return data
    for w in window_list:
        if w < 1:
            raise ValueError("each volume window size must be >= 1")
    name_fn = (
        column_name_fn
        if column_name_fn is not None
        else (lambda w: f"volume_roll_mean_{w}")
    )
    n_rows = len(data)
    trade_date = _trade_date_series(data)
    symbol = data.index.get_level_values("symbol")
    vol_all = data["volume"].to_numpy(dtype=np.float64, copy=False)
    columns = {name_fn(w): np.zeros(n_rows, dtype=np.float64) for w in window_list}

    for (_sym, _date), group in data.groupby([symbol, trade_date], sort=False):
        group = group.sort_index(level="timestamp")
        base = _index_position(data, group.index[0])
        n = len(group)
        seg = vol_all[base : base + n]
        for w in window_list:
            columns[name_fn(w)][base : base + n] = _rolling_volume_mean_1d(seg, w)

    new_df = pd.DataFrame(columns, index=data.index)
    return pd.concat([data, new_df], axis=1)


def _ema_1d(close: np.ndarray, span: int) -> np.ndarray:
    """EMA of ``close``; leading bars before ``span`` samples are 0."""
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
    if span < 1 or n == 0:
        return out
    s = pd.Series(close, dtype=np.float64, copy=False)
    ema = s.ewm(span=span, adjust=False, min_periods=span).mean()
    arr = ema.to_numpy(dtype=np.float64, copy=False)
    out[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _pct_diff_close_vs_reference(close: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``(close - ref) / ref`` with 0 where ``ref`` is not finite or not positive."""
    out = np.zeros(len(close), dtype=np.float64)
    valid = np.isfinite(ref) & (ref > 0) & np.isfinite(close)
    out[valid] = (close[valid] - ref[valid]) / ref[valid]
    return out


def _close_lag_pct_diff_np(close: np.ndarray, lag: int) -> np.ndarray:
    """``(close[i] - close[i-lag]) / close[i-lag]``; first ``lag`` indices are 0 (incomplete lookback)."""
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
    if lag < 1 or n <= lag:
        return out
    prev = close[:-lag]
    cur = close[lag:]
    valid = np.isfinite(prev) & (prev > 0) & np.isfinite(cur)
    seg = np.zeros(n - lag, dtype=np.float64)
    seg[valid] = (cur[valid] - prev[valid]) / prev[valid]
    out[lag:] = seg
    return out


def _rolling_session_vwap_pct_diff_np(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """Session slice: ``(typical - VWAP) / VWAP`` for rolling VWAP over ``window`` bars.

    **Current** price is the bar's typical price ``(high + low + close) / 3``. **Average VWAP**
    for the period is ``sum(typical * volume) / sum(volume)`` over the same ``window`` bars
    (``min_periods=1`` at the session start), i.e. the volume-weighted average for that lookback.
    """
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or window < 1:
        return out
    h = high.astype(np.float64, copy=False)
    l = low.astype(np.float64, copy=False)
    c = close.astype(np.float64, copy=False)
    v = np.maximum(volume.astype(np.float64, copy=False), 0.0)
    tp = (h + l + c) / 3.0
    pv = tp * v
    s_pv = (
        pd.Series(pv, dtype=np.float64, copy=False)
        .rolling(window, min_periods=1)
        .sum()
        .to_numpy(dtype=np.float64, copy=False)
    )
    s_v = (
        pd.Series(v, dtype=np.float64, copy=False)
        .rolling(window, min_periods=1)
        .sum()
        .to_numpy(dtype=np.float64, copy=False)
    )
    vwap = np.full(n, np.nan, dtype=np.float64)
    positive = s_v > 0
    vwap[positive] = s_pv[positive] / s_v[positive]
    return _pct_diff_close_vs_reference(tp, vwap)


def add_feature_close_vwap_pct_diff(
    data: pd.DataFrame,
    window_list: list[int],
    *,
    column_name_fn: Callable[[int], str] | None = None,
) -> pd.DataFrame:
    """Add ``(typical - rolling_VWAP) / rolling_VWAP`` for each window, per (symbol, trade_date).

    For each bar, **rolling VWAP** is the volume-weighted average price (**average VWAP**) over
    the last ``window`` bars in the **current session** (``min_periods=1`` after the open).
    **Typical price** ``(high + low + close) / 3`` is the **current** bar price vs that average.
    A window as large as the session (e.g. 390 minute bars) matches session-to-date VWAP on
    most bars.

    Expects MultiIndex ``(symbol, timestamp)`` and columns ``high``, ``low``, ``close``, ``volume``.
    """
    if not window_list:
        return data
    for w in window_list:
        if w < 1:
            raise ValueError("each VWAP window must be >= 1")
    name_fn = (
        column_name_fn
        if column_name_fn is not None
        else (lambda w: f"typical_vwap_{w}_pct_diff")
    )
    trade_date = _trade_date_series(data)
    symbol = data.index.get_level_values("symbol")
    n_rows = len(data)
    columns = {name_fn(w): np.zeros(n_rows, dtype=np.float64) for w in window_list}

    for (_sym, _date), group in data.groupby([symbol, trade_date], sort=False):
        group = group.sort_index(level="timestamp")
        base = _index_position(data, group.index[0])
        n = len(group)
        h = group["high"].to_numpy(dtype=np.float64, copy=False)
        l = group["low"].to_numpy(dtype=np.float64, copy=False)
        c = group["close"].to_numpy(dtype=np.float64, copy=False)
        v = group["volume"].to_numpy(dtype=np.float64, copy=False)
        for w in window_list:
            columns[name_fn(w)][base : base + n] = _rolling_session_vwap_pct_diff_np(h, l, c, v, w)

    new_df = pd.DataFrame(columns, index=data.index)
    return pd.concat([data, new_df], axis=1)


def add_feature_close_ema_pct_diff(
    data: pd.DataFrame,
    span_list: list[int],
    *,
    column_name_fn: Callable[[int], str] | None = None,
) -> pd.DataFrame:
    """Add ``(close - EMA) / EMA`` per span, timestamp order per symbol (continuous across days)."""
    if not span_list:
        return data
    for sp in span_list:
        if sp < 1:
            raise ValueError("each EMA span must be >= 1")
    name_fn = (
        column_name_fn
        if column_name_fn is not None
        else (lambda sp: f"close_ema_{sp}_pct_diff")
    )
    n_rows = len(data)
    symbol = data.index.get_level_values("symbol")
    columns = {name_fn(sp): np.zeros(n_rows, dtype=np.float64) for sp in span_list}

    for _sym, group in data.groupby(symbol, sort=False):
        group = group.sort_index(level="timestamp")
        base = _index_position(data, group.index[0])
        n = len(group)
        close = group["close"].to_numpy(dtype=np.float64, copy=False)
        for sp in span_list:
            ema = _ema_1d(close, sp)
            columns[name_fn(sp)][base : base + n] = _pct_diff_close_vs_reference(close, ema)

    new_df = pd.DataFrame(columns, index=data.index)
    return pd.concat([data, new_df], axis=1)


def add_feature_close_lag_pct_diff(
    data: pd.DataFrame,
    lag_list: list[int],
    *,
    column_name_fn: Callable[[int], str] | None = None,
) -> pd.DataFrame:
    """Add ``(close - close_lag) / close_lag`` for each lag, timestamp order per symbol.

    Same grouping as :func:`add_feature_close_sma_pct_diff`: chronological bars per symbol
    (lag may cross session boundaries).
    """
    if not lag_list:
        return data
    for lag in lag_list:
        if lag < 1:
            raise ValueError("each lag must be >= 1")
    name_fn = (
        column_name_fn
        if column_name_fn is not None
        else (lambda n: f"close_lag_{n}_pct_diff")
    )
    n_rows = len(data)
    symbol = data.index.get_level_values("symbol")
    columns = {name_fn(lag): np.zeros(n_rows, dtype=np.float64) for lag in lag_list}

    for _sym, group in data.groupby(symbol, sort=False):
        group = group.sort_index(level="timestamp")
        base = _index_position(data, group.index[0])
        n = len(group)
        close = group["close"].to_numpy(dtype=np.float64, copy=False)
        for lag in lag_list:
            columns[name_fn(lag)][base : base + n] = _close_lag_pct_diff_np(close, lag)

    new_df = pd.DataFrame(columns, index=data.index)
    return pd.concat([data, new_df], axis=1)


def add_forward_absolute_move_target(
    data: pd.DataFrame,
    pct_threshold: float,
    max_forward_bars: int,
    *,
    column_name: str = "move_target",
) -> pd.DataFrame:
    """Binary label: 1 if within the next ``max_forward_bars`` bars (same session) price moves
    at least ``pct_threshold`` in **either** direction vs current ``close``.

    Uses forward highs and lows only (bars after the current bar). Rows with no forward bars
    in the session are labeled 0.
    """
    if pct_threshold < 0:
        raise ValueError("pct_threshold must be non-negative")
    if max_forward_bars < 1:
        raise ValueError("max_forward_bars must be >= 1")

    targets = np.zeros(len(data), dtype=np.int64)
    trade_date = _trade_date_series(data)
    symbol = data.index.get_level_values("symbol")
    thr = float(pct_threshold)
    h_max = int(max_forward_bars)

    for (_sym, _date), group in data.groupby([symbol, trade_date], sort=False):
        group = group.sort_index(level="timestamp")
        locs = group.index
        highs = group["high"].to_numpy(dtype=np.float64, copy=False)
        lows = group["low"].to_numpy(dtype=np.float64, copy=False)
        closes = group["close"].to_numpy(dtype=np.float64, copy=False)
        n = len(group)
        base = _index_position(data, locs[0])
        day = np.zeros(n, dtype=np.int64)

        for i in range(n):
            start = i + 1
            end = min(n, i + 1 + h_max)
            if start >= end:
                continue
            c = float(closes[i])
            if not np.isfinite(c) or c <= 0:
                continue
            fh = float(np.max(highs[start:end]))
            fl = float(np.min(lows[start:end]))
            if not np.isfinite(fh) or not np.isfinite(fl):
                continue
            up = fh / c - 1.0
            down = 1.0 - fl / c
            if max(up, down) >= thr:
                day[i] = 1

        targets[base : base + n] = day

    data[column_name] = targets
    return data


def add_forward_move_eligibility_and_direction(
    data: pd.DataFrame,
    pct_threshold: float,
    max_forward_bars: int,
    *,
    move_eligible_column: str = "move_eligible",
    direction_column: str = "direction_target",
) -> pd.DataFrame:
    """Label rows where a ≥``pct_threshold`` move occurs within the forward window, and the direction.

    **Eligibility** matches :func:`add_forward_absolute_move_target`: within the next
    ``max_forward_bars`` bars of the same session, either ``max(high)/close - 1`` or
    ``1 - min(low)/close`` (vs **current** bar close) reaches ``pct_threshold``.

    **Direction** (only meaningful when eligible): ``1`` if the **up** threshold is first
    reached by the running forward maximum high, ``0`` if the **down** threshold is first
    reached by the running forward minimum low. Scan is chronological; if both first occur
    on the same bar, the larger proportional excursion from ``close`` wins; if equal, ``1``.

    Rows with no forward bars in the session are ineligible; ``direction_target`` is NaN there.

    Expects OHLC columns and MultiIndex ``(symbol, timestamp)``.
    """
    if pct_threshold < 0:
        raise ValueError("pct_threshold must be non-negative")
    if max_forward_bars < 1:
        raise ValueError("max_forward_bars must be >= 1")

    n_rows = len(data)
    eligible = np.zeros(n_rows, dtype=np.int64)
    direction = np.full(n_rows, np.nan, dtype=np.float64)
    trade_date = _trade_date_series(data)
    symbol = data.index.get_level_values("symbol")
    thr = float(pct_threshold)
    h_max = int(max_forward_bars)

    for (_sym, _date), group in data.groupby([symbol, trade_date], sort=False):
        group = group.sort_index(level="timestamp")
        locs = group.index
        highs = group["high"].to_numpy(dtype=np.float64, copy=False)
        lows = group["low"].to_numpy(dtype=np.float64, copy=False)
        closes = group["close"].to_numpy(dtype=np.float64, copy=False)
        n = len(group)
        base = _index_position(data, locs[0])

        for i in range(n):
            start = i + 1
            end = min(n, i + 1 + h_max)
            if start >= end:
                continue
            c = float(closes[i])
            if not np.isfinite(c) or c <= 0:
                continue

            hi = highs[start:end]
            lo = lows[start:end]
            m = len(hi)
            fh = float(np.max(hi))
            fl = float(np.min(lo))
            if not np.isfinite(fh) or not np.isfinite(fl):
                continue
            max_up_full = fh / c - 1.0
            max_down_full = 1.0 - fl / c
            if max(max_up_full, max_down_full) < thr:
                continue

            eligible[base + i] = 1
            run_max = float(hi[0])
            run_min = float(lo[0])
            tu: int | None = None
            td: int | None = None
            for k in range(m):
                run_max = max(run_max, float(hi[k]))
                run_min = min(run_min, float(lo[k]))
                up_now = run_max / c - 1.0
                down_now = 1.0 - run_min / c
                if tu is None and up_now >= thr:
                    tu = k
                if td is None and down_now >= thr:
                    td = k
                if tu is not None and td is not None:
                    break

            if tu is None:
                direction[base + i] = 0.0
            elif td is None:
                direction[base + i] = 1.0
            elif tu < td:
                direction[base + i] = 1.0
            elif td < tu:
                direction[base + i] = 0.0
            else:
                run_max_t = float(hi[0])
                run_min_t = float(lo[0])
                for k in range(tu + 1):
                    run_max_t = max(run_max_t, float(hi[k]))
                    run_min_t = min(run_min_t, float(lo[k]))
                u_exc = run_max_t / c - 1.0
                d_exc = 1.0 - run_min_t / c
                direction[base + i] = 1.0 if u_exc >= d_exc else 0.0

    data[move_eligible_column] = eligible
    data[direction_column] = direction
    return data
