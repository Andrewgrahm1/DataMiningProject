"""Compare TQQQ vs SQQQ: when TQQQ moves about -X%, does SQQQ move about +X% over the same span?

Uses cleaned minute bars (``elib.pull_and_clean``), aligned on timestamp. Reports:
- Daily close-to-close returns: events where TQQQ is near a target drop (e.g. -2%), distribution of SQQQ.
- Same-trading-day forward returns over a few bar horizons (no overnight gap in the window).

Run from project root::

    python -m experiments.orion.comp
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.orion.config import DEFAULT_ORION_CONFIG
from experiments.orion.elib import pull_and_clean


def _close_by_timestamp(df: pd.DataFrame) -> pd.Series:
    """Single-symbol frame: (symbol, timestamp) -> close series indexed by timestamp."""
    sym = df.index.get_level_values("symbol")[0]
    s = df["close"].xs(sym, level="symbol").sort_index()
    return s


def _align_closes(
    start, end, sym_a: str = "TQQQ", sym_b: str = "SQQQ"
) -> pd.DataFrame:
    a = _close_by_timestamp(pull_and_clean(sym_a, start, end))
    b = _close_by_timestamp(pull_and_clean(sym_b, start, end))
    out = pd.DataFrame({sym_a.lower(): a, sym_b.lower(): b})
    out = out.dropna(how="any")
    return out


def _daily_last_close(close: pd.Series) -> pd.Series:
    ts = close.index
    if getattr(ts, "tz", None) is not None:
        dates = ts.tz_convert("America/New_York").normalize().date
    else:
        dates = ts.date
    return close.groupby(pd.Index(dates, name="day")).last()


def _ny_trade_date_index(ts: pd.DatetimeIndex) -> np.ndarray:
    if getattr(ts, "tz", None) is not None:
        ny = ts.tz_convert("America/New_York")
    else:
        ny = ts.tz_localize("UTC").tz_convert("America/New_York")
    return ny.date


def _same_day_forward_simple_return(
    closes: pd.DataFrame, trade_dates: np.ndarray, horizon_bars: int
) -> tuple[pd.Series, pd.Series]:
    """Per-row simple return from current close to close ``horizon_bars`` later, same session only."""
    tqqq = closes["tqqq"].to_numpy(dtype=np.float64, copy=False)
    sqqq = closes["sqqq"].to_numpy(dtype=np.float64, copy=False)
    n = len(closes)
    rt = np.full(n, np.nan, dtype=np.float64)
    rs = np.full(n, np.nan, dtype=np.float64)
    idx = closes.index
    td = trade_dates

    bounds = np.concatenate([[0], np.nonzero(td[1:] != td[:-1])[0] + 1, [n]])
    for bi in range(len(bounds) - 1):
        lo, hi = bounds[bi], bounds[bi + 1]
        seg_t = tqqq[lo:hi]
        seg_s = sqqq[lo:hi]
        m = hi - lo
        if m <= horizon_bars:
            continue
        for i in range(m - horizon_bars):
            j = i + horizon_bars
            if seg_t[i] > 0 and seg_s[i] > 0:
                rt[lo + i] = seg_t[j] / seg_t[i] - 1.0
                rs[lo + i] = seg_s[j] / seg_s[i] - 1.0

    return (
        pd.Series(rt, index=idx, name=f"tqqq_fwd_{horizon_bars}"),
        pd.Series(rs, index=idx, name=f"sqqq_fwd_{horizon_bars}"),
    )


def _print_band_stats(
    label: str,
    r_tqqq: pd.Series,
    r_sqqq: pd.Series,
    target_tqqq: float,
    band: float,
) -> None:
    """Rows where r_tqqq is in [target - band, target + band]."""
    lo, hi = target_tqqq - band, target_tqqq + band
    m = r_tqqq.notna() & r_sqqq.notna() & (r_tqqq >= lo) & (r_tqqq <= hi)
    n = int(m.sum())
    print(f"\n{label}")
    print(f"  TQQQ return in [{lo*100:.3f}%, {hi*100:.3f}%]: n = {n:,}")
    if n == 0:
        return
    s = r_sqqq[m]
    expected_up = -target_tqqq
    sym_err = s + r_tqqq[m]  # want ~0 if perfect inverse same magnitude
    in_mirror_band = (s >= expected_up - band) & (s <= expected_up + band)
    print(f"  SQQQ return: mean {s.mean()*100:.4f}%  median {s.median()*100:.4f}%  std {s.std()*100:.4f}%")
    print(
        f"  (TQQQ + SQQQ) mean {sym_err.mean()*100:.4f}%  (0 = perfect inverse, same magnitude)"
    )
    print(
        f"  SQQQ within [{expected_up - band:.4f}, {expected_up + band:.4f}] "
        f"(mirror of TQQQ band): {in_mirror_band.sum():,} ({100*in_mirror_band.mean():.2f}%)"
    )


def main() -> None:
    cfg = DEFAULT_ORION_CONFIG
    start, end = cfg.start_date, cfg.end_date
    print(f"TQQQ vs SQQQ comparison  |  {start.date()} .. {end.date()}")
    print("(Leveraged ETFs rebalance; same-magnitude inverse is approximate.)\n")

    aligned = _align_closes(start, end)
    print(f"Aligned minute bars: {len(aligned):,} rows")

    # Daily close-to-close
    d_t = _daily_last_close(aligned["tqqq"])
    d_s = _daily_last_close(aligned["sqqq"])
    daily = pd.DataFrame({"tqqq": d_t, "sqqq": d_s}).sort_index()
    daily_ret = daily.pct_change().dropna()
    valid_d = daily_ret["tqqq"].notna() & daily_ret["sqqq"].notna()
    dt = daily_ret.loc[valid_d, "tqqq"]
    ds = daily_ret.loc[valid_d, "sqqq"]
    if len(dt) > 2:
        corr = float(np.corrcoef(dt, ds)[0, 1])
        print(f"\nDaily close-to-close correlation(TQQQ, SQQQ): {corr:.4f}")

    target = -0.02
    band = 0.002
    _print_band_stats("Daily", dt, ds, target, band)

    # Same-day intraday horizons (minutes)
    td_arr = _ny_trade_date_index(aligned.index)
    for h in (1, 30, 60, 90, 120, 180, 390):
        rt, rs = _same_day_forward_simple_return(aligned, td_arr, h)
        _print_band_stats(f"Same session, forward {h} bar(s)", rt, rs, target, band)


if __name__ == "__main__":
    main()
