from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit


@njit(cache=True)
def compute_swing_levels(high: np.ndarray, low: np.ndarray, window: int):
    n = len(high)
    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)

    for i in range(window, n - window):
        is_swing_high = True
        is_swing_low = True
        for j in range(1, window + 1):
            if high[i] <= high[i - j] or high[i] <= high[i + j]:
                is_swing_high = False
            if low[i] >= low[i - j] or low[i] >= low[i + j]:
                is_swing_low = False
        if is_swing_high:
            swing_highs[i] = high[i]
        if is_swing_low:
            swing_lows[i] = low[i]

    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    last_high = np.nan
    last_low = np.nan
    for i in range(n):
        if not np.isnan(swing_highs[i]):
            last_high = swing_highs[i]
        sh[i] = last_high
        if not np.isnan(swing_lows[i]):
            last_low = swing_lows[i]
        sl[i] = last_low

    return sh, sl


@njit(cache=True)
def first_barrier_hit_swing(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    swing_high_level: np.ndarray,
    swing_low_level: np.ndarray,
    atr: np.ndarray,
    start: int,
    horizon: int,
    fallback_tp_atr: float,
    fallback_sl_atr: float,
) -> tuple[int, int]:
    if not np.isfinite(atr[start]) or atr[start] <= 0:
        return 0, start

    if np.isfinite(swing_high_level[start]) and swing_high_level[start] > close[start]:
        upper = swing_high_level[start]
    else:
        upper = close[start] + fallback_tp_atr * atr[start]

    if np.isfinite(swing_low_level[start]) and swing_low_level[start] < close[start]:
        lower = swing_low_level[start]
    else:
        lower = close[start] - fallback_sl_atr * atr[start]

    horizon_end = start + horizon

    for current in range(start + 1, horizon_end + 1):
        if high[current] >= upper:
            return 1, current
        if low[current] <= lower:
            return -1, current
    return 0, horizon_end


@njit(cache=True)
def scan_barrier_arrays_swing(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    swing_high_level: np.ndarray,
    swing_low_level: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    fallback_tp_atr: float,
    fallback_sl_atr: float,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros(len(close), dtype=np.int64)
    event_end = np.arange(len(close), dtype=np.int64)
    for start in range(len(close) - horizon):
        labels[start], event_end[start] = first_barrier_hit_swing(
            close, high, low, swing_high_level, swing_low_level, atr,
            start, horizon, fallback_tp_atr, fallback_sl_atr,
        )
    return labels, event_end


def scan_barriers(
    frame: pl.DataFrame,
    horizon: int,
    fallback_tp_atr: float,
    fallback_sl_atr: float,
    swing_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    close = frame["close"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    atr = (frame["atr_14"] * frame["close"]).to_numpy()
    swing_high_level, swing_low_level = compute_swing_levels(high, low, swing_window)
    return scan_barrier_arrays_swing(
        close, high, low, swing_high_level, swing_low_level, atr,
        horizon, fallback_tp_atr, fallback_sl_atr,
    )


def triple_barrier_labels(
    frame: pl.DataFrame,
    horizon: int = 12,
    fallback_tp_atr: float = 2.0,
    fallback_sl_atr: float = 1.5,
    swing_window: int = 5,
) -> pl.DataFrame:
    labels, event_end = scan_barriers(
        frame, horizon, fallback_tp_atr, fallback_sl_atr, swing_window,
    )
    labeled = frame.with_columns([
        pl.Series("label", labels),
        pl.Series("event_end", event_end),
    ])
    return labeled.head(-horizon)
