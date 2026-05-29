from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit

from .swing import derive_trailing_swing_levels


@njit(cache=True)
def detect_first_barrier_breach(
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
def scan_triple_barrier_arrays(
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
        labels[start], event_end[start] = detect_first_barrier_breach(
            close, high, low, swing_high_level, swing_low_level, atr,
            start, horizon, fallback_tp_atr, fallback_sl_atr,
        )
    # Map time-expiry (0) to -1: unresolved = assume failure
    labels[labels == 0] = -1
    return labels, event_end


def scan_barriers_from_frame(
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
    swing_high_level, swing_low_level = derive_trailing_swing_levels(high, low, swing_window)
    return scan_triple_barrier_arrays(
        close, high, low, swing_high_level, swing_low_level, atr,
        horizon, fallback_tp_atr, fallback_sl_atr,
    )
