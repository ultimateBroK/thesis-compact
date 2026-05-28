from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def detect_swing_extremes(high: np.ndarray, low: np.ndarray, window: int):
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
def derive_trailing_swing_levels(high: np.ndarray, low: np.ndarray, window: int):
    sh_full, sl_full = detect_swing_extremes(high, low, window)
    n = len(high)
    lag = window + 1
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    last_h = np.nan
    last_l = np.nan
    for i in range(n):
        src = i - lag
        if src >= 0:
            if np.isfinite(sh_full[src]):
                last_h = sh_full[src]
            if np.isfinite(sl_full[src]):
                last_l = sl_full[src]
        sh[i] = last_h
        sl[i] = last_l
    return sh, sl
