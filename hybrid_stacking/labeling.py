from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


def triple_barrier_labels(
    frame: pd.DataFrame,
    horizon: int = 12,
    take_profit_atr: float = 1.5,
    stop_loss_atr: float = 1.0,
) -> pd.DataFrame:
    labels, event_end = scan_barriers(frame, horizon, take_profit_atr, stop_loss_atr)
    labeled = frame.copy()
    labeled["label"] = labels
    labeled["event_end"] = event_end
    return labeled.iloc[:-horizon]


def scan_barriers(
    frame: pd.DataFrame,
    horizon: int,
    take_profit_atr: float,
    stop_loss_atr: float,
) -> tuple[np.ndarray, np.ndarray]:
    close = frame["close"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    atr = frame["atr_14"].to_numpy()
    return scan_barrier_arrays(close, high, low, atr, horizon, take_profit_atr, stop_loss_atr)


@njit(cache=True)
def scan_barrier_arrays(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    take_profit_atr: float,
    stop_loss_atr: float,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros(len(close), dtype=np.int64)
    event_end = np.arange(len(close), dtype=np.int64)
    for start in range(len(close) - horizon):
        labels[start], event_end[start] = first_barrier_hit(
            close, high, low, atr, start, horizon, take_profit_atr, stop_loss_atr
        )
    return labels, event_end


@njit(cache=True)
def first_barrier_hit(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    start: int,
    horizon: int,
    take_profit_atr: float,
    stop_loss_atr: float,
) -> tuple[int, int]:
    if not np.isfinite(atr[start]) or atr[start] <= 0:
        return 0, start

    upper = close[start] + take_profit_atr * atr[start]
    lower = close[start] - stop_loss_atr * atr[start]
    horizon_end = start + horizon

    for current in range(start + 1, horizon_end + 1):
        if high[current] >= upper:
            return 1, current
        if low[current] <= lower:
            return -1, current
    return 0, horizon_end
