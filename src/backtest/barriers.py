from __future__ import annotations

import numpy as np

from src.config import FALLBACK_SL_ATR, FALLBACK_TP_ATR, MAX_LOSS_ATR


def compute_atr_from_raw_ohlc(n: int, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1)),
    ))
    atr = np.full(n, np.nan)
    for j in range(13, n):
        atr[j] = tr[j] if j == 13 else (atr[j - 1] * 13 + tr[j]) / 14
    return atr


def detect_barrier_breach(
    i: int,
    direction: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tp_price: float,
    sl_price: float,
    deadline: int,
) -> float | None:
    if direction > 0:
        if high[i] >= tp_price:
            return tp_price
        if low[i] <= sl_price:
            return sl_price
        if i >= deadline:
            return close[i]
    else:
        if low[i] <= tp_price:
            return tp_price
        if high[i] >= sl_price:
            return sl_price
        if i >= deadline:
            return close[i]
    return None


def derive_barrier_levels(
    i: int,
    direction: float,
    close: float,
    entry_price: float,
    atr_abs: np.ndarray,
    swing_high: np.ndarray,
    swing_low: np.ndarray,
    fallback_tp_atr: float = FALLBACK_TP_ATR,
    fallback_sl_atr: float = FALLBACK_SL_ATR,
    max_loss_atr: float = MAX_LOSS_ATR,
) -> tuple[float, float]:
    atr_i = atr_abs[i]
    if not (np.isfinite(atr_i) and atr_i > 0):
        return (np.inf if direction > 0 else -np.inf), (-np.inf if direction > 0 else np.inf)

    if direction > 0:
        tp = swing_high[i] if np.isfinite(swing_high[i]) and swing_high[i] > close else close + fallback_tp_atr * atr_i
        sl = swing_low[i] if np.isfinite(swing_low[i]) and swing_low[i] < close else close - fallback_sl_atr * atr_i
        hard_sl = entry_price - max_loss_atr * atr_i
        sl = max(sl, hard_sl)
    else:
        tp = swing_low[i] if np.isfinite(swing_low[i]) and swing_low[i] < close else close - fallback_tp_atr * atr_i
        sl = swing_high[i] if np.isfinite(swing_high[i]) and swing_high[i] > close else close + fallback_sl_atr * atr_i
        hard_sl = entry_price + max_loss_atr * atr_i
        sl = min(sl, hard_sl)
    return tp, sl
