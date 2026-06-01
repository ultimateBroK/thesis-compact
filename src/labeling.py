"""Labeling: swing highs/lows, barrier scanning, triple-barrier label assignment."""

from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit


# ---------------------------------------------------------------------------
# Swing high/low detection (numba @njit)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Barrier scanning (numba @njit)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------


def assign_triple_barrier_labels(
    frame: pl.DataFrame,
    horizon: int = 12,
    fallback_tp_atr: float = 2.0,
    fallback_sl_atr: float = 1.5,
    swing_window: int = 5,
) -> pl.DataFrame:
    """Assign triple-barrier labels to OHLC data.

    DESIGN DECISION: Time-expiry events (label=0) are mapped to -1 (failure).
    Rationale: This converts the problem to binary classification (-1, +1).
    Unresolved horizons are treated conservatively as failed signals —
    no trade would be generated for these cases.
    The thesis documents this choice explicitly.
    """
    labels, event_end = scan_barriers_from_frame(
        frame, horizon, fallback_tp_atr, fallback_sl_atr, swing_window,
    )
    labeled = frame.with_columns([
        pl.Series("label", labels),
        pl.Series("event_end", event_end),
    ])
    return labeled.head(-horizon)


def summarize_label_distribution(labels: np.ndarray) -> dict[str, int | float]:
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    mapping = {1: "TP (+1)", -1: "SL (-1)"}
    dist: dict[str, int | float] = {}
    for lbl, cnt in zip(unique, counts):
        dist[mapping.get(int(lbl), str(lbl))] = cnt
    dist["total"] = total
    dist["balance_ratio"] = float(round(min(counts) / max(counts), 4) if len(counts) > 1 else 0.0)
    return dist


def search_optimal_barrier_widths(
    frame: pl.DataFrame,
    horizon: int = 12,
    swing_window: int = 5,
    tp_range: tuple[float, float, float] = (0.5, 4.0, 0.25),
    sl_range: tuple[float, float, float] = (0.5, 4.0, 0.25),
    target_balance: float = 0.50,
) -> tuple[float, float, float, dict[str, int | float]]:
    best_balance = 0.0
    best_tp = 2.0
    best_sl = 1.5
    best_dist: dict[str, int | float] = {}

    tp_start, tp_end, tp_step = tp_range
    sl_start, sl_end, sl_step = sl_range
    tp_vals = np.arange(tp_start, tp_end + tp_step / 2, tp_step)
    sl_vals = np.arange(sl_start, sl_end + sl_step / 2, sl_step)

    for tp in tp_vals:
        for sl in sl_vals:
            if round(tp, 4) <= round(sl, 4):
                continue
            labels, _ = scan_barriers_from_frame(frame, horizon, round(tp, 4), round(sl, 4), swing_window)
            labels_clean = labels[: len(labels) - horizon]
            _, counts = np.unique(labels_clean, return_counts=True)
            if len(counts) < 2:
                balance = 0.0
            else:
                balance = float(min(counts) / max(counts))
            if balance > best_balance:
                best_balance = balance
                best_tp = round(tp, 4)
                best_sl = round(sl, 4)
                best_dist = summarize_label_distribution(labels_clean)
            if best_balance >= target_balance:
                break
        if best_balance >= target_balance:
            break

    return best_tp, best_sl, best_balance, best_dist


__all__ = [
    "assign_triple_barrier_labels",
    "derive_trailing_swing_levels",
    "detect_swing_extremes",
    "scan_barriers_from_frame",
    "search_optimal_barrier_widths",
    "summarize_label_distribution",
]
