"""Gán nhãn tín hiệu nhị phân từ lợi suất tương lai theo fixed horizon."""

from __future__ import annotations

import numpy as np
import polars as pl
from src.config import BUY_LABEL, SELL_LABEL


# ---------------------------------------------------------------------------
# Gán nhãn
# ---------------------------------------------------------------------------


def compute_future_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    """Tính close[t + horizon] / close[t] - 1; dùng NaN khi thiếu dữ liệu tương lai."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    future_return = np.full(len(close), np.nan, dtype=np.float64)
    if len(close) <= horizon:
        return future_return

    base = close[:-horizon]
    future = close[horizon:]
    np.divide(future, base, out=future_return[:-horizon], where=base != 0)
    future_return[:-horizon] -= 1.0
    return future_return


def compute_future_time_gaps_hours(timestamps: np.ndarray, horizon: int) -> np.ndarray:
    """Tính độ dài gap thời gian giữa t và t+horizon theo giờ."""
    gaps = np.full(len(timestamps), np.nan, dtype=np.float64)
    if len(timestamps) <= horizon:
        return gaps
    delta = timestamps[horizon:] - timestamps[:-horizon]
    gaps[:-horizon] = delta.astype("timedelta64[s]").astype(np.float64) / 3600.0
    return gaps


def assign_future_return_labels(
    frame: pl.DataFrame,
    horizon: int = 4,
    threshold: float = 0.0005,
    max_gap_hours: float | None = None,
) -> pl.DataFrame:
    """Gán nhãn nhị phân từ lợi suất close-to-close theo fixed horizon.

    Ý nghĩa nhãn:
      * +1: close[t + horizon] / close[t] - 1 > threshold
      * -1: close[t + horizon] / close[t] - 1 < -threshold
      * Mẫu có |return| <= threshold bị bỏ (không gán nhãn).

    ``event_start`` và ``event_end`` lưu tọa độ vertical barrier để dùng trong
    purged CV. ``horizon`` dòng cuối bị bỏ vì nhãn cần giá tương lai ngoài mẫu.
    """
    close = frame["close"].to_numpy()
    future_return = compute_future_returns(close, horizon)

    valid = np.isfinite(future_return)
    valid &= np.abs(future_return) > threshold

    if max_gap_hours is not None and "timestamp" in frame.columns:
        timestamps = frame["timestamp"].to_numpy()
        gap_hours = compute_future_time_gaps_hours(timestamps, horizon)
        valid &= gap_hours <= max_gap_hours
    else:
        gap_hours = np.full(len(frame), np.nan, dtype=np.float64)

    labels = np.where(future_return > threshold, BUY_LABEL, SELL_LABEL).astype(np.int64)
    event_start = np.arange(len(frame), dtype=np.int64)
    event_end = np.minimum(event_start + horizon, len(frame) - 1)

    labeled = frame.with_columns(
        [
            pl.Series("future_return", future_return),
            pl.Series("future_gap_hours", gap_hours),
            pl.Series("label", labels),
            pl.Series("event_start", event_start),
            pl.Series("event_end", event_end),
            pl.Series("label_is_valid", valid),
        ]
    )
    labeled = labeled.filter(pl.col("label_is_valid")).drop("label_is_valid")
    return labeled


def summarize_label_distribution(labels: np.ndarray) -> dict[str, int | float]:
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    mapping = {BUY_LABEL: "Buy (+1)", SELL_LABEL: "Sell (-1)"}
    dist: dict[str, int | float] = {}
    for label, count in zip(unique, counts):
        dist[mapping.get(int(label), str(label))] = int(count)
    dist["total"] = total
    dist["balance_ratio"] = float(
        round(min(counts) / max(counts), 4) if len(counts) > 1 else 0.0
    )
    return dist
