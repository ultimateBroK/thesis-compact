"""Labeling: fixed-horizon future-return labels for binary signal prediction."""

from __future__ import annotations

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------


def compute_future_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    """Return close[t + horizon] / close[t] - 1, NaN where unavailable."""
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


def assign_future_return_labels(
    frame: pl.DataFrame,
    horizon: int = 24,
) -> pl.DataFrame:
    """Assign binary labels from fixed-horizon close-to-close returns.

    Label semantics:
      * +1: close[t + horizon] > close[t]
      * -1: close[t + horizon] <= close[t]

    ``event_end`` stores the vertical barrier index used by purged CV. The
    final ``horizon`` rows are dropped because their label would require future
    prices outside the available sample.
    """
    close = frame["close"].to_numpy()
    future_return = compute_future_returns(close, horizon)
    labels = np.where(future_return > 0.0, 1, -1).astype(np.int64)
    event_end = np.minimum(np.arange(len(frame), dtype=np.int64) + horizon, len(frame) - 1)

    labeled = frame.with_columns([
        pl.Series("future_return", future_return),
        pl.Series("label", labels),
        pl.Series("event_end", event_end),
    ])
    return labeled.head(-horizon) if len(labeled) > horizon else labeled.head(0)


def summarize_label_distribution(labels: np.ndarray) -> dict[str, int | float]:
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    mapping = {1: "Buy (+1)", -1: "Sell (-1)"}
    dist: dict[str, int | float] = {}
    for label, count in zip(unique, counts):
        dist[mapping.get(int(label), str(label))] = int(count)
    dist["total"] = total
    dist["balance_ratio"] = float(round(min(counts) / max(counts), 4) if len(counts) > 1 else 0.0)
    return dist


__all__ = [
    "assign_future_return_labels",
    "compute_future_returns",
    "summarize_label_distribution",
]
