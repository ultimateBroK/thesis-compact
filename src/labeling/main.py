"""
Labeling pipeline: OHLC frame → triple-barrier labels + event_end.

Orchestration: scan_barriers_from_frame → assign_triple_barrier_labels.
"""
from __future__ import annotations

import polars as pl

from .barriers import scan_barriers_from_frame


def assign_triple_barrier_labels(
    frame: pl.DataFrame,
    horizon: int = 12,
    fallback_tp_atr: float = 2.0,
    fallback_sl_atr: float = 1.5,
    swing_window: int = 5,
) -> pl.DataFrame:
    labels, event_end = scan_barriers_from_frame(
        frame, horizon, fallback_tp_atr, fallback_sl_atr, swing_window,
    )
    labeled = frame.with_columns([
        pl.Series("label", labels),
        pl.Series("event_end", event_end),
    ])
    return labeled.head(-horizon)
