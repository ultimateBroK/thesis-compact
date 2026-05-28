from __future__ import annotations

import numpy as np
import polars as pl

from src.config import (
    FALLBACK_SL_ATR,
    FALLBACK_TP_ATR,
    LABELING_HORIZON,
    SWING_WINDOW,
    TUNE_SL_RANGE,
    TUNE_TP_RANGE,
    TUNE_TARGET_BALANCE,
)
from src.labeling import assign_triple_barrier_labels, search_optimal_barrier_widths


def forward_fill_infinite_values(frame: pl.DataFrame) -> pl.DataFrame:
    num_cols = [c for c in frame.columns if frame[c].dtype in pl.NUMERIC_DTYPES]
    return frame.with_columns([
        pl.when(pl.col(c).is_infinite()).then(np.nan).otherwise(pl.col(c)).alias(c)
        for c in num_cols
    ])


def auto_calibrate_barrier_widths(
    train_frame: pl.DataFrame,
    horizon: int = LABELING_HORIZON,
    swing_window: int = SWING_WINDOW,
) -> tuple[float, float, float, dict[str, int | float]]:
    tp_atr, sl_atr, balance, dist = search_optimal_barrier_widths(
        train_frame,
        horizon=horizon,
        swing_window=swing_window,
        tp_range=TUNE_TP_RANGE,
        sl_range=TUNE_SL_RANGE,
        target_balance=TUNE_TARGET_BALANCE,
    )
    print(f"Auto-tuned barriers: TP_ATR={tp_atr}, SL_ATR={sl_atr}, balance={balance}")
    print(f"Label distribution: {dist}")
    return tp_atr, sl_atr, balance, dist


def apply_labels_to_frame(
    frame: pl.DataFrame,
    tp_atr: float = FALLBACK_TP_ATR,
    sl_atr: float = FALLBACK_SL_ATR,
    horizon: int = LABELING_HORIZON,
    swing_window: int = SWING_WINDOW,
) -> pl.DataFrame:
    labeled = assign_triple_barrier_labels(
        frame,
        horizon=horizon,
        fallback_tp_atr=tp_atr,
        fallback_sl_atr=sl_atr,
        swing_window=swing_window,
    )
    return forward_fill_infinite_values(labeled).drop_nulls()
