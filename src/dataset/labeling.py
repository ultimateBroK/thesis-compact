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
from src.labeling import assign_triple_barrier_labels, search_optimal_barrier_widths, summarize_label_distribution


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
    """Grid-search barriers on first 60% of train, validate on next 20%, apply best to all."""
    n = len(train_frame)
    search_end = int(n * 0.60)
    val_end = min(int(n * 0.80), n)

    if search_end < 100 or val_end - search_end < 50:
        # Not enough data for split: fall back to full train
        tp_atr, sl_atr, balance, dist = search_optimal_barrier_widths(
            train_frame,
            horizon=horizon,
            swing_window=swing_window,
            tp_range=TUNE_TP_RANGE,
            sl_range=TUNE_SL_RANGE,
            target_balance=TUNE_TARGET_BALANCE,
        )
        print(f"Auto-tuned barriers (full train): TP_ATR={tp_atr}, SL_ATR={sl_atr}, balance={balance}")
        print(f"Label distribution: {dist}")
        return tp_atr, sl_atr, balance, dist

    search_frame = train_frame.head(search_end)
    val_frame = train_frame.slice(search_end, val_end - search_end)

    tp_atr, sl_atr, balance, dist = search_optimal_barrier_widths(
        search_frame,
        horizon=horizon,
        swing_window=swing_window,
        tp_range=TUNE_TP_RANGE,
        sl_range=TUNE_SL_RANGE,
        target_balance=TUNE_TARGET_BALANCE,
    )

    # Validate on hold-out
    val_labels = apply_labels_to_frame(val_frame, tp_atr=tp_atr, sl_atr=sl_atr)
    val_dist = summarize_label_distribution(val_labels["label"].to_numpy())
    print(f"Auto-tuned barriers: TP_ATR={tp_atr}, SL_ATR={sl_atr}, search_balance={balance}")
    print(f"Validation distribution: {val_dist}")

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
