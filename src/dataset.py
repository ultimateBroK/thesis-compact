from __future__ import annotations

import numpy as np
import polars as pl

from src.config import (
    DATA_DIR,
    FALLBACK_SL_ATR,
    FALLBACK_TP_ATR,
    FRACTIONAL_D,
    LABELING_HORIZON,
    SWING_WINDOW,
    TIMEFRAME,
    PipelineConfig,
)
from src.data import load_xauusd_candles
from src.features import add_technical_features
from src.labeling import triple_barrier_labels


def clean_labeled_frame(frame: pl.DataFrame) -> pl.DataFrame:
    num_cols = [c for c in frame.columns if frame[c].dtype in pl.NUMERIC_DTYPES]
    return frame.with_columns([
        pl.when(pl.col(c).is_infinite())
          .then(np.nan)
          .otherwise(pl.col(c))
          .alias(c)
        for c in num_cols
    ]).drop_nulls()


def feature_columns(frame: pl.DataFrame) -> list[str]:
    excluded = {"label", "event_end", "open", "high", "low", "close", "timestamp"}
    return [column for column in frame.columns if column not in excluded]


def train_test_time_split(
    frame: pl.DataFrame,
    test_size: float = 0.2,
    purge_pct: float = 0.02,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    split = int(len(frame) * (1 - test_size))
    purge = int(np.ceil(len(frame) * purge_pct))

    if "event_end" in frame.columns:
        event_end = frame["event_end"].to_numpy()
        test_start_idx = split + purge
        max_train_event_end = int(event_end[:split].max())
        if max_train_event_end >= test_start_idx:
            extra = max_train_event_end - split + 1
            purge = max(purge, extra)

    train = frame.head(split)
    test = frame.slice(split + purge, None)

    if "timestamp" in frame.columns:
        split_ts = str(frame["timestamp"][split])
        gap_rows = purge
        print(f"Split at row {split} | timestamp: {split_ts} | purge gap: {gap_rows} rows")
        print(f"Train: {train['timestamp'][0]} → {train['timestamp'][-1]}")
        print(f"Test:  {test['timestamp'][0]} → {test['timestamp'][-1]}")

    return train, test


def build_dataset(config: PipelineConfig) -> pl.DataFrame:
    candles = load_xauusd_candles(DATA_DIR, config.months, TIMEFRAME)
    featured = add_technical_features(candles, frac_d=FRACTIONAL_D)
    return clean_labeled_frame(triple_barrier_labels(
        featured,
        horizon=LABELING_HORIZON,
        fallback_tp_atr=FALLBACK_TP_ATR,
        fallback_sl_atr=FALLBACK_SL_ATR,
        swing_window=SWING_WINDOW,
    ))
