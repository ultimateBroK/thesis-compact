from __future__ import annotations

import numpy as np
import polars as pl

from src.config import (
    AUTO_TUNE_BARRIERS,
    DATA_DIR,
    FALLBACK_SL_ATR,
    FALLBACK_TP_ATR,
    FRACTIONAL_D,
    TEST_SIZE,
    PipelineConfig,
)
from src.data import load_candles_from_parquet
from src.features import enrich_with_technical_features
from src.labeling import summarize_label_distribution

from .labeling import apply_labels_to_frame, auto_calibrate_barrier_widths


def load_featured_candles(config: PipelineConfig) -> pl.DataFrame:
    candles = load_candles_from_parquet(DATA_DIR, config.months, config.timeframe)
    return enrich_with_technical_features(candles, frac_d=FRACTIONAL_D)


def compute_purge_gap(event_end: np.ndarray, split: int, purge: int) -> int:
    max_event_end_in_train = int(event_end[:split].max())
    test_start = split + purge
    if max_event_end_in_train >= test_start:
        return max(purge, max_event_end_in_train - split + 1)
    return purge


def derive_train_test_split(
    frame: pl.DataFrame,
    test_size: float = 0.2,
    purge_pct: float = 0.02,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    split = int(len(frame) * (1 - test_size))
    purge = int(np.ceil(len(frame) * purge_pct))

    if "event_end" in frame.columns:
        purge = compute_purge_gap(frame["event_end"].to_numpy(), split, purge)

    train = frame.head(split)
    test = frame.slice(split + purge, None)

    if "timestamp" in frame.columns:
        split_ts = str(frame["timestamp"][split])
        print(f"Split at row {split} | timestamp: {split_ts} | purge gap: {purge} rows")
        print(f"Train: {train['timestamp'][0]} -> {train['timestamp'][-1]}")
        print(f"Test:  {test['timestamp'][0]} -> {test['timestamp'][-1]}")

    return train, test, purge


def extract_feature_columns(frame: pl.DataFrame) -> list[str]:
    excluded = {"label", "event_end", "open", "high", "low", "close", "timestamp"}
    return [c for c in frame.columns if c not in excluded]


def assemble_labeled_dataset(config: PipelineConfig) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    featured = load_featured_candles(config)

    tune_cut = int(len(featured) * (1 - TEST_SIZE))
    train_portion = featured.head(tune_cut)
    test_portion = featured.slice(tune_cut, None)

    tp_atr = FALLBACK_TP_ATR
    sl_atr = FALLBACK_SL_ATR

    if AUTO_TUNE_BARRIERS:
        tp_atr, sl_atr, _, _ = auto_calibrate_barrier_widths(train_portion)

    train_labeled = apply_labels_to_frame(train_portion, tp_atr, sl_atr)
    test_labeled = apply_labels_to_frame(test_portion, tp_atr, sl_atr)

    print(f"Train label distribution: {summarize_label_distribution(train_labeled['label'].to_numpy())}")
    print(f"Test label distribution: {summarize_label_distribution(test_labeled['label'].to_numpy())}")

    return featured, train_labeled, test_labeled
