"""
Dataset assembly pipeline: config → labeled train/test split.

Orchestration: load/data → tune barriers → label → split.
"""
from __future__ import annotations

import polars as pl

from src.config import AUTO_TUNE_BARRIERS, FALLBACK_SL_ATR, FALLBACK_TP_ATR, TEST_SIZE, PipelineConfig
from src.labeling import summarize_label_distribution

from .builder import load_featured_candles
from .labeling import apply_labels_to_frame, auto_calibrate_barrier_widths


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
