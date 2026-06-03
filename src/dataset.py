"""Dataset: data loading, purge-aware splitting, label application, feature assembly."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.config import (
    DATA_DIR,
    FRACTIONAL_D,
    LABELING_HORIZON,
    PURGE_PCT,
    SWING_WINDOW,
    TEST_SIZE,
    TUNE_SL_RANGE,
    TUNE_TP_RANGE,
    TUNE_TARGET_BALANCE,
    PipelineConfig,
)
from src.data import load_candles_from_parquet
from src.features import build_feature_frame
from src.labeling import (
    assign_triple_barrier_labels,
    search_optimal_barrier_widths,
    summarize_label_distribution,
)


# ── Data loading ──────────────────────────────────────────────────────


def load_featured_candles(config: PipelineConfig) -> pl.DataFrame:
    candles = load_candles_from_parquet(DATA_DIR, config.months, config.timeframe)
    return build_feature_frame(candles, frac_d=FRACTIONAL_D)


# ── Purge-aware split ────────────────────────────────────────────────


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


# ── Label helpers ────────────────────────────────────────────────────


def forward_fill_infinite_values(frame: pl.DataFrame) -> pl.DataFrame:
    num_cols = [c for c in frame.columns if frame[c].dtype in pl.NUMERIC_DTYPES]
    return frame.with_columns([
        pl.when(pl.col(c).is_infinite()).then(np.nan).otherwise(pl.col(c)).alias(c)
        for c in num_cols
    ])


def apply_labels_to_frame(
    frame: pl.DataFrame,
    tp_atr: float,
    sl_atr: float,
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


# ── Calibration ──────────────────────────────────────────────────────


def calibrate_barrier_params(
    train_frame: pl.DataFrame,
    horizon: int = LABELING_HORIZON,
    swing_window: int = SWING_WINDOW,
) -> tuple[float, float, float, dict[str, int | float]]:
    """Grid-search barriers on first 60% of train, validate on next 20%, apply best to all."""
    n = len(train_frame)
    search_end = int(n * 0.60)
    val_end = min(int(n * 0.80), n)

    if search_end < 100 or val_end - search_end < 50:
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

    val_labels = apply_labels_to_frame(val_frame, tp_atr=tp_atr, sl_atr=sl_atr)
    val_dist = summarize_label_distribution(val_labels["label"].to_numpy())
    print(f"Auto-tuned barriers: TP_ATR={tp_atr}, SL_ATR={sl_atr}, search_balance={balance}")
    print(f"Validation distribution: {val_dist}")

    return tp_atr, sl_atr, balance, dist


# ── Public: build labeled dataset ────────────────────────────────────


def build_labeled_dataset(
    config: PipelineConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, float, float]:
    """Return (featured, train_labeled, test_labeled, tp_atr, sl_atr)."""
    featured = load_featured_candles(config)

    tune_cut = int(len(featured) * (1 - TEST_SIZE))
    train_portion = featured.head(tune_cut)

    # Calibrate barriers on training data only
    tp_atr, sl_atr, _, _ = calibrate_barrier_params(train_portion)

    # Label training data first — we need event_end to compute purge gap
    train_labeled = apply_labels_to_frame(train_portion, tp_atr, sl_atr)

    # Compute purge gap from event_end column
    base_purge = int(np.ceil(len(featured) * PURGE_PCT))
    purge = base_purge
    if "event_end" in train_labeled.columns:
        event_end_train = train_labeled["event_end"].to_numpy()
        max_event_end = int(event_end_train.max())
        if max_event_end >= tune_cut:
            purge = max(purge, max_event_end - tune_cut + 1)

    test_start = tune_cut + purge
    assert test_start < len(featured), (
        f"Test set empty after purge: test_start={test_start} >= len={len(featured)}"
    )

    test_portion = featured.slice(test_start, None)
    test_labeled = apply_labels_to_frame(test_portion, tp_atr, sl_atr)

    print(f"Split point: {tune_cut} | purge gap: {purge} | test start: {test_start}")
    if "timestamp" in featured.columns:
        print(f"Train range: {featured['timestamp'][0]} -> {featured['timestamp'][tune_cut - 1]}")
        print(f"Test  range: {featured['timestamp'][test_start]} -> {featured['timestamp'][-1]}")
    print(f"Train label distribution: {summarize_label_distribution(train_labeled['label'].to_numpy())}")
    print(f"Test  label distribution: {summarize_label_distribution(test_labeled['label'].to_numpy())}")

    return featured, train_labeled, test_labeled, tp_atr, sl_atr


# ── Feature columns ─────────────────────────────────────────────────


def get_feature_columns(frame: pl.DataFrame) -> list[str]:
    excluded = {"label", "event_end", "open", "high", "low", "close", "timestamp"}
    return [c for c in frame.columns if c not in excluded]
