"""Data loading: parquet → OHLC candles, labeling, chronological split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.config import DATA_DIR, LABELING_HORIZON, PURGE_PCT, TEST_SIZE, PipelineConfig
from src.features import combine_market_features
from src.labeling import assign_future_return_labels, summarize_label_distribution


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------


def collect_parquet_paths(data_dir: Path, months: int | None) -> list[Path]:
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    return files if months is None else files[-months:]


def load_candles_from_parquet(data_dir: Path, months: int | None, timeframe: str) -> pl.DataFrame:
    candles = (
        pl.scan_parquet([str(path) for path in collect_parquet_paths(data_dir, months)])
        .select(
            "timestamp",
            ((pl.col("ask") + pl.col("bid")) / 2).alias("mid"),
            (pl.col("ask") - pl.col("bid")).alias("spread"),
            (pl.col("ask_volume") + pl.col("bid_volume")).alias("tick_volume"),
        )
        .sort("timestamp")
        .group_by_dynamic("timestamp", every=timeframe)
        .agg(
            pl.col("mid").first().alias("open"),
            pl.col("mid").max().alias("high"),
            pl.col("mid").min().alias("low"),
            pl.col("mid").last().alias("close"),
            pl.col("tick_volume").sum().alias("volume"),
            pl.col("spread").mean().alias("spread"),
        )
        .drop_nulls()
        .collect(engine="streaming")
    )
    return candles


# ---------------------------------------------------------------------------
# Featured candles
# ---------------------------------------------------------------------------


def load_featured_candles(config: PipelineConfig) -> pl.DataFrame:
    candles = load_candles_from_parquet(DATA_DIR, config.months, config.timeframe)
    return combine_market_features(candles)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def forward_fill_infinite_values(frame: pl.DataFrame) -> pl.DataFrame:
    num_cols = [c for c in frame.columns if frame[c].dtype in pl.NUMERIC_DTYPES]
    return frame.with_columns([
        pl.when(pl.col(c).is_infinite()).then(np.nan).otherwise(pl.col(c)).alias(c)
        for c in num_cols
    ])


def apply_labels_to_frame(
    frame: pl.DataFrame,
    horizon: int = LABELING_HORIZON,
) -> pl.DataFrame:
    labeled = assign_future_return_labels(frame, horizon=horizon)
    cleaned = forward_fill_infinite_values(labeled).drop_nulls()
    event_end = np.minimum(
        np.arange(len(cleaned), dtype=np.int64) + horizon,
        max(len(cleaned) - 1, 0),
    )
    return cleaned.with_columns(pl.Series("event_end", event_end))


# ---------------------------------------------------------------------------
# Build labeled dataset (train/test split)
# ---------------------------------------------------------------------------


def compute_test_start(featured_len: int, split: int, horizon: int) -> tuple[int, int]:
    purge = max(int(np.ceil(featured_len * PURGE_PCT)), horizon)
    return split + purge, purge


def build_labeled_dataset(
    config: PipelineConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return (featured, train_labeled, test_labeled)."""
    featured = load_featured_candles(config)
    split = int(len(featured) * (1 - TEST_SIZE))
    test_start, purge = compute_test_start(len(featured), split, LABELING_HORIZON)
    if test_start >= len(featured):
        raise ValueError(
            f"Test set empty after purge: test_start={test_start} >= len={len(featured)}"
        )

    train_labeled = apply_labels_to_frame(featured.head(split))
    test_labeled = apply_labels_to_frame(featured.slice(test_start, None))
    if train_labeled.is_empty() or test_labeled.is_empty():
        raise ValueError("Labeled train/test set is empty; reduce horizon or load more data")

    print(f"Split point: {split} | purge gap: {purge} | test start: {test_start}")
    if "timestamp" in featured.columns:
        print(f"Train range: {featured['timestamp'][0]} -> {featured['timestamp'][split - 1]}")
        print(f"Test  range: {featured['timestamp'][test_start]} -> {featured['timestamp'][-1]}")
    print(f"Train label distribution: {summarize_label_distribution(train_labeled['label'].to_numpy())}")
    print(f"Test  label distribution: {summarize_label_distribution(test_labeled['label'].to_numpy())}")

    return featured, train_labeled, test_labeled
