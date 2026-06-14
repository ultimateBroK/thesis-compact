"""Tải dữ liệu: parquet → nến OHLC, gán nhãn, chia theo thời gian."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from src.config import (
    DATA_DIR,
    LABELING_HORIZON,
    LABEL_RETURN_THRESHOLD,
    MAX_LABEL_GAP_HOURS,
    PipelineConfig,
)
from src.features.engineering import combine_market_features
from .labeling import assign_future_return_labels, summarize_label_distribution


@dataclass(frozen=True)
class DatasetSplitInfo:
    split: int
    purge: int
    test_start: int
    train_range: tuple[pl.Datetime, pl.Datetime] | None
    test_range: tuple[pl.Datetime, pl.Datetime] | None
    train_label_distribution: dict[str, int | float]
    test_label_distribution: dict[str, int | float]


@dataclass(frozen=True)
class LabeledDataset:
    featured: pl.DataFrame
    train_labeled: pl.DataFrame
    test_labeled: pl.DataFrame
    test_continuous: pl.DataFrame
    split_info: DatasetSplitInfo

    # Cho phép tuple unpacking: featured, train, test_labeled, test_continuous = dataset
    def __iter__(self):
        yield self.featured
        yield self.train_labeled
        yield self.test_labeled
        yield self.test_continuous


def build_dataset_split_info(
    featured: pl.DataFrame,
    train_labeled: pl.DataFrame,
    test_labeled: pl.DataFrame,
    split: int,
    purge: int,
    test_start: int,
) -> DatasetSplitInfo:
    train_range = None
    test_range = None
    if "timestamp" in featured.columns:
        train_range = (featured["timestamp"][0], featured["timestamp"][split - 1])
        test_range = (featured["timestamp"][test_start], featured["timestamp"][-1])
    return DatasetSplitInfo(
        split=split,
        purge=purge,
        test_start=test_start,
        train_range=train_range,
        test_range=test_range,
        train_label_distribution=summarize_label_distribution(
            train_labeled["label"].to_numpy()
        ),
        test_label_distribution=summarize_label_distribution(
            test_labeled["label"].to_numpy()
        ),
    )


# ---------------------------------------------------------------------------
# Tải dữ liệu thô
# ---------------------------------------------------------------------------


def collect_parquet_paths(data_dir: Path, months: int | None) -> list[Path]:
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    return files if months is None else files[-months:]


def load_candles_from_parquet(
    data_dir: Path, months: int | None, timeframe: str
) -> pl.DataFrame:
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
            pl.len().alias("tick_count"),
        )
        .drop_nulls()
        .collect(engine="streaming")
    )
    return pl.DataFrame(candles)


# ---------------------------------------------------------------------------
# Nến đã có đặc trưng
# ---------------------------------------------------------------------------


def load_featured_candles(config: PipelineConfig) -> pl.DataFrame:
    candles = load_candles_from_parquet(DATA_DIR, config.months, config.timeframe)
    return combine_market_features(candles)


# ---------------------------------------------------------------------------
# Hàm hỗ trợ gán nhãn
# ---------------------------------------------------------------------------


def replace_infinite_with_nan(frame: pl.DataFrame) -> pl.DataFrame:
    num_cols = [name for name, dtype in frame.schema.items() if dtype.is_numeric()]
    return frame.with_columns(
        [
            pl.when(pl.col(c).is_infinite()).then(np.nan).otherwise(pl.col(c)).alias(c)
            for c in num_cols
        ]
    )


def apply_labels_to_frame(
    frame: pl.DataFrame,
    horizon: int = LABELING_HORIZON,
    threshold: float = LABEL_RETURN_THRESHOLD,
    max_gap_hours: float = MAX_LABEL_GAP_HOURS,
) -> pl.DataFrame:
    labeled = assign_future_return_labels(
        frame,
        horizon=horizon,
        threshold=threshold,
        max_gap_hours=max_gap_hours,
    )
    return replace_infinite_with_nan(labeled).drop_nulls()


# ---------------------------------------------------------------------------
# Tạo dataset đã gán nhãn (chia train/test)
# ---------------------------------------------------------------------------


def compute_test_start(split: int, purge_bars: int) -> tuple[int, int]:
    """Tính dòng test đầu tiên sau purge gap để tránh rò rỉ nhãn."""
    return split + purge_bars, purge_bars


def build_labeled_dataset(config: PipelineConfig) -> LabeledDataset:
    """Trả về feature frame, train/test đã gán nhãn, test liên tục và metadata chia tập."""
    featured = load_featured_candles(config)
    split = int(len(featured) * (1 - config.test_size))
    test_start, purge = compute_test_start(split, config.purge_bars)
    if test_start >= len(featured):
        raise ValueError(
            f"Test set empty after purge: test_start={test_start}"
            f" >= len={len(featured)}"
        )

    train_labeled = apply_labels_to_frame(
        featured.head(split),
        horizon=config.labeling_horizon,
        threshold=config.label_return_threshold,
        max_gap_hours=config.max_label_gap_hours,
    )
    test_labeled = apply_labels_to_frame(
        featured.slice(test_start, None),
        horizon=config.labeling_horizon,
        threshold=config.label_return_threshold,
        max_gap_hours=config.max_label_gap_hours,
    )
    test_continuous = replace_infinite_with_nan(
        featured.slice(test_start, None)
    ).drop_nulls()
    if train_labeled.is_empty() or test_labeled.is_empty():
        raise ValueError(
            "Labeled train/test set is empty; reduce horizon or load more data"
        )
    if test_continuous.is_empty():
        raise ValueError("Continuous test set is empty after invalid-value cleanup")

    split_info = build_dataset_split_info(
        featured,
        train_labeled,
        test_labeled,
        split,
        purge,
        test_start,
    )
    return LabeledDataset(
        featured=featured,
        train_labeled=train_labeled,
        test_labeled=test_labeled,
        test_continuous=test_continuous,
        split_info=split_info,
    )
