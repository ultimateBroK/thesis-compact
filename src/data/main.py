"""
Data loading pipeline: parquet → OHLC candles.

Orchestration: collect_parquet_file_paths → load_candles_from_parquet.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from .loader import collect_parquet_file_paths


def load_candles_from_parquet(data_dir: Path, months: int | None, timeframe: str) -> pl.DataFrame:
    """Read parquet files and resample tick data into OHLC candles."""
    candles = (
        pl.scan_parquet([str(path) for path in collect_parquet_file_paths(data_dir, months)])
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
