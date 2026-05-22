from __future__ import annotations

from pathlib import Path

import polars as pl


def parquet_files(data_dir: Path, months: int | None) -> list[Path]:
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    return files if months is None else files[:months]


def load_xauusd_candles(data_dir: Path, months: int | None, timeframe: str) -> pl.DataFrame:
    candles = (
        pl.scan_parquet([str(path) for path in parquet_files(data_dir, months)])
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
