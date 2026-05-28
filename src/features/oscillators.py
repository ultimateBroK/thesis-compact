from __future__ import annotations

import polars as pl


def compute_rsi(close: pl.Series, window: int) -> pl.Series:
    delta = close.diff()
    gain = delta.clip(lower_bound=0).rolling_mean(window)
    loss = (-delta.clip(upper_bound=0)).rolling_mean(window)
    return 100 - 100 / (1 + gain / loss)


def compute_average_true_range(frame: pl.DataFrame, window: int) -> pl.Series:
    prev_close = frame["close"].shift(1)
    true_range = pl.max_horizontal(
        frame["high"] - frame["low"],
        (frame["high"] - prev_close).abs(),
        (frame["low"] - prev_close).abs(),
    )
    return true_range.rolling_mean(window)
