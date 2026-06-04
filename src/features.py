"""Feature engineering: technical indicators and feature assembly."""

from __future__ import annotations

import numpy as np
import polars as pl


# Low-level: oscillators
# ---------------------------------------------------------------------------


def compute_rsi(close: pl.Series, window: int) -> pl.Series:
    delta = close.diff()
    gain = delta.clip(lower_bound=0).rolling_mean(window)
    loss = (-delta.clip(upper_bound=0)).rolling_mean(window)
    return 100 - 100 / (1 + gain / loss)


def compute_average_true_range(frame: pl.DataFrame, window: int) -> pl.Series:
    """Compute Average True Range in raw price units (USD/oz).

    Note: Feature builder normalizes this by dividing by close price
    to produce a relative ATR for feature scaling.
    """
    prev_close = frame["close"].shift(1)
    true_range = pl.max_horizontal(
        frame["high"] - frame["low"],
        (frame["high"] - prev_close).abs(),
        (frame["low"] - prev_close).abs(),
    )
    return true_range.rolling_mean(window)


# ---------------------------------------------------------------------------
# Feature generators
# ---------------------------------------------------------------------------


def add_return_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    return frame.with_columns(
        [
            (close / close.shift(4) - 1).alias("return_4"),
            (close / close.shift(12) - 1).alias("return_12"),
        ]
    )


def add_trend_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    ema_12 = close.ewm_mean(span=12, adjust=False)
    ema_26 = close.ewm_mean(span=26, adjust=False)
    macd = ema_12 - ema_26
    return frame.with_columns(
        [
            (ema_12 / close - 1).alias("ema_12"),
            (ema_26 / close - 1).alias("ema_26"),
            (macd / close).alias("macd_pct"),
        ]
    )


def add_momentum_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    prev_close = close.shift(1)
    plus_dm = (
        pl.when((high - high.shift(1)) > (low.shift(1) - low))
        .then((high - high.shift(1)).clip(lower_bound=0))
        .otherwise(0.0)
    )
    minus_dm = (
        pl.when((low.shift(1) - low) > (high - high.shift(1)))
        .then((low.shift(1) - low).clip(lower_bound=0))
        .otherwise(0.0)
    )
    tr = pl.max_horizontal(
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    )
    atr_14_price = tr.rolling_mean(14)
    plus_di = 100 * plus_dm.rolling_mean(14) / atr_14_price
    minus_di = 100 * minus_dm.rolling_mean(14) / atr_14_price
    dx = (
        (plus_di - minus_di).abs() / (plus_di + minus_di).clip(lower_bound=1e-8)
    ) * 100
    adx = dx.rolling_mean(14)
    return frame.with_columns(
        [
            compute_rsi(close, 14).alias("rsi_14"),
            adx.alias("adx_14"),
        ]
    )


def add_volatility_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    bb_mid = close.rolling_mean(20)
    bb_std = close.rolling_std(20)
    ret = close / close.shift(1) - 1
    spread = frame["spread"]
    return frame.with_columns(
        [
            # Normalize ATR to relative (fraction of close) for feature scaling
            (compute_average_true_range(frame, 14) / close).alias("atr_14"),
            (4 * bb_std / bb_mid).alias("bb_width"),
            ((close - bb_mid) / (2 * bb_std)).alias("bb_position"),
            ret.rolling_std(24).alias("volatility_24"),
            (ret.rolling_std(6) / ret.rolling_std(24)).alias("vol_ratio_6_24"),
            ((spread - spread.rolling_mean(24)) / spread.rolling_std(24)).alias(
                "spread_z_24"
            ),
            (
                (close - frame["low"].rolling_min(24))
                / (frame["high"].rolling_max(24) - frame["low"].rolling_min(24))
            ).alias("close_in_range_24"),
        ]
    )


def add_volume_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    volume = frame["volume"]
    direction = (
        pl.when(close > close.shift(1))
        .then(1)
        .otherwise(pl.when(close < close.shift(1)).then(-1).otherwise(0))
    )
    obv = (direction * volume).cum_sum()
    obv_delta = obv - obv.shift(12)
    obv_z_48 = (obv - obv.rolling_mean(48)) / obv.rolling_std(48)
    return frame.with_columns(
        [
            obv_z_48.alias("obv_z_48"),
            obv_delta.alias("obv_delta_12"),
        ]
    )


def add_candle_structure_features(frame: pl.DataFrame) -> pl.DataFrame:
    open_ = frame["open"]
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    body = close - open_
    upper = high - pl.max_horizontal(open_, close)
    lower = pl.min_horizontal(open_, close) - low
    range_ = high - low
    return frame.with_columns(
        [
            (body / close).alias("body_pct"),
            (upper / close).alias("upper_wick_pct"),
            (lower / close).alias("lower_wick_pct"),
            (range_ / close).alias("range_pct"),
        ]
    )


def add_microstructure_features(frame: pl.DataFrame) -> pl.DataFrame:
    tick_count = frame["tick_count"]
    spread = frame["spread"]
    close = frame["close"]
    return frame.with_columns(
        [
            (spread / close).alias("spread_pct"),
            tick_count.log1p().alias("log_tick_count"),
            (
                (tick_count - tick_count.rolling_mean(24)) / tick_count.rolling_std(24)
            ).alias("tick_count_z_24"),
        ]
    )


def add_calendar_features(frame: pl.DataFrame) -> pl.DataFrame:
    ts = frame["timestamp"]
    hour = ts.dt.hour().cast(pl.Float64)
    dow = ts.dt.weekday().cast(pl.Float64)
    return frame.with_columns(
        [
            (2 * np.pi * hour / 24).sin().alias("hour_sin"),
            (2 * np.pi * hour / 24).cos().alias("hour_cos"),
            (2 * np.pi * dow / 7).sin().alias("dow_sin"),
            (2 * np.pi * dow / 7).cos().alias("dow_cos"),
        ]
    )


def get_feature_columns(frame: pl.DataFrame) -> list[str]:
    """Return column names usable as model features (exclude labels, OHLC, metadata)."""
    excluded = {
        "label",
        "event_end",
        "future_return",
        "future_gap_hours",
        "open",
        "high",
        "low",
        "close",
        "timestamp",
    }
    return [column for column in frame.columns if column not in excluded]


def combine_market_features(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.pipe(add_return_features)
        .pipe(add_trend_features)
        .pipe(add_momentum_features)
        .pipe(add_volatility_features)
        .pipe(add_volume_features)
        .pipe(add_candle_structure_features)
        .pipe(add_microstructure_features)
        .pipe(add_calendar_features)
    )


__all__ = [
    "add_calendar_features",
    "add_candle_structure_features",
    "add_microstructure_features",
    "add_momentum_features",
    "add_return_features",
    "add_trend_features",
    "add_volume_features",
    "add_volatility_features",
    "combine_market_features",
    "compute_average_true_range",
    "compute_rsi",
    "get_feature_columns",
]
