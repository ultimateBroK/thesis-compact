"""Feature engineering: technical indicators and feature assembly."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.config import (
    ADX_WINDOW,
    ATR_WINDOW,
    BB_WINDOW,
    BUY_LABEL,
    DAYS_PER_WEEK,
    EMA_FAST_WINDOW,
    EMA_SLOW_WINDOW,
    HOURS_PER_DAY,
    OBV_DELTA_WINDOW,
    OBV_Z_WINDOW,
    RANGE_WINDOW,
    RETURN_LONG_WINDOW,
    RETURN_SHORT_WINDOW,
    RSI_WINDOW,
    SELL_LABEL,
    SPREAD_Z_WINDOW,
    TICK_COUNT_Z_WINDOW,
    VOL_LONG_WINDOW,
    VOL_SHORT_WINDOW,
)

FEATURE_COLUMNS = (
    "volume",
    "spread",
    "tick_count",
    "return_4",
    "return_12",
    "ema_12",
    "ema_26",
    "macd_pct",
    "rsi_14",
    "adx_14",
    "atr_14",
    "bb_width",
    "bb_position",
    "volatility_24",
    "vol_ratio_6_24",
    "spread_z_24",
    "close_in_range_24",
    "obv_z_48",
    "obv_delta_12",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "range_pct",
    "spread_pct",
    "log_tick_count",
    "tick_count_z_24",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)


def compute_rsi(close: pl.Series, window: int) -> pl.Series | pl.Expr:
    delta = close.diff()
    gain = delta.clip(lower_bound=0).rolling_mean(window)
    loss = (-delta.clip(upper_bound=0)).rolling_mean(window)
    return 100 - 100 / (1 + gain / loss)


def compute_average_true_range(frame: pl.DataFrame, window: int) -> pl.Series | pl.Expr:
    """Compute Average True Range in raw price units (USD/oz)."""
    prev_close = frame["close"].shift(1)
    true_range = pl.max_horizontal(
        frame["high"] - frame["low"],
        (frame["high"] - prev_close).abs(),
        (frame["low"] - prev_close).abs(),
    )
    return true_range.rolling_mean(window)


def add_return_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    return frame.with_columns(
        [
            (close / close.shift(RETURN_SHORT_WINDOW) - 1).alias("return_4"),
            (close / close.shift(RETURN_LONG_WINDOW) - 1).alias("return_12"),
        ]
    )


def add_trend_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    ema_fast = close.ewm_mean(span=EMA_FAST_WINDOW, adjust=False)
    ema_slow = close.ewm_mean(span=EMA_SLOW_WINDOW, adjust=False)
    macd = ema_fast - ema_slow
    return frame.with_columns(
        [
            (ema_fast / close - 1).alias("ema_12"),
            (ema_slow / close - 1).alias("ema_26"),
            (macd / close).alias("macd_pct"),
        ]
    )


def add_momentum_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
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
    atr_price = compute_average_true_range(frame, ATR_WINDOW)
    plus_di = 100 * plus_dm.rolling_mean(ADX_WINDOW) / atr_price
    minus_di = 100 * minus_dm.rolling_mean(ADX_WINDOW) / atr_price
    dx = (
        (plus_di - minus_di).abs() / (plus_di + minus_di).clip(lower_bound=1e-8)
    ) * 100
    adx = dx.rolling_mean(ADX_WINDOW)
    return frame.with_columns(
        [
            compute_rsi(close, RSI_WINDOW).alias("rsi_14"),
            adx.alias("adx_14"),
        ]
    )


def add_volatility_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    bb_mid = close.rolling_mean(BB_WINDOW)
    bb_std = close.rolling_std(BB_WINDOW)
    ret = close / close.shift(1) - 1
    spread = frame["spread"]
    low = frame["low"]
    high = frame["high"]
    return frame.with_columns(
        [
            (compute_average_true_range(frame, ATR_WINDOW) / close).alias("atr_14"),
            (4 * bb_std / bb_mid).alias("bb_width"),
            ((close - bb_mid) / (2 * bb_std)).alias("bb_position"),
            ret.rolling_std(VOL_LONG_WINDOW).alias("volatility_24"),
            (
                ret.rolling_std(VOL_SHORT_WINDOW) / ret.rolling_std(VOL_LONG_WINDOW)
            ).alias("vol_ratio_6_24"),
            (
                (spread - spread.rolling_mean(SPREAD_Z_WINDOW))
                / spread.rolling_std(SPREAD_Z_WINDOW)
            ).alias("spread_z_24"),
            (
                (close - low.rolling_min(RANGE_WINDOW))
                / (high.rolling_max(RANGE_WINDOW) - low.rolling_min(RANGE_WINDOW))
            ).alias("close_in_range_24"),
        ]
    )


def add_volume_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    volume = frame["volume"]
    direction = (
        pl.when(close > close.shift(1))
        .then(BUY_LABEL)
        .otherwise(pl.when(close < close.shift(1)).then(SELL_LABEL).otherwise(0))
    )
    obv = (direction * volume).cum_sum()
    obv_delta = obv - obv.shift(OBV_DELTA_WINDOW)
    obv_z = (obv - obv.rolling_mean(OBV_Z_WINDOW)) / obv.rolling_std(OBV_Z_WINDOW)
    return frame.with_columns(
        [
            obv_z.alias("obv_z_48"),
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
                (tick_count - tick_count.rolling_mean(TICK_COUNT_Z_WINDOW))
                / tick_count.rolling_std(TICK_COUNT_Z_WINDOW)
            ).alias("tick_count_z_24"),
        ]
    )


def add_calendar_features(frame: pl.DataFrame) -> pl.DataFrame:
    ts = frame["timestamp"]
    hour = ts.dt.hour().cast(pl.Float64)
    dow = ts.dt.weekday().cast(pl.Float64)
    return frame.with_columns(
        [
            (2 * np.pi * hour / HOURS_PER_DAY).sin().alias("hour_sin"),
            (2 * np.pi * hour / HOURS_PER_DAY).cos().alias("hour_cos"),
            (2 * np.pi * dow / DAYS_PER_WEEK).sin().alias("dow_sin"),
            (2 * np.pi * dow / DAYS_PER_WEEK).cos().alias("dow_cos"),
        ]
    )


def get_feature_columns(frame: pl.DataFrame) -> list[str]:
    """Return known model feature columns present in ``frame``."""
    return [column for column in FEATURE_COLUMNS if column in frame.columns]


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
