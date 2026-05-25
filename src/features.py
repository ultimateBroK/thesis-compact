from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit


@njit(cache=True)
def fractional_diff_values(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    output = np.full(len(values), np.nan)
    for i in range(len(weights) - 1, len(values)):
        total = 0.0
        offset = i - len(weights) + 1
        for j in range(len(weights)):
            total += weights[j] * values[offset + j]
        output[i] = total
    return output


def fractional_diff_weights(d: float, threshold: float) -> np.ndarray:
    weights = [1.0]
    k = 1
    while True:
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            return np.array(weights[::-1])
        weights.append(weight)
        k += 1


def fractional_diff(series: pl.Series, d: float, threshold: float = 1e-4) -> pl.Series:
    output = fractional_diff_values(series.to_numpy(), fractional_diff_weights(d, threshold))
    return pl.Series(series.name, output)


def rsi(close: pl.Series, window: int) -> pl.Series:
    delta = close.diff()
    gain = delta.clip(lower_bound=0).rolling_mean(window)
    loss = (-delta.clip(upper_bound=0)).rolling_mean(window)
    return 100 - 100 / (1 + gain / loss)


def average_true_range(frame: pl.DataFrame, window: int) -> pl.Series:
    prev_close = frame["close"].shift(1)
    true_range = pl.max_horizontal(
        frame["high"] - frame["low"],
        (frame["high"] - prev_close).abs(),
        (frame["low"] - prev_close).abs(),
    )
    return true_range.rolling_mean(window)


def add_returns(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    return frame.with_columns([
        (close / close.shift(1) - 1).alias("return_1"),
        (close / close.shift(4) - 1).alias("return_4"),
        (close / close.shift(12) - 1).alias("return_12"),
    ])


def add_trend_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    ema_12 = close.ewm_mean(span=12, adjust=False)
    ema_26 = close.ewm_mean(span=26, adjust=False)
    macd = ema_12 - ema_26
    return frame.with_columns([
        (ema_12 / close - 1).alias("ema_12"),
        (ema_26 / close - 1).alias("ema_26"),
        macd.alias("macd"),
    ])


def add_momentum_features(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns([
        rsi(frame["close"], 14).alias("rsi_14"),
    ])


def add_volatility_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    bb_mid = close.rolling_mean(20)
    bb_std = close.rolling_std(20)
    ret = frame["return_1"]
    spread = frame["spread"]
    return frame.with_columns([
        (average_true_range(frame, 14) / close).alias("atr_14"),
        (4 * bb_std / bb_mid).alias("bb_width"),
        ((close - bb_mid) / (2 * bb_std)).alias("bb_position"),
        ret.rolling_std(24).alias("volatility_24"),
        (ret.rolling_std(6) / ret.rolling_std(24)).alias("vol_ratio_6_24"),
        ((spread - spread.rolling_mean(24)) / spread.rolling_std(24)).alias("spread_z_24"),
        ((close - frame["low"].rolling_min(24)) / (frame["high"].rolling_max(24) - frame["low"].rolling_min(24))).alias("close_in_range_24"),
    ])


def add_calendar_features(frame: pl.DataFrame) -> pl.DataFrame:
    ts = frame["timestamp"]
    hour = ts.dt.hour()
    frame = frame.with_columns([
        hour.alias("hour"),
        ts.dt.weekday().alias("dayofweek"),
        hour.is_between(0, 8, closed="left").cast(pl.Int8).alias("session_asia"),
        hour.is_between(8, 17, closed="left").cast(pl.Int8).alias("session_london"),
        hour.is_between(13, 22, closed="left").cast(pl.Int8).alias("session_us"),
        hour.is_between(8, 9, closed="left").cast(pl.Int8).alias("session_asia_london_overlap"),
        hour.is_between(13, 17, closed="left").cast(pl.Int8).alias("session_london_us_overlap"),
    ])
    session_label = (
        pl.when(pl.col("session_us") == 1).then(pl.lit("us"))
        .when(pl.col("session_london") == 1).then(pl.lit("london"))
        .otherwise(pl.lit("asia"))
    )
    frame = frame.with_columns(session_label.alias("_session"))
    vol_z = ((pl.col("volume") - pl.col("volume").mean().over("_session")) / pl.col("volume").std().over("_session"))
    return frame.with_columns(vol_z.alias("volume_session_z")).drop("_session")


def add_market_features(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.pipe(add_returns)
        .pipe(add_trend_features)
        .pipe(add_momentum_features)
        .pipe(add_volatility_features)
        .pipe(add_calendar_features)
    )


def add_technical_features(
    candles: pl.DataFrame,
    frac_d: float = 0.4,
) -> pl.DataFrame:
    close = candles["close"]
    frac = fractional_diff(close, frac_d).alias("close_fracdiff")
    return add_market_features(candles).with_columns(frac.fill_nan(None).fill_null(strategy="forward"))
