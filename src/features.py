"""Feature engineering: technical indicators, fractional differentiation, feature assembly."""

from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit

# ---------------------------------------------------------------------------
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
# Low-level: fractional differentiation
# ---------------------------------------------------------------------------


@njit(cache=True)
def apply_fractional_diff(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    output = np.full(len(values), np.nan)
    for i in range(len(weights) - 1, len(values)):
        total = 0.0
        offset = i - len(weights) + 1
        for j in range(len(weights)):
            total += weights[j] * values[offset + j]
        output[i] = total
    return output


def compute_fractional_diff_weights(d: float, threshold: float) -> np.ndarray:
    weights = [1.0]
    k = 1
    while True:
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            return np.array(weights[::-1])
        weights.append(weight)
        k += 1


def derive_fractionally_differentiated_series(series: pl.Series, d: float, threshold: float = 1e-4) -> pl.Series:
    output = apply_fractional_diff(series.to_numpy(), compute_fractional_diff_weights(d, threshold))
    return pl.Series(series.name, output)


# ---------------------------------------------------------------------------
# Feature generators
# ---------------------------------------------------------------------------


def add_return_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    return frame.with_columns([
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
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).clip(lower_bound=1e-8)) * 100
    adx = dx.rolling_mean(14)
    return frame.with_columns([
        compute_rsi(close, 14).alias("rsi_14"),
        adx.alias("adx_14"),
    ])


def add_volatility_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    bb_mid = close.rolling_mean(20)
    bb_std = close.rolling_std(20)
    ret = close / close.shift(1) - 1
    spread = frame["spread"]
    return frame.with_columns([
        # Normalize ATR to relative (fraction of close) for feature scaling
        (compute_average_true_range(frame, 14) / close).alias("atr_14"),
        (4 * bb_std / bb_mid).alias("bb_width"),
        ((close - bb_mid) / (2 * bb_std)).alias("bb_position"),
        ret.rolling_std(24).alias("volatility_24"),
        (ret.rolling_std(6) / ret.rolling_std(24)).alias("vol_ratio_6_24"),
        ((spread - spread.rolling_mean(24)) / spread.rolling_std(24)).alias("spread_z_24"),
        ((close - frame["low"].rolling_min(24)) / (frame["high"].rolling_max(24) - frame["low"].rolling_min(24))).alias("close_in_range_24"),
    ])


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
    return frame.with_columns([
        obv.alias("obv"),
        obv_delta.alias("obv_delta_12"),
    ])


def add_calendar_features(frame: pl.DataFrame) -> pl.DataFrame:
    ts = frame["timestamp"]
    return frame.with_columns([
        ts.dt.hour().alias("hour"),
        ts.dt.weekday().alias("dayofweek"),
    ])


def combine_market_features(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame
        .pipe(add_return_features)
        .pipe(add_trend_features)
        .pipe(add_momentum_features)
        .pipe(add_volatility_features)
        .pipe(add_volume_features)
        .pipe(add_calendar_features)
    )


def build_feature_frame(
    candles: pl.DataFrame,
    frac_d: float = 0.4,
) -> pl.DataFrame:
    close = candles["close"]
    frac = derive_fractionally_differentiated_series(close, frac_d).alias("close_fracdiff")
    return combine_market_features(candles).with_columns(frac.fill_nan(None).fill_null(strategy="forward"))


__all__ = [
    "add_calendar_features",
    "add_momentum_features",
    "add_return_features",
    "add_trend_features",
    "add_volume_features",
    "add_volatility_features",
    "apply_fractional_diff",
    "build_feature_frame",
    "combine_market_features",
    "compute_average_true_range",
    "compute_fractional_diff_weights",
    "compute_rsi",
    "derive_fractionally_differentiated_series",
]
