from __future__ import annotations

import polars as pl

from .fractional import derive_fractionally_differentiated_series
from .oscillators import compute_average_true_range, compute_rsi


def generate_returns_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    return frame.with_columns([
        (close / close.shift(4) - 1).alias("return_4"),
        (close / close.shift(12) - 1).alias("return_12"),
    ])


def generate_trend_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    ema_12 = close.ewm_mean(span=12, adjust=False)
    ema_26 = close.ewm_mean(span=26, adjust=False)
    macd = ema_12 - ema_26
    return frame.with_columns([
        (ema_12 / close - 1).alias("ema_12"),
        (ema_26 / close - 1).alias("ema_26"),
        macd.alias("macd"),
    ])


def generate_momentum_features(frame: pl.DataFrame) -> pl.DataFrame:
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
    atr_14 = tr.rolling_mean(14)
    plus_di = 100 * plus_dm.rolling_mean(14) / atr_14
    minus_di = 100 * minus_dm.rolling_mean(14) / atr_14
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).clip(lower_bound=1e-8)) * 100
    adx = dx.rolling_mean(14)
    return frame.with_columns([
        compute_rsi(close, 14).alias("rsi_14"),
        adx.alias("adx_14"),
    ])


def generate_volatility_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    bb_mid = close.rolling_mean(20)
    bb_std = close.rolling_std(20)
    ret = close / close.shift(1) - 1
    spread = frame["spread"]
    return frame.with_columns([
        (compute_average_true_range(frame, 14) / close).alias("atr_14"),
        (4 * bb_std / bb_mid).alias("bb_width"),
        ((close - bb_mid) / (2 * bb_std)).alias("bb_position"),
        ret.rolling_std(24).alias("volatility_24"),
        (ret.rolling_std(6) / ret.rolling_std(24)).alias("vol_ratio_6_24"),
        ((spread - spread.rolling_mean(24)) / spread.rolling_std(24)).alias("spread_z_24"),
        ((close - frame["low"].rolling_min(24)) / (frame["high"].rolling_max(24) - frame["low"].rolling_min(24))).alias("close_in_range_24"),
    ])


def generate_volume_features(frame: pl.DataFrame) -> pl.DataFrame:
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


def generate_calendar_features(frame: pl.DataFrame) -> pl.DataFrame:
    ts = frame["timestamp"]
    return frame.with_columns([
        ts.dt.hour().alias("hour"),
        ts.dt.weekday().alias("dayofweek"),
    ])


def combine_market_features(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame
        .pipe(generate_returns_features)
        .pipe(generate_trend_features)
        .pipe(generate_momentum_features)
        .pipe(generate_volatility_features)
        .pipe(generate_volume_features)
        .pipe(generate_calendar_features)
    )


def enrich_with_technical_features(
    candles: pl.DataFrame,
    frac_d: float = 0.4,
) -> pl.DataFrame:
    close = candles["close"]
    frac = derive_fractionally_differentiated_series(close, frac_d).alias("close_fracdiff")
    return combine_market_features(candles).with_columns(frac.fill_nan(None).fill_null(strategy="forward"))
