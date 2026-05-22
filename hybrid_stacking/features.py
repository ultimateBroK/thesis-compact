from __future__ import annotations

import numpy as np
import polars as pl
import pywt
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


def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = "sym4",
    level: int = 3,
    mode: str = "soft",
) -> np.ndarray:
    n = len(signal)
    if n < 2:
        return signal.copy()
    padded = signal.copy()
    if n % 2 != 0:
        padded = np.append(padded, padded[-1])
    level = min(level, pywt.swt_max_level(len(padded)))
    coeffs = pywt.swt(padded, wavelet, level=level, trim_approx=True, norm=True)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(padded)))
    denoised = [coeffs[0]] + [pywt.threshold(d, threshold, mode=mode) for d in coeffs[1:]]
    result = pywt.iswt(denoised, wavelet, norm=True)
    return result[:n]


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
        macd.ewm_mean(span=9, adjust=False).alias("macd_signal"),
    ])


def add_momentum_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    low_14 = frame["low"].rolling_min(14)
    high_14 = frame["high"].rolling_max(14)
    stoch = 100 * (close - low_14) / (high_14 - low_14)
    stoch = stoch.replace(0, np.nan)
    median_price = (frame["high"] + frame["low"]) / 2
    ao = median_price.rolling_mean(5) - median_price.rolling_mean(34)
    return frame.with_columns([
        rsi(close, 14).alias("rsi_14"),
        stoch.alias("stoch_14"),
        ao.alias("ao"),
    ])


def add_volatility_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = frame["close"]
    bb_mid = close.rolling_mean(20)
    bb_std = close.rolling_std(20)
    return frame.with_columns([
        (average_true_range(frame, 14) / close).alias("atr_14"),
        (4 * bb_std / bb_mid).alias("bb_width"),
        ((close - bb_mid) / (2 * bb_std)).alias("bb_position"),
        frame["return_1"].rolling_std(24).alias("volatility_24"),
    ])


def add_calendar_features(frame: pl.DataFrame) -> pl.DataFrame:
    ts = frame["timestamp"]
    return frame.with_columns([
        ts.dt.hour().alias("hour"),
        ts.dt.weekday().alias("dayofweek"),
    ])


def add_market_features(frame: pl.DataFrame) -> pl.DataFrame:
    frame = add_returns(frame)
    frame = add_trend_features(frame)
    frame = add_momentum_features(frame)
    frame = add_volatility_features(frame)
    return add_calendar_features(frame)


def add_technical_features(
    candles: pl.DataFrame,
    frac_d: float = 0.4,
    wavelet: str = "sym4",
    wavelet_level: int = 3,
) -> pl.DataFrame:
    close = candles["close"]
    return add_market_features(candles).with_columns(
        pl.Series("close_denoised", wavelet_denoise(close.to_numpy(), wavelet, wavelet_level)),
        fractional_diff(close, frac_d).alias("close_fracdiff"),
    )
