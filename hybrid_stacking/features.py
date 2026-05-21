from __future__ import annotations

import numpy as np
import pandas as pd
import pywt
from statsmodels.tsa.stattools import adfuller


def wavelet_denoise(series: pd.Series, wavelet: str = "coif5", level: int = 3) -> pd.Series:
    values = series.to_numpy(dtype=float).copy()
    chosen_level = wavelet_level(len(values), wavelet, level)
    coeffs = pywt.wavedec(values, wavelet, mode="symmetric", level=chosen_level)
    coeffs[1:] = threshold_detail_coefficients(coeffs[1:], len(values))
    denoised = pywt.waverec(coeffs, wavelet, mode="symmetric")[: len(values)]
    return pd.Series(denoised, index=series.index, name=f"{series.name}_denoised")


def wavelet_level(sample_count: int, wavelet: str, requested_level: int) -> int:
    max_level = pywt.dwt_max_level(sample_count, pywt.Wavelet(wavelet).dec_len)
    return max(1, min(requested_level, max_level))


def threshold_detail_coefficients(coefficients: list[np.ndarray], sample_count: int) -> list[np.ndarray]:
    sigma = np.median(np.abs(coefficients[-1])) / 0.6745 if len(coefficients[-1]) else 0.0
    threshold = sigma * np.sqrt(2 * np.log(sample_count))
    return [pywt.threshold(coef, threshold, mode="soft") for coef in coefficients]


def fractional_diff(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    weights = fractional_diff_weights(d, threshold)
    values = series.to_numpy(dtype=float)
    output = np.full(len(values), np.nan)
    for i in range(len(weights) - 1, len(values)):
        output[i] = np.dot(weights, values[i - len(weights) + 1 : i + 1])
    return pd.Series(output, index=series.index, name=f"{series.name}_fracdiff")


def fractional_diff_weights(d: float, threshold: float) -> np.ndarray:
    weights = [1.0]
    k = 1
    while True:
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            return np.array(weights[::-1])
        weights.append(weight)
        k += 1


def choose_fractional_d(series: pd.Series) -> float:
    clean = series.dropna()
    for d in np.arange(0.0, 1.01, 0.05):
        transformed = fractional_diff(clean, float(d)).dropna()
        if len(transformed) >= 50 and adfuller(transformed, autolag="AIC")[1] < 0.01:
            return round(float(d), 2)
    return 1.0


def add_technical_features(candles: pd.DataFrame) -> pd.DataFrame:
    featured = candles.copy()
    close = featured["close"]
    featured["close_denoised"] = wavelet_denoise(close)
    featured.attrs["fractional_d"] = choose_fractional_d(featured["close_denoised"])
    featured["close_fracdiff"] = fractional_diff(
        featured["close_denoised"], featured.attrs["fractional_d"]
    )
    return add_market_features(featured)


def add_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = add_returns(frame)
    frame = add_trend_features(frame)
    frame = add_momentum_features(frame)
    frame = add_volatility_features(frame)
    return add_calendar_features(frame)


def add_returns(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"]
    frame["return_1"] = close.pct_change()
    frame["return_4"] = close.pct_change(4)
    frame["return_12"] = close.pct_change(12)
    return frame


def add_trend_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"]
    frame["ema_12"] = close.ewm(span=12, adjust=False).mean()
    frame["ema_26"] = close.ewm(span=26, adjust=False).mean()
    frame["macd"] = frame["ema_12"] - frame["ema_26"]
    frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False).mean()
    return frame


def add_momentum_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"]
    low_14 = frame["low"].rolling(14).min()
    high_14 = frame["high"].rolling(14).max()
    frame["rsi_14"] = rsi(close, 14)
    frame["stoch_14"] = 100 * (close - low_14) / (high_14 - low_14).replace(0, np.nan)
    frame["ao"] = awesome_oscillator(frame["high"], frame["low"])
    return frame


def rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def awesome_oscillator(high: pd.Series, low: pd.Series) -> pd.Series:
    median_price = (high + low) / 2
    return median_price.rolling(5).mean() - median_price.rolling(34).mean()


def add_volatility_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"]
    frame["atr_14"] = average_true_range(frame, 14)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    frame["bb_width"] = 4 * bb_std / bb_mid
    frame["bb_position"] = (close - bb_mid) / (2 * bb_std).replace(0, np.nan)
    frame["volatility_24"] = frame["return_1"].rolling(24).std()
    return frame


def average_true_range(frame: pd.DataFrame, window: int) -> pd.Series:
    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame["hour"] = frame.index.hour
    frame["dayofweek"] = frame.index.dayofweek
    return frame
