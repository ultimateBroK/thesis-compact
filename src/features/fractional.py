from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit


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
