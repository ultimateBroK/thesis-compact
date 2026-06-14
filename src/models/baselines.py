"""Các baseline Buy/Sell đơn giản để đối chiếu với Hybrid Stacking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BUY_LABEL, LABELS, RANDOM_STATE, SELL_LABEL


def _class_counts(labels: np.ndarray) -> np.ndarray:
    return np.array([(labels == label).sum() for label in LABELS], dtype=np.float64)


def class_prior_probabilities(y_train: np.ndarray, n_rows: int) -> np.ndarray:
    """Trả về cột P(Sell), P(Buy) cố định từ prior của nhãn train."""
    counts = _class_counts(np.asarray(y_train))
    total = counts.sum()
    if total <= 0.0:
        return np.full((n_rows, len(LABELS)), 1.0 / len(LABELS), dtype=np.float64)
    return np.tile(counts / total, (n_rows, 1))


def one_hot_probabilities(predictions: np.ndarray) -> np.ndarray:
    """Trả về các cột xác suất tất định căn theo ``LABELS``."""
    pred = np.asarray(predictions)
    proba = np.zeros((len(pred), len(LABELS)), dtype=np.float64)
    for col, label in enumerate(LABELS):
        proba[:, col] = pred == label
    return proba


def majority_baseline(y_train: np.ndarray, n_rows: int) -> np.ndarray:
    """Luôn dự đoán lớp Buy/Sell chiếm đa số trong train."""
    y = np.asarray(y_train)
    counts = _class_counts(y)
    majority_label = int(LABELS[int(np.argmax(counts))])
    return np.full(n_rows, majority_label, dtype=np.int64)


def random_baseline(
    y_train: np.ndarray,
    n_rows: int,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Lấy mẫu nhãn Buy/Sell theo empirical prior của train."""
    priors = class_prior_probabilities(np.asarray(y_train), 1)[0]
    rng = np.random.default_rng(random_state)
    return rng.choice(LABELS, size=n_rows, p=priors).astype(np.int64)


def momentum_baseline(X_test: pd.DataFrame) -> np.ndarray:
    """Dùng momentum 4 bar: return_4 >= 0 → Buy, ngược lại Sell."""
    if "return_4" not in X_test.columns:
        raise KeyError("momentum_baseline requires a return_4 feature")
    values = X_test["return_4"].to_numpy(dtype=np.float64)
    return np.where(values >= 0.0, BUY_LABEL, SELL_LABEL).astype(np.int64)


def buy_hold_baseline(n_rows: int) -> np.ndarray:
    """Luôn dự đoán Buy (+1)."""
    return np.full(n_rows, BUY_LABEL, dtype=np.int64)
