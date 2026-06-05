"""Naive Buy/Sell baselines for context around Hybrid Stacking metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BUY_LABEL, LABELS, RANDOM_STATE, SELL_LABEL


def _class_counts(labels: np.ndarray) -> np.ndarray:
    return np.array([(labels == label).sum() for label in LABELS], dtype=np.float64)


def class_prior_probabilities(y_train: np.ndarray, n_rows: int) -> np.ndarray:
    """Return constant P(Sell), P(Buy) columns from train-label priors."""
    counts = _class_counts(np.asarray(y_train))
    total = counts.sum()
    if total <= 0.0:
        return np.full((n_rows, len(LABELS)), 1.0 / len(LABELS), dtype=np.float64)
    return np.tile(counts / total, (n_rows, 1))


def one_hot_probabilities(predictions: np.ndarray) -> np.ndarray:
    """Return deterministic probability columns aligned to ``LABELS``."""
    pred = np.asarray(predictions)
    proba = np.zeros((len(pred), len(LABELS)), dtype=np.float64)
    for col, label in enumerate(LABELS):
        proba[:, col] = pred == label
    return proba


def majority_baseline(y_train: np.ndarray, n_rows: int) -> np.ndarray:
    """Always predict the majority Buy/Sell class observed in train data."""
    y = np.asarray(y_train)
    counts = _class_counts(y)
    majority_label = int(LABELS[int(np.argmax(counts))])
    return np.full(n_rows, majority_label, dtype=np.int64)


def random_baseline(
    y_train: np.ndarray,
    n_rows: int,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Sample Buy/Sell labels using train-label empirical priors."""
    priors = class_prior_probabilities(np.asarray(y_train), 1)[0]
    rng = np.random.default_rng(random_state)
    return rng.choice(LABELS, size=n_rows, p=priors).astype(np.int64)


def momentum_baseline(X_test: pd.DataFrame) -> np.ndarray:
    """Use 4-bar realized momentum: return_4 >= 0 → Buy, else Sell."""
    if "return_4" not in X_test.columns:
        raise KeyError("momentum_baseline requires a return_4 feature")
    values = X_test["return_4"].to_numpy(dtype=np.float64)
    return np.where(values >= 0.0, BUY_LABEL, SELL_LABEL).astype(np.int64)


def buy_hold_baseline(n_rows: int) -> np.ndarray:
    """Always predict Buy (+1)."""
    return np.full(n_rows, BUY_LABEL, dtype=np.int64)
