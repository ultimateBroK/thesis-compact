"""Hàm hỗ trợ purged time-series cross-validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl


def _to_int_array(values) -> np.ndarray:
    if isinstance(values, pl.Series):
        return values.to_numpy().astype(int)
    if hasattr(values, "to_numpy"):
        return values.to_numpy(dtype=int)
    return np.asarray(values, dtype=int)


def compute_purged_train_indices(
    indices: np.ndarray,
    event_start_pos: np.ndarray,
    event_end_pos: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    test_start, test_end = int(test_idx[0]), int(test_idx[-1])
    test_time_start = int(event_start_pos[test_start])
    test_time_end = int(event_end_pos[test_end])
    train_mask = np.ones(len(indices), dtype=bool)
    train_mask[test_idx] = False
    train_mask[
        (event_start_pos <= test_time_end) & (event_end_pos >= test_time_start)
    ] = False
    return indices[train_mask]


class PurgedTimeSeriesSplit:
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, X: pd.DataFrame, event_start: np.ndarray, event_end: np.ndarray):
        indices = np.arange(len(X))
        event_start_pos = _to_int_array(event_start)
        event_end_pos = _to_int_array(event_end)

        n_rows = len(indices)
        test_size = max(1, n_rows // (self.n_splits + 1))

        for index in range(self.n_splits):
            train_end = (index + 1) * test_size
            test_start = train_end
            test_end = test_start + test_size if index < self.n_splits - 1 else n_rows

            test_idx = indices[test_start:test_end]
            candidate_idx = compute_purged_train_indices(
                indices, event_start_pos, event_end_pos, test_idx
            )
            train_idx = candidate_idx[candidate_idx < test_start]

            if len(train_idx):
                yield train_idx, test_idx
