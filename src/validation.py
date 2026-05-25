from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl


def purged_embargo_train_indices(
    indices: np.ndarray,
    event_end_pos: np.ndarray,
    test_idx: np.ndarray,
    embargo: int,
) -> np.ndarray:
    test_start, test_end = int(test_idx[0]), int(test_idx[-1])
    train_mask = np.ones(len(indices), dtype=bool)
    train_mask[test_idx] = False
    train_mask[(indices <= test_end) & (event_end_pos >= test_start)] = False
    train_mask[test_end + 1 : min(len(indices), test_end + embargo + 1)] = False
    return indices[train_mask]


class PurgedEmbargoTimeSeriesSplit:
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.02):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame, event_end: pd.Series | pl.Series):
        indices = np.arange(len(X))
        if isinstance(event_end, pl.Series):
            event_end_pos = event_end.to_numpy().astype(int)
        else:
            event_end_pos = event_end.to_numpy(dtype=int)
        embargo = int(np.ceil(len(X) * self.embargo_pct))

        n = len(indices)
        test_size = max(1, n // (self.n_splits + 1))

        for i in range(self.n_splits):
            train_end = (i + 1) * test_size
            test_start = train_end
            test_end = test_start + test_size if i < self.n_splits - 1 else n

            test_idx = indices[test_start:test_end]
            candidate_idx = purged_embargo_train_indices(indices, event_end_pos, test_idx, embargo)
            train_idx = candidate_idx[candidate_idx < test_start]

            if len(train_idx):
                yield train_idx, test_idx
