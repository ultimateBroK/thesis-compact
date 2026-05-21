from __future__ import annotations

import numpy as np
import pandas as pd


class PurgedEmbargoTimeSeriesSplit:
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.02):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame, event_end: pd.Series):
        indices = np.arange(len(X))
        event_end_pos = event_end.to_numpy(dtype=int)
        embargo = int(np.ceil(len(X) * self.embargo_pct))

        for test_idx in np.array_split(indices, self.n_splits):
            train_idx = purged_embargo_train_indices(indices, event_end_pos, test_idx, embargo)
            if len(train_idx):
                yield train_idx, test_idx


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
