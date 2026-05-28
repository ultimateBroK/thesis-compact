"""
Cross-validation pipeline: purged-embargo time series split.

Orchestration: PurgedEmbargoTimeSeriesSplit.split yields (train_idx, val_idx).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from .split import compute_embargo_clean_train_indices


class PurgedEmbargoTimeSeriesSplit:
    """Cross-validation split with purge and embargo to prevent data leakage."""

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.02):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame, event_end: pd.Series | pl.Series):
        indices = np.arange(len(X))
        event_end_pos = (
            event_end.to_numpy().astype(int)
            if isinstance(event_end, pl.Series)
            else event_end.to_numpy(dtype=int)
        )
        embargo = int(np.ceil(len(X) * self.embargo_pct))

        n = len(indices)
        test_size = max(1, n // (self.n_splits + 1))

        for i in range(self.n_splits):
            train_end = (i + 1) * test_size
            test_start = train_end
            test_end = test_start + test_size if i < self.n_splits - 1 else n

            test_idx = indices[test_start:test_end]
            candidate_idx = compute_embargo_clean_train_indices(indices, event_end_pos, test_idx, embargo)
            train_idx = candidate_idx[candidate_idx < test_start]

            if len(train_idx):
                yield train_idx, test_idx
