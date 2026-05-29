from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl


def compute_embargo_clean_train_indices(
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


def walk_forward_split(
    dates: np.ndarray,
    n_windows: int = 3,
) -> list[tuple[np.ndarray, np.ndarray, int, str, str]]:
    """Generate expanding walk-forward windows by year boundaries.

    Each window yields (train_idx, test_idx, window_id, train_range, test_range).
    Expanding: each window's train set includes all data before the test year.
    """
    date_objs = pd.to_datetime(dates)
    years = np.unique(date_objs.year)
    if len(years) < 2:
        raise ValueError(f"Need at least 2 distinct years, got {len(years)}")

    # Use last n_windows+1 years: first n_windows for training progression, last for test
    usable_years = years[-(n_windows + 1):]
    if len(usable_years) < 2:
        usable_years = years

    # Sort years and create expanding windows
    unique_years = sorted(np.unique(usable_years))
    n = min(n_windows, len(unique_years) - 1)

    windows = []
    for w in range(n):
        test_year = unique_years[-(n - w)]
        train_years = [y for y in unique_years if y < test_year]
        train_mask = np.isin(date_objs.year, train_years)
        test_mask = date_objs.year == test_year
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        train_range = f"{min(train_years)}-{max(train_years)}" if train_years else str(test_year)
        test_range = str(test_year)
        windows.append((train_idx, test_idx, w, train_range, test_range))
    return windows