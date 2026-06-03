"""Signal conversion: translate model probabilities into trading positions."""

from __future__ import annotations

import numpy as np


def probabilities_to_positions(
    probas: np.ndarray,
    threshold: float = 0.55,
    long_only: bool = False,
) -> np.ndarray:
    """Convert Buy/Sell probabilities to {-1, 0, +1} positions.

    A class must exceed *threshold* to open a position; otherwise flat.

    Parameters
    ----------
    probas : ndarray, shape (n_samples, 2)
        Column 0 = P(Sell), column 1 = P(Buy).
    threshold : float
        Minimum probability to act on.
    long_only : bool
        If True, suppress all SHORT positions (set to 0).
    """
    positions = np.zeros(len(probas), dtype=np.int64)
    sell = probas[:, 0]
    buy = probas[:, 1]
    buy_mask = (buy >= threshold) & (buy > sell)
    sell_mask = (sell >= threshold) & (sell > buy)
    positions[buy_mask] = 1
    positions[sell_mask] = -1
    if long_only:
        positions[positions < 0] = 0
    return positions
