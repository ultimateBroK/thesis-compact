"""Grid-search backtest hyperparameters (TP, SL, min_hold) on training data.

Pure downstream consumer of model outputs — no model retraining required.
Tunes on train data; final metrics MUST come from test set.
"""

from __future__ import annotations

import itertools
import warnings
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from src.backtest.engine import backtest_signal_positions
from src.models.main import enforce_minimum_position_hold

if TYPE_CHECKING:
    from src.models import HybridStackingSignalClassifier


def tune_backtest_hyperparameters(
    model: HybridStackingSignalClassifier,
    train_data: pl.DataFrame,
    features: list[str],
    close_series: np.ndarray,
    tp_range: tuple[float, float, float] = (1.5, 4.0, 0.5),
    sl_range: tuple[float, float, float] = (1.0, 3.0, 0.5),
    min_hold_values: list[int] | None = None,
) -> dict[str, float | int]:
    """Grid-search (min_hold, tp_atr, sl_atr) for best Sharpe on train data.

    Args:
        model: Trained HybridStackingSignalClassifier.
        train_data: Training DataFrame with OHLC and atr_14 columns.
        features: Feature column names.
        close_series: Close prices for trend/EMA filtering in predict_positions.
        tp_range: (start, stop, step) in ATR multiples.
        sl_range: (start, stop, step) in ATR multiples.
        min_hold_values: List of min_hold values to try.

    Returns:
        Dict with keys: score, tp, sl, min_hold, trades, win_rate, profit_factor.
    """
    if min_hold_values is None:
        min_hold_values = [4, 6, 8, 12, 16, 24]

    tp_start, tp_stop, tp_step = tp_range
    sl_start, sl_stop, sl_step = sl_range

    tp_values = np.arange(tp_start, tp_stop + tp_step / 2, tp_step)
    sl_values = np.arange(sl_start, sl_stop + sl_step / 2, sl_step)

    raw_positions = model.predict_positions(
        train_data[features], close_series, skip_min_hold=True
    )

    best: dict[str, float | int] = {
        "score": -np.inf,
        "tp": 1.5,
        "sl": 1.0,
        "min_hold": 24,
        "trades": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
    }

    total_combos = len(min_hold_values) * len(tp_values) * len(sl_values)
    print(f"  Tuning backtest params: {total_combos} combos ...")

    for min_hold in min_hold_values:
        positions = enforce_minimum_position_hold(raw_positions.copy(), min_hold)

        for tp, sl in itertools.product(tp_values, sl_values):
            if sl >= tp:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics, trades, _ = backtest_signal_positions(
                    train_data, positions, tp_atr=float(tp), sl_atr=float(sl),
                )

            score = float(metrics.get("sharpe", -999))
            if score > best["score"]:
                best = {
                    "score": score,
                    "tp": float(tp),
                    "sl": float(sl),
                    "min_hold": min_hold,
                    "trades": len(trades),
                    "win_rate": float(metrics.get("win_rate", 0)),
                    "profit_factor": float(metrics.get("profit_factor", 0)),
                }

    print(
        f"  Best: tp={best['tp']:.1f} sl={best['sl']:.1f} "
        f"min_hold={best['min_hold']} "
        f"sharpe={best['score']:.3f} "
        f"trades={best['trades']} "
        f"pf={best['profit_factor']:.2f}"
    )
    return best
