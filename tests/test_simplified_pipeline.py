from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from src.backtest import compute_strategy_bar_returns, run_signal_backtest
from src.dataset import get_feature_columns
from src.labeling import assign_future_return_labels, compute_future_returns
from src.models import HybridStackingSignalClassifier, select_base_oof_predictions


class LabelingTests(unittest.TestCase):
    def test_future_return_labels_use_fixed_horizon(self) -> None:
        frame = pl.DataFrame({"close": [100.0, 110.0, 99.0, 101.0]})

        labeled = assign_future_return_labels(frame, horizon=1)

        self.assertEqual(labeled["label"].to_list(), [1, -1, 1])
        self.assertEqual(labeled["event_end"].to_list(), [1, 2, 3])
        np.testing.assert_allclose(
            labeled["future_return"].to_numpy(),
            [0.10, -0.10, 101.0 / 99.0 - 1.0],
        )

    def test_future_returns_are_nan_without_enough_horizon(self) -> None:
        future_returns = compute_future_returns(np.array([1.0, 2.0]), horizon=2)
        self.assertTrue(np.isnan(future_returns).all())

    def test_future_return_column_is_not_a_feature(self) -> None:
        frame = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "future_return": [0.01],
            "label": [1],
            "event_end": [1],
            "rsi_14": [55.0],
        })

        self.assertEqual(get_feature_columns(frame), ["rsi_14"])


class BacktestTests(unittest.TestCase):
    def test_signal_backtest_applies_position_to_next_bar(self) -> None:
        frame = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(3)],
            "close": [100.0, 110.0, 121.0],
            "spread": [0.0, 0.0, 0.0],
        })
        positions = np.array([1, 1, 0])

        metrics, trades, equity = run_signal_backtest(frame, positions, initial_balance=100.0)

        np.testing.assert_allclose(equity, [100.0, 110.0, 121.0])
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(trades[0]["trade_return"], 0.21)
        self.assertAlmostEqual(metrics["total_return"], 0.21)
        self.assertEqual(metrics["trades"], 1.0)

    def test_spread_cost_is_charged_on_position_change(self) -> None:
        close = np.array([100.0, 100.0])
        spread = np.array([0.1, 0.1])
        positions = np.array([1, 0])

        bar_returns = compute_strategy_bar_returns(close, spread, positions)

        np.testing.assert_allclose(bar_returns, [0.0, -0.001])


class PositionAssignmentTests(unittest.TestCase):
    def test_probability_threshold_creates_hold_zone(self) -> None:
        model = HybridStackingSignalClassifier(signal_probability_threshold=0.55)
        model.predict_proba = lambda _: np.array([
            [0.60, 0.40],
            [0.54, 0.46],
            [0.30, 0.70],
            [0.50, 0.50],
        ])

        positions = model.predict_positions(pl.DataFrame({"x": [1, 2, 3, 4]}))

        np.testing.assert_array_equal(positions, [-1, 0, 1, 0])


class StackingSelectionTests(unittest.TestCase):
    def test_all_base_models_remain_in_stacking(self) -> None:
        oofs = {
            "logistic_regression": np.zeros((2, 2)),
            "lightgbm": np.ones((2, 2)),
            "random_forest": np.full((2, 2), 0.5),
        }

        selected = select_base_oof_predictions(oofs)

        self.assertEqual(list(selected), ["logistic_regression", "lightgbm", "random_forest"])


if __name__ == "__main__":
    unittest.main()
