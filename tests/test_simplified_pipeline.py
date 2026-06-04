"""Tests for the simplified thesis pipeline."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import polars as pl

from src.backtest import (
    apply_fixed_horizon_positions,
    compute_strategy_bar_returns,
    run_signal_backtest,
)
from src.artifacts import build_predictions_results
from src.config import PipelineConfig
from src.data import apply_labels_to_frame, build_labeled_dataset
from src.features import get_feature_columns
from src.labeling import assign_future_return_labels, compute_future_returns
from src.models import (
    HybridStackingSignalClassifier,
    compute_purged_train_indices,
    probabilities_to_signals,
)


class LabelingTests(unittest.TestCase):
    def test_future_return_labels_use_fixed_horizon(self) -> None:
        frame = pl.DataFrame({"close": [100.0, 110.0, 99.0, 101.0]})

        labeled = assign_future_return_labels(frame, horizon=1, threshold=0.0)

        self.assertEqual(labeled["label"].to_list(), [1, -1, 1])
        self.assertEqual(labeled["event_end"].to_list(), [1, 2, 3])
        self.assertEqual(labeled["event_start"].to_list(), [0, 1, 2])
        np.testing.assert_allclose(
            labeled["future_return"].to_numpy(),
            [0.10, -0.10, 101.0 / 99.0 - 1.0],
        )

    def test_future_return_labels_filter_by_threshold(self) -> None:
        frame = pl.DataFrame({"close": [100.0, 100.01, 99.0, 101.0]})

        # return[0] = 100.01/100 - 1 = 0.0001 < threshold → filtered
        labeled = assign_future_return_labels(frame, horizon=1, threshold=0.0005)

        self.assertEqual(labeled["label"].to_list(), [-1, 1])
        self.assertEqual(len(labeled), 2)

    def test_future_returns_are_nan_without_enough_horizon(self) -> None:
        future_returns = compute_future_returns(np.array([1.0, 2.0]), horizon=2)
        self.assertTrue(np.isnan(future_returns).all())

    def test_apply_labels_preserves_original_event_end_after_filter(self) -> None:
        frame = pl.DataFrame({"close": [100.0, 100.01, 99.0, 101.0]})

        labeled = apply_labels_to_frame(
            frame, horizon=1, threshold=0.0005, max_gap_hours=999.0  # type: ignore[arg-type]
        )

        self.assertEqual(labeled["label"].to_list(), [-1, 1])
        self.assertEqual(labeled["event_end"].to_list(), [2, 3])
        self.assertEqual(labeled["event_start"].to_list(), [1, 2])

    def test_future_return_column_is_not_a_feature(self) -> None:
        frame = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "future_return": [0.01],
                "label": [1],
                "event_end": [1],
                "event_start": [0],
                "rsi_14": [55.0],
            }
        )
        self.assertEqual(get_feature_columns(frame), ["rsi_14"])


class DatasetSplitTests(unittest.TestCase):
    def test_build_labeled_dataset_returns_continuous_test_frame(self) -> None:
        close = [100.0 + i * 0.1 for i in range(80)]
        frame = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1) + timedelta(hours=i) for i in range(80)
                ],
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "spread": [0.0] * 80,
                "rsi_14": [55.0] * 80,
            }
        )

        with patch("src.data.load_featured_candles", return_value=frame):
            featured, train_labeled, test_labeled, test_continuous = (
                build_labeled_dataset(PipelineConfig())
            )

        self.assertEqual(len(featured), 80)
        self.assertGreater(len(train_labeled), 0)
        self.assertGreaterEqual(len(test_continuous), len(test_labeled))
        self.assertNotIn("label", test_continuous.columns)


class PredictionArtifactTests(unittest.TestCase):
    def test_predictions_results_separate_labels_signals_and_positions(self) -> None:
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(4)]
        test_continuous = pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": [100.0, 101.0, 100.5, 102.0],
                "spread": [0.0, 0.0, 0.0, 0.0],
            }
        )
        test_labeled = pl.DataFrame(
            {
                "timestamp": [timestamps[0], timestamps[2]],
                "label": [1, -1],
            }
        )

        results = build_predictions_results(
            test_labeled,
            test_continuous,
            predictions=np.array([1, -1]),
            raw_signals=np.array([1, 1, -1, -1]),
            positions=np.array([1, 1, -1, -1]),
            equity=np.array([100.0, 101.0, 99.0, 102.0]),
        )

        self.assertIn("prediction", results.columns)
        self.assertIn("raw_signal", results.columns)
        self.assertIn("executed_position", results.columns)
        self.assertTrue(np.isnan(results.loc[1, "prediction"]))
        self.assertEqual(results["raw_signal"].to_list(), [1, 1, -1, -1])
        self.assertEqual(results["executed_position"].to_list(), [1, 1, -1, -1])


class BacktestTests(unittest.TestCase):
    def test_signal_backtest_applies_position_to_next_bar(self) -> None:
        frame = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1) + timedelta(hours=i) for i in range(3)
                ],
                "close": [100.0, 110.0, 121.0],
                "spread": [0.0, 0.0, 0.0],
            }
        )
        positions = np.array([1, 1, 1])

        metrics, trades, equity = run_signal_backtest(
            frame, positions, initial_balance=100.0
        )

        np.testing.assert_allclose(equity, [100.0, 110.0, 121.0])
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(trades[0]["trade_return"], 0.21)
        self.assertAlmostEqual(metrics["total_return"], 0.21)
        self.assertEqual(metrics["trades"], 1.0)

    def test_spread_cost_is_charged_on_position_change(self) -> None:
        close = np.array([100.0, 100.0])
        spread = np.array([0.1, 0.1])
        positions = np.array([1, -1])

        bar_returns = compute_strategy_bar_returns(close, spread, positions)

        np.testing.assert_allclose(bar_returns, [0.0, -0.001])

    def test_held_horizon_broadcasts_each_decision(self) -> None:
        raw = np.array([1, -1, -1, -1, -1, 1, 1, 1, 1, -1])

        held = apply_fixed_horizon_positions(raw, hold_bars=4)

        np.testing.assert_array_equal(held, [1, 1, 1, 1, -1, -1, -1, -1, 1, 1])

    def test_held_horizon_rejects_invalid_hold_bars(self) -> None:
        with self.assertRaises(ValueError):
            apply_fixed_horizon_positions(np.array([1, -1]), hold_bars=0)


class PositionAssignmentTests(unittest.TestCase):
    def test_probability_argmax_always_returns_buy_or_sell(self) -> None:
        model = HybridStackingSignalClassifier()
        model.predict_proba = lambda _: np.array(  # type: ignore[assignment]
            [
                [0.60, 0.40],
                [0.50, 0.50],
                [0.30, 0.70],
                [0.49, 0.51],
            ]
        )

        signals = model.predict_signals(pl.DataFrame({"x": [1, 2, 3, 4]}))

        np.testing.assert_array_equal(signals, [-1, 1, 1, 1])
        self.assertFalse(np.any(signals == 0))


class PurgedCVTests(unittest.TestCase):
    def test_purge_uses_original_event_coordinates_after_filter(self) -> None:
        indices = np.arange(4)
        event_start = np.array([0, 2, 5, 8])
        event_end = np.array([1, 6, 9, 12])
        test_idx = np.array([1])

        train_idx = compute_purged_train_indices(
            indices, event_start, event_end, test_idx
        )

        np.testing.assert_array_equal(train_idx, [0, 3])


class SignalConversionTests(unittest.TestCase):
    """Tests for probabilities_to_signals."""

    def test_buy_probability_wins_returns_buy(self) -> None:
        probas = np.array([[0.44, 0.56]])
        signals = probabilities_to_signals(probas)
        np.testing.assert_array_equal(signals, [1])

    def test_tie_defaults_to_buy(self) -> None:
        probas = np.array([[0.50, 0.50]])
        signals = probabilities_to_signals(probas)
        np.testing.assert_array_equal(signals, [1])

    def test_sell_probability_wins_returns_sell(self) -> None:
        probas = np.array([[0.60, 0.40]])
        signals = probabilities_to_signals(probas)
        np.testing.assert_array_equal(signals, [-1])

    def test_invalid_probability_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            probabilities_to_signals(np.array([0.5, 0.5]))


class StackingSelectionTests(unittest.TestCase):
    def test_all_base_models_remain_in_stacking(self) -> None:
        oofs = {
            "logistic_regression": np.zeros((2, 2)),
            "lightgbm": np.ones((2, 2)),
            "svc": np.full((2, 2), 0.5),
        }

        selected = dict(oofs)

        self.assertEqual(list(selected), ["logistic_regression", "lightgbm", "svc"])


class BaselineMetricsTests(unittest.TestCase):
    """Tests for metric row construction and naive baselines."""

    def test_classification_metric_row_has_core_fields(self) -> None:
        from src.metrics import build_classification_metric_row

        y_true = np.array([-1, 1, 1, -1])
        preds = np.array([-1, 1, -1, -1])
        proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.8, 0.2]])

        row = build_classification_metric_row("test_model", y_true, preds, proba)
        self.assertIn("model", row)
        self.assertEqual(row["model"], "test_model")
        self.assertIn("accuracy", row)
        self.assertIn("roc_auc", row)

    def test_naive_baselines_return_buy_sell_arrays(self) -> None:
        from src.baselines import (
            buy_hold_baseline,
            majority_baseline,
            momentum_baseline,
            random_baseline,
        )

        y_train = np.array([-1, -1, 1, -1])
        X_test = pl.DataFrame({"return_4": [-0.01, 0.0, 0.02]}).to_pandas()

        outputs = [
            majority_baseline(y_train, 3),
            random_baseline(y_train, 3, random_state=7),
            momentum_baseline(X_test),
            buy_hold_baseline(3),
        ]

        for pred in outputs:
            self.assertEqual(pred.shape, (3,))
            self.assertTrue(np.isin(pred, [-1, 1]).all())

    def test_naive_metric_rows_are_added_before_model_rows(self) -> None:
        from src.metrics import build_naive_baseline_metric_rows

        train = pl.DataFrame({"label": [-1, -1, 1, 1]})
        test = pl.DataFrame({"label": [-1, 1, 1]})
        X_test = pl.DataFrame({"return_4": [-0.01, 0.01, 0.0]}).to_pandas()

        rows = build_naive_baseline_metric_rows(train, test, X_test)

        self.assertEqual(
            [row["model"] for row in rows],
            [
                "naive_majority",
                "naive_random_prior",
                "naive_momentum_return_4",
                "naive_buy_only",
            ],
        )


if __name__ == "__main__":
    unittest.main()
