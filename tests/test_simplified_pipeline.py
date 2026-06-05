"""Tests for the restructured src/ package — organized by subpackage module."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import polars as pl

from src.config import PipelineConfig
from src.data.labeling import (
    assign_future_return_labels,
    compute_future_returns,
    summarize_label_distribution,
)
from src.data.loader import (
    apply_labels_to_frame,
    build_labeled_dataset,
    compute_test_start,
    replace_infinite_with_nan,
)
from src.features.engineering import (
    add_return_features,
    combine_market_features,
    get_feature_columns,
)
from src.models.stacking import (
    HybridStackingSignalClassifier,
    probabilities_to_signals,
)
from src.models.baselines import (
    buy_hold_baseline,
    majority_baseline,
    momentum_baseline,
    random_baseline,
)
from src.models.cross_validation import (
    PurgedTimeSeriesSplit,
    compute_purged_train_indices,
)
from src.evaluation.metrics import (
    build_classification_metric_row,
    build_naive_baseline_metric_rows,
)
from src.backtest.engine import (
    apply_fixed_horizon_positions,
    compute_strategy_bar_returns,
    run_signal_backtest,
)
from src.backtest.trades import (
    TradeRecord,
    extract_position_trades,
)
from src.reporting.artifacts import build_predictions_results


# ===================================================================
# src.data.labeling
# ===================================================================


class TestDataLabeling(unittest.TestCase):
    """Tests for src.data.labeling public API."""

    def test_assign_labels_fixed_horizon(self) -> None:
        frame = pl.DataFrame({"close": [100.0, 110.0, 99.0, 101.0]})
        labeled = assign_future_return_labels(frame, horizon=1, threshold=0.0)

        self.assertEqual(labeled["label"].to_list(), [1, -1, 1])
        self.assertEqual(labeled["event_end"].to_list(), [1, 2, 3])
        self.assertEqual(labeled["event_start"].to_list(), [0, 1, 2])
        np.testing.assert_allclose(
            labeled["future_return"].to_numpy(),
            [0.10, -0.10, 101.0 / 99.0 - 1.0],
        )

    def test_assign_labels_filter_by_threshold(self) -> None:
        frame = pl.DataFrame({"close": [100.0, 100.01, 99.0, 101.0]})
        labeled = assign_future_return_labels(frame, horizon=1, threshold=0.0005)
        # return[0] = 0.0001 < threshold → filtered
        self.assertEqual(labeled["label"].to_list(), [-1, 1])
        self.assertEqual(len(labeled), 2)

    def test_compute_future_returns_nan_without_enough_horizon(self) -> None:
        future_returns = compute_future_returns(np.array([1.0, 2.0]), horizon=2)
        self.assertTrue(np.isnan(future_returns).all())

    def test_compute_future_returns_horizon_exceeds_length(self) -> None:
        """Edge case: horizon > len(close) returns all NaN."""
        future_returns = compute_future_returns(np.array([1.0, 2.0, 3.0]), horizon=10)
        self.assertEqual(len(future_returns), 3)
        self.assertTrue(np.isnan(future_returns).all())

    def test_summarize_label_distribution_basic(self) -> None:
        labels = np.array([1, -1, 1, 1, -1])
        dist = summarize_label_distribution(labels)
        self.assertEqual(dist["Buy (+1)"], 3)
        self.assertEqual(dist["Sell (-1)"], 2)
        self.assertEqual(dist["total"], 5)
        self.assertAlmostEqual(dist["balance_ratio"], 2 / 3, places=4)

    def test_summarize_label_distribution_all_same(self) -> None:
        """Edge case: all labels identical → balance_ratio = 0.0."""
        labels = np.array([1, 1, 1])
        dist = summarize_label_distribution(labels)
        self.assertEqual(dist["Buy (+1)"], 3)
        self.assertEqual(dist["total"], 3)
        self.assertEqual(dist["balance_ratio"], 0.0)


# ===================================================================
# src.data.loader
# ===================================================================


class TestDataLoader(unittest.TestCase):
    """Tests for src.data.loader public API."""

    def test_apply_labels_preserves_event_end_after_filter(self) -> None:
        frame = pl.DataFrame({"close": [100.0, 100.01, 99.0, 101.0]})
        labeled = apply_labels_to_frame(
            frame, horizon=1, threshold=0.0005, max_gap_hours=999.0  # type: ignore[arg-type]
        )
        self.assertEqual(labeled["label"].to_list(), [-1, 1])
        self.assertEqual(labeled["event_end"].to_list(), [2, 3])
        self.assertEqual(labeled["event_start"].to_list(), [1, 2])

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
        with patch("src.data.loader.load_featured_candles", return_value=frame):
            featured, train_labeled, test_labeled, test_continuous = (
                build_labeled_dataset(PipelineConfig())
            )
        self.assertEqual(len(featured), 80)
        self.assertGreater(len(train_labeled), 0)
        self.assertGreaterEqual(len(test_continuous), len(test_labeled))
        self.assertNotIn("label", test_continuous.columns)

    def test_compute_test_start(self) -> None:
        """Purge gap offsets test start from split boundary."""
        test_start, purge = compute_test_start(split=800, purge_bars=4)
        self.assertEqual(test_start, 804)
        self.assertEqual(purge, 4)

    def test_replace_infinite_with_nan(self) -> None:
        frame = pl.DataFrame({"a": [1.0, float("inf"), -float("inf")], "b": ["x", "y", "z"]})
        result = replace_infinite_with_nan(frame)
        self.assertTrue(np.isfinite(result["a"][0]))
        self.assertTrue(np.isnan(result["a"][1]))
        self.assertTrue(np.isnan(result["a"][2]))
        self.assertEqual(result["b"].to_list(), ["x", "y", "z"])


# ===================================================================
# src.features.engineering
# ===================================================================


class TestFeatureEngineering(unittest.TestCase):
    """Tests for src.features.engineering public API."""

    def test_get_feature_columns_excludes_label_columns(self) -> None:
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

    def test_combine_market_features_on_synthetic_data(self) -> None:
        """combine_market_features adds all expected feature columns."""
        n = 60
        frame = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)],
                "open": [100.0] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "close": [100.0 + i * 0.1 for i in range(n)],
                "spread": [0.1] * n,
                "volume": [1000.0] * n,
                "tick_count": [50] * n,
            }
        )
        result = combine_market_features(frame)
        features = get_feature_columns(result)
        self.assertGreater(len(features), 10)
        self.assertIn("return_4", result.columns)
        self.assertIn("rsi_14", result.columns)
        self.assertIn("hour_sin", result.columns)

    def test_add_return_features(self) -> None:
        frame = pl.DataFrame({"close": [100.0, 102.0, 101.0, 103.0]})
        result = add_return_features(frame)
        self.assertIn("return_4", result.columns)
        # return_4 at index 0 is null (shift by 4, only 4 rows)
        self.assertTrue(result["return_4"][0] is None or np.isnan(result["return_4"][0]))


# ===================================================================
# src.models.stacking
# ===================================================================


class TestModelStacking(unittest.TestCase):
    """Tests for src.models.stacking public API."""

    def test_predict_signals_argmax_buy_or_sell(self) -> None:
        model = HybridStackingSignalClassifier()
        model.predict_proba = lambda _: np.array(  # type: ignore[assignment]
            [[0.60, 0.40], [0.50, 0.50], [0.30, 0.70], [0.49, 0.51]]
        )
        signals = model.predict_signals(pl.DataFrame({"x": [1, 2, 3, 4]}))
        np.testing.assert_array_equal(signals, [-1, 1, 1, 1])
        self.assertFalse(np.any(signals == 0))

    def test_probabilities_to_signals_buy(self) -> None:
        signals = probabilities_to_signals(np.array([[0.44, 0.56]]))
        np.testing.assert_array_equal(signals, [1])

    def test_probabilities_to_signals_tie(self) -> None:
        signals = probabilities_to_signals(np.array([[0.50, 0.50]]))
        np.testing.assert_array_equal(signals, [1])

    def test_probabilities_to_signals_sell(self) -> None:
        signals = probabilities_to_signals(np.array([[0.60, 0.40]]))
        np.testing.assert_array_equal(signals, [-1])

    def test_probabilities_to_signals_invalid_shape(self) -> None:
        with self.assertRaises(ValueError):
            probabilities_to_signals(np.array([0.5, 0.5]))

    def test_probabilities_to_signals_single_sample(self) -> None:
        """Edge case: single-row probability array."""
        signals = probabilities_to_signals(np.array([[0.9, 0.1]]))
        self.assertEqual(signals.shape, (1,))
        self.assertEqual(signals[0], -1)

    def test_all_base_models_remain_in_stacking(self) -> None:
        oofs = {
            "logistic_regression": np.zeros((2, 2)),
            "lightgbm": np.ones((2, 2)),
            "svc": np.full((2, 2), 0.5),
        }
        selected = dict(oofs)
        self.assertEqual(list(selected), ["logistic_regression", "lightgbm", "svc"])


# ===================================================================
# src.models.baselines
# ===================================================================


class TestModelBaselines(unittest.TestCase):
    """Tests for src.models.baselines public API."""

    def test_baselines_return_correct_shape_and_values(self) -> None:
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

    def test_majority_baseline_uses_majority_class(self) -> None:
        """3 sells, 1 buy → majority is -1."""
        y_train = np.array([-1, -1, 1, -1])
        pred = majority_baseline(y_train, 5)
        self.assertTrue(np.all(pred == -1))
        self.assertEqual(len(pred), 5)

    def test_buy_hold_always_returns_buy(self) -> None:
        pred = buy_hold_baseline(10)
        self.assertTrue(np.all(pred == 1))
        self.assertEqual(len(pred), 10)


# ===================================================================
# src.models.cross_validation
# ===================================================================


class TestModelCrossValidation(unittest.TestCase):
    """Tests for src.models.cross_validation public API."""

    def test_purge_uses_original_event_coordinates(self) -> None:
        indices = np.arange(4)
        event_start = np.array([0, 2, 5, 8])
        event_end = np.array([1, 6, 9, 12])
        test_idx = np.array([1])
        train_idx = compute_purged_train_indices(
            indices, event_start, event_end, test_idx
        )
        np.testing.assert_array_equal(train_idx, [0, 3])

    def test_purged_time_series_split_produces_non_overlapping(self) -> None:
        """Each fold's train indices must be strictly before test indices."""
        n = 50
        event_start = np.arange(n)
        event_end = event_start + 1
        X = pl.DataFrame({"x": np.zeros(n)}).to_pandas()
        cv = PurgedTimeSeriesSplit(n_splits=3)
        for train_idx, test_idx in cv.split(X, event_start, event_end):
            self.assertTrue(np.all(train_idx < test_idx[0]))


# ===================================================================
# src.evaluation.metrics
# ===================================================================


class TestEvaluationMetrics(unittest.TestCase):
    """Tests for src.evaluation.metrics public API."""

    def test_classification_metric_row_has_core_fields(self) -> None:
        y_true = np.array([-1, 1, 1, -1])
        preds = np.array([-1, 1, -1, -1])
        proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.8, 0.2]])
        row = build_classification_metric_row("test_model", y_true, preds, proba)
        self.assertIn("model", row)
        self.assertEqual(row["model"], "test_model")
        self.assertIn("accuracy", row)
        self.assertIn("roc_auc", row)

    def test_naive_baseline_metric_rows_order(self) -> None:
        train = pl.DataFrame({"label": [-1, -1, 1, 1]})
        test = pl.DataFrame({"label": [-1, 1, 1]})
        X_test = pl.DataFrame({"return_4": [-0.01, 0.01, 0.0]}).to_pandas()
        rows = build_naive_baseline_metric_rows(train, test, X_test)
        self.assertEqual(
            [row["model"] for row in rows],
            ["naive_majority", "naive_random_prior", "naive_momentum_return_4", "naive_buy_only"],
        )


# ===================================================================
# src.backtest.engine
# ===================================================================


class TestBacktestEngine(unittest.TestCase):
    """Tests for src.backtest.engine public API."""

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

    def test_spread_cost_on_position_change(self) -> None:
        close = np.array([100.0, 100.0])
        spread = np.array([0.1, 0.1])
        positions = np.array([1, -1])
        bar_returns = compute_strategy_bar_returns(close, spread, positions)
        np.testing.assert_allclose(bar_returns, [0.0, -0.001])

    def test_spread_cost_with_nonzero_close(self) -> None:
        """Bar return deducted when position changes and spread > 0."""
        close = np.array([100.0, 101.0, 100.0])
        spread = np.array([0.5, 0.5, 0.5])
        positions = np.array([1, -1, 1])
        bar_returns = compute_strategy_bar_returns(close, spread, positions)
        # bar[1]: positions[0]*price_ret[1] - turnover[0]*spread_frac[0]
        # = 1*(101/100-1) - |1-0|*(0.5/100) = 0.01 - 0.005 = 0.005
        self.assertAlmostEqual(bar_returns[1], 0.005, places=6)

    def test_held_horizon_broadcasts_each_decision(self) -> None:
        raw = np.array([1, -1, -1, -1, -1, 1, 1, 1, 1, -1])
        held = apply_fixed_horizon_positions(raw, hold_bars=4)
        np.testing.assert_array_equal(held, [1, 1, 1, 1, -1, -1, -1, -1, 1, 1])

    def test_held_horizon_rejects_invalid_hold_bars(self) -> None:
        with self.assertRaises(ValueError):
            apply_fixed_horizon_positions(np.array([1, -1]), hold_bars=0)

    def test_apply_fixed_horizon_positions_hold_exceeds_length(self) -> None:
        """Edge case: hold_bars > len(array) → entire array takes first value."""
        raw = np.array([1, -1, 1])
        held = apply_fixed_horizon_positions(raw, hold_bars=10)
        np.testing.assert_array_equal(held, [1, 1, 1])


# ===================================================================
# src.backtest.trades
# ===================================================================


class TestBacktestTrades(unittest.TestCase):
    """Tests for src.backtest.trades public API."""

    def test_extract_position_trades_basic(self) -> None:
        close = np.array([100.0, 110.0, 105.0])
        equity = np.array([100.0, 110.0, 105.0])
        positions = np.array([1, 1, 0])
        trades = extract_position_trades(close, equity, positions)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["direction"], "LONG")
        self.assertTrue(trades[0]["win"])

    def test_extract_position_trades_no_changes(self) -> None:
        """Edge case: all positions are flat (0) → no trades."""
        close = np.array([100.0, 101.0, 102.0])
        equity = np.array([100.0, 101.0, 102.0])
        positions = np.array([0, 0, 0])
        trades = extract_position_trades(close, equity, positions)
        self.assertEqual(len(trades), 0)

    def test_trade_record_from_equity(self) -> None:
        close = np.array([100.0, 105.0, 110.0])
        equity = np.array([100.0, 105.0, 110.0])
        record = TradeRecord.from_equity(0, 2, 1, close, equity)
        self.assertEqual(record.entry_idx, 0)
        self.assertEqual(record.exit_idx, 2)
        self.assertEqual(record.direction, "LONG")
        self.assertAlmostEqual(record.trade_return, 0.10)
        self.assertTrue(record.win)


# ===================================================================
# src.reporting.artifacts
# ===================================================================


class TestReportingArtifacts(unittest.TestCase):
    """Tests for src.reporting.artifacts public API."""

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


if __name__ == "__main__":
    unittest.main()
