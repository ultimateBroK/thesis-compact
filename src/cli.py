"""CLI: argument parsing and simplified hybrid-stacking pipeline orchestration."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import polars as pl

from src.backtest import run_signal_backtest
from src.config import (
    CV_SPLITS,
    DATA_DIR,
    EMBARGO_PCT,
    FRACTIONAL_D,
    INITIAL_BALANCE,
    LABELING_HORIZON,
    MIN_OOF_F1,
    PURGE_PCT,
    RANDOM_STATE,
    SIGNAL_PROBABILITY_THRESHOLD,
    PipelineConfig,
)
from src.data import collect_parquet_paths
from src.dataset import apply_labels_to_frame, build_labeled_dataset, get_feature_columns, load_featured_candles
from src.models import HybridStackingSignalClassifier
from src.reporting import publish_pipeline_results
from src.validation import walk_forward_split


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingResults:
    """Immutable pipeline step timings in seconds."""

    data_loading: float = 0.0
    model_training: float = 0.0
    prediction: float = 0.0
    positions: float = 0.0
    backtesting: float = 0.0
    reporting: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "data_loading": self.data_loading,
            "model_training": self.model_training,
            "prediction": self.prediction,
            "positions": self.positions,
            "backtesting": self.backtesting,
            "reporting": self.reporting,
            "total": self.total,
        }


@dataclass(frozen=True)
class RunConfigPayload:
    """Serializable pipeline configuration metadata for reporting."""

    months: str = ""
    data_range: str = ""
    cv_splits: int = 0
    embargo_pct: float = 0.0
    purge_pct: float = 0.0
    fractional_d: float = 0.0
    min_oof_f1: float = 0.0
    random_state: int = 0
    timeframe: str = "1h"
    initial_balance: float = 10_000.0
    labeling_method: str = "fixed_horizon_future_return"
    labeling_horizon: int = 24
    signal_probability_threshold: float = 0.55
    timing: TimingResults | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "months": self.months,
            "data_range": self.data_range,
            "cv_splits": self.cv_splits,
            "embargo_pct": self.embargo_pct,
            "purge_pct": self.purge_pct,
            "fractional_d": self.fractional_d,
            "min_oof_f1": self.min_oof_f1,
            "random_state": self.random_state,
            "timeframe": self.timeframe,
            "initial_balance": self.initial_balance,
            "labeling_method": self.labeling_method,
            "labeling_horizon": self.labeling_horizon,
            "signal_probability_threshold": self.signal_probability_threshold,
            "timing": self.timing.as_dict() if self.timing else {},
        }


@dataclass(frozen=True)
class PipelineOutputs:
    """Output artifact bundle from a single pipeline execution."""

    train: pl.DataFrame = field(repr=False)
    test: pl.DataFrame = field(repr=False)
    features: list[str]
    model: HybridStackingSignalClassifier = field(repr=False)
    predictions: np.ndarray = field(repr=False)
    positions: np.ndarray = field(repr=False)
    backtest_metrics: dict[str, float]
    equity: np.ndarray = field(repr=False)
    executed_trades: list[dict] = field(repr=False)
    pred_proba: np.ndarray = field(repr=False, default=None)

    def to_dict(
        self,
        window_id: int | None = None,
        window_train_range: str = "",
        window_test_range: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "train": self.train,
            "test": self.test,
            "features": self.features,
            "model": self.model,
            "predictions": self.predictions,
            "positions": self.positions,
            "backtest_metrics": self.backtest_metrics,
            "executed_trades": self.executed_trades,
            "equity": self.equity,
            "pred_proba": self.pred_proba,
        }
        if window_id is not None:
            payload["window_id"] = window_id
            payload["window_train_range"] = window_train_range
            payload["window_test_range"] = window_test_range
        return payload


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def validate_positive_month_count(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--months must be >= 1; use --full for all data")
    return parsed


def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid Stacking gold signal prediction")
    parser.add_argument(
        "--months",
        type=validate_positive_month_count,
        default=PipelineConfig().months,
        help="Number of months to load from first month",
    )
    parser.add_argument("--full", action="store_true", help="Use all available parquet data")
    parser.add_argument("--long-only", action="store_true", help="Disable all SHORT positions")
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run expanding walk-forward evaluation instead of single train/test split",
    )
    return parser.parse_args()


def derive_config_from_arguments(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        months=None if args.full else args.months,
        long_only=args.long_only,
        walk_forward=args.walk_forward,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def measure_step_duration(
    name: str,
    step: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, float]:
    """Execute *step* and return (result, elapsed_seconds)."""
    del name
    started = time.perf_counter()
    result = step(*args, **kwargs)
    return result, time.perf_counter() - started


def format_parquet_file_range(config: PipelineConfig) -> str:
    files = collect_parquet_paths(DATA_DIR, config.months)
    return f"{files[0].stem} -> {files[-1].stem}"


# ---------------------------------------------------------------------------
# Model and evaluation
# ---------------------------------------------------------------------------


def train_hybrid_stacking_model(
    train: pl.DataFrame,
    features: list[str],
    config: PipelineConfig,
) -> HybridStackingSignalClassifier:
    return HybridStackingSignalClassifier(
        n_splits=CV_SPLITS,
        embargo_pct=EMBARGO_PCT,
        min_oof_f1=MIN_OOF_F1,
        signal_probability_threshold=SIGNAL_PROBABILITY_THRESHOLD,
        random_state=RANDOM_STATE,
        long_only=config.long_only,
    ).fit(train[features], train["label"], train["event_end"])


def build_run_config_payload(config: PipelineConfig, timing: TimingResults) -> RunConfigPayload:
    return RunConfigPayload(
        months="full" if config.months is None else f"{config.months} months",
        data_range=format_parquet_file_range(config),
        cv_splits=CV_SPLITS,
        embargo_pct=EMBARGO_PCT,
        purge_pct=PURGE_PCT,
        fractional_d=FRACTIONAL_D,
        min_oof_f1=MIN_OOF_F1,
        random_state=RANDOM_STATE,
        timeframe=config.timeframe,
        initial_balance=INITIAL_BALANCE,
        labeling_horizon=LABELING_HORIZON,
        signal_probability_threshold=SIGNAL_PROBABILITY_THRESHOLD,
        timing=timing,
    )


def run_prediction_stage(
    model: HybridStackingSignalClassifier,
    test: pl.DataFrame,
    features: list[str],
    timing: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if timing is None:
        predictions = model.predict(test[features])
        positions = model.predict_positions(test[features])
    else:
        predictions, timing["prediction"] = measure_step_duration(
            "prediction",
            model.predict,
            test[features],
        )
        positions, timing["positions"] = measure_step_duration(
            "positions",
            model.predict_positions,
            test[features],
        )
    return predictions, positions


def run_backtest_stage(
    test_frame: pl.DataFrame,
    positions: np.ndarray,
    timing: dict[str, float] | None = None,
) -> tuple[dict[str, float], list[dict], np.ndarray]:
    if timing is None:
        return run_signal_backtest(test_frame, positions)
    result, timing["backtesting"] = measure_step_duration(
        "backtesting",
        run_signal_backtest,
        test_frame,
        positions,
    )
    return result


def run_evaluation_pipeline(
    model: HybridStackingSignalClassifier,
    data: tuple[pl.DataFrame, pl.DataFrame],
    features: list[str],
    timing: dict[str, float] | None = None,
) -> PipelineOutputs:
    train, test = data
    predictions, positions = run_prediction_stage(model, test, features, timing)
    pred_proba = model.predict_proba(test[features])
    backtest_metrics, executed_trades, equity = run_backtest_stage(test, positions, timing)

    return PipelineOutputs(
        train=train,
        test=test,
        features=features,
        model=model,
        predictions=predictions,
        positions=positions,
        backtest_metrics=backtest_metrics,
        equity=equity,
        executed_trades=executed_trades,
        pred_proba=pred_proba,
    )


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


def run_walk_forward_pipeline(config: PipelineConfig) -> list[PipelineOutputs]:
    """Run expanding walk-forward windows, returning per-window outputs."""
    featured = load_featured_candles(config)
    timestamps = featured["timestamp"].to_numpy()
    windows = walk_forward_split(timestamps, n_windows=config.n_windows)

    print(f"Walk-forward: {len(windows)} windows")
    window_outputs: list[PipelineOutputs] = []

    for train_idx, test_idx, window_id, train_range, test_range in windows:
        print(f"\n--- Window {window_id}: train={train_range}, test={test_range} ---")
        train_labeled = apply_labels_to_frame(featured[train_idx])
        test_labeled = apply_labels_to_frame(featured[test_idx])
        features = get_feature_columns(train_labeled)
        model = train_hybrid_stacking_model(train_labeled, features, config)
        outputs = run_evaluation_pipeline(model, (train_labeled, test_labeled), features)
        window_outputs.append(outputs)
    return window_outputs


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------


def run_model_pipeline(config: PipelineConfig) -> tuple[PipelineOutputs, dict[str, float]]:
    """Run the ML pipeline and return outputs with accumulated timing."""
    timing: dict[str, float] = {}

    (_, train, test), timing["data_loading"] = measure_step_duration(
        "data_loading",
        build_labeled_dataset,
        config,
    )
    features = get_feature_columns(train)
    model, timing["model_training"] = measure_step_duration(
        "model_training",
        train_hybrid_stacking_model,
        train,
        features,
        config,
    )
    outputs = run_evaluation_pipeline(model, (train, test), features, timing=timing)
    return outputs, timing


def print_timing_summary(timing: TimingResults) -> None:
    print("\n=== PIPELINE TIMING ===")
    for step, secs in timing.as_dict().items():
        print(f"  {step:<22s} {secs:>8.3f}s")
    print("========================\n")


def run_pipeline(config: PipelineConfig) -> None:
    t_total = time.perf_counter()

    if config.walk_forward:
        window_outputs = run_walk_forward_pipeline(config)
        timing = TimingResults(total=time.perf_counter() - t_total)
        config_payload = build_run_config_payload(config, timing)
        for window_id, outputs in enumerate(window_outputs):
            print(f"\n=== Window {window_id} ===")
            publish_pipeline_results(config_payload.as_dict(), outputs, window_id=window_id)
        return

    outputs, ml_timing = run_model_pipeline(config)
    ml_timing["reporting"] = 0.0
    ml_timing["total"] = 0.0
    timing = TimingResults(**ml_timing)
    config_payload = build_run_config_payload(config, timing)

    _, reporting_secs = measure_step_duration(
        "reporting",
        publish_pipeline_results,
        config_payload.as_dict(),
        outputs,
    )

    ml_timing["reporting"] = reporting_secs
    ml_timing["total"] = time.perf_counter() - t_total
    print_timing_summary(TimingResults(**ml_timing))


def main() -> None:
    run_pipeline(derive_config_from_arguments(parse_command_line_arguments()))
