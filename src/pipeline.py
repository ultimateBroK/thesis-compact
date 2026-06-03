"""Pipeline orchestration: train, predict, backtest, evaluate, publish."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable

import numpy as np
import polars as pl

from src.backtest import run_signal_backtest
from src.config import (
    CV_SPLITS,
    DATA_DIR,
    EMBARGO_PCT,
    INITIAL_BALANCE,
    LABELING_HORIZON,
    MIN_OOF_F1,
    PURGE_PCT,
    RANDOM_STATE,
    SIGNAL_PROBABILITY_THRESHOLD,
    REPORT_DIR,
    PipelineConfig,
)
from src.console import (
    print_backtest_metrics_report,
    print_base_model_oof_report,
    print_classification_report,
    print_dataset_report,
    print_feature_importance_report,
)
from src.artifacts import extract_lightgbm_feature_importance, save_run_artifacts
from src.data import collect_parquet_paths
from src.dataset import (
    apply_labels_to_frame,
    build_labeled_dataset,
    get_feature_columns,
    load_featured_candles,
)
from src.models import HybridStackingSignalClassifier
from src.types import PipelineOutputs, RunConfigPayload, TimingResults
from src.validation import walk_forward_split
from types import SimpleNamespace


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
            "prediction", model.predict, test[features],
        )
        positions, timing["positions"] = measure_step_duration(
            "positions", model.predict_positions, test[features],
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
        "backtesting", run_signal_backtest, test_frame, positions,
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
        "data_loading", build_labeled_dataset, config,
    )
    features = get_feature_columns(train)
    model, timing["model_training"] = measure_step_duration(
        "model_training", train_hybrid_stacking_model, train, features, config,
    )
    outputs = run_evaluation_pipeline(model, (train, test), features, timing=timing)
    return outputs, timing


def print_timing_summary(timing: TimingResults) -> None:
    print("\n=== PIPELINE TIMING ===")
    for step, secs in timing.as_dict().items():
        print(f"  {step:<22s} {secs:>8.3f}s")
    print("========================\n")


def publish_pipeline_results(
    config_payload: dict[str, Any],
    outputs: PipelineOutputs | dict[str, Any],
    window_id: int | None = None,
    window_train_range: str = "",
    window_test_range: str = "",
) -> None:
    if hasattr(outputs, "to_dict"):
        output_payload = outputs.to_dict(
            window_id=window_id,
            window_train_range=window_train_range,
            window_test_range=window_test_range,
        )
        artifact_outputs = outputs
    else:
        output_payload = dict(outputs)
        if window_id is not None:
            output_payload["window_id"] = window_id
            output_payload["window_train_range"] = window_train_range
            output_payload["window_test_range"] = window_test_range
        artifact_outputs = SimpleNamespace(
            train=output_payload["train"],
            test=output_payload["test"],
            features=output_payload["features"],
            model=output_payload["model"],
            predictions=output_payload["predictions"],
            positions=output_payload["positions"],
            backtest_metrics=output_payload.get("backtest_metrics"),
            equity=output_payload.get("equity", np.full(len(output_payload["test"]), 10_000.0)),
            executed_trades=output_payload.get("executed_trades"),
            pred_proba=output_payload.get("pred_proba"),
        )
    train = output_payload["train"]
    test = output_payload["test"]
    features = output_payload["features"]
    model = output_payload["model"]
    predictions = output_payload["predictions"]
    backtest_metrics = output_payload["backtest_metrics"]

    labeled_full = pl.concat([train, test])

    print_dataset_report(labeled_full, train, test, len(features))
    print_base_model_oof_report(model)
    print_classification_report(test["label"], predictions)
    print_feature_importance_report(extract_lightgbm_feature_importance(model, features))
    print_backtest_metrics_report(backtest_metrics)

    save_run_artifacts(
        run_dir=config_payload.get("run_dir", REPORT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        outputs=artifact_outputs,
        config_payload=config_payload,
        window_id=window_id,
        window_train_range=window_train_range,
        window_test_range=window_test_range,
    )


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
