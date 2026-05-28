"""
CLI pipeline: parse args → accelerate → data → train → predict → backtest → report.

Orchestration: run_pipeline is the top-level entry; execute_model_pipeline drives the ML flow.
"""
from __future__ import annotations

import time
from typing import Any

import polars as pl
from accelerate import Accelerator
from accelerate.utils import set_seed

from src.backtest import backtest_signal_positions
from src.config import (
    ADX_THRESHOLD,
    BB_WIDTH_MIN_MULT,
    CONFIDENCE_THRESHOLD,
    CV_SPLITS,
    DATA_DIR,
    EMBARGO_PCT,
    FRACTIONAL_D,
    INITIAL_BALANCE,
    META_LABEL_THRESHOLD,
    MIN_OOF_F1,
    PURGE_PCT,
    RANDOM_STATE,
    SHORT_META_LABEL_THRESHOLD,
    USE_META_LABELING,
    PipelineConfig,
)
from src.data import collect_parquet_file_paths
from src.dataset import assemble_labeled_dataset, extract_feature_columns
from src.models import HybridStackingSignalClassifier
from src.reporting import publish_pipeline_results


def start_accelerator_with_seed(random_state: int) -> Accelerator:
    set_seed(random_state)
    return Accelerator()


def measure_step_duration(timing: dict[str, float], name: str, step, *args, **kwargs) -> Any:
    started = time.perf_counter()
    result = step(*args, **kwargs)
    timing[name] = time.perf_counter() - started
    return result


def format_parquet_file_range(config: PipelineConfig) -> str:
    files = collect_parquet_file_paths(DATA_DIR, config.months)
    return f"{files[0].stem} → {files[-1].stem}"


def build_position_strategy_kwargs() -> dict[str, Any]:
    return {
        "min_oof_f1": MIN_OOF_F1,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "use_meta_labeling": USE_META_LABELING,
        "meta_label_threshold": META_LABEL_THRESHOLD,
        "short_meta_label_threshold": SHORT_META_LABEL_THRESHOLD,
        "adx_threshold": ADX_THRESHOLD,
        "bb_width_min_mult": BB_WIDTH_MIN_MULT,
        "random_state": RANDOM_STATE,
    }


def train_hybrid_stacking_model(train: pl.DataFrame, features: list[str]) -> HybridStackingSignalClassifier:
    return HybridStackingSignalClassifier(
        n_splits=CV_SPLITS, embargo_pct=EMBARGO_PCT, **build_position_strategy_kwargs(),
    ).fit(train[features], train["label"], train["event_end"])


def build_run_config_payload(config: PipelineConfig, timing: dict[str, float]) -> dict[str, Any]:
    return {
        "months": "full" if config.months is None else f"{config.months} months",
        "data_range": format_parquet_file_range(config),
        "cv_splits": CV_SPLITS,
        "embargo_pct": EMBARGO_PCT,
        "purge_pct": PURGE_PCT,
        "fractional_d": FRACTIONAL_D,
        "min_oof_f1": MIN_OOF_F1,
        "random_state": RANDOM_STATE,
        "timeframe": config.timeframe,
        "initial_balance": INITIAL_BALANCE,
        "use_meta_labeling": USE_META_LABELING,
        "meta_label_threshold": META_LABEL_THRESHOLD,
        "timing": timing,
    }


def execute_model_pipeline(config: PipelineConfig, timing: dict[str, float]) -> dict[str, Any]:
    _, train, test = measure_step_duration(timing, "data_loading", assemble_labeled_dataset, config)
    features = extract_feature_columns(train)

    model = measure_step_duration(timing, "model_training", train_hybrid_stacking_model, train, features)
    predictions = measure_step_duration(timing, "prediction", model.predict, test[features])
    positions = measure_step_duration(timing, "positions", model.predict_positions, test[features])
    backtest_metrics, executed_trades = measure_step_duration(
        timing, "backtesting", backtest_signal_positions, test, positions,
    )
    return {
        "train": train,
        "test": test,
        "features": features,
        "model": model,
        "predictions": predictions,
        "positions": positions,
        "backtest_metrics": backtest_metrics,
        "executed_trades": executed_trades,
    }


def print_timing_summary(timing: dict[str, float]) -> None:
    print("\n=== PIPELINE TIMING ===")
    for step, secs in timing.items():
        print(f"  {step:<22s} {secs:>8.3f}s")
    print("========================\n")


def run_pipeline(config: PipelineConfig) -> None:
    accelerator = start_accelerator_with_seed(RANDOM_STATE)
    if not accelerator.is_local_main_process:
        return

    t_total = time.perf_counter()
    timing: dict[str, float] = {}
    outputs = execute_model_pipeline(config, timing)
    config_payload = build_run_config_payload(config, timing)
    measure_step_duration(timing, "reporting", publish_pipeline_results, accelerator, config_payload, outputs)
    timing["total"] = time.perf_counter() - t_total
    print_timing_summary(timing)
