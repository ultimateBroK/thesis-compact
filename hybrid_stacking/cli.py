from __future__ import annotations

import argparse
import time
from datetime import datetime
from typing import Any, Callable

import polars as pl
from accelerate import Accelerator
from accelerate.utils import set_seed

from hybrid_stacking.backtest import backtest_signals
from hybrid_stacking.config import (
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
    REPORT_DIR,
    USE_META_LABELING,
    WAVELET,
    WAVELET_LEVEL,
    PipelineConfig,
)
from hybrid_stacking.data import parquet_files
from hybrid_stacking.dataset import build_dataset, feature_columns, train_test_time_split
from hybrid_stacking.models import HybridStackingSignalClassifier
from hybrid_stacking.reporting import (
    print_acceleration_report,
    print_backtest_report,
    print_classification_report,
    print_dataset_report,
    print_model_report,
    save_run_artifacts,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--months phải >= 1; dùng --full để chạy toàn bộ")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid Stacking dự báo tín hiệu CFD vàng")
    parser.add_argument(
        "--months",
        type=positive_int,
        default=PipelineConfig().months,
        help="Số tháng dữ liệu tính từ tháng đầu",
    )
    parser.add_argument("--full", action="store_true", help="Dùng toàn bộ dữ liệu parquet")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(months=None if args.full else args.months)


def configure_accelerator(random_state: int) -> Accelerator:
    set_seed(random_state)
    return Accelerator()


def train_model(train: pl.DataFrame, features: list[str]) -> HybridStackingSignalClassifier:
    return HybridStackingSignalClassifier(
        n_splits=CV_SPLITS,
        embargo_pct=EMBARGO_PCT,
        min_oof_f1=MIN_OOF_F1,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        use_meta_labeling=USE_META_LABELING,
        meta_label_threshold=META_LABEL_THRESHOLD,
        random_state=RANDOM_STATE,
    ).fit(train[features], train["label"], train["event_end"])


def timed_step(
    timing: dict[str, float],
    name: str,
    step: Callable[..., Any],
    *args,
    **kwargs,
) -> Any:
    started = time.perf_counter()
    result = step(*args, **kwargs)
    timing[name] = time.perf_counter() - started
    return result


def file_range(config: PipelineConfig) -> str:
    files = parquet_files(DATA_DIR, config.months)
    return f"{files[0].stem} → {files[-1].stem}"


def config_payload(config: PipelineConfig, timing: dict[str, float]) -> dict[str, Any]:
    return {
        "months": "full" if config.months is None else f"{config.months} months",
        "data_range": file_range(config),
        "cv_splits": CV_SPLITS,
        "embargo_pct": EMBARGO_PCT,
        "purge_pct": PURGE_PCT,
        "fractional_d": FRACTIONAL_D,
        "wavelet": WAVELET,
        "wavelet_level": WAVELET_LEVEL,
        "min_oof_f1": MIN_OOF_F1,
        "random_state": RANDOM_STATE,
        "timeframe": "1h",
        "initial_balance": INITIAL_BALANCE,
        "use_meta_labeling": USE_META_LABELING,
        "meta_label_threshold": META_LABEL_THRESHOLD,
        "timing": timing,
    }


def run_model_pipeline(config: PipelineConfig, timing: dict[str, float]) -> dict[str, Any]:
    dataset = timed_step(timing, "data_loading", build_dataset, config)
    train, test = timed_step(
        timing,
        "train_test_split",
        train_test_time_split,
        dataset,
        purge_pct=PURGE_PCT,
    )
    features = timed_step(timing, "feature_extraction", feature_columns, dataset)
    model = timed_step(timing, "model_training", train_model, train, features)
    predictions = timed_step(timing, "prediction", model.predict, test[features])
    positions = timed_step(timing, "positions", model.predict_positions, test[features])
    backtest_metrics = timed_step(timing, "backtesting", backtest_signals, test, positions)
    return {
        "dataset": dataset,
        "train": train,
        "test": test,
        "features": features,
        "model": model,
        "predictions": predictions,
        "positions": positions,
        "backtest_metrics": backtest_metrics,
    }


def publish_pipeline_results(
    accelerator: Accelerator,
    config: PipelineConfig,
    timing: dict[str, float],
    outputs: dict[str, Any],
) -> None:
    dataset = outputs["dataset"]
    train = outputs["train"]
    test = outputs["test"]
    features = outputs["features"]
    model = outputs["model"]
    predictions = outputs["predictions"]
    positions = outputs["positions"]
    backtest_metrics = outputs["backtest_metrics"]

    print_acceleration_report(accelerator)
    print_dataset_report(dataset, train, test, len(features))
    print_model_report(model)
    print_classification_report(test["label"], predictions)
    print_backtest_report(backtest_metrics)

    save_run_artifacts(
        run_dir=REPORT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        model=model,
        test=test,
        predictions=predictions,
        positions=positions,
        config_payload=config_payload(config, timing),
        dataset=dataset,
        train=train,
        test_df=test,
        features=features,
        backtest_metrics=backtest_metrics,
    )


def print_timing_report(timing: dict[str, float]) -> None:
    print("\n=== PIPELINE TIMING ===")
    for step, secs in timing.items():
        print(f"  {step:<22s} {secs:>8.3f}s")
    print("========================\n")


def run(config: PipelineConfig) -> None:
    accelerator = configure_accelerator(RANDOM_STATE)
    if not accelerator.is_local_main_process:
        return

    t_total = time.perf_counter()
    timing: dict[str, float] = {}
    outputs = run_model_pipeline(config, timing)
    timed_step(
        timing,
        "reporting",
        publish_pipeline_results,
        accelerator,
        config,
        timing,
        outputs,
    )
    timing["total"] = time.perf_counter() - t_total
    print_timing_report(timing)


def main() -> None:
    run(config_from_args(parse_args()))
