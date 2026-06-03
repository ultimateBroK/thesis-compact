"""Entry point: parse args, run pipeline, publish results."""

from __future__ import annotations

import argparse
import time

from src.config import PipelineConfig
from src.pipeline import (
    PipelineOutputs,
    TimingResults,
    build_run_config_payload,
    run_model_pipeline,
    run_walk_forward_pipeline,
)
from src.reporting import publish_pipeline_results, print_timing_summary


def validate_positive_month_count(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--months must be >= 1; use --full for all data")
    return parsed


def parse_args() -> argparse.Namespace:
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
        help="Run expanding walk-forward evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        months=None if args.full else args.months,
        long_only=args.long_only,
        walk_forward=args.walk_forward,
    )
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
    config_payload = build_run_config_payload(config, TimingResults(**ml_timing))

    t_report = time.perf_counter()
    publish_pipeline_results(config_payload.as_dict(), outputs)
    ml_timing["reporting"] = time.perf_counter() - t_report
    ml_timing["total"] = time.perf_counter() - t_total
    print_timing_summary(TimingResults(**ml_timing))


if __name__ == "__main__":
    main()
