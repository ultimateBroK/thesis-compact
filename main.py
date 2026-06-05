"""Entry point: parse args, run pipeline, publish results."""

from __future__ import annotations

import argparse
import warnings
import time

from src.config import PipelineConfig
from src.pipeline import (
    TimingResults,
    build_run_config_payload,
    run_model_pipeline,
)
from src.reporting import publish_pipeline_results
from src.reporting.console import print_timing_summary


def validate_positive_month_count(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "--months must be >= 1; use --full for all data"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid Stacking gold signal prediction"
    )
    parser.add_argument(
        "--months",
        type=validate_positive_month_count,
        default=PipelineConfig().months,
        help="Number of latest monthly parquet files to load",
    )
    parser.add_argument(
        "--full", action="store_true", help="Use all available parquet data"
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*probability.*parameter.*deprecated.*",
        category=FutureWarning,
    )
    args = parse_args()
    config = PipelineConfig(months=None if args.full else args.months)
    t_total = time.perf_counter()

    outputs, ml_timing = run_model_pipeline(config)
    config_payload = build_run_config_payload(config, TimingResults(**ml_timing))

    publish_pipeline_results(
        config_payload.as_dict(),
        outputs,
        timing=ml_timing,
        total_start=t_total,
    )
    print_timing_summary(TimingResults(**ml_timing))


if __name__ == "__main__":
    main()
