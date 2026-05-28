from __future__ import annotations

import argparse

from src.config import PipelineConfig


def validate_positive_month_count(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--months must be >= 1; use --full for all data")
    return parsed


def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid Stacking CFD gold signal prediction")
    parser.add_argument(
        "--months",
        type=validate_positive_month_count,
        default=PipelineConfig().months,
        help="Number of months to load from first month",
    )
    parser.add_argument("--full", action="store_true", help="Use all available parquet data")
    return parser.parse_args()


def derive_config_from_arguments(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(months=None if args.full else args.months)
