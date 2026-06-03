"""CLI: argument parsing and pipeline entry point."""

from __future__ import annotations

import argparse

from src.config import PipelineConfig
from src.pipeline import run_pipeline


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_positive_month_count(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--months must be >= 1; use --full for all data")
    return parsed


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


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
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    run_pipeline(derive_config_from_arguments(parse_command_line_arguments()))
