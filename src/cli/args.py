from __future__ import annotations

import argparse

from src.config import PipelineConfig


def validate_positive_month_count(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "--months must be >= 1; use --full for all data"
        )
    return parsed


def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid Stacking CFD gold signal prediction"
    )
    parser.add_argument(
        "--months",
        type=validate_positive_month_count,
        default=PipelineConfig().months,
        help="Number of months to load from first month",
    )
    parser.add_argument(
        "--full", action="store_true", help="Use all available parquet data"
    )
    parser.add_argument(
        "--long-only", action="store_true", help="Disable all SHORT positions"
    )
    parser.add_argument(
        "--backtest-tp",
        type=float,
        default=PipelineConfig().backtest_tp_atr,
        help=f"Backtest TP distance in ATR multiples (default: {PipelineConfig().backtest_tp_atr})",
    )
    parser.add_argument(
        "--backtest-sl",
        type=float,
        default=PipelineConfig().backtest_sl_atr,
        help=f"Backtest SL distance in ATR multiples (default: {PipelineConfig().backtest_sl_atr})",
    )
    parser.add_argument(
        "--min-hold",
        type=int,
        default=PipelineConfig().min_position_hold,
        help=f"Minimum bars to hold a position before allowing exit (default: {PipelineConfig().min_position_hold})",
    )

    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run expanding walk-forward evaluation instead of single train/test split",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip backtest hyperparameter tuning (default: tune is enabled)",
    )
    return parser.parse_args()


def derive_config_from_arguments(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        months=None if args.full else args.months,
        long_only=args.long_only,
        backtest_tp_atr=args.backtest_tp,
        backtest_sl_atr=args.backtest_sl,
        min_position_hold=args.min_hold,
        tune_backtest=not args.no_tune,
        walk_forward=args.walk_forward,
    )
