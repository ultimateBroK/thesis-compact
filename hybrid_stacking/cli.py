from __future__ import annotations

import argparse

from hybrid_stacking.acceleration import configure_accelerator
from hybrid_stacking.backtest import backtest_signals, cost_adjusted_returns, equity_curve
from hybrid_stacking.config import (
    CV_SPLITS,
    EMBARGO_PCT,
    MIN_OOF_F1,
    RANDOM_STATE,
    REPORT_DIR,
    PipelineConfig,
)
from hybrid_stacking.dataset import build_dataset, feature_columns, train_test_time_split
from hybrid_stacking.models import HybridStackingSignalClassifier
from hybrid_stacking.reporting import (
    print_acceleration_report,
    print_backtest_report,
    print_classification_report,
    print_dataset_report,
    print_model_report,
    save_run_plots,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--months phải >= 1; dùng --full để chạy toàn bộ")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid Stacking dự báo tín hiệu CFD vàng")
    parser.add_argument("--months", type=positive_int, default=12, help="Số tháng dữ liệu gần nhất")
    parser.add_argument("--full", action="store_true", help="Dùng toàn bộ dữ liệu parquet")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(months=None if args.full else args.months)


def train_model(train, features: list[str]) -> HybridStackingSignalClassifier:
    return HybridStackingSignalClassifier(
        n_splits=CV_SPLITS,
        embargo_pct=EMBARGO_PCT,
        min_oof_f1=MIN_OOF_F1,
        random_state=RANDOM_STATE,
    ).fit(train[features], train["label"], train["event_end"])


def run(config: PipelineConfig) -> None:
    accelerator = configure_accelerator(RANDOM_STATE)
    if not accelerator.is_local_main_process:
        return

    dataset = build_dataset(config)
    train, test = train_test_time_split(dataset)
    features = feature_columns(dataset)
    model = train_model(train, features)
    predictions = model.predict(test[features])
    strategy_returns = cost_adjusted_returns(test, predictions)
    equity = equity_curve(strategy_returns, test.index)
    backtest_metrics = backtest_signals(test, predictions)
    print_acceleration_report(accelerator)
    print_dataset_report(dataset, train, test, len(features))
    print_model_report(model)
    print_classification_report(test["label"], predictions)
    print_backtest_report(backtest_metrics)
    save_run_plots(model, equity, REPORT_DIR)


def main() -> None:
    run(config_from_args(parse_args()))
