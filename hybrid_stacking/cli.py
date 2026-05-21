from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from hybrid_stacking.acceleration import configure_accelerator
from hybrid_stacking.backtest import backtest_signals, cost_adjusted_returns, equity_curve
from hybrid_stacking.config import (
    CV_SPLITS,
    EMBARGO_PCT,
    FRACTIONAL_D,
    MIN_OOF_F1,
    PURGE_PCT,
    RANDOM_STATE,
    REPORT_DIR,
    PipelineConfig,
    TradingCosts,
)
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
    train, test = train_test_time_split(dataset, purge_pct=PURGE_PCT)
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

    run_dir = REPORT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_run_artifacts(
        run_dir=run_dir,
        model=model,
        test=test,
        predictions=predictions,
        strategy_returns=strategy_returns,
        equity=equity,
        backtest_metrics=backtest_metrics,
        config_payload={
            "months": config.months,
            "cv_splits": CV_SPLITS,
            "embargo_pct": EMBARGO_PCT,
            "purge_pct": PURGE_PCT,
            "fractional_d": FRACTIONAL_D,
            "min_oof_f1": MIN_OOF_F1,
            "random_state": RANDOM_STATE,
            "timeframe": "1h",
            "trading_costs": {
                "slippage_points": TradingCosts().slippage_points,
                "spread_multiplier": TradingCosts().spread_multiplier,
            },
        },
        dataset=dataset,
        train=train,
        test_df=test,
        features=features,
    )


def main() -> None:
    run(config_from_args(parse_args()))
