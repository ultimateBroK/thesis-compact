"""Điều phối pipeline: tải dữ liệu, gán nhãn, train model, dự đoán và backtest.

Luồng xử lý:
1. Tải dữ liệu tick parquet XAU/USD → nến OHLC 1H
2. Tạo đặc trưng kỹ thuật
3. Tạo nhãn lợi suất tương lai theo fixed horizon
4. Chia train/test theo thời gian, có purge gap để tránh rò rỉ nhãn
5. Huấn luyện mô hình Hybrid Stacking
6. Dự đoán tín hiệu Buy/Sell trên tập test
7. Chạy backtest vector hóa đơn giản
8. Trả kết quả cho báo cáo
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np
import polars as pl

from src.backtest import apply_fixed_horizon_positions, run_signal_backtest
from src.config import DATA_DIR, LABELING_METHOD, PipelineConfig
from src.reporting.console import print_dataset_split_report
from src.data import (
    build_labeled_dataset,
    collect_parquet_paths,
)
from src.features import get_feature_columns
from src.models import HybridStackingSignalClassifier


# ── Data structures ──────────────────────────────────────────────


@dataclass(frozen=True)
class TimingResults:
    """Thời gian từng bước quy trình, tính bằng giây."""

    data_loading: float = 0.0
    model_training: float = 0.0
    prediction: float = 0.0
    positions: float = 0.0
    backtesting: float = 0.0
    reporting: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {key: value for key, value in asdict(self).items() if value > 0.0}


@dataclass(frozen=True)
class RunConfigPayload:
    """Payload cấu hình có thể serialize để lưu vào báo cáo."""

    months: str = ""
    data_range: str = ""
    cv_splits: int = 0
    purge_bars: int = 0
    random_state: int = 0
    timeframe: str = "1h"
    initial_balance: float = 10_000.0
    labeling_method: str = LABELING_METHOD
    labeling_horizon: int = 4
    label_return_threshold: float = 0.0005
    max_label_gap_hours: float = 5.0
    timing: TimingResults | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timing"] = self.timing.as_dict() if self.timing else {}
        return payload


@dataclass(frozen=True)
class PipelineOutputs:
    """Gói kết quả của một lần chạy pipeline."""

    train: pl.DataFrame = field(repr=False)
    test_labeled: pl.DataFrame = field(repr=False)
    test_continuous: pl.DataFrame = field(repr=False)
    features: list[str]
    model: HybridStackingSignalClassifier = field(repr=False)
    predictions: np.ndarray = field(repr=False)
    raw_signals: np.ndarray = field(repr=False)
    positions: np.ndarray = field(repr=False)
    backtest_metrics: dict[str, float]
    equity: np.ndarray = field(repr=False)
    executed_trades: list[dict] = field(repr=False)
    pred_proba: np.ndarray | None = field(repr=False, default=None)

    def as_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


# ── Config helpers ───────────────────────────────────────────────


def format_parquet_file_range(config: PipelineConfig) -> str:
    files = collect_parquet_paths(DATA_DIR, config.months)
    return f"{files[0].stem} -> {files[-1].stem}"


def build_run_config_payload(
    config: PipelineConfig, timing: TimingResults
) -> RunConfigPayload:
    return RunConfigPayload(
        months="full" if config.months is None else f"{config.months} months",
        data_range=format_parquet_file_range(config),
        cv_splits=config.cv_splits,
        purge_bars=config.purge_bars,
        random_state=config.random_state,
        timeframe=config.timeframe,
        initial_balance=config.initial_balance,
        labeling_horizon=config.labeling_horizon,
        label_return_threshold=config.label_return_threshold,
        max_label_gap_hours=config.max_label_gap_hours,
        timing=timing,
    )


# ── Model training ───────────────────────────────────────────────


def train_hybrid_stacking_model(
    train: pl.DataFrame,
    features: list[str],
    config: PipelineConfig,
) -> HybridStackingSignalClassifier:
    return HybridStackingSignalClassifier(
        n_splits=config.cv_splits,
        random_state=config.random_state,
        labels=config.labels,
    ).fit(train[features], train["label"], train["event_end"], train["event_start"])


# ── Single-run pipeline ──────────────────────────────────────────


def run_model_pipeline(
    config: PipelineConfig,
) -> tuple[PipelineOutputs, dict[str, float]]:
    """Chạy pipeline một lần; trả về kết quả và thời gian từng bước."""
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    dataset = build_labeled_dataset(config)
    print_dataset_split_report(dataset.split_info)
    train = dataset.train_labeled
    test_labeled = dataset.test_labeled
    test_continuous = dataset.test_continuous
    features = get_feature_columns(train)
    timing["data_loading"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = train_hybrid_stacking_model(train, features, config)
    timing["model_training"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    predictions = model.predict(
        test_labeled[features]
    )  # nhãn dự đoán cho các test bar đã có label
    pred_proba = model.predict_proba(test_labeled[features])
    raw_signals = model.predict_signals(
        test_continuous[features]
    )  # tín hiệu {-1,+1} liên tục cho toàn bộ test bar
    timing["prediction"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    positions = apply_fixed_horizon_positions(
        raw_signals, hold_bars=config.backtest_hold_bars
    )
    timing["positions"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    backtest_metrics, executed_trades, equity = run_signal_backtest(
        test_continuous, positions, initial_balance=config.initial_balance
    )
    timing["backtesting"] = time.perf_counter() - t0

    outputs = PipelineOutputs(
        train=train,
        test_labeled=test_labeled,
        test_continuous=test_continuous,
        features=features,
        model=model,
        predictions=predictions,
        raw_signals=raw_signals,
        positions=positions,
        backtest_metrics=backtest_metrics,
        equity=equity,
        executed_trades=executed_trades,
        pred_proba=pred_proba,
    )
    return outputs, timing
