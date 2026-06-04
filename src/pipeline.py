"""Pipeline orchestration: load, label, split, train, predict, backtest.

Story:
1. Load XAU/USD parquet ticks → 1H OHLC candles
2. Build technical features
3. Create fixed-horizon future-return labels
4. Split train/test chronologically (with purge gap)
5. Train hybrid stacking ensemble
6. Predict test signals → positions
7. Run simple vectorized backtest
8. Return results for reporting
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from src.backtest import apply_fixed_horizon_positions, run_signal_backtest
from src.config import (
    BACKTEST_HOLD_BARS,
    CV_SPLITS,
    DATA_DIR,
    EMBARGO_PCT,
    INITIAL_BALANCE,
    LABELING_HORIZON,
    LABEL_RETURN_THRESHOLD,
    MAX_LABEL_GAP_HOURS,
    MIN_OOF_F1,
    PURGE_BARS,
    RANDOM_STATE,
    SIGNAL_PROBABILITY_MARGIN,
    SIGNAL_PROBABILITY_THRESHOLD,
    PipelineConfig,
)
from src.data import (
    apply_labels_to_frame,
    build_labeled_dataset,
    collect_parquet_paths,
    load_featured_candles,
)
from src.features import get_feature_columns
from src.models import HybridStackingSignalClassifier


# ── Data structures ──────────────────────────────────────────────


@dataclass(frozen=True)
class TimingResults:
    """Pipeline step timings in seconds."""

    data_loading: float = 0.0
    model_training: float = 0.0
    prediction: float = 0.0
    positions: float = 0.0
    backtesting: float = 0.0
    reporting: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            k: v for k, v in {
                "data_loading": self.data_loading,
                "model_training": self.model_training,
                "prediction": self.prediction,
                "positions": self.positions,
                "backtesting": self.backtesting,
                "reporting": self.reporting,
                "total": self.total,
            }.items()
            if v > 0.0
        }


@dataclass(frozen=True)
class RunConfigPayload:
    """Serializable pipeline configuration for reporting."""

    months: str = ""
    data_range: str = ""
    cv_splits: int = 0
    embargo_pct: float = 0.0
    purge_bars: int = 0
    min_oof_f1: float = 0.0
    random_state: int = 0
    timeframe: str = "1h"
    initial_balance: float = 10_000.0
    labeling_method: str = "fixed_horizon_future_return"
    labeling_horizon: int = 4
    label_return_threshold: float = 0.0005
    max_label_gap_hours: float = 5.0
    signal_probability_threshold: float = 0.50
    signal_probability_margin: float = 0.02
    timing: TimingResults | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "months": self.months,
            "data_range": self.data_range,
            "cv_splits": self.cv_splits,
            "embargo_pct": self.embargo_pct,
            "purge_bars": self.purge_bars,
            "min_oof_f1": self.min_oof_f1,
            "random_state": self.random_state,
            "timeframe": self.timeframe,
            "initial_balance": self.initial_balance,
            "labeling_method": self.labeling_method,
            "labeling_horizon": self.labeling_horizon,
            "label_return_threshold": self.label_return_threshold,
            "max_label_gap_hours": self.max_label_gap_hours,
            "signal_probability_threshold": self.signal_probability_threshold,
            "signal_probability_margin": self.signal_probability_margin,
            "timing": self.timing.as_dict() if self.timing else {},
        }


@dataclass(frozen=True)
class PipelineOutputs:
    """Output bundle from a single pipeline execution."""

    train: pl.DataFrame = field(repr=False)
    test: pl.DataFrame = field(repr=False)
    features: list[str]
    model: HybridStackingSignalClassifier = field(repr=False)
    predictions: np.ndarray = field(repr=False)
    positions: np.ndarray = field(repr=False)
    backtest_metrics: dict[str, float]
    equity: np.ndarray = field(repr=False)
    executed_trades: list[dict] = field(repr=False)
    pred_proba: np.ndarray = field(repr=False, default=None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "train": self.train,
            "test": self.test,
            "features": self.features,
            "model": self.model,
            "predictions": self.predictions,
            "positions": self.positions,
            "backtest_metrics": self.backtest_metrics,
            "executed_trades": self.executed_trades,
            "equity": self.equity,
            "pred_proba": self.pred_proba,
        }


# ── Config helpers ───────────────────────────────────────────────


def format_parquet_file_range(config: PipelineConfig) -> str:
    files = collect_parquet_paths(DATA_DIR, config.months)
    return f"{files[0].stem} -> {files[-1].stem}"


def build_run_config_payload(config: PipelineConfig, timing: TimingResults) -> RunConfigPayload:
    return RunConfigPayload(
        months="full" if config.months is None else f"{config.months} months",
        data_range=format_parquet_file_range(config),
        cv_splits=CV_SPLITS,
        embargo_pct=EMBARGO_PCT,
        purge_bars=PURGE_BARS,
        min_oof_f1=MIN_OOF_F1,
        random_state=RANDOM_STATE,
        timeframe=config.timeframe,
        initial_balance=INITIAL_BALANCE,
        labeling_horizon=LABELING_HORIZON,
        label_return_threshold=LABEL_RETURN_THRESHOLD,
        max_label_gap_hours=MAX_LABEL_GAP_HOURS,
        signal_probability_threshold=SIGNAL_PROBABILITY_THRESHOLD,
        signal_probability_margin=SIGNAL_PROBABILITY_MARGIN,
        timing=timing,
    )


# ── Model training ───────────────────────────────────────────────


def train_hybrid_stacking_model(
    train: pl.DataFrame,
    features: list[str],
    config: PipelineConfig,
) -> HybridStackingSignalClassifier:
    return HybridStackingSignalClassifier(
        n_splits=CV_SPLITS,
        embargo_pct=EMBARGO_PCT,
        min_oof_f1=MIN_OOF_F1,
        signal_probability_threshold=SIGNAL_PROBABILITY_THRESHOLD,
        signal_probability_margin=SIGNAL_PROBABILITY_MARGIN,
        random_state=RANDOM_STATE,
        long_only=config.long_only,
    ).fit(train[features], train["label"], train["event_end"])


# ── Single-run pipeline ──────────────────────────────────────────


def run_model_pipeline(config: PipelineConfig) -> tuple[PipelineOutputs, dict[str, float]]:
    """Load data, train model, predict, backtest. Returns (outputs, timing)."""
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    _, train, test = build_labeled_dataset(config)
    features = get_feature_columns(train)
    timing["data_loading"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = train_hybrid_stacking_model(train, features, config)
    timing["model_training"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    predictions = model.predict(test[features])
    raw_positions = model.predict_positions(test[features])
    positions = apply_fixed_horizon_positions(raw_positions, hold_bars=BACKTEST_HOLD_BARS)
    pred_proba = model.predict_proba(test[features])
    timing["prediction"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    backtest_metrics, executed_trades, equity = run_signal_backtest(test, positions)
    timing["backtesting"] = time.perf_counter() - t0

    outputs = PipelineOutputs(
        train=train,
        test=test,
        features=features,
        model=model,
        predictions=predictions,
        positions=positions,
        backtest_metrics=backtest_metrics,
        equity=equity,
        executed_trades=executed_trades,
        pred_proba=pred_proba,
    )
    return outputs, timing


