"""Shared data structures for pipeline execution and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from src.models import HybridStackingSignalClassifier


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingResults:
    """Immutable pipeline step timings in seconds."""

    data_loading: float = 0.0
    model_training: float = 0.0
    prediction: float = 0.0
    positions: float = 0.0
    backtesting: float = 0.0
    reporting: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "data_loading": self.data_loading,
            "model_training": self.model_training,
            "prediction": self.prediction,
            "positions": self.positions,
            "backtesting": self.backtesting,
            "reporting": self.reporting,
            "total": self.total,
        }


# ---------------------------------------------------------------------------
# Config payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfigPayload:
    """Serializable pipeline configuration metadata for reporting."""

    months: str = ""
    data_range: str = ""
    cv_splits: int = 0
    embargo_pct: float = 0.0
    purge_pct: float = 0.0
    min_oof_f1: float = 0.0
    random_state: int = 0
    timeframe: str = "1h"
    initial_balance: float = 10_000.0
    labeling_method: str = "fixed_horizon_future_return"
    labeling_horizon: int = 4
    signal_probability_threshold: float = 0.55
    timing: TimingResults | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "months": self.months,
            "data_range": self.data_range,
            "cv_splits": self.cv_splits,
            "embargo_pct": self.embargo_pct,
            "purge_pct": self.purge_pct,
            "min_oof_f1": self.min_oof_f1,
            "random_state": self.random_state,
            "timeframe": self.timeframe,
            "initial_balance": self.initial_balance,
            "labeling_method": self.labeling_method,
            "labeling_horizon": self.labeling_horizon,
            "signal_probability_threshold": self.signal_probability_threshold,
            "timing": self.timing.as_dict() if self.timing else {},
        }


# ---------------------------------------------------------------------------
# Pipeline outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineOutputs:
    """Output artifact bundle from a single pipeline execution."""

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

    def to_dict(
        self,
        window_id: int | None = None,
        window_train_range: str = "",
        window_test_range: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        if window_id is not None:
            payload["window_id"] = window_id
            payload["window_train_range"] = window_train_range
            payload["window_test_range"] = window_test_range
        return payload
