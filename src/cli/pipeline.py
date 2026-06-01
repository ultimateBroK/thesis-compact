"""
CLI pipeline: parse args -> accelerate -> data -> train -> predict -> backtest -> report.

Orchestration: run_pipeline is the top-level entry; execute_model_pipeline drives the ML flow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import polars as pl
from accelerate import Accelerator
from accelerate.utils import set_seed
from src.backtest import backtest_signal_positions, tune_backtest_hyperparameters
from src.config import (
    ADX_THRESHOLD,
    BB_WIDTH_MIN_MULT,
    CONFIDENCE_THRESHOLD,
    CV_SPLITS,
    DATA_DIR,
    EMBARGO_PCT,
    FRACTIONAL_D,
    INITIAL_BALANCE,
    META_LABEL_THRESHOLD,
    MIN_OOF_F1,
    PURGE_PCT,
    RANDOM_STATE,
    SHORT_META_LABEL_THRESHOLD,
    USE_META_LABELING,
    TREND_EMA_PERIOD,
    TREND_FILTER_ENABLED,
    TUNE_HOLD_VALUES,
    TUNE_SL_RANGE_BT,
    TUNE_TP_RANGE_BT,
    PipelineConfig,
)
from src.data import collect_parquet_file_paths
from src.dataset import (
    apply_labels_to_frame,
    assemble_labeled_dataset,
    auto_calibrate_barrier_widths,
    extract_feature_columns,
    load_featured_candles,
)
from src.models import HybridStackingSignalClassifier
from src.models.main import enforce_minimum_position_hold
from src.reporting import publish_pipeline_results
from src.validation import walk_forward_split


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingResults:
    """Immutable pipeline step timings in seconds."""

    data_loading: float = 0.0
    model_training: float = 0.0
    tuning: float = 0.0
    prediction: float = 0.0
    positions: float = 0.0
    backtesting: float = 0.0
    reporting: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "data_loading": self.data_loading,
            "model_training": self.model_training,
            "tuning": self.tuning,
            "prediction": self.prediction,
            "positions": self.positions,
            "backtesting": self.backtesting,
            "reporting": self.reporting,
            "total": self.total,
        }


@dataclass(frozen=True)
class RunConfigPayload:
    """Serializable pipeline configuration metadata for reporting."""

    months: str = ""
    data_range: str = ""
    cv_splits: int = 0
    embargo_pct: float = 0.0
    purge_pct: float = 0.0
    fractional_d: float = 0.0
    min_oof_f1: float = 0.0
    random_state: int = 0
    timeframe: str = "1h"
    initial_balance: float = 10_000.0
    use_meta_labeling: bool = True
    meta_label_threshold: float = 0.55
    timing: TimingResults | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "months": self.months,
            "data_range": self.data_range,
            "cv_splits": self.cv_splits,
            "embargo_pct": self.embargo_pct,
            "purge_pct": self.purge_pct,
            "fractional_d": self.fractional_d,
            "min_oof_f1": self.min_oof_f1,
            "random_state": self.random_state,
            "timeframe": self.timeframe,
            "initial_balance": self.initial_balance,
            "use_meta_labeling": self.use_meta_labeling,
            "meta_label_threshold": self.meta_label_threshold,
            "timing": self.timing.as_dict() if self.timing else {},
        }


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
        }
        if window_id is not None:
            payload["window_id"] = window_id
            payload["window_train_range"] = window_train_range
            payload["window_test_range"] = window_test_range
        return payload


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------


def start_accelerator_with_seed(random_state: int) -> Accelerator:
    set_seed(random_state)
    return Accelerator()


def measure_step_duration(
    name: str, step: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[Any, float]:
    """Execute *step* and return (result, elapsed_seconds). Pure -- no side effects."""
    started = time.perf_counter()
    result = step(*args, **kwargs)
    return result, time.perf_counter() - started


def format_parquet_file_range(config: PipelineConfig) -> str:
    files = collect_parquet_file_paths(DATA_DIR, config.months)
    return f"{files[0].stem} -> {files[-1].stem}"


def build_position_strategy_kwargs(config: PipelineConfig) -> dict[str, Any]:
    return {
        "min_oof_f1": MIN_OOF_F1,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "use_meta_labeling": USE_META_LABELING,
        "meta_label_threshold": META_LABEL_THRESHOLD,
        "short_meta_label_threshold": SHORT_META_LABEL_THRESHOLD,
        "adx_threshold": ADX_THRESHOLD,
        "bb_width_min_mult": BB_WIDTH_MIN_MULT,
        "random_state": RANDOM_STATE,
        "long_only": config.long_only,
        "trend_filter_enabled": TREND_FILTER_ENABLED,
        "trend_ema_period": TREND_EMA_PERIOD,
        "min_position_hold": config.min_position_hold,
    }


def train_hybrid_stacking_model(
    train: pl.DataFrame, features: list[str], config: PipelineConfig
) -> HybridStackingSignalClassifier:
    return HybridStackingSignalClassifier(
        n_splits=CV_SPLITS,
        embargo_pct=EMBARGO_PCT,
        **build_position_strategy_kwargs(config),
    ).fit(train[features], train["label"], train["event_end"])


def build_run_config_payload(
    config: PipelineConfig, timing: TimingResults
) -> RunConfigPayload:
    return RunConfigPayload(
        months="full" if config.months is None else f"{config.months} months",
        data_range=format_parquet_file_range(config),
        cv_splits=CV_SPLITS,
        embargo_pct=EMBARGO_PCT,
        purge_pct=PURGE_PCT,
        fractional_d=FRACTIONAL_D,
        min_oof_f1=MIN_OOF_F1,
        random_state=RANDOM_STATE,
        timeframe=config.timeframe,
        initial_balance=INITIAL_BALANCE,
        use_meta_labeling=USE_META_LABELING,
        meta_label_threshold=META_LABEL_THRESHOLD,
        timing=timing,
    )


def run_prediction_and_backtest(
    model: HybridStackingSignalClassifier,
    data: tuple[pl.DataFrame, pl.DataFrame],
    features: list[str],
    close_series: np.ndarray,
    config: PipelineConfig,
    tuned_min_hold: int | None = None,
    timing: dict[str, float] | None = None,
) -> PipelineOutputs:
    """Tune backtest parameters, predict positions, and run the backtest."""
    train, test = data
    backtest_tp = config.backtest_tp_atr
    backtest_sl = config.backtest_sl_atr
    min_hold = config.min_position_hold if tuned_min_hold is None else tuned_min_hold

    if config.tune_backtest:
        if timing is None:
            best = tune_backtest_hyperparameters(
                model,
                train,
                features,
                close_series,
                tp_range=TUNE_TP_RANGE_BT,
                sl_range=TUNE_SL_RANGE_BT,
                min_hold_values=TUNE_HOLD_VALUES,
            )
        else:
            best, timing["tuning"] = measure_step_duration(
                "tuning",
                tune_backtest_hyperparameters,
                model,
                train,
                features,
                close_series,
                tp_range=TUNE_TP_RANGE_BT,
                sl_range=TUNE_SL_RANGE_BT,
                min_hold_values=TUNE_HOLD_VALUES,
            )
        backtest_tp = best["tp"]
        backtest_sl = best["sl"]
        min_hold = best["min_hold"]
        if timing is not None:
            print(
                f"  Tuned: tp={backtest_tp:.1f} sl={backtest_sl:.1f} min_hold={min_hold} sharpe={best['score']:.3f}"
            )

    if timing is None:
        predictions = model.predict(test[features])
        raw_positions = model.predict_positions(
            test[features],
            test["close"].to_numpy(),
            skip_min_hold=True,
        )
        positions = enforce_minimum_position_hold(raw_positions, min_hold)
        backtest_metrics, executed_trades, equity = backtest_signal_positions(
            test,
            positions,
            backtest_tp,
            backtest_sl,
        )
    else:
        predictions, timing["prediction"] = measure_step_duration(
            "prediction",
            model.predict,
            test[features],
        )
        raw_positions, timing["positions"] = measure_step_duration(
            "positions",
            model.predict_positions,
            test[features],
            test["close"].to_numpy(),
            skip_min_hold=True,
        )
        positions = enforce_minimum_position_hold(raw_positions, min_hold)
        (backtest_metrics, executed_trades, equity), timing["backtesting"] = (
            measure_step_duration(
                "backtesting",
                backtest_signal_positions,
                test,
                positions,
                backtest_tp,
                backtest_sl,
            )
        )

    return PipelineOutputs(
        train=train,
        test=test,
        features=features,
        model=model,
        predictions=predictions,
        positions=positions,
        backtest_metrics=backtest_metrics,
        equity=equity,
        executed_trades=executed_trades,
    )


def execute_model_pipeline(
    config: PipelineConfig,
) -> tuple[PipelineOutputs, dict[str, float]]:
    """Run the ML pipeline and return outputs with accumulated timing."""
    timing: dict[str, float] = {}

    (featured, train, test, tp_atr, sl_atr), timing["data_loading"] = (
        measure_step_duration(
            "data_loading",
            assemble_labeled_dataset,
            config,
        )
    )
    features = extract_feature_columns(train)
    model, timing["model_training"] = measure_step_duration(
        "model_training",
        train_hybrid_stacking_model,
        train,
        features,
        config,
    )
    outputs = run_prediction_and_backtest(
        model,
        (train, test),
        features,
        train["close"].to_numpy(),
        config,
        timing=timing,
    )
    return outputs, timing


def execute_walk_forward_pipeline(config: PipelineConfig) -> list[PipelineOutputs]:
    """Run expanding walk-forward windows, returning per-window outputs."""
    featured = load_featured_candles(config)
    timestamps = featured["timestamp"].to_numpy()
    windows = walk_forward_split(timestamps, n_windows=config.n_windows)

    print(f"Walk-forward: {len(windows)} windows")
    window_outputs: list[PipelineOutputs] = []

    for train_idx, test_idx, w_id, train_range, test_range in windows:
        print(f"\n--- Window {w_id}: train={train_range}, test={test_range} ---")

        train_frame = featured[train_idx]
        test_frame = featured[test_idx]

        # Calibrate barriers on train data only
        tp_atr, sl_atr, _, _ = auto_calibrate_barrier_widths(train_frame)
        train_labeled = apply_labels_to_frame(train_frame, tp_atr=tp_atr, sl_atr=sl_atr)
        test_labeled = apply_labels_to_frame(test_frame, tp_atr=tp_atr, sl_atr=sl_atr)

        features = extract_feature_columns(train_labeled)

        model = train_hybrid_stacking_model(train_labeled, features, config)
        outputs = run_prediction_and_backtest(
            model,
            (train_labeled, test_labeled),
            features,
            train_labeled["close"].to_numpy(),
            config,
        )
        window_outputs.append(outputs)
    return window_outputs


def print_timing_summary(timing: TimingResults) -> None:
    print("\n=== PIPELINE TIMING ===")
    for step, secs in timing.as_dict().items():
        print(f"  {step:<22s} {secs:>8.3f}s")
    print("========================\n")


def run_pipeline(config: PipelineConfig) -> None:
    accelerator = start_accelerator_with_seed(RANDOM_STATE)
    if not accelerator.is_local_main_process:
        return

    t_total = time.perf_counter()

    if config.walk_forward:
        window_outputs = execute_walk_forward_pipeline(config)
        ml_timing = {
            "data_loading": 0,
            "model_training": 0,
            "tuning": 0,
            "prediction": 0,
            "positions": 0,
            "backtesting": 0,
        }
        timing = TimingResults(**ml_timing, reporting=0, total=0)
        config_payload = build_run_config_payload(config, timing)

        # Report each window
        for w_id, outputs in enumerate(window_outputs):
            print(f"\n=== Window {w_id} ===")
            publish_pipeline_results(
                accelerator,
                config_payload.as_dict(),
                outputs,
                window_id=w_id,
            )

        # Aggregate across windows
        all_metrics = [o.backtest_metrics for o in window_outputs]
        total_trades = sum(m.get("trades", 0) for m in all_metrics)
        agg = {
            "windows": len(window_outputs),
            "total_trades": total_trades,
            "avg_sharpe": np.mean([m.get("sharpe", 0) for m in all_metrics]),
            "avg_profit_factor": np.mean(
                [m.get("profit_factor", 0) for m in all_metrics]
            ),
            "avg_win_rate": np.mean([m.get("win_rate", 0) for m in all_metrics]),
            "total_return_sum": sum(m.get("total_return", 0) for m in all_metrics),
            "avg_max_drawdown": np.mean(
                [m.get("max_drawdown", 0) for m in all_metrics]
            ),
        }
        print("\n=== WALK-FORWARD AGGREGATE ===")
        for k, v in agg.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    else:
        outputs, ml_timing = execute_model_pipeline(config)

        ml_timing["reporting"] = 0.0
        ml_timing["total"] = 0.0
        timing = TimingResults(**ml_timing)

        config_payload = build_run_config_payload(config, timing)

        _, reporting_secs = measure_step_duration(
            "reporting",
            publish_pipeline_results,
            accelerator,
            config_payload.as_dict(),
            outputs,
        )

        ml_timing["reporting"] = reporting_secs
        ml_timing["total"] = time.perf_counter() - t_total
        final_timing = TimingResults(**ml_timing)

        print_timing_summary(final_timing)
