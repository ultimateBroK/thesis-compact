"""
Reporting pipeline: publish results to console and persist run artifacts.

Orchestration: publish_pipeline_results -> persist_run_artifacts -> _build_run_data.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.backtest import simulate_equity_barrier
from src.config import LABELS
from src.models import HybridStackingSignalClassifier

from .console import (
    print_backtest_metrics_report,
    print_classification_report,
    print_dataset_report,
    print_device_acceleration_report,
    print_feature_importance_report,
    print_model_filtering_report,
)
from .importance import (
    extract_lightgbm_feature_importance,
    save_equity_curve_plot,
    save_feature_importance_bar_plot,
    save_feature_importance_csv,
    save_oof_scores_bar_plot,
)
from .trades import convert_executed_trades_to_dataframe, extract_trades_from_results


# ---------------------------------------------------------------------------
# Metadata dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DatasetMeta:
    total_rows: int
    train_rows: int
    test_rows: int
    feature_count: int
    features: list[str]
    fractional_d: float | None
    data_range: dict[str, str]
    train_date_range: dict[str, str]
    test_date_range: dict[str, str]
    label_distribution_total: dict[str, int]
    label_distribution_train: dict[str, int]
    label_distribution_test: dict[str, int]


@dataclass
class TrainingMeta:
    oof_scores: dict[str, float]
    per_class_oof_f1: dict[str, dict[str, float]]
    active_models: list[str]
    filtered_models: list[str]


@dataclass
class EvalMeta:
    accuracy: float
    f1_macro: float
    confusion_matrix: dict[str, Any]


@dataclass
class WinRateMeta:
    win_rate: float
    turnover: float


@dataclass
class RunMetadata:
    run_id: str
    timestamp: str
    config: dict[str, Any]
    dataset: DatasetMeta
    training: TrainingMeta
    evaluation: EvalMeta
    backtest: dict[str, Any]
    feature_importance: dict[str, float]
    trade_summary: dict[str, Any]
    artifacts: dict[str, Any]
    reproducibility: dict[str, Any]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def publish_pipeline_results(
    accelerator: Any,
    config_payload: dict[str, Any],
    outputs: dict[str, Any],
) -> None:
    train = outputs["train"]
    test = outputs["test"]
    features = outputs["features"]
    model = outputs["model"]
    predictions = outputs["predictions"]
    positions = outputs["positions"]
    backtest_metrics = outputs["backtest_metrics"]

    labeled_full = pl.concat([train, test])

    print_device_acceleration_report(accelerator)
    print_dataset_report(labeled_full, train, test, len(features))
    print_model_filtering_report(model)
    print_classification_report(test["label"], predictions)
    print_feature_importance_report(extract_lightgbm_feature_importance(model, features))
    print_backtest_metrics_report(backtest_metrics)

    persist_run_artifacts(
        run_dir=config_payload.get("run_dir", Path("reports") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        model=model,
        test=test,
        predictions=predictions,
        positions=positions,
        config_payload=config_payload,
        dataset=labeled_full,
        train=train,
        test_df=test,
        features=features,
        backtest_metrics=backtest_metrics,
        executed_trades=outputs.get("executed_trades"),
        window_id=outputs.get("window_id"),
        window_train_range=outputs.get("window_train_range", ""),
        window_test_range=outputs.get("window_test_range", ""),
    )


def persist_run_artifacts(
    run_dir: Path,
    model: HybridStackingSignalClassifier,
    test: pl.DataFrame,
    predictions: np.ndarray,
    positions: np.ndarray,
    config_payload: dict[str, Any],
    dataset: pl.DataFrame,
    train: pl.DataFrame,
    test_df: pl.DataFrame,
    features: list[str],
    backtest_metrics: dict[str, float] | None = None,
    executed_trades: list[dict] | None = None,
    equity: np.ndarray | None = None,
    window_id: int | None = None,
    window_train_range: str = "",
    window_test_range: str = "",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    if equity is not None:
        equity_arr = equity
    else:
        close = test["close"].to_numpy()
        high = test["high"].to_numpy()
        low = test["low"].to_numpy()
        spread = test["spread"].to_numpy()
        atr_rel = test["atr_14"].to_numpy()
        equity_arr, _, _ = simulate_equity_barrier(close, high, low, positions, spread, atr_rel=atr_rel)

    results = test.select(["timestamp", "close", "spread", "label"]).to_pandas()
    results["prediction"] = predictions
    results["position"] = positions
    results["pnl_usd"] = np.diff(equity_arr, prepend=equity_arr[0])
    results["equity"] = equity_arr
    results.to_csv(run_dir / "predictions.csv", index=False)

    if executed_trades is not None:
        timestamps = test["timestamp"].to_numpy()
        trades_df = convert_executed_trades_to_dataframe(executed_trades, timestamps)
    else:
        trades_df = extract_trades_from_results(results)
    trades_df.to_csv(run_dir / "trades.csv", index=False)

    if backtest_metrics:
        pd.Series(backtest_metrics).to_csv(run_dir / "backtest_metrics.csv")

    importance_df = save_feature_importance_csv(model, features, run_dir / "feature_importance.csv")
    save_feature_importance_bar_plot(importance_df, figures_dir / "feature_importance.png")
    save_oof_scores_bar_plot(model, figures_dir / "oof_scores.png")
    save_equity_curve_plot(equity_arr, figures_dir / "equity_curve.png")

    artifact_files = _collect_artifact_files(run_dir, figures_dir) + ["run_data.json"]
    run_data = _build_run_data(
        run_dir, model, config_payload, dataset, train, test_df,
        predictions, positions, results, features, backtest_metrics,
        artifact_files, executed_trades=executed_trades,
        window_id=window_id,
        window_train_range=window_train_range,
        window_test_range=window_test_range,
    )
    with open(run_dir / "run_data.json", "w", encoding="utf-8") as f:
        json.dump(asdict(run_data), f, indent=2, ensure_ascii=False, default=str)

    print(f"\nRun dir: {run_dir.resolve()}")
    print(f"Files: predictions.csv, trades.csv ({len(trades_df)} trades), backtest_metrics.csv, feature_importance.csv, run_data.json, figures/*.png")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_artifact_files(run_dir: Path, figures_dir: Path) -> list[str]:
    root_files = [f.name for f in run_dir.iterdir() if f.is_file()]
    fig_files = [f"figures/{f.name}" for f in figures_dir.iterdir() if f.is_file()]
    return sorted(root_files + fig_files)


def _build_run_data(
    run_dir: Path,
    model: HybridStackingSignalClassifier,
    config_payload: dict[str, Any],
    dataset: pl.DataFrame,
    train: pl.DataFrame,
    test: pl.DataFrame,
    predictions: np.ndarray,
    positions: np.ndarray,
    results: pd.DataFrame,
    features: list[str],
    backtest_metrics: dict[str, float] | None,
    artifact_files: list[str],
    executed_trades: list[dict] | None = None,
    window_id: int | None = None,
    window_train_range: str | None = None,
    window_test_range: str | None = None,
) -> RunMetadata:
    import platform as _platform
    import subprocess
    import sys

    def _git_value(args):
        try:
            return subprocess.check_output(args, cwd=Path.cwd(), text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    feature_imp = extract_lightgbm_feature_importance(model, features)
    if executed_trades is not None:
        timestamps = test["timestamp"].to_numpy()
        trades_df = convert_executed_trades_to_dataframe(executed_trades, timestamps)
    else:
        trades_df = extract_trades_from_results(results)

    return RunMetadata(
        run_id=run_dir.name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config=config_payload,
        dataset=_dataset_meta(dataset, train, test, features),
        training=_training_meta(model),
        evaluation=_eval_meta(test, predictions),
        backtest={
            **{k: round(float(v), 6) for k, v in (backtest_metrics or {}).items()},
            **asdict(_win_rate_meta(results, executed_trades)),
            "window_id": window_id,
            "window_train_range": window_train_range or "",
            "window_test_range": window_test_range or "",
        },
        feature_importance={
            row["feature"]: round(float(row["pct"]), 2)
            for _, row in feature_imp.iterrows()
        },
        trade_summary={
            "total_trades": len(trades_df),
            "wins": int(trades_df["win"].sum()) if len(trades_df) else 0,
            "losses": int((~trades_df["win"]).sum()) if len(trades_df) else 0,
            "avg_bars_held": round(float(trades_df["bars_held"].mean()), 1) if len(trades_df) else 0,
            "avg_pnl_usd": round(float(trades_df["pnl_usd"].mean()), 2) if len(trades_df) else 0,
        },
        artifacts={"files": artifact_files, "figure_count": sum(".png" in n for n in artifact_files)},
        reproducibility={
            "python_version": sys.version.split()[0],
            "python_version_full": sys.version,
            "python_build": _platform.python_build(),
            "platform": _platform.platform(),
            "git_commit": _git_value(["git", "rev-parse", "HEAD"]),
            "git_branch": _git_value(["git", "branch", "--show-current"]),
            "git_dirty": bool(_git_value(["git", "status", "--short"])),
            "run_entrypoint": "cli",
        },
    )


def _dataset_meta(
    dataset: pl.DataFrame,
    train: pl.DataFrame,
    test: pl.DataFrame,
    features: list[str],
) -> DatasetMeta:
    def _date_range(frame: pl.DataFrame) -> dict[str, str]:
        ts = frame["timestamp"]
        return {"start": str(ts.min()), "end": str(ts.max())}

    def _label_counts(frame: pl.DataFrame) -> dict[str, int]:
        vc = frame["label"].value_counts()
        return {str(row["label"]): int(row["count"]) for row in vc.iter_rows(named=True)}

    return DatasetMeta(
        total_rows=len(dataset),
        train_rows=len(train),
        test_rows=len(test),
        feature_count=len(features),
        features=features,
        fractional_d=None,
        data_range=_date_range(dataset),
        train_date_range=_date_range(train),
        test_date_range=_date_range(test),
        label_distribution_total=_label_counts(dataset),
        label_distribution_train=_label_counts(train),
        label_distribution_test=_label_counts(test),
    )


def _training_meta(model: HybridStackingSignalClassifier) -> TrainingMeta:
    return TrainingMeta(
        oof_scores=model.oof_scores_,
        per_class_oof_f1=getattr(model, "per_class_oof_", {}),
        active_models=getattr(model, "active_model_names_", []),
        filtered_models=[
            n for n in model.base_models
            if n not in getattr(model, "active_model_names_", [])
        ],
    )


def _eval_meta(test: pl.DataFrame, predictions: np.ndarray) -> EvalMeta:
    y_true = test["label"].to_numpy()
    labels = LABELS.tolist()
    return EvalMeta(
        accuracy=round(float(accuracy_score(y_true, predictions)), 6),
        f1_macro=round(float(f1_score(y_true, predictions, average="macro", zero_division=0)), 6),
        confusion_matrix={
            "labels": labels,
            "matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
        },
    )


def _win_rate_meta(results: pd.DataFrame, executed_trades: list[dict] | None = None) -> WinRateMeta:
    if executed_trades:
        wins = sum(1 for t in executed_trades if t.get("win", t.get("pnl_usd", 0) > 0))
        total = len(executed_trades)
        win_rate = round(wins / total, 6) if total else 0.0
    else:
        pnl = results["pnl_usd"]
        nonzero_pnl = pnl[pnl != 0]
        wins = float((nonzero_pnl[nonzero_pnl > 0]).sum())
        win_rate = round(wins / len(nonzero_pnl), 6) if len(nonzero_pnl) else 0.0
    trades_cnt = float(np.sum(np.diff(results["position"], prepend=0) != 0))
    return WinRateMeta(
        win_rate=win_rate,
        turnover=round(trades_cnt / len(results), 6) if len(results) else 0.0,
    )
