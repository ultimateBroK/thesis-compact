"""
Reporting pipeline: publish results to console and persist run artifacts.

Orchestration: publish_pipeline_results -> persist_run_artifacts -> build_run_metadata.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.config import LABELS
from src.models import HybridStackingSignalClassifier

if TYPE_CHECKING:
    from src.cli.pipeline import PipelineOutputs

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
    outputs: PipelineOutputs | dict[str, Any],
    window_id: int | None = None,
    window_train_range: str = "",
    window_test_range: str = "",
) -> None:
    if hasattr(outputs, "to_dict"):
        output_payload = outputs.to_dict(
            window_id=window_id,
            window_train_range=window_train_range,
            window_test_range=window_test_range,
        )
        artifact_outputs = outputs
    else:
        output_payload = dict(outputs)
        if window_id is not None:
            output_payload["window_id"] = window_id
            output_payload["window_train_range"] = window_train_range
            output_payload["window_test_range"] = window_test_range
        artifact_outputs = SimpleNamespace(
            train=output_payload["train"],
            test=output_payload["test"],
            features=output_payload["features"],
            model=output_payload["model"],
            predictions=output_payload["predictions"],
            positions=output_payload["positions"],
            backtest_metrics=output_payload.get("backtest_metrics"),
            equity=output_payload.get("equity", np.full(len(output_payload["test"]), 10_000.0)),
            executed_trades=output_payload.get("executed_trades"),
        )
    train = output_payload["train"]
    test = output_payload["test"]
    features = output_payload["features"]
    model = output_payload["model"]
    predictions = output_payload["predictions"]
    backtest_metrics = output_payload["backtest_metrics"]

    labeled_full = pl.concat([train, test])

    print_device_acceleration_report(accelerator)
    print_dataset_report(labeled_full, train, test, len(features))
    print_model_filtering_report(model)
    print_classification_report(test["label"], predictions)
    print_feature_importance_report(extract_lightgbm_feature_importance(model, features))
    print_backtest_metrics_report(backtest_metrics)

    persist_run_artifacts(
        run_dir=config_payload.get("run_dir", Path("reports") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        outputs=artifact_outputs,
        config_payload=config_payload,
        window_id=window_id,
        window_train_range=window_train_range,
        window_test_range=window_test_range,
    )


def persist_run_artifacts(
    run_dir: Path,
    outputs: PipelineOutputs,
    config_payload: dict[str, Any],
    window_id: int | None = None,
    window_train_range: str = "",
    window_test_range: str = "",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    train = outputs.train
    test = outputs.test
    model = outputs.model
    features = outputs.features
    predictions = outputs.predictions
    positions = outputs.positions
    backtest_metrics = outputs.backtest_metrics
    executed_trades = outputs.executed_trades
    equity_arr = outputs.equity
    dataset = pl.concat([train, test])

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

    artifact_files = collect_artifact_files(run_dir, figures_dir) + ["run_data.json"]
    run_data = build_run_metadata(
        run_dir, model, config_payload, dataset, train, test,
        predictions, positions, results, features, backtest_metrics,
        artifact_files, trades_df, executed_trades=executed_trades,
        window_id=window_id,
        window_train_range=window_train_range,
        window_test_range=window_test_range,
    )
    with open(run_dir / "run_data.json", "w", encoding="utf-8") as f:
        json.dump(asdict(run_data), f, indent=2, ensure_ascii=False, default=str)

    print(f"\nRun dir: {run_dir.resolve()}")
    print(f"Files: predictions.csv, trades.csv ({len(trades_df)} trades), backtest_metrics.csv, feature_importance.csv, run_data.json, figures/*.png")


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def collect_artifact_files(run_dir: Path, figures_dir: Path) -> list[str]:
    root_files = [f.name for f in run_dir.iterdir() if f.is_file()]
    fig_files = [f"figures/{f.name}" for f in figures_dir.iterdir() if f.is_file()]
    return sorted(root_files + fig_files)


def build_run_metadata(
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
    trades_df: pd.DataFrame,
    executed_trades: list[dict] | None = None,
    window_id: int | None = None,
    window_train_range: str | None = None,
    window_test_range: str | None = None,
) -> RunMetadata:
    import platform
    import subprocess
    import sys

    def get_git_value(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=Path.cwd(), text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    feature_imp = extract_lightgbm_feature_importance(model, features)

    return RunMetadata(
        run_id=run_dir.name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config=config_payload,
        dataset=build_dataset_metadata(dataset, train, test, features),
        training=build_training_metadata(model),
        evaluation=build_evaluation_metadata(test, predictions),
        backtest={
            **{k: round(float(v), 6) for k, v in (backtest_metrics or {}).items()},
            **asdict(build_win_rate_metadata(results, executed_trades)),
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
            "python_build": platform.python_build(),
            "platform": platform.platform(),
            "git_commit": get_git_value(["git", "rev-parse", "HEAD"]),
            "git_branch": get_git_value(["git", "branch", "--show-current"]),
            "git_dirty": bool(get_git_value(["git", "status", "--short"])),
            "run_entrypoint": "cli",
        },
    )


def build_date_range(frame: pl.DataFrame) -> dict[str, str]:
    ts = frame["timestamp"]
    return {"start": str(ts.min()), "end": str(ts.max())}


def build_label_counts(frame: pl.DataFrame) -> dict[str, int]:
    vc = frame["label"].value_counts()
    return {str(row["label"]): int(row["count"]) for row in vc.iter_rows(named=True)}


def build_dataset_metadata(
    dataset: pl.DataFrame,
    train: pl.DataFrame,
    test: pl.DataFrame,
    features: list[str],
) -> DatasetMeta:
    return DatasetMeta(
        total_rows=len(dataset),
        train_rows=len(train),
        test_rows=len(test),
        feature_count=len(features),
        features=features,
        fractional_d=None,
        data_range=build_date_range(dataset),
        train_date_range=build_date_range(train),
        test_date_range=build_date_range(test),
        label_distribution_total=build_label_counts(dataset),
        label_distribution_train=build_label_counts(train),
        label_distribution_test=build_label_counts(test),
    )


def build_training_metadata(model: HybridStackingSignalClassifier) -> TrainingMeta:
    return TrainingMeta(
        oof_scores=model.oof_scores_,
        per_class_oof_f1=getattr(model, "per_class_oof_", {}),
        active_models=getattr(model, "active_model_names_", []),
        filtered_models=[
            n for n in model.base_models
            if n not in getattr(model, "active_model_names_", [])
        ],
    )


def build_evaluation_metadata(test: pl.DataFrame, predictions: np.ndarray) -> EvalMeta:
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


def build_win_rate_metadata(results: pd.DataFrame, executed_trades: list[dict] | None = None) -> WinRateMeta:
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