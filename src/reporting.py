from __future__ import annotations

import json
import platform as _platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.backtest import simulate_equity
from src.config import LABELS
from src.models import HybridStackingSignalClassifier


def model_status(name: str, model: HybridStackingSignalClassifier) -> str:
    return "ACTIVE" if name in model.active_model_names_ else "FILTERED"


def print_dataset_report(
    frame: pl.DataFrame,
    train: pl.DataFrame,
    test: pl.DataFrame,
    feature_count: int,
) -> None:
    print("=== DATASET ===")
    print(f"Rows: {len(frame)} | Train: {len(train)} | Test: {len(test)}")
    print(f"Features: {feature_count}")
    label_vc = frame["label"].value_counts().sort("label")
    print("Label distribution:")
    for row in label_vc.iter_rows(named=True):
        print(f"  {row['label']}: {row['count']}")


def print_model_report(model: HybridStackingSignalClassifier) -> None:
    print("\n=== SMART FILTERING OOF F1 ===")
    for name, score in sorted(
        model.oof_scores_.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{name}: {score:.4f} [{model_status(name, model)}]")


def print_classification_report(y_true: pl.Series, y_pred) -> None:
    print("\n=== TEST CLASSIFICATION ===")
    y_np = y_true.to_numpy() if isinstance(y_true, pl.Series) else y_true
    print(f"Accuracy: {accuracy_score(y_np, y_pred):.4f}")
    print(f"F1 macro: {f1_score(y_np, y_pred, average='macro', zero_division=0):.4f}")


def print_backtest_report(metrics: dict[str, float]) -> None:
    # Note: Very negative Sharpe with small MDD is valid for strategies
    # with consistent small losses (high-frequency small PnL).
    print("\n=== COST-AWARE BACKTEST ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def print_acceleration_report(accelerator: Any) -> None:
    print("=== ACCELERATION ===")
    print(f"Device: {accelerator.device} | Processes: {accelerator.num_processes}")


def save_oof_scores_plot(model: HybridStackingSignalClassifier, path: Path) -> None:
    scores = pd.Series(model.oof_scores_).sort_values()
    colors = [
        "#2ca02c" if name in model.active_model_names_ else "#d62728"
        for name in scores.index
    ]
    figure = Figure(figsize=(8, 4))
    ax = figure.subplots()
    ax.barh(scores.index, scores.to_numpy(), color=colors)
    ax.set_title("OOF macro F1 by base model")
    ax.set_xlabel("Macro F1")
    figure.tight_layout()
    figure.savefig(path, dpi=160)


def save_equity_curve_plot(equity: np.ndarray, path: Path) -> None:
    figure = Figure(figsize=(9, 4))
    ax = figure.subplots()
    ax.plot(equity, color="#1f77b4")
    ax.set_title("Cost-aware equity curve")
    ax.set_ylabel("Equity (USD)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            args,
            cwd=Path.cwd(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def reproducibility_data() -> dict[str, Any]:
    status = git_value(["git", "status", "--short"])
    return {
        "python_version": sys.version.split()[0],
        "python_version_full": sys.version,
        "python_build": _platform.python_build(),
        "platform": _platform.platform(),
        "git_commit": git_value(["git", "rev-parse", "HEAD"]),
        "git_branch": git_value(["git", "branch", "--show-current"]),
        "git_dirty": bool(status),
        "run_entrypoint": "cli",
    }


def backtest_eval(results: pd.DataFrame) -> dict[str, float]:
    pnl = results["pnl_usd"]
    positions = results["position"]
    nonzero_pnl = pnl[pnl != 0]
    wins = nonzero_pnl[nonzero_pnl > 0]
    trades = float(np.sum(np.diff(positions, prepend=0) != 0))
    return {
        "win_rate": round(len(wins) / len(nonzero_pnl), 6) if len(nonzero_pnl) else 0.0,
        "turnover": round(trades / len(results), 6) if len(results) else 0.0,
    }


def prediction_results(
    test: pl.DataFrame,
    predictions: np.ndarray,
    positions: np.ndarray,
    equity: np.ndarray,
) -> pd.DataFrame:
    results = test.select(["timestamp", "close", "spread", "label"]).to_pandas()
    results["prediction"] = predictions
    results["position"] = positions
    results["pnl_usd"] = np.diff(equity, prepend=equity[0])
    results["equity"] = equity
    return results


def series_counts(values: Any) -> dict[str, int]:
    series = pd.Series(values.to_numpy() if isinstance(values, pl.Series) else values)
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def date_range(frame: pl.DataFrame) -> dict[str, str]:
    if not len(frame):
        return {"start": "", "end": ""}
    return {"start": str(frame["timestamp"][0]), "end": str(frame["timestamp"][-1])}


def dataset_run_data(
    dataset: pl.DataFrame,
    train: pl.DataFrame,
    test: pl.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    frac_d = (
        dataset.get_attribute("fractional_d")
        if hasattr(dataset, "get_attribute")
        else None
    )
    return {
        "total_rows": len(dataset),
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": len(features),
        "features": features,
        "fractional_d": frac_d,
        "data_range": date_range(dataset),
        "train_date_range": date_range(train),
        "test_date_range": date_range(test),
        "label_distribution_total": series_counts(dataset["label"]),
        "label_distribution_train": series_counts(train["label"]),
        "label_distribution_test": series_counts(test["label"]),
        "split_gap_info": {
            "train_end": str(train["timestamp"][-1]) if len(train) else "",
            "test_start": str(test["timestamp"][0]) if len(test) else "",
            "purge_rows": int(len(dataset) - len(train) - len(test)),
        },
    }


def training_run_data(model: HybridStackingSignalClassifier) -> dict[str, Any]:
    return {
        "oof_scores": {k: round(v, 6) for k, v in model.oof_scores_.items()},
        "per_class_oof_f1": {
            k: {str(c): round(v, 4) for c, v in cls.items()}
            for k, cls in model.per_class_oof_.items()
        },
        "active_models": model.active_model_names_,
        "filtered_models": [
            n for n in model.oof_scores_ if n not in model.active_model_names_
        ],
    }


def evaluation_run_data(
    test: pl.DataFrame,
    predictions: np.ndarray,
    positions: np.ndarray,
) -> dict[str, Any]:
    y_true = test["label"].to_numpy()
    labels = LABELS.tolist()
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 6),
        "f1_macro": round(
            float(f1_score(y_true, predictions, average="macro", zero_division=0)),
            6,
        ),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
        },
    }


def artifact_run_data(artifact_files: list[str]) -> dict[str, Any]:
    return {
        "files": artifact_files,
        "figure_count": sum(name.endswith(".png") for name in artifact_files),
    }


def build_run_data(
    run_dir: Path,
    model: HybridStackingSignalClassifier,
    config_payload: dict,
    dataset: pl.DataFrame,
    train: pl.DataFrame,
    test: pl.DataFrame,
    predictions: np.ndarray,
    positions: np.ndarray,
    results: pd.DataFrame,
    features: list[str],
    backtest_metrics: dict[str, float] | None,
    artifact_files: list[str],
) -> dict:
    return {
        "run_id": run_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config_payload,
        "dataset": dataset_run_data(dataset, train, test, features),
        "training": training_run_data(model),
        "evaluation": evaluation_run_data(test, predictions, positions),
        "backtest": {
            **{k: round(float(v), 6) for k, v in (backtest_metrics or {}).items()},
            **backtest_eval(results),
        },
        "artifacts": artifact_run_data(artifact_files),
        "reproducibility": reproducibility_data(),
    }


def save_run_artifacts(
    run_dir: Path,
    model: HybridStackingSignalClassifier,
    test: pl.DataFrame,
    predictions: np.ndarray,
    positions: np.ndarray,
    config_payload: dict,
    dataset: pl.DataFrame,
    train: pl.DataFrame,
    test_df: pl.DataFrame,
    features: list[str],
    backtest_metrics: dict[str, float] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    close = test["close"].to_numpy()
    spread = test["spread"].to_numpy() if "spread" in test.columns else None
    equity_arr = simulate_equity(close, positions, spread)
    results = prediction_results(test, predictions, positions, equity_arr)
    results.to_csv(run_dir / "predictions.csv", index=False)

    if backtest_metrics:
        pd.Series(backtest_metrics).to_csv(run_dir / "backtest_metrics.csv")

    save_oof_scores_plot(model, run_dir / "model_oof_f1.png")
    save_equity_curve_plot(equity_arr, run_dir / "equity_curve.png")
    artifact_files = sorted(path.name for path in run_dir.iterdir())

    run_data = build_run_data(
        run_dir,
        model,
        config_payload,
        dataset,
        train,
        test_df,
        predictions,
        positions,
        results,
        features,
        backtest_metrics,
        artifact_files + ["run_data.json"],
    )
    with open(run_dir / "run_data.json", "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nRun dir: {run_dir.resolve()}")
    print("Files: predictions.csv, backtest_metrics.csv, run_data.json, *.png")
