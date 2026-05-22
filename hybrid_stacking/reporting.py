from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, f1_score

from hybrid_stacking.models import HybridStackingSignalClassifier
from hybrid_stacking.backtest import simulate_equity
from hybrid_stacking.config import INITIAL_BALANCE


def model_status(name: str, model: HybridStackingSignalClassifier) -> str:
    return "ACTIVE" if name in model.active_model_names_ else "FILTERED"


def print_dataset_report(frame: pl.DataFrame, train: pl.DataFrame, test: pl.DataFrame, feature_count: int) -> None:
    print("=== DATASET ===")
    print(f"Rows: {len(frame)} | Train: {len(train)} | Test: {len(test)}")
    print(f"Features: {feature_count}")
    label_vc = frame["label"].value_counts().sort("label")
    print("Label distribution:")
    for row in label_vc.iter_rows(named=True):
        print(f"  {row['label']}: {row['count']}")


def print_model_report(model: HybridStackingSignalClassifier) -> None:
    print("\n=== SMART FILTERING OOF F1 ===")
    for name, score in sorted(model.oof_scores_.items(), key=lambda item: item[1], reverse=True):
        print(f"{name}: {score:.4f} [{model_status(name, model)}]")


def print_classification_report(y_true: pl.Series, y_pred) -> None:
    print("\n=== TEST CLASSIFICATION ===")
    y_np = y_true.to_numpy() if isinstance(y_true, pl.Series) else y_true
    print(f"Accuracy: {accuracy_score(y_np, y_pred):.4f}")
    print(f"F1 macro: {f1_score(y_np, y_pred, average='macro', zero_division=0):.4f}")


def print_backtest_report(metrics: dict[str, float]) -> None:
    print("\n=== COST-AWARE BACKTEST ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def print_acceleration_report(accelerator: Any) -> None:
    print("=== ACCELERATION ===")
    print(f"Device: {accelerator.device} | Processes: {accelerator.num_processes}")


def save_oof_scores_plot(model: HybridStackingSignalClassifier, path: Path) -> None:
    scores = pd.Series(model.oof_scores_).sort_values()
    colors = ["#2ca02c" if name in model.active_model_names_ else "#d62728" for name in scores.index]
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


def _build_run_data(
    run_dir: Path,
    model: HybridStackingSignalClassifier,
    config_payload: dict,
    dataset: pl.DataFrame,
    train: pl.DataFrame,
    test: pl.DataFrame,
    predictions: np.ndarray,
    features: list[str],
    backtest_metrics: dict[str, float] | None,
) -> dict:
    frac_d = dataset.get_attribute("fractional_d") if hasattr(dataset, "get_attribute") else None
    first_ts, last_ts = dataset["timestamp"][0], dataset["timestamp"][-1]
    label_vc = dataset["label"].value_counts().sort("label")
    label_dist = {row["label"]: row["count"] for row in label_vc.iter_rows(named=True)}

    return {
        "run_id": run_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config_payload,
        "dataset": {
            "total_rows": len(dataset),
            "train_rows": len(train),
            "test_rows": len(test),
            "feature_count": len(features),
            "features": features,
            "fractional_d": frac_d,
            "data_range": {"start": str(first_ts), "end": str(last_ts)},
            "label_distribution": label_dist,
        },
        "training": {
            "oof_scores": {k: round(v, 6) for k, v in model.oof_scores_.items()},
            "active_models": model.active_model_names_,
            "filtered_models": [n for n in model.oof_scores_ if n not in model.active_model_names_],
        },
        "evaluation": {
            "accuracy": round(float(accuracy_score(test["label"].to_numpy(), predictions)), 6),
            "f1_macro": round(float(f1_score(test["label"].to_numpy(), predictions, average="macro", zero_division=0)), 6),
        },
        "backtest": {k: round(float(v), 6) for k, v in (backtest_metrics or {}).items()},
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
    equity_arr = simulate_equity(close, positions)

    results = test.select(["close", "spread", "label"]).to_pandas()
    results["prediction"] = predictions
    results["position"] = positions
    results["equity"] = equity_arr
    results.to_csv(run_dir / "predictions.csv")

    metrics_to_save = backtest_metrics or {"initial_balance": float(INITIAL_BALANCE)}
    pd.Series(metrics_to_save).to_csv(run_dir / "backtest_metrics.csv")

    save_oof_scores_plot(model, run_dir / "model_oof_f1.png")
    save_equity_curve_plot(equity_arr, run_dir / "equity_curve.png")

    run_data = _build_run_data(run_dir, model, config_payload, dataset, train, test_df, predictions, features, backtest_metrics)
    with open(run_dir / "run_data.json", "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nRun dir: {run_dir.resolve()}")
    print("Files: predictions.csv, backtest_metrics.csv, run_data.json, *.png")
