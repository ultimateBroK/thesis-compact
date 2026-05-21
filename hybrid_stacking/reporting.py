from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, classification_report, f1_score

from hybrid_stacking.models import HybridStackingSignalClassifier


def print_dataset_report(frame: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, feature_count: int) -> None:
    print("=== DATASET ===")
    print(f"Rows: {len(frame)} | Train: {len(train)} | Test: {len(test)}")
    print(f"Fractional d: {frame.attrs.get('fractional_d', 'n/a')}")
    print(f"Features: {feature_count}")
    print("Label distribution:")
    print(frame["label"].value_counts(normalize=True).sort_index().round(3))


def print_model_report(model: HybridStackingSignalClassifier) -> None:
    print("\n=== SMART FILTERING OOF F1 ===")
    for name, score in sorted(model.oof_scores_.items(), key=lambda item: item[1], reverse=True):
        print(f"{name}: {score:.4f} [{model_status(name, model)}]")


def model_status(name: str, model: HybridStackingSignalClassifier) -> str:
    return "ACTIVE" if name in model.active_model_names_ else "FILTERED"


def print_classification_report(y_true: pd.Series, y_pred) -> None:
    print("\n=== TEST CLASSIFICATION ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 macro: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))


def print_backtest_report(metrics: dict[str, float]) -> None:
    print("\n=== COST-AWARE BACKTEST ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def print_acceleration_report(accelerator: Any) -> None:
    print("=== ACCELERATION ===")
    print(f"Device: {accelerator.device} | Processes: {accelerator.num_processes}")


def save_run_artifacts(
    run_dir: Path,
    model: HybridStackingSignalClassifier,
    test: pd.DataFrame,
    predictions: np.ndarray,
    strategy_returns: np.ndarray,
    equity: pd.Series,
    backtest_metrics: dict[str, float],
    config_payload: dict,
    dataset: pd.DataFrame,
    train: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    results = test[["close", "spread", "label"]].copy()
    results["prediction"] = predictions
    results["strategy_return"] = strategy_returns
    results["equity"] = equity
    results.to_csv(run_dir / "predictions.csv")
    pd.Series(backtest_metrics).to_csv(run_dir / "backtest_metrics.csv")

    save_oof_scores_plot(model, run_dir / "model_oof_f1.png")
    save_equity_curve_plot(equity, run_dir / "equity_curve.png")

    run_data = {
        "run_id": run_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config_payload,
        "dataset": {
            "total_rows": len(dataset),
            "train_rows": len(train),
            "test_rows": len(test_df),
            "feature_count": len(features),
            "features": features,
            "fractional_d": dataset.attrs.get("fractional_d"),
            "data_range": {
                "start": str(dataset.index[0]),
                "end": str(dataset.index[-1]),
            },
            "train_range": {
                "start": str(train.index[0]),
                "end": str(train.index[-1]),
            },
            "test_range": {
                "start": str(test_df.index[0]),
                "end": str(test_df.index[-1]),
            },
            "label_distribution": dataset["label"].value_counts().sort_index().to_dict(),
        },
        "training": {
            "oof_scores": {k: round(v, 6) for k, v in model.oof_scores_.items()},
            "active_models": model.active_model_names_,
            "filtered_models": [n for n in model.oof_scores_ if n not in model.active_model_names_],
        },
        "evaluation": {
            "accuracy": round(float(accuracy_score(test["label"], predictions)), 6),
            "f1_macro": round(float(f1_score(test["label"], predictions, average="macro", zero_division=0)), 6),
            "classification_report": classification_report(test["label"], predictions, zero_division=0, output_dict=True),
        },
        "backtest": {k: round(float(v), 6) for k, v in backtest_metrics.items()},
    }

    with open(run_dir / "run_data.json", "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nRun dir: {run_dir.resolve()}")
    print(f"Files: predictions.csv, backtest_metrics.csv, run_data.json, *.png")


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


def save_equity_curve_plot(equity: pd.Series, path: Path) -> None:
    figure = Figure(figsize=(9, 4))
    ax = figure.subplots()
    ax.plot(equity.index, equity.to_numpy(), color="#1f77b4")
    ax.set_title("Cost-aware equity curve")
    ax.set_ylabel("Equity")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
