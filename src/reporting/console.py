from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import accuracy_score, f1_score

from src.models import HybridStackingSignalClassifier


def determine_model_status(name: str, model: HybridStackingSignalClassifier) -> str:
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


def print_model_filtering_report(model: HybridStackingSignalClassifier) -> None:
    print("\n=== SMART FILTERING OOF F1 ===")
    for name, score in sorted(
        model.oof_scores_.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{name}: {score:.4f} [{determine_model_status(name, model)}]")


def print_classification_report(y_true: pl.Series, y_pred: np.ndarray | pl.Series) -> None:
    print("\n=== TEST CLASSIFICATION ===")
    y_np = y_true.to_numpy() if isinstance(y_true, pl.Series) else y_true
    print(f"Accuracy: {accuracy_score(y_np, y_pred):.4f}")
    print(f"F1 macro: {f1_score(y_np, y_pred, average='macro', zero_division=0):.4f}")


def print_backtest_metrics_report(metrics: dict[str, float]) -> None:
    print("\n=== COST-AWARE BACKTEST ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def print_device_acceleration_report(accelerator: Any) -> None:
    print("=== ACCELERATION ===")
    print(f"Device: {accelerator.device} | Processes: {accelerator.num_processes}")


def print_feature_importance_report(importance_df: pd.DataFrame) -> None:
    print("\n=== FEATURE IMPORTANCE (LightGBM) ===")
    for idx, row in importance_df.head(10).iterrows():
        bar = "#" * int(row["pct"] * 2)
        print(f"  {idx:>2d}. {row['feature']:<25s} {row['importance']:>6d}  {row['pct']:>5.1f}%  {bar}")
