from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from hybrid_stacking.models import HybridStackingSignalClassifier


def print_dataset_report(frame: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, feature_count: int) -> None:
    print("=== DATASET ===")
    print(f"Rows: {len(frame)} | Train: {len(train)} | Test: {len(test)}")
    print(f"Fractional d*: {frame.attrs.get('fractional_d', 'n/a')}")
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
