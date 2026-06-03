"""Metrics: classification scores, baseline comparison, ROC AUC."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import LABELS
from src.models import HybridStackingSignalClassifier, derive_aligned_probabilities


def compute_roc_auc(y_true: np.ndarray, pred_proba: np.ndarray | None) -> float:
    if pred_proba is None or pred_proba.shape[1] < 2:
        return float("nan")
    y_bin = (y_true == LABELS[1]).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return float("nan")
    return float(roc_auc_score(y_bin, pred_proba[:, 1]))


def build_classification_metric_row(
    model_name: str,
    y_true: np.ndarray,
    predictions: np.ndarray,
    pred_proba: np.ndarray | None,
) -> dict[str, float | str]:
    sell_label = int(LABELS[0])
    buy_label = int(LABELS[1])
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "precision_sell": float(precision_score(y_true, predictions, pos_label=sell_label, zero_division=0)),
        "recall_sell": float(recall_score(y_true, predictions, pos_label=sell_label, zero_division=0)),
        "precision_buy": float(precision_score(y_true, predictions, pos_label=buy_label, zero_division=0)),
        "recall_buy": float(recall_score(y_true, predictions, pos_label=buy_label, zero_division=0)),
        "roc_auc": compute_roc_auc(y_true, pred_proba),
    }


def build_baseline_metrics_dataframe(
    model: HybridStackingSignalClassifier,
    test: pl.DataFrame,
    features: list[str],
    hybrid_predictions: np.ndarray,
    hybrid_proba: np.ndarray | None,
) -> pd.DataFrame:
    X_test = test[features].to_pandas()
    y_true = test["label"].to_numpy()
    rows: list[dict[str, float | str]] = []

    for name, base_model in model.active_models.items():
        encoded_pred = base_model.predict(X_test)
        predictions = model.label_encoder.inverse_transform(encoded_pred.astype(int))
        pred_proba = derive_aligned_probabilities(base_model, X_test)
        rows.append(build_classification_metric_row(name, y_true, predictions, pred_proba))

    rows.append(
        build_classification_metric_row(
            "hybrid_stacking",
            y_true,
            hybrid_predictions,
            hybrid_proba,
        )
    )
    return pd.DataFrame(rows)


def save_baseline_metrics_csv(
    model: HybridStackingSignalClassifier,
    test: pl.DataFrame,
    features: list[str],
    predictions: np.ndarray,
    pred_proba: np.ndarray | None,
    path: "Path",
) -> pd.DataFrame:
    df = build_baseline_metrics_dataframe(model, test, features, predictions, pred_proba)
    df.to_csv(path, index=False)
    return df
