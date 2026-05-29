from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.config import LABELS
from src.validation import PurgedEmbargoTimeSeriesSplit


def combine_model_probabilities(model_probas: list[np.ndarray]) -> np.ndarray:
    return np.hstack(model_probas)


def build_finite_oof_mask(oof: np.ndarray) -> np.ndarray:
    return ~np.isnan(oof).any(axis=1)


def build_shared_valid_oof_mask(oofs: list[np.ndarray]) -> np.ndarray:
    valid = None
    for oof in oofs:
        current = build_finite_oof_mask(oof)
        valid = current if valid is None else valid & current
    return valid


def derive_aligned_probabilities(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    aligned = np.zeros((len(X), len(LABELS)))
    for source_col, label_idx in enumerate(model[-1].classes_):
        aligned[:, int(label_idx)] = proba[:, source_col]
    return aligned


def evaluate_oof_predictions(
    oof: np.ndarray,
    y: pd.Series | pl.Series,
    label_encoder: LabelEncoder,
) -> tuple[float, dict[int, float]]:
    valid = build_finite_oof_mask(oof)
    y_np = y.to_numpy() if isinstance(y, pl.Series) else y
    if not valid.any() or len(y_np[valid]) == 0:
        return 0.0, {}
    pred = label_encoder.inverse_transform(np.argmax(oof[valid], axis=1))
    macro_f1 = f1_score(y_np[valid], pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_np[valid], pred, average=None, zero_division=0)
    per_class = dict(zip([int(c) for c in label_encoder.classes_], per_class_f1.tolist()))
    return macro_f1, per_class


def select_qualified_oof_predictions(
    oof_by_model: dict[str, np.ndarray],
    scores: dict[str, float],
    min_oof_f1: float,
) -> dict[str, np.ndarray]:
    selected = {name: oof for name, oof in oof_by_model.items() if scores[name] >= min_oof_f1}
    if selected:
        return selected
    best_name = max(scores, key=scores.get)
    return {best_name: oof_by_model[best_name]}


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    weight_map = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[v] for v in y])


def extract_sample_weight_key(model: Pipeline) -> str:
    last_step = list(model.named_steps.keys())[-1]
    return f"{last_step}__sample_weight"


def cross_validate_oof_probabilities(
    model: Pipeline,
    cv: PurgedEmbargoTimeSeriesSplit,
    X: pd.DataFrame,
    y_enc: np.ndarray,
    event_end: pd.Series,
) -> np.ndarray:
    oof = np.full((len(X), len(LABELS)), np.nan)
    weight_key = extract_sample_weight_key(model)
    for train_idx, val_idx in cv.split(X, event_end):
        weights = compute_class_weights(y_enc[train_idx])
        fold_model = clone(model).fit(X.iloc[train_idx], y_enc[train_idx], **{weight_key: weights})
        oof[val_idx] = derive_aligned_probabilities(fold_model, X.iloc[val_idx])
    return oof
