"""Models: classical base learners plus a logistic stacking meta-learner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import LABELS, MIN_OOF_F1, SIGNAL_PROBABILITY_THRESHOLD
from src.signals import probabilities_to_positions
from src.validation import PurgedEmbargoTimeSeriesSplit


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def create_scaled_pipeline(estimator: BaseEstimator) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        estimator,
    ).set_output(transform="pandas")


def create_tree_pipeline(estimator: BaseEstimator) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        estimator,
    ).set_output(transform="pandas")


def create_meta_classifier(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
    )


def create_logistic_classifier(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
    )


def create_lightgbm_classifier(random_state: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.035,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        class_weight="balanced",
        random_state=random_state,
        verbosity=-1,
    )


def create_random_forest_classifier(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def assemble_base_model_registry(random_state: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": create_scaled_pipeline(create_logistic_classifier(random_state)),
        "random_forest": create_tree_pipeline(create_random_forest_classifier(random_state)),
        "lightgbm": create_tree_pipeline(create_lightgbm_classifier(random_state)),
    }


# ---------------------------------------------------------------------------
# Stacking helpers
# ---------------------------------------------------------------------------


def combine_model_probabilities(model_probas: list[np.ndarray]) -> np.ndarray:
    return np.hstack(model_probas)


def build_finite_oof_mask(oof: np.ndarray) -> np.ndarray:
    return ~np.isnan(oof).any(axis=1)


def build_shared_valid_oof_mask(oofs: list[np.ndarray]) -> np.ndarray:
    valid = None
    for oof in oofs:
        current = build_finite_oof_mask(oof)
        valid = current if valid is None else valid & current
    return np.zeros(0, dtype=bool) if valid is None else valid


def derive_aligned_probabilities(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    aligned = np.zeros((len(X), len(LABELS)), dtype=np.float64)
    classes = getattr(model[-1], "classes_", np.arange(proba.shape[1]))
    for source_col, encoded_label in enumerate(classes):
        aligned[:, int(encoded_label)] = proba[:, source_col]
    return aligned


def evaluate_oof_predictions(
    oof: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
) -> tuple[float, dict[int, float]]:
    valid = build_finite_oof_mask(oof)
    if not valid.any():
        return 0.0, {}
    pred = label_encoder.inverse_transform(np.argmax(oof[valid], axis=1))
    macro_f1 = f1_score(y[valid], pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y[valid], pred, labels=label_encoder.classes_, average=None, zero_division=0)
    per_class = dict(zip([int(c) for c in label_encoder.classes_], per_class_f1.tolist()))
    return float(macro_f1), per_class


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    weight_map = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[v] for v in y], dtype=np.float64)


def extract_sample_weight_key(model: Pipeline) -> str:
    return f"{list(model.named_steps)[-1]}__sample_weight"


def fill_single_class_probabilities(oof: np.ndarray, val_idx: np.ndarray, class_id: int) -> None:
    oof[val_idx, :] = 0.0
    oof[val_idx, int(class_id)] = 1.0


def cross_validate_oof_probabilities(
    model: Pipeline,
    cv: PurgedEmbargoTimeSeriesSplit,
    X: pd.DataFrame,
    y_enc: np.ndarray,
    event_end: pd.Series,
) -> np.ndarray:
    oof = np.full((len(X), len(LABELS)), np.nan, dtype=np.float64)
    weight_key = extract_sample_weight_key(model)

    for train_idx, val_idx in cv.split(X, event_end):
        train_y = y_enc[train_idx]
        unique = np.unique(train_y)
        if len(unique) < 2:
            fill_single_class_probabilities(oof, val_idx, int(unique[0]))
            continue
        weights = compute_class_weights(train_y)
        fold_model = clone(model).fit(X.iloc[train_idx], train_y, **{weight_key: weights})
        oof[val_idx] = derive_aligned_probabilities(fold_model, X.iloc[val_idx])
    return oof


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class HybridStackingSignalClassifier:
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.02,
        min_oof_f1: float = MIN_OOF_F1,
        signal_probability_threshold: float = SIGNAL_PROBABILITY_THRESHOLD,
        random_state: int = 42,
        long_only: bool = False,
        base_models: dict[str, Pipeline] | None = None,
    ):
        self.cv = PurgedEmbargoTimeSeriesSplit(n_splits, embargo_pct)
        self.min_oof_f1 = min_oof_f1
        self.signal_probability_threshold = signal_probability_threshold
        self.random_state = random_state
        self.long_only = long_only
        self.label_encoder = LabelEncoder().fit(LABELS)
        self.base_models = base_models if base_models is not None else assemble_base_model_registry(random_state)
        self.active_models: dict[str, Pipeline] = {}
        self.meta_model = create_meta_classifier(random_state)

    def fit(self, X: pl.DataFrame, y: pl.Series | pd.Series | np.ndarray, event_end: pl.Series | pd.Series):
        X_pdf = X.to_pandas()
        y_np = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        y_enc = self.label_encoder.transform(y_np)

        oof_by_model, scores = self.compute_base_model_oof_scores(X_pdf, y_np, y_enc, event_end)
        selected_oof = dict(oof_by_model)
        self.train_meta_classifier(selected_oof, y_enc)
        self.train_active_base_models(selected_oof, X_pdf, y_enc)
        self.oof_scores_ = scores
        self.active_model_names_ = list(self.active_models)
        self.fallback_fit_meta_classifier(X_pdf, y_enc)
        return self

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        if not self.active_models:
            return np.full((len(X), len(LABELS)), 1.0 / len(LABELS), dtype=np.float64)

        X_pdf = X.to_pandas()
        meta_feats = self.compute_meta_features(X_pdf)
        if self.check_estimator_fitted(self.meta_model):
            return self.meta_model.predict_proba(meta_feats)

        probas = [derive_aligned_probabilities(m, X_pdf) for m in self.active_models.values()]
        return np.mean(probas, axis=0)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        probas = self.predict_proba(X)
        return self.label_encoder.inverse_transform(probas.argmax(axis=1))

    def predict_positions(self, X: pl.DataFrame) -> np.ndarray:
        """Convert class probabilities to {-1, 0, +1} positions."""
        probas = self.predict_proba(X)
        return probabilities_to_positions(
            probas,
            threshold=self.signal_probability_threshold,
            long_only=self.long_only,
        )

    def compute_base_model_oof_scores(self, X: pd.DataFrame, y: np.ndarray, y_enc: np.ndarray, event_end):
        oof_by_model = {}
        scores = {}
        per_class = {}
        for name, model in self.base_models.items():
            oof = cross_validate_oof_probabilities(model, self.cv, X, y_enc, event_end)
            macro, cls_scores = evaluate_oof_predictions(oof, y, self.label_encoder)
            scores[name] = macro
            per_class[name] = cls_scores
            oof_by_model[name] = oof
        self.per_class_oof_ = per_class
        return oof_by_model, scores

    def filter_finite_predictions(self, oofs, y_enc):
        valid = build_shared_valid_oof_mask(list(oofs))
        if len(valid) == 0 or not valid.any():
            return None, None
        stacked = combine_model_probabilities([oof[valid] for oof in oofs])
        finite = np.isfinite(stacked).all(axis=1)
        if not finite.any():
            return None, None
        return stacked[finite], y_enc[valid][finite]

    def train_meta_classifier(self, selected_oof, y_enc) -> None:
        X_meta, y_meta = self.filter_finite_predictions(selected_oof.values(), y_enc)
        if X_meta is not None and len(np.unique(y_meta)) > 1:
            self.meta_model.fit(X_meta, y_meta)

    def train_active_base_models(self, selected_oof, X, y_enc) -> None:
        if len(np.unique(y_enc)) < 2:
            self.active_models = {}
            return
        weights = compute_class_weights(y_enc)
        self.active_models = {
            name: clone(self.base_models[name]).fit(
                X,
                y_enc,
                **{extract_sample_weight_key(self.base_models[name]): weights},
            )
            for name in selected_oof
        }

    def check_estimator_fitted(self, estimator) -> bool:
        try:
            from sklearn.utils.validation import check_is_fitted

            check_is_fitted(estimator)
            return True
        except Exception:
            return False

    def fallback_fit_meta_classifier(self, X, y_enc) -> None:
        if self.check_estimator_fitted(self.meta_model) or not self.active_models:
            return
        stacked = combine_model_probabilities([
            derive_aligned_probabilities(m, X) for m in self.active_models.values()
        ])
        finite = np.isfinite(stacked).all(axis=1)
        if finite.any() and len(np.unique(y_enc[finite])) > 1:
            self.meta_model.fit(stacked[finite], y_enc[finite])

    def compute_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        if not self.active_models:
            return np.zeros((len(X), len(LABELS)), dtype=np.float64)
        return combine_model_probabilities([
            derive_aligned_probabilities(m, X) for m in self.active_models.values()
        ])


__all__ = [
    "HybridStackingSignalClassifier",
    "assemble_base_model_registry",
    "combine_model_probabilities",
    "create_lightgbm_classifier",
    "create_logistic_classifier",
    "create_meta_classifier",
    "create_random_forest_classifier",
    "derive_aligned_probabilities",
]
