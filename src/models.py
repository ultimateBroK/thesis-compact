"""Models: classical base learners plus a logistic stacking meta-learner."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, clone

from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import LABELS

# Suppress sklearn SVC probability deprecation (1.9→1.11)
warnings.filterwarnings("ignore", message=".*probability.*parameter.*deprecated.*", category=FutureWarning)


# ---------------------------------------------------------------------------
# Signal conversion
# ---------------------------------------------------------------------------


def probabilities_to_signals(probas: np.ndarray) -> np.ndarray:
    """Convert aligned P(Sell), P(Buy) probabilities to {-1, +1} signals."""
    if probas.ndim != 2 or probas.shape[1] < 2:
        raise ValueError("probas must be a 2D array with Sell and Buy columns")
    return np.where(probas[:, 1] >= probas[:, 0], 1, -1).astype(np.int64)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def _to_int_array(values) -> np.ndarray:
    if isinstance(values, pl.Series):
        return values.to_numpy().astype(int)
    if hasattr(values, "to_numpy"):
        return values.to_numpy(dtype=int)
    return np.asarray(values, dtype=int)


def compute_purged_train_indices(
    indices: np.ndarray,
    event_start_pos: np.ndarray,
    event_end_pos: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    test_start, test_end = int(test_idx[0]), int(test_idx[-1])
    test_time_start = int(event_start_pos[test_start])
    test_time_end = int(event_end_pos[test_end])
    train_mask = np.ones(len(indices), dtype=bool)
    train_mask[test_idx] = False
    train_mask[
        (event_start_pos <= test_time_end) & (event_end_pos >= test_time_start)
    ] = False
    return indices[train_mask]


class PurgedTimeSeriesSplit:
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, X: pd.DataFrame, event_start: np.ndarray, event_end: np.ndarray):
        indices = np.arange(len(X))
        event_start_pos = _to_int_array(event_start)
        event_end_pos = _to_int_array(event_end)

        n = len(indices)
        test_size = max(1, n // (self.n_splits + 1))

        for i in range(self.n_splits):
            train_end = (i + 1) * test_size
            test_start = train_end
            test_end = test_start + test_size if i < self.n_splits - 1 else n

            test_idx = indices[test_start:test_end]
            candidate_idx = compute_purged_train_indices(
                indices, event_start_pos, event_end_pos, test_idx
            )
            train_idx = candidate_idx[candidate_idx < test_start]

            if len(train_idx):
                yield train_idx, test_idx


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


def create_svc_classifier(random_state: int) -> SVC:
    return SVC(
        C=1.0,
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=random_state,
    )


def assemble_base_model_registry(random_state: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": create_scaled_pipeline(
            create_logistic_classifier(random_state)
        ),
        "svc": create_scaled_pipeline(create_svc_classifier(random_state)),
        "lightgbm": create_tree_pipeline(create_lightgbm_classifier(random_state)),
    }


# ---------------------------------------------------------------------------
# Stacking helpers
# ---------------------------------------------------------------------------


def combine_model_probabilities(model_probas: list[np.ndarray]) -> np.ndarray:
    """Stack only the Buy-class probability per base learner.

    Binary classifiers output P(Sell), P(Buy) with P(Sell) + P(Buy) = 1, so
    the two columns are perfectly collinear. Keeping P(Buy) alone is
    sufficient for the meta feature space and avoids redundancy.
    """
    return np.hstack([proba[:, [1]] for proba in model_probas])


def build_finite_oof_mask(oof: np.ndarray) -> np.ndarray:
    return ~np.isnan(oof).any(axis=1)


def build_shared_valid_oof_mask(oofs: list[np.ndarray]) -> np.ndarray:
    valid: np.ndarray | None = None
    for oof in oofs:
        current = build_finite_oof_mask(oof)
        valid = current if valid is None else valid & current
    if valid is None:
        return np.zeros(0, dtype=bool)
    return valid


def derive_aligned_probabilities(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return full 2-col P(Sell), P(Buy) probabilities aligned to LABELS order.

    Consumers that build meta features should slice column 1 (P(Buy)) via
    ``combine_model_probabilities`` to avoid the collinearity of binary
    classifier outputs.
    """
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
    per_class_f1 = f1_score(
        y[valid], pred, labels=label_encoder.classes_, average=None, zero_division=0
    )
    per_class = dict(
        zip([int(c) for c in label_encoder.classes_], per_class_f1.tolist())
    )
    return float(macro_f1), per_class


def fill_single_class_probabilities(
    oof: np.ndarray, val_idx: np.ndarray, class_id: int
) -> None:
    oof[val_idx, :] = 0.0
    oof[val_idx, int(class_id)] = 1.0


def cross_validate_oof_probabilities(
    model: Pipeline,
    cv: PurgedTimeSeriesSplit,
    X: pd.DataFrame,
    y_enc: np.ndarray,
    event_start: np.ndarray,
    event_end: np.ndarray,
) -> np.ndarray:
    oof = np.full((len(X), len(LABELS)), np.nan, dtype=np.float64)

    for train_idx, val_idx in cv.split(X, event_start, event_end):
        train_y = y_enc[train_idx]
        unique = np.unique(train_y)
        if len(unique) < 2:
            fill_single_class_probabilities(oof, val_idx, int(unique[0]))
            continue
        fold_model = clone(model).fit(X.iloc[train_idx], train_y)
        oof[val_idx] = derive_aligned_probabilities(fold_model, X.iloc[val_idx])
    return oof


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class HybridStackingSignalClassifier:
    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        base_models: dict[str, Pipeline] | None = None,
    ):
        self.cv = PurgedTimeSeriesSplit(n_splits)
        self.random_state = random_state
        self.label_encoder = LabelEncoder().fit(LABELS)
        self.base_models = (
            base_models
            if base_models is not None
            else assemble_base_model_registry(random_state)
        )
        self.active_models: dict[str, Pipeline] = {}
        self.meta_model = create_meta_classifier(random_state)

    def fit(
        self,
        X: pl.DataFrame,
        y: pl.Series | pd.Series | np.ndarray,
        event_end: pl.Series | pd.Series,
        event_start: pl.Series | pd.Series | np.ndarray | None = None,
    ):
        X_pdf = X.to_pandas()
        y_np = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        y_enc = self.label_encoder.transform(y_np)
        event_start = (
            np.arange(len(X_pdf), dtype=np.int64)
            if event_start is None
            else event_start
        )

        oof_by_model, scores = self.compute_base_model_oof_scores(
            X_pdf, y_np, y_enc, event_start, event_end
        )
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

        probas = [
            derive_aligned_probabilities(m, X_pdf) for m in self.active_models.values()
        ]
        return np.mean(probas, axis=0)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        probas = self.predict_proba(X)
        return self.label_encoder.inverse_transform(probas.argmax(axis=1))

    def predict_signals(self, X: pl.DataFrame) -> np.ndarray:
        """Convert class probabilities to {-1, +1} Buy/Sell signals."""
        return probabilities_to_signals(self.predict_proba(X))

    def compute_base_model_oof_scores(
        self, X: pd.DataFrame, y: np.ndarray, y_enc: np.ndarray,
        event_start: np.ndarray, event_end: np.ndarray,
    ):
        oof_by_model = {}
        scores = {}
        per_class = {}
        for name, model in self.base_models.items():
            oof = cross_validate_oof_probabilities(
                model, self.cv, X, y_enc, event_start, event_end
            )
            macro, cls_scores = evaluate_oof_predictions(oof, y, self.label_encoder)
            scores[name] = macro
            per_class[name] = cls_scores
            oof_by_model[name] = oof
        self.per_class_oof_ = per_class
        return oof_by_model, scores

    def filter_finite_predictions(self, oofs, y_enc: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        valid = build_shared_valid_oof_mask(list(oofs))
        if len(valid) == 0 or not valid.any():
            return None, None
        stacked = combine_model_probabilities([oof[valid] for oof in oofs])
        finite = np.isfinite(stacked).all(axis=1)
        if not finite.any():
            return None, None
        return stacked[finite], y_enc[valid][finite]

    def train_meta_classifier(self, selected_oof, y_enc: np.ndarray) -> None:
        X_meta, y_meta = self.filter_finite_predictions(selected_oof.values(), y_enc)
        if X_meta is not None and len(np.unique(y_meta)) > 1:
            self.meta_model.fit(X_meta, y_meta)

    def train_active_base_models(self, selected_oof, X: pd.DataFrame, y_enc: np.ndarray) -> None:
        if len(np.unique(y_enc)) < 2:
            self.active_models = {}
            return
        self.active_models = {
            name: clone(self.base_models[name]).fit(X, y_enc) for name in selected_oof
        }

    def check_estimator_fitted(self, estimator) -> bool:
        try:
            from sklearn.utils.validation import check_is_fitted

            check_is_fitted(estimator)
            return True
        except Exception:
            return False

    def fallback_fit_meta_classifier(self, X: pd.DataFrame, y_enc: np.ndarray) -> None:
        if self.check_estimator_fitted(self.meta_model) or not self.active_models:
            return
        stacked = combine_model_probabilities(
            [derive_aligned_probabilities(m, X) for m in self.active_models.values()]
        )
        finite = np.isfinite(stacked).all(axis=1)
        if finite.any() and len(np.unique(y_enc[finite])) > 1:
            self.meta_model.fit(stacked[finite], y_enc[finite])

    def compute_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        if not self.active_models:
            return np.zeros((len(X), len(LABELS)), dtype=np.float64)
        return combine_model_probabilities(
            [derive_aligned_probabilities(m, X) for m in self.active_models.values()]
        )


__all__ = [
    "HybridStackingSignalClassifier",
    "assemble_base_model_registry",
    "combine_model_probabilities",
    "create_lightgbm_classifier",
    "create_logistic_classifier",
    "create_meta_classifier",
    "create_svc_classifier",
    "derive_aligned_probabilities",
    "probabilities_to_signals",
]
