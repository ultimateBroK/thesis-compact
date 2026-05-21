from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from hybrid_stacking.config import LABELS
from hybrid_stacking.validation import PurgedEmbargoTimeSeriesSplit


class HybridStackingSignalClassifier:
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.02,
        min_oof_f1: float = 0.34,
        random_state: int = 42,
    ):
        self.cv = PurgedEmbargoTimeSeriesSplit(n_splits, embargo_pct)
        self.min_oof_f1 = min_oof_f1
        self.random_state = random_state
        self.label_encoder = LabelEncoder().fit(LABELS)
        self.base_models = build_base_models(random_state)
        self.active_models: dict[str, object] = {}
        self.meta_model = build_meta_model(random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series, event_end: pd.Series):
        y_enc = self.label_encoder.transform(y)
        oof_by_model, scores = self.oof_predictions_by_model(X, y, y_enc, event_end)
        selected_oof = select_oof_predictions(oof_by_model, scores, self.min_oof_f1)
        self.fit_meta_model(selected_oof, y_enc)
        self.fit_active_models(selected_oof, X, y_enc)
        self.oof_scores_ = scores
        self.active_model_names_ = list(self.active_models)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.meta_model.predict_proba(self.meta_features(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred_enc = self.meta_model.predict(self.meta_features(X))
        return self.label_encoder.inverse_transform(pred_enc)

    def oof_predictions_by_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        y_enc: np.ndarray,
        event_end: pd.Series,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        oof_by_model = {}
        scores = {}
        for name, model in self.base_models.items():
            oof = oof_predictions(model, self.cv, X, y_enc, event_end)
            scores[name] = score_oof_predictions(oof, y, self.label_encoder)
            oof_by_model[name] = oof
        return oof_by_model, scores

    def fit_meta_model(self, selected_oof: dict[str, np.ndarray], y_enc: np.ndarray) -> None:
        valid = shared_valid_oof_mask(selected_oof.values())
        self.meta_model.fit(stack_probas([oof[valid] for oof in selected_oof.values()]), y_enc[valid])

    def fit_active_models(
        self,
        selected_oof: dict[str, np.ndarray],
        X: pd.DataFrame,
        y_enc: np.ndarray,
    ) -> None:
        self.active_models = {}
        for name in selected_oof:
            self.active_models[name] = clone(self.base_models[name]).fit(X, y_enc)

    def meta_features(self, X: pd.DataFrame) -> np.ndarray:
        return stack_probas([aligned_proba(model, X) for model in self.active_models.values()])


def build_meta_model(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        C=0.7,
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
    )


def build_base_models(random_state: int) -> dict[str, object]:
    return {
        "xgboost": pandas_pipeline(build_xgboost(random_state)),
        "lightgbm": pandas_pipeline(build_lightgbm(random_state)),
        "random_forest": pandas_pipeline(build_random_forest(random_state)),
        "extra_trees": pandas_pipeline(build_extra_trees(random_state)),
        "svc_rbf": pandas_pipeline(build_svc(random_state)),
    }


def pandas_pipeline(estimator):
    pipeline = make_pipeline(KNNImputer(n_neighbors=5), MinMaxScaler(), estimator)
    return pipeline.set_output(transform="pandas")


def build_xgboost(random_state: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=random_state,
    )


def build_lightgbm(random_state: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=220,
        max_depth=5,
        learning_rate=0.035,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        class_weight="balanced",
        random_state=random_state,
        verbosity=-1,
    )


def build_random_forest(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=220,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def build_extra_trees(random_state: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=220,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def build_svc(random_state: int) -> SVC:
    return SVC(
        C=1.2,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=random_state,
    )


def oof_predictions(
    model,
    cv: PurgedEmbargoTimeSeriesSplit,
    X: pd.DataFrame,
    y_enc: np.ndarray,
    event_end: pd.Series,
) -> np.ndarray:
    oof = np.full((len(X), len(LABELS)), np.nan)
    for train_idx, val_idx in cv.split(X, event_end):
        fold_model = clone(model).fit(X.iloc[train_idx], y_enc[train_idx])
        oof[val_idx] = aligned_proba(fold_model, X.iloc[val_idx])
    return oof


def aligned_proba(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    aligned = np.zeros((len(X), len(LABELS)))
    for source_col, label_idx in enumerate(model[-1].classes_):
        aligned[:, int(label_idx)] = proba[:, source_col]
    return aligned


def score_oof_predictions(oof: np.ndarray, y: pd.Series, label_encoder: LabelEncoder) -> float:
    valid = valid_oof_mask(oof)
    pred = label_encoder.inverse_transform(np.argmax(oof[valid], axis=1))
    return f1_score(y.iloc[valid], pred, average="macro", zero_division=0)


def select_oof_predictions(
    oof_by_model: dict[str, np.ndarray],
    scores: dict[str, float],
    min_oof_f1: float,
) -> dict[str, np.ndarray]:
    selected = {name: oof for name, oof in oof_by_model.items() if scores[name] >= min_oof_f1}
    if selected:
        return selected
    best_name = max(scores, key=scores.get)
    return {best_name: oof_by_model[best_name]}


def shared_valid_oof_mask(oofs) -> np.ndarray:
    valid = None
    for oof in oofs:
        current = valid_oof_mask(oof)
        valid = current if valid is None else valid & current
    return valid


def valid_oof_mask(oof: np.ndarray) -> np.ndarray:
    return ~np.isnan(oof).any(axis=1)


def stack_probas(model_probas: list[np.ndarray]) -> np.ndarray:
    return np.hstack(model_probas)
