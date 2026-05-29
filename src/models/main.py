"""
Model pipeline: cross-validate base models -> stack -> meta-label -> predict positions.

Orchestration: HybridStackingSignalClassifier.fit -> predict -> predict_positions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

from src.config import LABELS
from src.validation import PurgedEmbargoTimeSeriesSplit

from .builders import (
    assemble_base_model_registry,
    create_meta_classifier,
    create_meta_label_classifier,
)
from .stacking import (
    build_shared_valid_oof_mask,
    combine_model_probabilities,
    compute_class_weights,
    cross_validate_oof_probabilities,
    derive_aligned_probabilities,
    evaluate_oof_predictions,
    extract_sample_weight_key,
    select_qualified_oof_predictions,
)


class HybridStackingSignalClassifier:
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.02,
        min_oof_f1: float = 0.34,
        confidence_threshold: float = 0.15,
        use_meta_labeling: bool = False,
        meta_label_threshold: float = 0.35,
        short_meta_label_threshold: float = 0.60,
        adx_threshold: float = 20.0,
        bb_width_min_mult: float = 1.0,
        random_state: int = 42,
    ):
        self.cv = PurgedEmbargoTimeSeriesSplit(n_splits, embargo_pct)
        self.min_oof_f1 = min_oof_f1
        self.confidence_threshold = confidence_threshold
        self.use_meta_labeling = use_meta_labeling
        self.meta_label_threshold = meta_label_threshold
        self.short_meta_label_threshold = short_meta_label_threshold
        self.adx_threshold = adx_threshold
        self.bb_width_min_mult = bb_width_min_mult
        self.random_state = random_state
        self.label_encoder = LabelEncoder().fit(LABELS)
        self.base_models = assemble_base_model_registry(random_state)
        self.active_models: dict[str, object] = {}
        self.meta_model = create_meta_classifier(random_state)
        self.meta_label_model_ = create_meta_label_classifier(random_state)

    def fit(self, X: pl.DataFrame, y: pd.Series, event_end: pd.Series):
        X_pdf = X.to_pandas()
        y_enc = self.label_encoder.transform(y)
        oof_by_model, scores = self.collect_base_model_oof_scores(X_pdf, y, y_enc, event_end)
        selected_oof = select_qualified_oof_predictions(oof_by_model, scores, self.min_oof_f1)
        if not selected_oof:
            best_name = max(scores, key=scores.get)
            selected_oof = {best_name: oof_by_model[best_name]}
            print(f"No models passed OOF threshold; using best: {best_name} (F1={scores[best_name]:.4f})")
        self.train_meta_classifier(selected_oof, y_enc)
        self.train_active_base_models(selected_oof, X_pdf, y_enc)
        self.oof_scores_ = scores
        self.active_model_names_ = list(self.active_models)
        self.train_meta_label_corrector(selected_oof, y_enc)
        if not self._is_fitted(self.meta_model):
            # Fallback: fit meta on full base model predictions
            self._fallback_fit_meta(X_pdf, y_enc)
        return self

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        if not self.active_models:
            return np.full((len(X), len(LABELS)), 1.0 / len(LABELS))
        meta_feats = combine_model_probabilities([
            derive_aligned_probabilities(m, X.to_pandas())
            for m in self.active_models.values()
        ])
        try:
            return self.meta_model.predict_proba(meta_feats)
        except Exception:
            # Meta model unfitted: average base model predictions
            probas = [derive_aligned_probabilities(m, X.to_pandas()) for m in self.active_models.values()]
            return np.mean(probas, axis=0)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        if not self.active_models:
            return np.full(len(X), LABELS[0])
        try:
            pred_enc = self.meta_model.predict(self.derive_meta_features(X.to_pandas()))
            return self.label_encoder.inverse_transform(pred_enc)
        except Exception:
            probas = self.predict_proba(X)
            return self.label_encoder.inverse_transform(probas.argmax(axis=1))

    def detect_market_regime_filters(self, X: pl.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        adx = X["adx_14"].to_numpy() if "adx_14" in X.columns else None
        bb_width = X["bb_width"].to_numpy() if "bb_width" in X.columns else None
        if bb_width is not None:
            bb_mean = np.nanmean(bb_width[np.isfinite(bb_width)])
            bb_floor = bb_mean * self.bb_width_min_mult
        else:
            bb_floor = 0.0
        return adx, bb_width, bb_floor

    def check_market_regime_pass(self, i: int, adx: np.ndarray | None, bb_width: np.ndarray | None, bb_floor: float) -> bool:
        if adx is not None and np.isfinite(adx[i]) and adx[i] < self.adx_threshold:
            return False
        if bb_width is not None and np.isfinite(bb_width[i]) and bb_width[i] < bb_floor:
            return False
        return True

    def derive_positions_by_confidence(self, probas: np.ndarray, X: pl.DataFrame) -> np.ndarray:
        """Binary: prob_buy > prob_sell + threshold -> long; inverse -> short; else flat."""
        adx, bb_width, bb_floor = self.detect_market_regime_filters(X)
        positions = np.zeros(len(X), dtype=np.int64)
        threshold = self.confidence_threshold
        for i, row in enumerate(probas):
            if not self.check_market_regime_pass(i, adx, bb_width, bb_floor):
                continue
            prob_sell, prob_buy = row[0], row[1]
            if prob_buy > prob_sell + threshold:
                positions[i] = 1
            elif prob_sell > prob_buy + threshold:
                positions[i] = -1
        return positions

    def derive_positions_by_meta_label(self, probas: np.ndarray, X: pl.DataFrame) -> np.ndarray:
        """Binary meta-label: predict if thesis is correct; filter by single threshold."""
        adx, bb_width, bb_floor = self.detect_market_regime_filters(X)
        positions = np.zeros(len(X), dtype=np.int64)
        try:
            P_correct = self.meta_label_model_.predict_proba(
                self.derive_meta_label_features(X.to_pandas())
            )[:, 1]
        except Exception:
            P_correct = np.full(len(X), 0.5)
        for i, row in enumerate(probas):
            if not self.check_market_regime_pass(i, adx, bb_width, bb_floor):
                continue
            prob_sell, prob_buy = row[0], row[1]
            if prob_buy > prob_sell:
                if P_correct[i] >= self.meta_label_threshold:
                    positions[i] = 1
            elif prob_sell > prob_buy:
                if P_correct[i] >= self.meta_label_threshold:
                    positions[i] = -1
        return positions

    def predict_positions(self, X: pl.DataFrame) -> np.ndarray:
        probas = self.predict_proba(X)
        if self.use_meta_labeling:
            return self.derive_positions_by_meta_label(probas, X)
        return self.derive_positions_by_confidence(probas, X)

    def collect_base_model_oof_scores(self, X, y, y_enc, event_end):
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

    def _filter_finite(self, oofs, y_enc):
        """Return (stacked, y) with NaN rows dropped."""
        valid = build_shared_valid_oof_mask(oofs)
        if not valid.any():
            return None, None
        stacked = combine_model_probabilities([oof[valid] for oof in oofs])
        finite = np.isfinite(stacked).all(axis=1)
        if not finite.any():
            return None, None
        return stacked[finite], y_enc[valid][finite]

    def train_meta_classifier(self, selected_oof, y_enc):
        X, y = self._filter_finite(selected_oof.values(), y_enc)
        if X is not None:
            self.meta_model.fit(X, y)

    def train_active_base_models(self, selected_oof, X, y_enc):
        weights = compute_class_weights(y_enc)
        self.active_models = {
            name: clone(self.base_models[name]).fit(
                X, y_enc,
                **{extract_sample_weight_key(self.base_models[name]): weights},
            )
            for name in selected_oof
        }

    def train_meta_label_corrector(self, selected_oof, y_enc):
        X, y = self._filter_finite(selected_oof.values(), y_enc)
        if X is None:
            return
        meta_probas = self.meta_model.predict_proba(X)
        meta_X = np.column_stack([meta_probas, X])
        primary_pred = self.meta_model.predict(X)
        meta_y = (primary_pred == y).astype(np.int64)
        self.meta_label_model_.fit(meta_X, meta_y)

    def _is_fitted(self, estimator) -> bool:
        """Check if sklearn estimator has been fitted."""
        try:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(estimator)
            return True
        except Exception:
            return False

    def _fallback_fit_meta(self, X, y_enc):
        """Fit meta model on full training set when OOF is unusable."""
        stacked = combine_model_probabilities([
            derive_aligned_probabilities(m, X) for m in self.active_models.values()
        ])
        finite = np.isfinite(stacked).all(axis=1)
        if finite.any():
            self.meta_model.fit(stacked[finite], y_enc[finite])

    def derive_meta_label_features(self, X: pd.DataFrame) -> np.ndarray:
        stacked = combine_model_probabilities([
            derive_aligned_probabilities(m, X) for m in self.active_models.values()
        ])
        try:
            meta_probas = self.meta_model.predict_proba(stacked)
        except Exception:
            meta_probas = np.full((len(stacked), len(LABELS)), 1.0 / len(LABELS))
        return np.column_stack([meta_probas, stacked])

    def derive_meta_features(self, X):
        if not self.active_models:
            return np.zeros((len(X), len(LABELS)))
        return combine_model_probabilities([
            derive_aligned_probabilities(m, X) for m in self.active_models.values()
        ])
