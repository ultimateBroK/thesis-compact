"""
Model pipeline: cross-validate base models → stack → meta-label → predict positions.

Orchestration: HybridStackingSignalClassifier.fit → predict → predict_positions.
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
        self.train_meta_classifier(selected_oof, y_enc)
        self.train_active_base_models(selected_oof, X_pdf, y_enc)
        self.oof_scores_ = scores
        self.active_model_names_ = list(self.active_models)
        self.train_meta_label_corrector(selected_oof, y_enc)
        return self

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        meta_feats = combine_model_probabilities([
            derive_aligned_probabilities(m, X.to_pandas())
            for m in self.active_models.values()
        ])
        return self.meta_model.predict_proba(meta_feats)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        pred_enc = self.meta_model.predict(self.derive_meta_features(X.to_pandas()))
        return self.label_encoder.inverse_transform(pred_enc)

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
        adx, bb_width, bb_floor = self.detect_market_regime_filters(X)
        pred_enc = np.full(len(X), 1, dtype=np.int64)
        threshold = self.confidence_threshold
        for i, row in enumerate(probas):
            if not self.check_market_regime_pass(i, adx, bb_width, bb_floor):
                continue
            prob_sell, prob_hold, prob_buy = row[0], row[1], row[2]
            if prob_buy > prob_hold + threshold and prob_buy > prob_sell:
                pred_enc[i] = 2
            elif prob_sell > prob_hold + threshold and prob_sell > prob_buy:
                pred_enc[i] = 0
        return self.label_encoder.inverse_transform(pred_enc)

    def derive_positions_by_meta_label(self, probas: np.ndarray, X: pl.DataFrame) -> np.ndarray:
        adx, bb_width, bb_floor = self.detect_market_regime_filters(X)
        pred_enc = np.full(len(X), 1, dtype=np.int64)
        P_correct = self.meta_label_model_.predict_proba(
            self.derive_meta_label_features(X.to_pandas())
        )[:, 1]
        for i, row in enumerate(probas):
            if not self.check_market_regime_pass(i, adx, bb_width, bb_floor):
                continue
            prob_sell, prob_hold, prob_buy = row[0], row[1], row[2]
            if prob_buy > prob_hold and prob_buy > prob_sell:
                if P_correct[i] < self.meta_label_threshold:
                    continue
                pred_enc[i] = 2
            elif prob_sell > prob_hold and prob_sell > prob_buy:
                if P_correct[i] < self.short_meta_label_threshold:
                    continue
                pred_enc[i] = 0
        return self.label_encoder.inverse_transform(pred_enc)

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

    def train_meta_classifier(self, selected_oof, y_enc):
        valid = build_shared_valid_oof_mask(selected_oof.values())
        stacked = combine_model_probabilities([oof[valid] for oof in selected_oof.values()])
        self.meta_model.fit(stacked, y_enc[valid])

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
        valid = build_shared_valid_oof_mask(selected_oof.values())
        stacked = combine_model_probabilities([oof[valid] for oof in selected_oof.values()])
        meta_probas = self.meta_model.predict_proba(stacked)
        meta_X = np.column_stack([meta_probas, stacked])
        primary_pred = self.meta_model.predict(stacked)
        meta_y = (primary_pred == y_enc[valid]).astype(np.int64)
        self.meta_label_model_.fit(meta_X, meta_y)

    def derive_meta_label_features(self, X: pd.DataFrame) -> np.ndarray:
        stacked = combine_model_probabilities([
            derive_aligned_probabilities(m, X) for m in self.active_models.values()
        ])
        meta_probas = self.meta_model.predict_proba(stacked)
        return np.column_stack([meta_probas, stacked])

    def derive_meta_features(self, X):
        return combine_model_probabilities([
            derive_aligned_probabilities(m, X) for m in self.active_models.values()
        ])
