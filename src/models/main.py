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


def enforce_minimum_position_hold(
    positions: np.ndarray, min_hold: int,
) -> np.ndarray:
    """Extend active position segments shorter than *min_hold* bars.

    Any consecutive non-zero segment shorter than *min_hold* is
    extended forward to at least *min_hold* bars, overwriting
    whatever follows (zeros or opposite-direction positions alike).

    Rationale: if the model cannot hold a direction for *min_hold*
    bars, any direction change within that window is noise, not
    signal.
    """
    if min_hold <= 1:
        return positions

    result = positions.copy()
    n = len(positions)
    i = 0

    # Scan contiguous position segments from the original signal.
    while i < n:
        # Flat bars do not start or extend a hold window.
        if positions[i] == 0:
            i += 1
            continue

        # Measure the current non-zero segment length.
        val = positions[i]
        j = i + 1
        while j < n and positions[j] == val:
            j += 1

        seg_len = j - i
        # Extend short segments unless the next segment is a true reversal.
        if seg_len < min_hold:
            end_val = positions[j] if j < n else 0
            if end_val == -val:
                # Genuine reversal — don't extend
                i = j
            else:
                # Extend forward, overwriting zeros (or same direction)
                extend_to = min(i + min_hold, n)
                result[i:extend_to] = val
                i = extend_to
        else:
            i = j
    return result


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
        long_only: bool = False,
        trend_filter_enabled: bool = True,
        trend_ema_period: int = 200,
        min_position_hold: int = 4,
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
        self.long_only = long_only
        self.trend_filter_enabled = trend_filter_enabled
        self.trend_ema_period = trend_ema_period
        self.min_position_hold = min_position_hold
        self.label_encoder = LabelEncoder().fit(LABELS)
        self.base_models = assemble_base_model_registry(random_state)
        self.active_models: dict[str, object] = {}
        self.meta_model = create_meta_classifier(random_state)
        self.meta_label_model_ = create_meta_label_classifier(random_state)

    # Stacking / cross-validation

    def fit(self, X: pl.DataFrame, y: pd.Series, event_end: pd.Series):
        X_pdf = X.to_pandas()
        y_enc = self.label_encoder.transform(y)
        oof_by_model, scores = self.compute_base_model_oof_scores(X_pdf, y, y_enc, event_end)
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
        if not self.check_estimator_fitted(self.meta_model):
            # Fallback: fit meta on full base model predictions
            self.fallback_fit_meta_classifier(X_pdf, y_enc)
        return self

    # Prediction

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

    # Regime filtering

    def detect_market_regime_filters(self, X: pl.DataFrame, close_series: np.ndarray | None = None) -> tuple[np.ndarray | None, np.ndarray | None, float, np.ndarray | None, np.ndarray | None]:
        adx = X["adx_14"].to_numpy() if "adx_14" in X.columns else None
        bb_width = X["bb_width"].to_numpy() if "bb_width" in X.columns else None
        if bb_width is not None:
            bb_mean = np.nanmean(bb_width[np.isfinite(bb_width)])
            bb_floor = bb_mean * self.bb_width_min_mult
        else:
            bb_floor = 0.0
        if close_series is not None:
            close = close_series
        elif "close" in X.columns:
            close = X["close"].to_numpy()
        else:
            close = None
        ema = pd.Series(close).ewm(span=self.trend_ema_period, adjust=False).mean().to_numpy() if close is not None else None
        return adx, bb_width, bb_floor, close, ema

    def check_market_regime_pass(
        self, i: int, position: int, adx: np.ndarray | None, bb_width: np.ndarray | None,
        bb_floor: float, close: np.ndarray | None, ema: np.ndarray | None,
    ) -> bool:
        if adx is not None and np.isfinite(adx[i]) and adx[i] < self.adx_threshold:
            return False
        if bb_width is not None and np.isfinite(bb_width[i]) and bb_width[i] < bb_floor:
            return False
        if position == -1 and self.trend_filter_enabled and close is not None and ema is not None:
            if np.isfinite(close[i]) and np.isfinite(ema[i]) and close[i] > ema[i]:
                return False
        return True

    # Position strategy

    def derive_positions_by_confidence(self, probas: np.ndarray, X: pl.DataFrame, close_series: np.ndarray | None = None) -> np.ndarray:
        """Binary: prob_buy > prob_sell + threshold -> long; inverse -> short; else flat."""
        adx, bb_width, bb_floor, close, ema = self.detect_market_regime_filters(X, close_series)
        positions = np.zeros(len(X), dtype=np.int64)
        for i, row in enumerate(probas):
            prob_sell, prob_buy = row[0], row[1]
            if prob_buy > prob_sell + self.confidence_threshold:
                if self.check_market_regime_pass(i, 1, adx, bb_width, bb_floor, close, ema):
                    positions[i] = 1
            elif prob_sell > prob_buy + self.short_meta_label_threshold:
                if self.check_market_regime_pass(i, -1, adx, bb_width, bb_floor, close, ema):
                    positions[i] = -1
        return positions

    def derive_positions_by_meta_label(self, probas: np.ndarray, X: pl.DataFrame, close_series: np.ndarray | None = None) -> np.ndarray:
        """Binary meta-label: predict if thesis is correct; asymmetric threshold for SHORT."""
        adx, bb_width, bb_floor, close, ema = self.detect_market_regime_filters(X, close_series)
        positions = np.zeros(len(X), dtype=np.int64)
        try:
            P_correct = self.meta_label_model_.predict_proba(
                self.derive_meta_label_features(X.to_pandas())
            )[:, 1]
        except Exception:
            P_correct = np.full(len(X), 0.5)
        for i, row in enumerate(probas):
            prob_sell, prob_buy = row[0], row[1]
            if prob_buy > prob_sell:
                if P_correct[i] >= self.meta_label_threshold and \
                        self.check_market_regime_pass(i, 1, adx, bb_width, bb_floor, close, ema):
                    positions[i] = 1
            elif prob_sell > prob_buy:
                if P_correct[i] >= self.short_meta_label_threshold and \
                        self.check_market_regime_pass(i, -1, adx, bb_width, bb_floor, close, ema):
                    positions[i] = -1
        return positions

    def predict_positions(
        self, X: pl.DataFrame, close_series: np.ndarray | None = None,
        skip_min_hold: bool = False,
    ) -> np.ndarray:
        probas = self.predict_proba(X)
        if self.use_meta_labeling:
            positions = self.derive_positions_by_meta_label(probas, X, close_series)
        else:
            positions = self.derive_positions_by_confidence(probas, X, close_series)
        if not skip_min_hold:
            positions = enforce_minimum_position_hold(positions, self.min_position_hold)
        if self.long_only:
            positions[positions == -1] = 0
        return positions

    # Stacking / cross-validation helpers

    def compute_base_model_oof_scores(self, X, y, y_enc, event_end):
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
        X, y = self.filter_finite_predictions(selected_oof.values(), y_enc)
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
        X, y = self.filter_finite_predictions(selected_oof.values(), y_enc)
        if X is None:
            return
        meta_probas = self.meta_model.predict_proba(X)
        meta_X = np.column_stack([meta_probas, X])
        primary_pred = self.meta_model.predict(X)
        meta_y = (primary_pred == y).astype(np.int64)
        self.meta_label_model_.fit(meta_X, meta_y)

    # Utilities

    def check_estimator_fitted(self, estimator) -> bool:
        """Check if sklearn estimator has been fitted."""
        try:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(estimator)
            return True
        except Exception:
            return False

    def fallback_fit_meta_classifier(self, X, y_enc):
        """Fit meta model on full training set when OOF is unusable."""
        stacked = combine_model_probabilities([
            derive_aligned_probabilities(m, X) for m in self.active_models.values()
        ])
        finite = np.isfinite(stacked).all(axis=1)
        if finite.any():
            self.meta_model.fit(stacked[finite], y_enc[finite])

    # Feature assembly

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
