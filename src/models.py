"""Models: LightGBM, GRU, stacking ensemble, position assignment."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import torch
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from src.config import LABELS, MIN_OOF_F1
from src.validation import PurgedEmbargoTimeSeriesSplit


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def wrap_sklearn_pipeline(estimator: BaseEstimator) -> Pipeline:
    pipeline = make_pipeline(KNNImputer(n_neighbors=5), StandardScaler(), estimator)
    return pipeline.set_output(transform="pandas")


def create_meta_classifier(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs",
        class_weight="balanced", random_state=random_state,
    )


def create_meta_label_classifier(random_state: int) -> CalibratedClassifierCV:
    base = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs",
        class_weight="balanced", random_state=random_state,
    )
    return CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)


def create_lightgbm_classifier(random_state: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=120, max_depth=5, learning_rate=0.035,
        num_leaves=31, subsample=0.85, colsample_bytree=0.85,
        class_weight="balanced", random_state=random_state, verbosity=-1,
    )


def create_gru_classifier(random_state: int) -> GRUClassifier:
    return GRUClassifier(
        sequence_length=8, hidden_size=128, num_layers=2,
        dropout=0.3, epochs=10, batch_size=64,
        bidirectional=True, random_state=random_state,
    )


def create_svm_classifier(random_state: int) -> SVC:
    return SVC(
        C=1.0, kernel="rbf", gamma="scale",
        class_weight="balanced", probability=True, random_state=random_state,
    )


def assemble_base_model_registry(random_state: int) -> dict[str, Pipeline]:
    return {
        "gru": wrap_sklearn_pipeline(create_gru_classifier(random_state)),
        "lightgbm": wrap_sklearn_pipeline(create_lightgbm_classifier(random_state)),
        "svc": wrap_sklearn_pipeline(create_svm_classifier(random_state)),
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


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------

def derive_rolling_sequences(X: pd.DataFrame, sequence_length: int) -> torch.Tensor:
    """Vectorized rolling sequence builder with pre-padding (repeat first row)."""
    values = X.to_numpy(dtype=np.float32)
    n, d = values.shape
    if n == 0:
        return torch.empty(0, sequence_length, d)
    # Pre-pad with sequence_length-1 copies of the first row
    padded = np.vstack([np.tile(values[0:1], (sequence_length - 1, 1)), values])
    # Strided windows of length sequence_length
    shape = (n, sequence_length, d)
    strides = (padded.strides[0], padded.strides[0], padded.strides[1])
    sequences = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return torch.from_numpy(sequences.copy())


class GRUNet(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = torch.nn.Dropout(dropout)
        dir_mult = 2 if bidirectional else 1
        self.output = torch.nn.Linear(hidden_size * dir_mult, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        # Take last time step; bidirectional output is already concatenated
        pooled = out[:, -1, :]
        return self.output(self.dropout(pooled))


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = torch.nn.functional.cross_entropy(pred, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


class GRUClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        sequence_length: int = 8,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 20,
        batch_size: int = 64,
        bidirectional: bool = True,
        random_state: int = 42,
    ):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.random_state = random_state
        self.focal_gamma = 1.0

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight=None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        if len(self.classes_) == 1:
            self.model_ = None
            return self

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = device
        use_amp = device.type == "cuda"

        train_x = derive_rolling_sequences(X, self.sequence_length).to(device)
        y_idx = np.searchsorted(self.classes_, y)
        train_y = torch.as_tensor(y_idx, dtype=torch.long, device=device)
        class_counts = np.bincount(y_idx, minlength=len(self.classes_))
        class_weights = len(y) / (len(self.classes_) * np.maximum(class_counts, 1))
        class_weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)

        model = GRUNet(
            self.n_features_in_, self.hidden_size, len(self.classes_),
            num_layers=self.num_layers, dropout=self.dropout,
            bidirectional=self.bidirectional,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = FocalLoss(gamma=self.focal_gamma, weight=class_weights_tensor)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        num_samples = len(train_x)
        indices = np.arange(num_samples)

        model.train()
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = loss_fn(model(train_x[batch_idx]), train_y[batch_idx])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        self.model_ = model
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            return np.full((len(X), max(len(self.classes_), 1)), 1.0 / max(len(self.classes_), 1))

        self.model_.eval()
        sequences = derive_rolling_sequences(X, self.sequence_length).to(self.device_)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                logits = self.model_(sequences[start : start + self.batch_size])
                chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(chunks)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# Position hold
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main class: HybridStackingSignalClassifier
# ---------------------------------------------------------------------------

class HybridStackingSignalClassifier:
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.02,
        min_oof_f1: float = MIN_OOF_F1,
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
        base_models: "dict[str, Pipeline] | None" = None,
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
        self.base_models = base_models if base_models is not None else assemble_base_model_registry(random_state)
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
            pred_enc = self.meta_model.predict(self.compute_meta_features(X.to_pandas()))
            return self.label_encoder.inverse_transform(pred_enc)
        except Exception:
            probas = self.predict_proba(X)
            return self.label_encoder.inverse_transform(probas.argmax(axis=1))

    # Regime filtering

    def extract_market_regime_inputs(self, X: pl.DataFrame, close_prices: np.ndarray | None = None) -> tuple[np.ndarray | None, np.ndarray | None, float, np.ndarray | None, np.ndarray | None]:
        adx = X["adx_14"].to_numpy() if "adx_14" in X.columns else None
        bb_width = X["bb_width"].to_numpy() if "bb_width" in X.columns else None
        if bb_width is not None:
            bb_mean = np.nanmean(bb_width[np.isfinite(bb_width)])
            bb_floor = bb_mean * self.bb_width_min_mult
        else:
            bb_floor = 0.0
        if close_prices is not None:
            close = close_prices
        elif "close" in X.columns:
            close = X["close"].to_numpy()
        else:
            close = None
        ema = pd.Series(close).ewm(span=self.trend_ema_period, adjust=False).mean().to_numpy() if close is not None else None
        return adx, bb_width, bb_floor, close, ema

    def passes_market_regime_filter(
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

    def assign_positions_by_confidence(self, probas: np.ndarray, X: pl.DataFrame, close_prices: np.ndarray | None = None) -> np.ndarray:
        """Binary: prob_buy > prob_sell + threshold -> long; inverse -> short; else flat."""
        adx, bb_width, bb_floor, close, ema = self.extract_market_regime_inputs(X, close_prices)
        positions = np.zeros(len(X), dtype=np.int64)
        for i, row in enumerate(probas):
            prob_sell, prob_buy = row[0], row[1]
            if prob_buy > prob_sell + self.confidence_threshold:
                if self.passes_market_regime_filter(i, 1, adx, bb_width, bb_floor, close, ema):
                    positions[i] = 1
            elif prob_sell > prob_buy + self.confidence_threshold:
                if self.passes_market_regime_filter(i, -1, adx, bb_width, bb_floor, close, ema):
                    positions[i] = -1
        return positions

    def assign_positions_by_meta_label(self, probas: np.ndarray, X: pl.DataFrame, close_prices: np.ndarray | None = None) -> np.ndarray:
        """Binary meta-label: predict if thesis is correct; asymmetric threshold for SHORT."""
        adx, bb_width, bb_floor, close, ema = self.extract_market_regime_inputs(X, close_prices)
        positions = np.zeros(len(X), dtype=np.int64)
        try:
            P_correct = self.meta_label_model_.predict_proba(
                self.compute_meta_label_features(X.to_pandas())
            )[:, 1]
        except Exception:
            P_correct = np.full(len(X), 0.5)
        for i, row in enumerate(probas):
            prob_sell, prob_buy = row[0], row[1]
            if prob_buy > prob_sell:
                if P_correct[i] >= self.meta_label_threshold and \
                        self.passes_market_regime_filter(i, 1, adx, bb_width, bb_floor, close, ema):
                    positions[i] = 1
            elif prob_sell > prob_buy:
                if P_correct[i] >= self.short_meta_label_threshold and \
                        self.passes_market_regime_filter(i, -1, adx, bb_width, bb_floor, close, ema):
                    positions[i] = -1
        return positions

    def predict_positions(
        self, X: pl.DataFrame, close_prices: np.ndarray | None = None,
        skip_min_hold: bool = False,
    ) -> np.ndarray:
        probas = self.predict_proba(X)
        if self.use_meta_labeling:
            positions = self.assign_positions_by_meta_label(probas, X, close_prices)
        else:
            positions = self.assign_positions_by_confidence(probas, X, close_prices)
        if self.long_only:
            positions[positions == -1] = 0
        if not skip_min_hold:
            positions = enforce_minimum_position_hold(positions, self.min_position_hold)
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

    def compute_meta_label_features(self, X: pd.DataFrame) -> np.ndarray:
        stacked = combine_model_probabilities([
            derive_aligned_probabilities(m, X) for m in self.active_models.values()
        ])
        try:
            meta_probas = self.meta_model.predict_proba(stacked)
        except Exception:
            meta_probas = np.full((len(stacked), len(LABELS)), 1.0 / len(LABELS))
        return np.column_stack([meta_probas, stacked])

    def compute_meta_features(self, X):
        if not self.active_models:
            return np.zeros((len(X), len(LABELS)))
        return combine_model_probabilities([
            derive_aligned_probabilities(m, X) for m in self.active_models.values()
        ])
