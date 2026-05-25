from __future__ import annotations

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import torch

from src.config import LABELS
from src.validation import PurgedEmbargoTimeSeriesSplit


def stack_probas(model_probas: list[np.ndarray]) -> np.ndarray:
    return np.hstack(model_probas)


def valid_oof_mask(oof: np.ndarray) -> np.ndarray:
    return ~np.isnan(oof).any(axis=1)


def shared_valid_oof_mask(oofs) -> np.ndarray:
    valid = None
    for oof in oofs:
        current = valid_oof_mask(oof)
        valid = current if valid is None else valid & current
    return valid


def aligned_proba(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    aligned = np.zeros((len(X), len(LABELS)))
    for source_col, label_idx in enumerate(model[-1].classes_):
        aligned[:, int(label_idx)] = proba[:, source_col]
    return aligned


def score_oof_predictions(
    oof: np.ndarray,
    y: pd.Series | pl.Series,
    label_encoder: LabelEncoder,
) -> tuple[float, dict[int, float]]:
    valid = valid_oof_mask(oof)
    y_np = y.to_numpy() if isinstance(y, pl.Series) else y
    pred = label_encoder.inverse_transform(np.argmax(oof[valid], axis=1))
    macro_f1 = f1_score(y_np[valid], pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_np[valid], pred, average=None, zero_division=0)
    per_class = dict(zip([int(c) for c in label_encoder.classes_], per_class_f1.tolist()))
    return macro_f1, per_class


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


def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    weight_map = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[v] for v in y])


def _pipeline_weight_key(model) -> str:
    last_step = list(model.named_steps.keys())[-1]
    return f"{last_step}__sample_weight"


def oof_predictions(
    model,
    cv: PurgedEmbargoTimeSeriesSplit,
    X: pd.DataFrame,
    y_enc: np.ndarray,
    event_end: pd.Series,
) -> np.ndarray:
    oof = np.full((len(X), len(LABELS)), np.nan)
    weight_key = _pipeline_weight_key(model)
    for train_idx, val_idx in cv.split(X, event_end):
        weights = _compute_sample_weights(y_enc[train_idx])
        fold_model = clone(model).fit(X.iloc[train_idx], y_enc[train_idx], **{weight_key: weights})
        oof[val_idx] = aligned_proba(fold_model, X.iloc[val_idx])
    return oof


def pandas_pipeline(estimator):
    pipeline = make_pipeline(KNNImputer(n_neighbors=5), StandardScaler(), estimator)
    return pipeline.set_output(transform="pandas")


def build_meta_model(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
    )


def build_meta_label_model(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
    )


def build_lightgbm(random_state: int) -> LGBMClassifier:
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


def build_gru(random_state: int) -> "GRUClassifier":
    return GRUClassifier(
        sequence_length=8,
        hidden_size=128,
        num_layers=2,
        dropout=0.25,
        epochs=20,
        batch_size=64,
        random_state=random_state,
    )


def build_svc(random_state: int) -> SVC:
    return SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=random_state,
    )


def build_base_models(random_state: int) -> dict[str, object]:
    return {
        "gru": pandas_pipeline(build_gru(random_state)),
        "lightgbm": pandas_pipeline(build_lightgbm(random_state)),
        "svc": pandas_pipeline(build_svc(random_state)),
    }


def make_gru_sequences(X: pd.DataFrame, sequence_length: int) -> torch.Tensor:
    values = X.to_numpy(dtype=np.float32)
    sequences = np.empty((len(values), sequence_length, values.shape[1]), dtype=np.float32)
    for i in range(len(values)):
        start = max(0, i - sequence_length + 1)
        window = values[start : i + 1]
        sequences[i, : sequence_length - len(window)] = window[0]
        sequences[i, sequence_length - len(window) :] = window
    return torch.from_numpy(sequences)


class GRUNet(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.output = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        pooled = hidden[-1]
        return self.output(self.dropout(pooled))


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = torch.nn.functional.cross_entropy(pred, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


class GRUClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        sequence_length: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.25,
        learning_rate: float = 0.001,
        epochs: int = 20,
        batch_size: int = 64,
        random_state: int = 42,
    ):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
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
        train_x = make_gru_sequences(X, self.sequence_length).to(device)
        y_idx = np.searchsorted(self.classes_, y)
        train_y = torch.as_tensor(y_idx, dtype=torch.long, device=device)
        class_counts = np.bincount(y_idx, minlength=len(self.classes_))
        class_weights = len(y) / (len(self.classes_) * np.maximum(class_counts, 1))
        class_weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        model = GRUNet(
            self.n_features_in_,
            self.hidden_size,
            len(self.classes_),
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = FocalLoss(gamma=self.focal_gamma, weight=class_weights_tensor)
        num_samples = len(train_x)
        indices = np.arange(num_samples)

        model.train()
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss_fn(model(train_x[batch_idx]), train_y[batch_idx]).backward()
                optimizer.step()

        self.model_ = model
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            return np.ones((len(X), 1))

        self.model_.eval()
        sequences = make_gru_sequences(X, self.sequence_length).to(self.device_)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                logits = self.model_(sequences[start : start + self.batch_size])
                chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(chunks)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class HybridStackingSignalClassifier:
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.02,
        min_oof_f1: float = 0.34,
        confidence_threshold: float = 0.15,
        use_meta_labeling: bool = True,
        meta_label_threshold: float = 0.5,
        random_state: int = 42,
    ):
        self.cv = PurgedEmbargoTimeSeriesSplit(n_splits, embargo_pct)
        self.min_oof_f1 = min_oof_f1
        self.confidence_threshold = confidence_threshold
        self.use_meta_labeling = use_meta_labeling
        self.meta_label_threshold = meta_label_threshold
        self.random_state = random_state
        self.label_encoder = LabelEncoder().fit(LABELS)
        self.base_models = build_base_models(random_state)
        self.active_models: dict[str, object] = {}
        self.meta_model = build_meta_model(random_state)
        self.meta_label_model_ = build_meta_label_model(random_state)

    def fit(self, X: pl.DataFrame, y: pd.Series, event_end: pd.Series):
        X_pdf = X.to_pandas()
        y_enc = self.label_encoder.transform(y)
        oof_by_model, scores = self._collect_oof(X_pdf, y, y_enc, event_end)
        selected_oof = select_oof_predictions(oof_by_model, scores, self.min_oof_f1)
        self._fit_meta(selected_oof, y_enc)
        self._fit_active(selected_oof, X_pdf, y_enc)
        self.oof_scores_ = scores
        self.active_model_names_ = list(self.active_models)
        self._fit_meta_label_model(selected_oof, y_enc)
        return self

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        return self.meta_model.predict_proba(self._meta_features(X.to_pandas()))

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        pred_enc = self.meta_model.predict(self._meta_features(X.to_pandas()))
        return self.label_encoder.inverse_transform(pred_enc)

    def predict_positions(self, X: pl.DataFrame) -> np.ndarray:
        probas = self.predict_proba(X)
        pred_enc = np.full(len(X), 1, dtype=np.int64)

        if self.use_meta_labeling:
            P_correct = self.meta_label_model_.predict_proba(
                self._meta_label_features(X.to_pandas())
            )[:, 1]
            for i, row in enumerate(probas):
                if P_correct[i] < self.meta_label_threshold:
                    continue
                prob_sell, prob_hold, prob_buy = row[0], row[1], row[2]
                if prob_buy > prob_hold and prob_buy > prob_sell:
                    pred_enc[i] = 2
                elif prob_sell > prob_hold and prob_sell > prob_buy:
                    pred_enc[i] = 0
        else:
            threshold = self.confidence_threshold
            for i, row in enumerate(probas):
                prob_sell, prob_hold, prob_buy = row[0], row[1], row[2]
                if prob_buy > prob_hold + threshold and prob_buy > prob_sell:
                    pred_enc[i] = 2
                elif prob_sell > prob_hold + threshold and prob_sell > prob_buy:
                    pred_enc[i] = 0

        return self.label_encoder.inverse_transform(pred_enc)

    def _collect_oof(self, X, y, y_enc, event_end):
        oof_by_model = {}
        scores = {}
        per_class = {}
        for name, model in self.base_models.items():
            oof = oof_predictions(model, self.cv, X, y_enc, event_end)
            macro, cls_scores = score_oof_predictions(oof, y, self.label_encoder)
            scores[name] = macro
            per_class[name] = cls_scores
            oof_by_model[name] = oof
        self.per_class_oof_ = per_class
        return oof_by_model, scores

    def _fit_meta(self, selected_oof, y_enc):
        valid = shared_valid_oof_mask(selected_oof.values())
        stacked = stack_probas([oof[valid] for oof in selected_oof.values()])
        self.meta_model.fit(stacked, y_enc[valid])

    def _fit_active(self, selected_oof, X, y_enc):
        weights = _compute_sample_weights(y_enc)
        self.active_models = {
            name: clone(self.base_models[name]).fit(X, y_enc, **{_pipeline_weight_key(self.base_models[name]): weights})
            for name in selected_oof
        }

    def _fit_meta_label_model(self, selected_oof, y_enc):
        valid = shared_valid_oof_mask(selected_oof.values())
        stacked = stack_probas([oof[valid] for oof in selected_oof.values()])
        meta_probas = self.meta_model.predict_proba(stacked)
        meta_X = np.column_stack([meta_probas, stacked])
        primary_pred = self.meta_model.predict(stacked)
        meta_y = (primary_pred == y_enc[valid]).astype(np.int64)
        self.meta_label_model_.fit(meta_X, meta_y)

    def _meta_label_features(self, X: pd.DataFrame) -> np.ndarray:
        stacked = self._meta_features(X)
        meta_probas = self.meta_model.predict_proba(stacked)
        return np.column_stack([meta_probas, stacked])

    def _meta_features(self, X):
        return stack_probas([aligned_proba(m, X) for m in self.active_models.values()])