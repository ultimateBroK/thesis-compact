"""Stacked signal classifier và các hàm xử lý xác suất."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.config import LABELS
from .cross_validation import PurgedTimeSeriesSplit
from .factories import assemble_base_model_registry, create_logistic_classifier


def probabilities_to_signals(probas: np.ndarray) -> np.ndarray:
    """Chuyển xác suất 2D đã căn chỉnh thành tín hiệu Buy/Sell {-1, +1}.

    Quy ước cột: ``probas[:, 0]`` là xác suất lớp Sell, ``probas[:, 1]`` là xác
    suất lớp Buy. Trả về +1 (Buy) khi P(Buy) >= P(Sell), ngược lại -1 (Sell).
    """
    if probas.ndim != 2 or probas.shape[1] < 2:
        raise ValueError("probas must be a 2D array with Sell and Buy columns")
    return np.where(probas[:, 1] >= probas[:, 0], 1, -1).astype(np.int64)


def _to_pandas_df(frame: pl.DataFrame | pd.DataFrame) -> pd.DataFrame:
    return frame if isinstance(frame, pd.DataFrame) else frame.to_pandas()


def combine_model_probabilities(model_probas: list[np.ndarray]) -> np.ndarray:
    """Chỉ stack xác suất lớp Buy của từng base learner.

    Binary classifier xuất P(Sell), P(Buy) với P(Sell) + P(Buy) = 1, nên hai cột
    đồng tuyến hoàn toàn. Giữ P(Buy) là đủ cho meta-feature và tránh dư thừa.
    """
    return np.hstack(
        [proba[:, [1]] for proba in model_probas]
    )  # chỉ giữ cột Buy; Sell dư thừa vì P(Sell) = 1 - P(Buy)


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


def derive_aligned_probabilities(
    model: Pipeline,
    X: pd.DataFrame,
    labels: tuple[int, ...] | np.ndarray = LABELS,
) -> np.ndarray:
    """Trả về xác suất căn theo thứ tự ``labels``.

    Ánh xạ ``classes_`` của mô hình sang vị trí trong ``labels``: nếu nhãn lớp có
    trong ``labels`` thì dùng vị trí đó; nếu không, lớp được xem là chỉ mục đã mã
    hóa vào ``labels``. Xử lý được cả mô hình huấn luyện trên nhãn thô (ví dụ
    ``[-1, +1]``) và nhãn đã mã hóa (ví dụ ``[0, 1]``).
    """
    proba = model.predict_proba(X)
    aligned = np.zeros((len(X), len(labels)), dtype=np.float64)
    classes = getattr(model[-1], "classes_", np.arange(proba.shape[1]))
    labels_list = list(labels)
    for source_col, class_label in enumerate(classes):
        if class_label in labels_list:
            target_col = labels_list.index(class_label)
        else:
            target_col = int(class_label)
        aligned[:, target_col] = proba[:, source_col]
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
        zip([int(label) for label in label_encoder.classes_], per_class_f1.tolist())
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
    labels: tuple[int, ...] | np.ndarray = LABELS,
) -> np.ndarray:
    """Tạo out-of-fold probabilities bằng purged time-series CV.

    Với mỗi fold, fit ``model`` trên train split và dự đoán xác suất trên
    validation split. Nếu train split chỉ có một lớp, validation rows được điền
    one-hot probability qua ``fill_single_class_probabilities``. Kết quả là mảng
    ``(n_samples, n_classes)`` dùng làm meta-feature không thiên lệch cho stacking.
    """
    oof = np.full((len(X), len(labels)), np.nan, dtype=np.float64)

    for train_idx, val_idx in cv.split(X, event_start, event_end):
        train_y = y_enc[train_idx]
        unique = np.unique(train_y)
        if len(unique) < 2:
            fill_single_class_probabilities(oof, val_idx, int(unique[0]))
            continue
        fold_model = clone(model).fit(X.iloc[train_idx], train_y)
        oof[val_idx] = derive_aligned_probabilities(fold_model, X.iloc[val_idx], labels)
    return oof


class HybridStackingSignalClassifier:
    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        base_models: dict[str, Pipeline] | None = None,
        labels: tuple[int, ...] | np.ndarray = LABELS,
    ):
        self.cv = PurgedTimeSeriesSplit(n_splits)
        self.random_state = random_state
        self.labels = tuple(labels)
        self.label_encoder = LabelEncoder().fit(self.labels)
        self.base_models = (
            base_models
            if base_models is not None
            else assemble_base_model_registry(random_state)
        )
        self.active_models: dict[str, Pipeline] = {}
        self.meta_model = create_logistic_classifier(random_state)

    def fit(
        self,
        X: pl.DataFrame | pd.DataFrame,
        y: pl.Series | pd.Series | np.ndarray,
        event_end: pl.Series | pd.Series,
        event_start: pl.Series | pd.Series | np.ndarray | None = None,
    ):
        """Huấn luyện ba giai đoạn: OOF scoring → meta-classifier → full retrain.

        1. Cross-validate từng base model để lấy out-of-fold probabilities.
        2. Fit meta-classifier (logistic regression) trên stacked OOF Buy-probabilities.
        3. Refit toàn bộ active base models trên full train set.

        Nếu meta-classifier chưa fit được (ví dụ mọi OOF đều NaN),
        ``fallback_fit_meta_classifier`` tạo lại meta-features từ base models đã
        fit đầy đủ rồi thử fit lại.
        """
        X_pdf = _to_pandas_df(X)
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
        # Fallback: nếu lọc OOF loại hết dòng, meta-model vẫn chưa được fit.
        # Tạo lại meta-features từ base models đã fit đầy đủ rồi thử fit lại.
        self.fallback_fit_meta_classifier(X_pdf, y_enc)
        return self

    def predict_proba(self, X: pl.DataFrame | pd.DataFrame) -> np.ndarray:
        if not self.active_models:
            return np.full(
                (len(X), len(self.labels)), 1.0 / len(self.labels), dtype=np.float64
            )

        X_pdf = _to_pandas_df(X)
        meta_feats = self.compute_meta_features(X_pdf)
        if self.check_estimator_fitted(self.meta_model):
            return self.meta_model.predict_proba(meta_feats)

        probas = [
            derive_aligned_probabilities(model, X_pdf, self.labels)
            for model in self.active_models.values()
        ]
        return np.mean(probas, axis=0)

    def predict(self, X: pl.DataFrame | pd.DataFrame) -> np.ndarray:
        probas = self.predict_proba(X)
        return self.label_encoder.inverse_transform(probas.argmax(axis=1))

    def predict_signals(self, X: pl.DataFrame | pd.DataFrame) -> np.ndarray:
        """Chuyển class probabilities thành tín hiệu Buy/Sell {-1, +1}."""
        return probabilities_to_signals(self.predict_proba(X))

    def compute_base_model_oof_scores(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        y_enc: np.ndarray,
        event_start: np.ndarray,
        event_end: np.ndarray,
    ):
        oof_by_model = {}
        scores = {}
        per_class = {}
        for name, model in self.base_models.items():
            oof = cross_validate_oof_probabilities(
                model, self.cv, X, y_enc, event_start, event_end, self.labels
            )
            macro, cls_scores = evaluate_oof_predictions(oof, y, self.label_encoder)
            scores[name] = macro
            per_class[name] = cls_scores
            oof_by_model[name] = oof
        self.per_class_oof_ = per_class
        return oof_by_model, scores

    def filter_finite_predictions(
        self, oofs, y_enc: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Tạo valid mask chung qua mọi OOF array, rồi stack Buy-probabilities.

        Trả về ``(stacked_meta_features, filtered_labels)`` hoặc ``(None, None)``
        nếu không có dòng nào có dự đoán hữu hạn trên mọi base model.
        """
        valid = build_shared_valid_oof_mask(list(oofs))
        if len(valid) == 0 or not valid.any():
            return None, None
        stacked = combine_model_probabilities([oof[valid] for oof in oofs])
        finite = np.isfinite(stacked).all(axis=1)
        if not finite.any():
            return None, None
        return stacked[finite], y_enc[valid][finite]

    def train_meta_classifier(self, selected_oof, y_enc: np.ndarray) -> None:
        """Fit meta-classifier trên stacked OOF Buy-probabilities.

        Chỉ fit khi nhãn sau lọc có hơn 1 lớp.
        """
        X_meta, y_meta = self.filter_finite_predictions(selected_oof.values(), y_enc)
        if X_meta is not None and len(np.unique(y_meta)) > 1:
            self.meta_model.fit(X_meta, y_meta)

    def train_active_base_models(
        self, selected_oof, X: pd.DataFrame, y_enc: np.ndarray
    ) -> None:
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
        """Tạo lại meta-features từ base models đã fit đầy đủ.

        Được gọi sau ``train_meta_classifier`` + ``train_active_base_models``.
        Nếu lọc OOF loại hết dòng nên meta-classifier vẫn chưa fit, hàm này dùng
        dự đoán từ các base models đã fit đầy đủ.
        """
        if self.check_estimator_fitted(self.meta_model) or not self.active_models:
            return
        stacked = combine_model_probabilities(
            [
                derive_aligned_probabilities(model, X, self.labels)
                for model in self.active_models.values()
            ]
        )
        finite = np.isfinite(stacked).all(axis=1)
        if finite.any() and len(np.unique(y_enc[finite])) > 1:
            self.meta_model.fit(stacked[finite], y_enc[finite])

    def compute_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        if not self.active_models:
            return np.zeros((len(X), len(self.labels)), dtype=np.float64)
        return combine_model_probabilities(
            [
                derive_aligned_probabilities(model, X, self.labels)
                for model in self.active_models.values()
            ]
        )
