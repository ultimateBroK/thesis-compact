from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .gru import GRUClassifier


def wrap_sklearn_pipeline(estimator):
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


def create_gru_classifier(random_state: int) -> "GRUClassifier":
    return GRUClassifier(
        sequence_length=8, hidden_size=128, num_layers=2,
        dropout=0.25, epochs=20, batch_size=64, random_state=random_state,
    )


def create_svm_classifier(random_state: int) -> SVC:
    return SVC(
        C=1.0, kernel="rbf", gamma="scale",
        class_weight="balanced", probability=True, random_state=random_state,
    )


def assemble_base_model_registry(random_state: int) -> dict[str, object]:
    return {
        "gru": wrap_sklearn_pipeline(create_gru_classifier(random_state)),
        "lightgbm": wrap_sklearn_pipeline(create_lightgbm_classifier(random_state)),
        "svc": wrap_sklearn_pipeline(create_svm_classifier(random_state)),
    }
