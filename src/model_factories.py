"""Factory functions for sklearn base models."""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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


__all__ = [
    "assemble_base_model_registry",
    "create_lightgbm_classifier",
    "create_logistic_classifier",
    "create_scaled_pipeline",
    "create_svc_classifier",
    "create_tree_pipeline",
]
