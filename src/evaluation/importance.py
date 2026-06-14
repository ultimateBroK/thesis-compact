"""Trích xuất feature importance từ model đã fit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.models import HybridStackingSignalClassifier


def extract_lightgbm_feature_importance(
    model: "HybridStackingSignalClassifier",
    features: list[str],
) -> pd.DataFrame:
    lgbm_pipeline = model.active_models.get("lightgbm")
    if lgbm_pipeline is None:
        return pd.DataFrame(columns=["rank", "feature", "importance", "pct"])
    lgbm_model = list(lgbm_pipeline.named_steps.values())[-1]
    importance = lgbm_model.feature_importances_
    total = importance.sum()
    df = (
        pd.DataFrame(
            {
                "feature": features,
                "importance": importance,
                "pct": importance / total * 100 if total > 0 else importance * 0,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    df.index = df.index + 1
    df.index.name = "rank"
    return df
