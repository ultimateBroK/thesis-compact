from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.models import HybridStackingSignalClassifier


def extract_lightgbm_feature_importance(
    model: HybridStackingSignalClassifier,
    features: list[str],
) -> pd.DataFrame:
    lgbm_pipeline = model.active_models.get("lightgbm")
    if lgbm_pipeline is None:
        return pd.DataFrame(columns=["rank", "feature", "importance", "pct"])
    lgbm_model = list(lgbm_pipeline.named_steps.values())[-1]
    imp = lgbm_model.feature_importances_
    total = imp.sum()
    df = pd.DataFrame({
        "feature": features,
        "importance": imp,
        "pct": imp / total * 100 if total > 0 else imp * 0,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "rank"
    return df


def save_feature_importance_csv(
    model: HybridStackingSignalClassifier,
    features: list[str],
    path: Path,
) -> pd.DataFrame:
    df = extract_lightgbm_feature_importance(model, features)
    df.to_csv(path)
    return df


def save_oof_scores_bar_plot(model: HybridStackingSignalClassifier, path: Path) -> None:
    scores = pd.Series(model.oof_scores_).sort_values()
    colors = [
        "#2ca02c" if name in model.active_model_names_ else "#d62728"
        for name in scores.index
    ]
    figure = Figure(figsize=(8, 4))
    ax = figure.subplots()
    ax.barh(scores.index, scores.to_numpy(), color=colors)
    ax.set_title("OOF Macro F1 — Base Models")
    ax.set_xlabel("Macro F1")
    figure.tight_layout()
    figure.savefig(path, dpi=160)


def save_equity_curve_plot(equity: np.ndarray, path: Path) -> None:
    figure = Figure(figsize=(9, 4))
    ax = figure.subplots()
    ax.plot(equity, color="#1f77b4")
    ax.set_title("Equity Curve (Cost-Aware)")
    ax.set_ylabel("Equity (USD)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)


def save_feature_importance_bar_plot(importance_df: pd.DataFrame, path: Path) -> None:
    figure = Figure(figsize=(10, 8))
    ax = figure.subplots()
    top = importance_df.head(20)
    colors = ["#1f77b4" if p >= 5.0 else "#aec7e8" for p in top["pct"]]
    ax.barh(top["feature"][::-1], top["pct"][::-1], color=colors[::-1])
    for i, (_, row) in enumerate(top[::-1].iterrows()):
        ax.text(row["pct"] + 0.2, i, f"{row['pct']:.1f}%", va="center", fontsize=8)
    ax.axvline(5.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_title("Feature Importance (LightGBM) — Top 20")
    ax.set_xlabel("Importance %")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
