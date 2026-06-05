"""Plotting helpers for report figures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from sklearn.metrics import confusion_matrix

if TYPE_CHECKING:
    from src.models import HybridStackingSignalClassifier

FIG_DPI = 200
FIG_SAVE_KWARGS = {"dpi": FIG_DPI, "bbox_inches": "tight"}


def _new_figure(figsize: tuple[float, float]) -> Figure:
    return Figure(figsize=figsize, dpi=FIG_DPI)


def _save_tight(figure: Figure, path: Path) -> None:
    figure.savefig(path, **FIG_SAVE_KWARGS)


def _annotate_vertical_bars(
    ax, bars, labels, *, y_offset: float, fontsize: int
) -> None:
    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            label,
            ha="center",
            fontsize=fontsize,
        )


def _display_model_name(name: str) -> str:
    return {
        "logistic_regression": "LR",
        "svc": "SVC",
        "lightgbm": "LightGBM",
        "LogisticRegression": "Logistic Regression",
    }.get(name, name.replace("_", " ").title())


def _format_model_names(names: list[str]) -> str:
    display_names = [_display_model_name(name) for name in names]
    if len(display_names) <= 2:
        return " + ".join(display_names)
    return f"{' + '.join(display_names[:-1])}\n+ {display_names[-1]}"


def _format_meta_model_name(name: str) -> str:
    """Format meta-model class name for the pipeline diagram box.

    Splits multi-word class names on word boundaries so the box stays compact.
    """
    known = {
        "LogisticRegression": "Logistic\nRegression",
        "XGBClassifier": "XGB\nClassifier",
        "LGBMClassifier": "LightGBM\nClassifier",
    }
    if name in known:
        return known[name]
    return name.replace("_", "\n", 1) if "_" in name else name


def _label_name(label: int) -> str:
    if label > 0:
        return f"Buy (+{label})"
    if label < 0:
        return f"Sell ({label})"
    return str(label)


def save_oof_scores_bar_plot(
    model: "HybridStackingSignalClassifier", path: Path
) -> None:
    scores = pd.Series(model.oof_scores_).sort_values()
    colors = [
        "#2ca02c" if name in model.active_model_names_ else "#d62728"
        for name in scores.index
    ]
    figure = _new_figure((8, 4))
    ax = figure.subplots()
    bars = ax.barh(scores.index, scores.to_numpy(), color=colors)
    for bar, value in zip(bars, scores.to_numpy()):
        ax.text(
            value + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )
    ax.set_title("OOF Macro F1 — Base Models", fontsize=11)
    ax.set_xlabel("Macro F1", fontsize=10)
    ax.set_xlim(0, max(scores.to_numpy()) + 0.05)
    figure.tight_layout()
    _save_tight(figure, path)


def save_feature_importance_bar_plot(importance_df: pd.DataFrame, path: Path) -> None:
    figure = _new_figure((11, 10))
    ax = figure.subplots()
    top = importance_df.head(30)
    colors = ["#1f77b4" if pct >= 5.0 else "#aec7e8" for pct in top["pct"]]
    ax.barh(top["feature"][::-1], top["pct"][::-1], color=colors[::-1])
    for index, (_, row) in enumerate(top[::-1].iterrows()):
        ax.text(
            row["pct"] + 0.2,
            index,
            f"{row['pct']:.1f}%",
            va="center",
            fontsize=8,
        )
    ax.axvline(5.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_title("Feature Importance (LightGBM) — Top 30", fontsize=11)
    ax.set_xlabel("Importance %", fontsize=10)
    figure.tight_layout()
    _save_tight(figure, path)


def _crop_whitespace(path: Path, margin: int = 8) -> None:
    """Crop whitespace from saved PNG using Pillow."""
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    non_white = np.where(
        ~(
            (arr[:, :, 0] > 240)
            & (arr[:, :, 1] > 240)
            & (arr[:, :, 2] > 240)
            & (arr[:, :, 3] > 240)
        )
    )
    if non_white[0].size == 0:
        return
    y_min, y_max = (
        max(non_white[0].min() - margin, 0),
        min(non_white[0].max() + margin, img.height),
    )
    x_min, x_max = (
        max(non_white[1].min() - margin, 0),
        min(non_white[1].max() + margin, img.width),
    )
    img.crop((x_min, y_min, x_max, y_max)).save(path)


def save_pipeline_overview_figure(
    path: Path,
    *,
    feature_count: int,
    labeling_horizon: int,
    timeframe: str,
    base_model_names: list[str],
    meta_model_name: str,
) -> None:
    """Figure 1: Overall Hybrid Stacking Pipeline as a flow diagram."""
    steps = [
        "Tick Parquet",
        f"OHLC {timeframe}",
        f"{feature_count} Model\nFeatures",
        f"{labeling_horizon}×{timeframe}\nFuture-Return\nLabels",
        "Train/Test\nSplit",
        _format_model_names(base_model_names),
        f"{_format_meta_model_name(meta_model_name)}\nMeta Model",
        "Metrics +\nBacktest",
    ]
    n_steps = len(steps)
    fig = _new_figure((15, 2.4))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)
    ax = fig.subplots()
    spacing = 1.8
    ax.set_xlim(-1, (n_steps - 1) * spacing + 1)
    ax.set_ylim(-0.6, 0.6)
    ax.axis("off")
    box_w, box_h = 1.55, 0.68
    colors = [
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
    ]
    for index, (label, color) in enumerate(zip(steps, colors)):
        x_pos = index * spacing
        box = FancyBboxPatch(
            (x_pos - box_w / 2, -box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.06",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(
            x_pos,
            0,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="white",
        )
        if index < n_steps - 1:
            ax.annotate(
                "",
                xy=(x_pos + box_w / 2 + 0.18, 0),
                xytext=(x_pos + box_w / 2 + 0.04, 0),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2),
            )
    fig.savefig(path, dpi=FIG_DPI)
    _crop_whitespace(path)


def save_train_test_split_figure(
    train: pl.DataFrame,
    test: pl.DataFrame,
    path: Path,
    purge_bars: int,
) -> None:
    """Figure 2: Chronological Train/Test Split timeline."""
    train_start = str(train["timestamp"].min())[:10]
    train_end = str(train["timestamp"].max())[:10]
    test_start = str(test["timestamp"].min())[:10]
    test_end = str(test["timestamp"].max())[:10]

    fig = _new_figure((14, 1.7))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.18)
    ax = fig.subplots()
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.15, 0.85)
    ax.axis("off")

    bar_h = 0.45
    ax.barh(
        0.5,
        0.65,
        left=0.0,
        height=bar_h,
        color="#4e79a7",
        alpha=0.85,
        edgecolor="white",
    )
    ax.text(
        0.325,
        0.5,
        f"TRAIN\nn={len(train)}\n{train_start} → {train_end}",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white",
    )

    purge_x = 0.65
    ax.barh(
        0.5,
        0.05,
        left=purge_x,
        height=bar_h,
        color="#e15759",
        alpha=0.9,
        edgecolor="white",
    )
    ax.text(
        purge_x + 0.025,
        0.5,
        f"Purge\n{purge_bars} bars",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color="white",
    )

    test_x = 0.70
    ax.barh(
        0.5,
        0.30,
        left=test_x,
        height=bar_h,
        color="#59a14f",
        alpha=0.85,
        edgecolor="white",
    )
    ax.text(
        test_x + 0.15,
        0.5,
        f"TEST\nn={len(test)}\n{test_start} → {test_end}",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white",
    )

    fig.text(
        0.5,
        0.05,
        "Chronological split — no shuffle, no data leakage",
        ha="center",
        fontsize=10,
        fontstyle="italic",
        color="#666666",
    )
    fig.savefig(path, dpi=FIG_DPI)
    _crop_whitespace(path)


def save_label_distribution_figure(
    train: pl.DataFrame,
    test: pl.DataFrame,
    path: Path,
    labels: tuple[int, int] = (-1, 1),
) -> None:
    """Figure 3: Buy/Sell Label Distribution grouped bar chart (%)."""
    train_vc = train["label"].value_counts()
    test_vc = test["label"].value_counts()
    sell_label, buy_label = labels
    train_sell = train_vc.filter(pl.col("label") == sell_label)["count"].item()
    train_buy = train_vc.filter(pl.col("label") == buy_label)["count"].item()
    test_sell = test_vc.filter(pl.col("label") == sell_label)["count"].item()
    test_buy = test_vc.filter(pl.col("label") == buy_label)["count"].item()
    train_total = train_sell + train_buy
    test_total = test_sell + test_buy

    fig = _new_figure((6, 4))
    ax = fig.subplots()
    x_axis = np.array([0, 1])
    width = 0.35
    train_pct = [train_sell / train_total * 100, train_buy / train_total * 100]
    test_pct = [test_sell / test_total * 100, test_buy / test_total * 100]
    bars1 = ax.bar(
        x_axis - width / 2,
        train_pct,
        width,
        label=f"Train (n={train_total})",
        color="#4e79a7",
    )
    bars2 = ax.bar(
        x_axis + width / 2,
        test_pct,
        width,
        label=f"Test (n={test_total})",
        color="#59a14f",
    )
    _annotate_vertical_bars(
        ax,
        bars1,
        [
            f"{pct:.1f}%\n({count})"
            for pct, count in zip(train_pct, [train_sell, train_buy])
        ],
        y_offset=1,
        fontsize=8,
    )
    _annotate_vertical_bars(
        ax,
        bars2,
        [
            f"{pct:.1f}%\n({count})"
            for pct, count in zip(test_pct, [test_sell, test_buy])
        ],
        y_offset=1,
        fontsize=8,
    )
    ax.set_xticks(x_axis)
    ax.set_xticklabels([_label_name(sell_label), _label_name(buy_label)], fontsize=10)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Buy/Sell Label Distribution")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(max(train_pct), max(test_pct)) + 22)
    fig.tight_layout()
    _save_tight(fig, path)


def save_baseline_comparison_figure(
    baseline_df: pd.DataFrame,
    path: Path,
) -> None:
    """Figure 4: Baseline Models vs Hybrid Stacking grouped horizontal bar chart."""
    metrics = ["accuracy", "f1_macro", "roc_auc"]
    model_names = baseline_df["model"].tolist()
    y_axis = np.arange(len(model_names))
    height = 0.25
    colors = ["#4e79a7", "#f28e2b", "#e15759"]

    fig = _new_figure((10, 6))
    ax = fig.subplots()
    all_values = []
    for index, metric in enumerate(metrics):
        values = baseline_df[metric].fillna(0).tolist()
        all_values.extend(values)
        bars = ax.barh(
            y_axis + index * height,
            values,
            height,
            label=metric.upper(),
            color=colors[index],
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=7,
            )
    ax.set_yticks(y_axis + height)
    ax.set_yticklabels([name.replace("_", "\n") for name in model_names], fontsize=9)
    ax.set_xlabel("Score", fontsize=10)
    ax.set_title("Baseline Models vs Hybrid Stacking", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    x_min = max(0, min(all_values) - 0.08)
    ax.set_xlim(x_min, max(all_values) + 0.08)
    fig.tight_layout()
    _save_tight(fig, path)


def save_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    labels: tuple[int, int] = (-1, 1),
) -> None:
    """Figure 5: Confusion Matrix of Hybrid Stacking on Test Set."""
    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = _new_figure((5, 4.5))
    ax = fig.subplots()
    image = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Pred {_label_name(label)}" for label in labels], fontsize=10)
    ax.set_yticklabels([f"True {_label_name(label)}" for label in labels], fontsize=10)
    for row in range(2):
        for col in range(2):
            color = "white" if cm[row, col] > cm.max() / 2 else "black"
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color=color,
            )
    ax.set_title("Confusion Matrix — Hybrid Stacking", fontsize=11)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    _save_tight(fig, path)


def save_equity_vs_buyhold_figure(
    equity: np.ndarray,
    test_close: np.ndarray,
    path: Path,
) -> None:
    """Figure 6: Equity Curve — Model Strategy vs Buy-and-Hold."""
    initial = equity[0]
    buyhold = initial * test_close / test_close[0]
    fig = _new_figure((10, 5))
    ax = fig.subplots()
    ax.plot(equity, label="Model Strategy", color="#1f77b4", linewidth=1.5)
    ax.plot(
        buyhold,
        label="Buy & Hold",
        color="#ff7f0e",
        linewidth=1.5,
        linestyle="--",
    )
    ax.axhline(initial, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_title("Equity Curve: Model Strategy vs Buy-and-Hold", fontsize=11)
    ax.set_ylabel("Equity (USD)", fontsize=10)
    ax.set_xlabel("Test Bar", fontsize=10)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save_tight(fig, path)


def save_position_exposure_figure(
    positions: np.ndarray,
    path: Path,
) -> None:
    """Figure 7: Long/Short Exposure Distribution bar chart."""
    total = len(positions)
    long_bars = int((positions > 0).sum())
    short_bars = int((positions < 0).sum())
    categories = ["Long", "Short"]
    counts = [long_bars, short_bars]
    pcts = [count / total * 100 for count in counts] if total else [0.0, 0.0]
    colors = ["#2ca02c", "#d62728"]

    fig = _new_figure((6, 4))
    ax = fig.subplots()
    bars = ax.bar(categories, pcts, color=colors, width=0.55)
    _annotate_vertical_bars(
        ax,
        bars,
        [f"{pct:.1f}%\n({count} bars)" for pct, count in zip(pcts, counts)],
        y_offset=1.5,
        fontsize=9,
    )
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_title("Buy/Sell Signal Exposure Distribution", fontsize=11)
    ax.set_ylim(0, max(pcts) + 18)
    fig.tight_layout()
    _save_tight(fig, path)
