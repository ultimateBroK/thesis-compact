"""Artifacts: CSV/JSON/PNG persistence for pipeline outputs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from sklearn.metrics import confusion_matrix

from src.config import INITIAL_BALANCE
from src.metadata import build_run_metadata, collect_artifact_files
from src.metrics import save_baseline_metrics_csv
from src.models import HybridStackingSignalClassifier, probabilities_to_positions


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Trade extraction
# ---------------------------------------------------------------------------


def extract_trades_from_positions(results: pd.DataFrame) -> pd.DataFrame:
    ts = results["timestamp"].values
    close = results["close"].values
    pos = results["position"].values
    pnl = results["bar_pnl_usd"].values

    trades = []
    in_trade = False
    entry_idx = 0
    entry_pos = 0

    for i in range(len(pos)):
        changed = (i == 0 and pos[i] != 0) or (i > 0 and pos[i] != pos[i - 1])
        if changed:
            if in_trade and entry_pos != 0:
                trade_pnl = float(np.sum(pnl[entry_idx : i + 1]))
                trades.append({
                    "entry_time": str(ts[entry_idx]),
                    "exit_time": str(ts[i]),
                    "direction": "LONG" if entry_pos > 0 else "SHORT",
                    "entry_price": float(close[entry_idx]),
                    "exit_price": float(close[i]),
                    "bars_held": i - entry_idx + 1,
                    "trade_pnl_usd": trade_pnl,
                    "win": trade_pnl > 0,
                })
            if pos[i] == 0:
                in_trade = False
            else:
                in_trade = True
                entry_idx = i
                entry_pos = int(pos[i])

    if in_trade and entry_pos != 0:
        trade_pnl = float(np.sum(pnl[entry_idx:]))
        trades.append({
            "entry_time": str(ts[entry_idx]),
            "exit_time": str(ts[-1]),
            "direction": "LONG" if entry_pos > 0 else "SHORT",
            "entry_price": float(close[entry_idx]),
            "exit_price": float(close[-1]),
            "bars_held": len(pos) - entry_idx,
            "trade_pnl_usd": trade_pnl,
            "win": trade_pnl > 0,
        })

    return pd.DataFrame(trades)


def build_trades_dataframe(
    executed_trades: list[dict],
    timestamps: np.ndarray,
) -> pd.DataFrame:
    cleaned = []
    for t in executed_trades:
        trade = t.copy()
        trade["entry_time"] = str(timestamps[t["entry_idx"]])
        trade["exit_time"] = str(timestamps[t["exit_idx"]])
        del trade["entry_idx"]
        del trade["exit_idx"]
        cleaned.append(trade)
    return pd.DataFrame(cleaned)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def save_oof_scores_bar_plot(model: HybridStackingSignalClassifier, path: Path) -> None:
    scores = pd.Series(model.oof_scores_).sort_values()
    colors = [
        "#2ca02c" if name in model.active_model_names_ else "#d62728"
        for name in scores.index
    ]
    figure = Figure(figsize=(8, 4), dpi=200)
    ax = figure.subplots()
    ax.barh(scores.index, scores.to_numpy(), color=colors)
    ax.set_title("OOF Macro F1 — Base Models", fontsize=11)
    ax.set_xlabel("Macro F1", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")


def save_feature_importance_bar_plot(importance_df: pd.DataFrame, path: Path) -> None:
    figure = Figure(figsize=(10, 8), dpi=200)
    ax = figure.subplots()
    top = importance_df.head(20)
    colors = ["#1f77b4" if p >= 5.0 else "#aec7e8" for p in top["pct"]]
    ax.barh(top["feature"][::-1], top["pct"][::-1], color=colors[::-1])
    for i, (_, row) in enumerate(top[::-1].iterrows()):
        ax.text(row["pct"] + 0.2, i, f"{row['pct']:.1f}%", va="center", fontsize=9)
    ax.axvline(5.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_title("Feature Importance (LightGBM) — Top 20", fontsize=11)
    ax.set_xlabel("Importance %", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Thesis figures (Fig 1–7, 9)
# ---------------------------------------------------------------------------


def _crop_whitespace(path: Path, margin: int = 8) -> None:
    """Crop whitespace from saved PNG using Pillow."""
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    non_white = np.where(~((arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240) & (arr[:, :, 3] > 240)))
    if non_white[0].size == 0:
        return
    y_min, y_max = max(non_white[0].min() - margin, 0), min(non_white[0].max() + margin, img.height)
    x_min, x_max = max(non_white[1].min() - margin, 0), min(non_white[1].max() + margin, img.width)
    img.crop((x_min, y_min, x_max, y_max)).save(path)


def save_pipeline_overview_figure(path: Path) -> None:
    """Figure 1: Overall Hybrid Stacking Pipeline as a flow diagram."""
    steps = [
        "Tick Parquet",
        "OHLC 1H",
        "20 Technical\nFeatures",
        "4H Future-Return\nLabels",
        "Train/Test\nSplit",
        "LR + SVC\n+ LightGBM",
        "Logistic\nMeta Model",
        "Metrics +\nBacktest",
    ]
    n = len(steps)
    fig = Figure(figsize=(14, 2.2), dpi=200)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)
    ax = fig.subplots()
    spacing = 1.7
    ax.set_xlim(-1, (n - 1) * spacing + 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    box_w, box_h = 1.3, 0.48
    colors = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
        "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    ]
    for i, (label, color) in enumerate(zip(steps, colors)):
        x = i * spacing
        box = FancyBboxPatch(
            (x - box_w / 2, -box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.06", facecolor=color, edgecolor="white", linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x, 0, label, ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        if i < n - 1:
            ax.annotate(
                "", xy=(x + box_w / 2 + 0.18, 0),
                xytext=(x + box_w / 2 + 0.04, 0),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2),
            )
    fig.savefig(path, dpi=200)
    _crop_whitespace(path)


def save_train_test_split_figure(
    train: pl.DataFrame,
    test: pl.DataFrame,
    path: Path,
) -> None:
    """Figure 2: Chronological Train/Test Split timeline."""
    train_start = str(train["timestamp"].min())[:10]
    train_end = str(train["timestamp"].max())[:10]
    test_start = str(test["timestamp"].min())[:10]
    test_end = str(test["timestamp"].max())[:10]

    fig = Figure(figsize=(12, 1.3), dpi=200)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.85, bottom=0.15)
    ax = fig.subplots()
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.15, 0.85)
    ax.axis("off")

    bar_h = 0.45
    ax.barh(0.5, 0.65, left=0.0, height=bar_h, color="#4e79a7", alpha=0.85, edgecolor="white")
    ax.text(0.325, 0.5, f"TRAIN  {train_start} — {train_end}", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")

    purge_x = 0.65
    ax.barh(0.5, 0.05, left=purge_x, height=bar_h, color="#e15759", alpha=0.9, edgecolor="white")
    ax.text(purge_x + 0.025, 0.5, "Purge\n4 bars", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")

    test_x = 0.70
    ax.barh(0.5, 0.30, left=test_x, height=bar_h, color="#59a14f", alpha=0.85, edgecolor="white")
    ax.text(test_x + 0.15, 0.5, f"TEST  {test_start} — {test_end}", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")

    fig.text(0.5, 0.05, "Chronological split — no shuffle, no data leakage",
             ha="center", fontsize=10, fontstyle="italic", color="#666666")
    fig.savefig(path, dpi=200)
    _crop_whitespace(path)


def save_label_distribution_figure(
    train: pl.DataFrame,
    test: pl.DataFrame,
    path: Path,
) -> None:
    """Figure 3: Buy/Sell Label Distribution grouped bar chart (%)."""
    train_vc = train["label"].value_counts()
    test_vc = test["label"].value_counts()
    train_sell = train_vc.filter(pl.col("label") == -1)["count"].item()
    train_buy = train_vc.filter(pl.col("label") == 1)["count"].item()
    test_sell = test_vc.filter(pl.col("label") == -1)["count"].item()
    test_buy = test_vc.filter(pl.col("label") == 1)["count"].item()
    train_total = train_sell + train_buy
    test_total = test_sell + test_buy

    fig = Figure(figsize=(6, 4), dpi=200)
    ax = fig.subplots()
    x = np.array([0, 1])
    width = 0.35
    train_pct = [train_sell / train_total * 100, train_buy / train_total * 100]
    test_pct = [test_sell / test_total * 100, test_buy / test_total * 100]
    bars1 = ax.bar(x - width / 2, train_pct, width, label=f"Train (n={train_total})", color="#4e79a7")
    bars2 = ax.bar(x + width / 2, test_pct, width, label=f"Test (n={test_total})", color="#59a14f")
    for bar, pct, cnt in zip(bars1, train_pct, [train_sell, train_buy]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%\n({cnt})", ha="center", fontsize=8)
    for bar, pct, cnt in zip(bars2, test_pct, [test_sell, test_buy]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%\n({cnt})", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Sell (-1)", "Buy (+1)"], fontsize=10)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Buy/Sell Label Distribution")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(max(train_pct), max(test_pct)) + 15)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")


def save_baseline_comparison_figure(
    baseline_df: pd.DataFrame,
    path: Path,
) -> None:
    """Figure 4: Baseline Models vs Hybrid Stacking grouped bar chart."""
    metrics = ["accuracy", "f1_macro", "roc_auc"]
    model_names = baseline_df["model"].tolist()
    n_models = len(model_names)
    x = np.arange(n_models)
    width = 0.25
    colors = ["#4e79a7", "#f28e2b", "#e15759"]

    fig = Figure(figsize=(10, 5), dpi=200)
    ax = fig.subplots()
    all_values = []
    for i, metric in enumerate(metrics):
        values = baseline_df[metric].fillna(0).tolist()
        all_values.extend(values)
        bars = ax.bar(x + i * width, values, width, label=metric.upper(), color=colors[i])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([n.replace("_", "\n") for n in model_names], fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Baseline Models vs Hybrid Stacking", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    y_min = max(0, min(all_values) - 0.08)
    ax.set_ylim(y_min, max(all_values) + 0.08)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")


def save_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
) -> None:
    """Figure 5: Confusion Matrix of Hybrid Stacking on Test Set."""
    labels = [-1, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = Figure(figsize=(5, 4.5), dpi=200)
    ax = fig.subplots()
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Sell", "Pred Buy"], fontsize=10)
    ax.set_yticklabels(["True Sell", "True Buy"], fontsize=10)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18, fontweight="bold", color=color)
    ax.set_title("Confusion Matrix — Hybrid Stacking", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")


def save_equity_vs_buyhold_figure(
    equity: np.ndarray,
    test_close: np.ndarray,
    path: Path,
) -> None:
    """Figure 6: Equity Curve — Model Strategy vs Buy-and-Hold."""
    initial = equity[0]
    buyhold = initial * test_close / test_close[0]
    fig = Figure(figsize=(10, 5), dpi=200)
    ax = fig.subplots()
    ax.plot(equity, label="Model Strategy", color="#1f77b4", linewidth=1.5)
    ax.plot(buyhold, label="Buy & Hold", color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax.axhline(initial, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_title("Equity Curve: Model Strategy vs Buy-and-Hold", fontsize=11)
    ax.set_ylabel("Equity (USD)", fontsize=10)
    ax.set_xlabel("Test Bar", fontsize=10)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")


def save_position_exposure_figure(
    positions: np.ndarray,
    path: Path,
) -> None:
    """Figure 7: Long/Short Exposure Distribution bar chart."""
    total = len(positions)
    long_bars = int((positions > 0).sum())
    short_bars = int((positions < 0).sum())
    flat_bars = total - long_bars - short_bars
    categories = ["Long", "Short", "Flat"]
    counts = [long_bars, short_bars, flat_bars]
    pcts = [c / total * 100 for c in counts]
    colors = ["#2ca02c", "#d62728", "#999999"]

    fig = Figure(figsize=(6, 4), dpi=200)
    ax = fig.subplots()
    bars = ax.bar(categories, pcts, color=colors, width=0.55)
    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{pct:.1f}%\n({cnt} bars)", ha="center", fontsize=9)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_title("Position Exposure Distribution", fontsize=11)
    ax.set_ylim(0, max(pcts) + 18)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")


def save_threshold_sensitivity_figure(
    model: HybridStackingSignalClassifier,
    test: pl.DataFrame,
    features: list[str],
    path: Path,
) -> None:
    """Figure 9: Threshold Sensitivity Analysis."""
    proba = model.predict_proba(test[features])
    close = test["close"].to_numpy().astype(np.float64)
    spread = test["spread"].to_numpy().astype(np.float64) if "spread" in test.columns else np.zeros(len(close))

    thresholds = np.arange(0.50, 0.58, 0.01)
    rows = []
    for thr in thresholds:
        pos = probabilities_to_positions(proba, threshold=thr, long_only=model.long_only)
        from src.backtest import compute_strategy_bar_returns, build_equity_curve, compute_backtest_metrics, extract_position_trades
        bar_ret = compute_strategy_bar_returns(close, spread, pos)
        eq = build_equity_curve(bar_ret, INITIAL_BALANCE)
        trades = extract_position_trades(close, eq, pos)
        metrics = compute_backtest_metrics(eq, INITIAL_BALANCE, trades, pos)
        rows.append({
            "threshold": round(thr, 2),
            "trades": metrics["trades"],
            "total_return": metrics["total_return"],
            "profit_factor": metrics["profit_factor"],
            "sharpe": metrics["sharpe"],
        })
    df = pd.DataFrame(rows)

    fig = Figure(figsize=(10, 6))
    axes = fig.subplots(nrows=2, ncols=2).flat
    metrics_map = [
        ("trades", "Number of Trades", "#4e79a7"),
        ("total_return", "Total Return", "#e15759"),
        ("profit_factor", "Profit Factor", "#59a14f"),
        ("sharpe", "Sharpe Ratio", "#f28e2b"),
    ]
    for ax, (col, title, color) in zip(axes, metrics_map):
        ax.plot(df["threshold"], df[col], marker="o", color=color, linewidth=1.5)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Threshold")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Threshold Sensitivity Analysis", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Main save function
# ---------------------------------------------------------------------------


def save_run_artifacts(
    run_dir: Path,
    outputs,
    config_payload: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    train = outputs.train
    test = outputs.test
    model = outputs.model
    features = outputs.features
    predictions = outputs.predictions
    positions = outputs.positions
    backtest_metrics = outputs.backtest_metrics
    executed_trades = outputs.executed_trades
    equity_arr = outputs.equity
    dataset = pl.concat([train, test])

    results = test.select(["timestamp", "close", "spread", "label"]).to_pandas()
    results["prediction"] = predictions
    results["position"] = positions
    results["bar_pnl_usd"] = np.diff(equity_arr, prepend=equity_arr[0])
    results["equity_usd"] = equity_arr
    results.to_csv(tables_dir / "predictions.csv", index=False)

    if executed_trades is not None:
        timestamps = test["timestamp"].to_numpy()
        trades_df = build_trades_dataframe(executed_trades, timestamps)
    else:
        trades_df = extract_trades_from_positions(results)
    trades_df.to_csv(tables_dir / "trades.csv", index=False)

    if backtest_metrics:
        pd.DataFrame([backtest_metrics]).to_csv(tables_dir / "backtest_metrics.csv", index=False)

    pred_proba = getattr(outputs, "pred_proba", None)
    baseline_metrics_df = save_baseline_metrics_csv(
        model, test, features, predictions, pred_proba,
        tables_dir / "baseline_metrics.csv",
    )
    print("\n=== BASELINE TEST METRICS ===")
    print(baseline_metrics_df.to_string(index=False))

    importance_df = save_feature_importance_csv(model, features, tables_dir / "feature_importance.csv")

    # Thesis figures (fig1–fig9)
    save_pipeline_overview_figure(figures_dir / "fig1_pipeline.png")
    save_train_test_split_figure(train, test, figures_dir / "fig2_split.png")
    save_label_distribution_figure(train, test, figures_dir / "fig3_labels.png")
    save_baseline_comparison_figure(baseline_metrics_df, figures_dir / "fig4_baselines.png")
    save_confusion_matrix_figure(
        test["label"].to_numpy(), predictions,
        figures_dir / "fig5_confusion.png",
    )
    save_equity_vs_buyhold_figure(
        equity_arr, test["close"].to_numpy().astype(np.float64),
        figures_dir / "fig6_equity.png",
    )
    save_position_exposure_figure(positions, figures_dir / "fig7_exposure.png")
    save_feature_importance_bar_plot(importance_df, figures_dir / "fig8_importance.png")
    save_oof_scores_bar_plot(model, figures_dir / "fig9_oof_scores.png")
    try:
        save_threshold_sensitivity_figure(model, test, features, figures_dir / "fig10_threshold.png")
    except Exception:
        pass

    artifact_files = collect_artifact_files(run_dir, figures_dir, tables_dir) + ["run_data.json"]
    run_data = build_run_metadata(
        run_dir, model, config_payload, dataset, train, test,
        predictions, positions, results, features, backtest_metrics,
        artifact_files, trades_df, executed_trades=executed_trades,
        pred_proba=pred_proba,
    )
    with open(run_dir / "run_data.json", "w", encoding="utf-8") as f:
        json.dump(asdict(run_data), f, indent=2, ensure_ascii=False, default=str)

    print(f"\nRun dir: {run_dir.resolve()}")
    print(f"Files: tables/predictions.csv, tables/trades.csv ({len(trades_df)} trades), "
          "tables/backtest_metrics.csv, tables/baseline_metrics.csv, "
          "tables/feature_importance.csv, run_data.json, figures/*.png")
