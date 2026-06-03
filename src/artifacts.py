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

from src.metadata import build_run_metadata, collect_artifact_files
from src.metrics import save_baseline_metrics_csv
from src.models import HybridStackingSignalClassifier


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
    ax.set_title("Equity Curve (Signal Backtest)")
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


# ---------------------------------------------------------------------------
# Main save function
# ---------------------------------------------------------------------------


def save_run_artifacts(
    run_dir: Path,
    outputs,
    config_payload: dict[str, Any],
    window_id: int | None = None,
    window_train_range: str = "",
    window_test_range: str = "",
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
    save_feature_importance_bar_plot(importance_df, figures_dir / "feature_importance.png")
    save_oof_scores_bar_plot(model, figures_dir / "oof_scores.png")
    save_equity_curve_plot(equity_arr, figures_dir / "equity_curve.png")

    artifact_files = collect_artifact_files(run_dir, figures_dir, tables_dir) + ["run_data.json"]
    run_data = build_run_metadata(
        run_dir, model, config_payload, dataset, train, test,
        predictions, positions, results, features, backtest_metrics,
        artifact_files, trades_df, executed_trades=executed_trades,
        window_id=window_id,
        window_train_range=window_train_range,
        window_test_range=window_test_range,
        pred_proba=pred_proba,
    )
    with open(run_dir / "run_data.json", "w", encoding="utf-8") as f:
        json.dump(asdict(run_data), f, indent=2, ensure_ascii=False, default=str)

    print(f"\nRun dir: {run_dir.resolve()}")
    print(f"Files: tables/predictions.csv, tables/trades.csv ({len(trades_df)} trades), "
          "tables/backtest_metrics.csv, tables/baseline_metrics.csv, "
          "tables/feature_importance.csv, run_data.json, figures/*.png")
