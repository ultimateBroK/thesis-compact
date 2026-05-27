from __future__ import annotations

import json
import platform as _platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.backtest import simulate_equity, simulate_equity_barrier
from src.config import LABELS
from src.models import HybridStackingSignalClassifier


def model_status(name: str, model: HybridStackingSignalClassifier) -> str:
    return "ACTIVE" if name in model.active_model_names_ else "FILTERED"


def print_dataset_report(
    frame: pl.DataFrame,
    train: pl.DataFrame,
    test: pl.DataFrame,
    feature_count: int,
) -> None:
    print("=== DATASET ===")
    print(f"Rows: {len(frame)} | Train: {len(train)} | Test: {len(test)}")
    print(f"Features: {feature_count}")
    label_vc = frame["label"].value_counts().sort("label")
    print("Label distribution:")
    for row in label_vc.iter_rows(named=True):
        print(f"  {row['label']}: {row['count']}")


def print_model_report(model: HybridStackingSignalClassifier) -> None:
    print("\n=== SMART FILTERING OOF F1 ===")
    for name, score in sorted(
        model.oof_scores_.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{name}: {score:.4f} [{model_status(name, model)}]")


def print_classification_report(y_true: pl.Series, y_pred) -> None:
    print("\n=== TEST CLASSIFICATION ===")
    y_np = y_true.to_numpy() if isinstance(y_true, pl.Series) else y_true
    print(f"Accuracy: {accuracy_score(y_np, y_pred):.4f}")
    print(f"F1 macro: {f1_score(y_np, y_pred, average='macro', zero_division=0):.4f}")


def print_backtest_report(metrics: dict[str, float]) -> None:
    print("\n=== COST-AWARE BACKTEST ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def print_acceleration_report(accelerator: Any) -> None:
    print("=== ACCELERATION ===")
    print(f"Device: {accelerator.device} | Processes: {accelerator.num_processes}")


def print_feature_importance_report(importance_df: pd.DataFrame) -> None:
    print("\n=== FEATURE IMPORTANCE (LightGBM) ===")
    for idx, row in importance_df.head(10).iterrows():
        bar = "#" * int(row["pct"] * 2)
        print(f"  {idx:>2d}. {row['feature']:<25s} {row['importance']:>6d}  {row['pct']:>5.1f}%  {bar}")


def extract_trades(results: pd.DataFrame) -> pd.DataFrame:
    ts = results["timestamp"].values
    close = results["close"].values
    pos = results["position"].values
    pnl = results["pnl_usd"].values

    trades = []
    in_trade = False
    entry_idx = 0
    entry_pos = 0

    for i in range(len(pos)):
        changed = (i == 0 and pos[i] != 0) or (i > 0 and pos[i] != pos[i - 1])
        if changed:
            if in_trade and entry_pos != 0:
                trade_pnl = float(np.sum(pnl[entry_idx:i]))
                trades.append({
                    "entry_time": str(ts[entry_idx]),
                    "exit_time": str(ts[i - 1]),
                    "direction": "LONG" if entry_pos > 0 else "SHORT",
                    "entry_price": float(close[entry_idx]),
                    "exit_price": float(close[i - 1]),
                    "bars_held": i - entry_idx,
                    "pnl_usd": trade_pnl,
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
            "pnl_usd": trade_pnl,
            "win": trade_pnl > 0,
        })

    return pd.DataFrame(trades)


def save_oof_scores_plot(model: HybridStackingSignalClassifier, path: Path) -> None:
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


def save_feature_importance_plot(importance_df: pd.DataFrame, path: Path) -> None:
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


def extract_feature_importance(
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


def save_feature_importance(
    model: HybridStackingSignalClassifier,
    features: list[str],
    path: Path,
) -> pd.DataFrame:
    df = extract_feature_importance(model, features)
    df.to_csv(path)
    return df


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            args,
            cwd=Path.cwd(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def reproducibility_data() -> dict[str, Any]:
    status = git_value(["git", "status", "--short"])
    return {
        "python_version": sys.version.split()[0],
        "python_version_full": sys.version,
        "python_build": _platform.python_build(),
        "platform": _platform.platform(),
        "git_commit": git_value(["git", "rev-parse", "HEAD"]),
        "git_branch": git_value(["git", "branch", "--show-current"]),
        "git_dirty": bool(status),
        "run_entrypoint": "cli",
    }


def backtest_eval(results: pd.DataFrame) -> dict[str, float]:
    pnl = results["pnl_usd"]
    positions = results["position"]
    nonzero_pnl = pnl[pnl != 0]
    wins = nonzero_pnl[nonzero_pnl > 0]
    trades = float(np.sum(np.diff(positions, prepend=0) != 0))
    return {
        "win_rate": round(len(wins) / len(nonzero_pnl), 6) if len(nonzero_pnl) else 0.0,
        "turnover": round(trades / len(results), 6) if len(results) else 0.0,
    }


def prediction_results(
    test: pl.DataFrame,
    predictions: np.ndarray,
    positions: np.ndarray,
    equity: np.ndarray,
) -> pd.DataFrame:
    results = test.select(["timestamp", "close", "spread", "label"]).to_pandas()
    results["prediction"] = predictions
    results["position"] = positions
    results["pnl_usd"] = np.diff(equity, prepend=equity[0])
    results["equity"] = equity
    return results


def series_counts(values: Any) -> dict[str, int]:
    series = pd.Series(values.to_numpy() if isinstance(values, pl.Series) else values)
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def date_range(frame: pl.DataFrame) -> dict[str, str]:
    if not len(frame):
        return {"start": "", "end": ""}
    return {"start": str(frame["timestamp"][0]), "end": str(frame["timestamp"][-1])}


def dataset_run_data(
    dataset: pl.DataFrame,
    train: pl.DataFrame,
    test: pl.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    frac_d = (
        dataset.get_attribute("fractional_d")
        if hasattr(dataset, "get_attribute")
        else None
    )
    return {
        "total_rows": len(dataset),
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": len(features),
        "features": features,
        "fractional_d": frac_d,
        "data_range": date_range(dataset),
        "train_date_range": date_range(train),
        "test_date_range": date_range(test),
        "label_distribution_total": series_counts(dataset["label"]),
        "label_distribution_train": series_counts(train["label"]),
        "label_distribution_test": series_counts(test["label"]),
        "split_gap_info": {
            "train_end": str(train["timestamp"][-1]) if len(train) else "",
            "test_start": str(test["timestamp"][0]) if len(test) else "",
            "purge_rows": int(len(dataset) - len(train) - len(test)),
        },
    }


def training_run_data(model: HybridStackingSignalClassifier) -> dict[str, Any]:
    return {
        "oof_scores": {k: round(v, 6) for k, v in model.oof_scores_.items()},
        "per_class_oof_f1": {
            k: {str(c): round(v, 4) for c, v in cls.items()}
            for k, cls in model.per_class_oof_.items()
        },
        "active_models": model.active_model_names_,
        "filtered_models": [
            n for n in model.oof_scores_ if n not in model.active_model_names_
        ],
    }


def evaluation_run_data(
    test: pl.DataFrame,
    predictions: np.ndarray,
    positions: np.ndarray,
) -> dict[str, Any]:
    y_true = test["label"].to_numpy()
    labels = LABELS.tolist()
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 6),
        "f1_macro": round(
            float(f1_score(y_true, predictions, average="macro", zero_division=0)),
            6,
        ),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
        },
    }


def artifact_run_data(artifact_files: list[str]) -> dict[str, Any]:
    return {
        "files": artifact_files,
        "figure_count": sum(".png" in name for name in artifact_files),
    }


def build_run_data(
    run_dir: Path,
    model: HybridStackingSignalClassifier,
    config_payload: dict,
    dataset: pl.DataFrame,
    train: pl.DataFrame,
    test: pl.DataFrame,
    predictions: np.ndarray,
    positions: np.ndarray,
    results: pd.DataFrame,
    features: list[str],
    backtest_metrics: dict[str, float] | None,
    artifact_files: list[str],
) -> dict:
    feature_imp = extract_feature_importance(model, features)
    trades_df = extract_trades(results)
    return {
        "run_id": run_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config_payload,
        "dataset": dataset_run_data(dataset, train, test, features),
        "training": training_run_data(model),
        "evaluation": evaluation_run_data(test, predictions, positions),
        "backtest": {
            **{k: round(float(v), 6) for k, v in (backtest_metrics or {}).items()},
            **backtest_eval(results),
        },
        "feature_importance": {
            row["feature"]: round(float(row["pct"]), 2)
            for _, row in feature_imp.iterrows()
        },
        "trade_summary": {
            "total_trades": len(trades_df),
            "wins": int(trades_df["win"].sum()) if len(trades_df) else 0,
            "losses": int((~trades_df["win"]).sum()) if len(trades_df) else 0,
            "avg_bars_held": round(float(trades_df["bars_held"].mean()), 1) if len(trades_df) else 0,
            "avg_pnl_usd": round(float(trades_df["pnl_usd"].mean()), 2) if len(trades_df) else 0,
        },
        "artifacts": artifact_run_data(artifact_files),
        "reproducibility": reproducibility_data(),
    }


def _collect_artifact_files(run_dir: Path, figures_dir: Path) -> list[str]:
    root_files = [f.name for f in run_dir.iterdir() if f.is_file()]
    fig_files = [f"figures/{f.name}" for f in figures_dir.iterdir() if f.is_file()]
    return sorted(root_files + fig_files)


def save_run_artifacts(
    run_dir: Path,
    model: HybridStackingSignalClassifier,
    test: pl.DataFrame,
    predictions: np.ndarray,
    positions: np.ndarray,
    config_payload: dict,
    dataset: pl.DataFrame,
    train: pl.DataFrame,
    test_df: pl.DataFrame,
    features: list[str],
    backtest_metrics: dict[str, float] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    close = test["close"].to_numpy()
    high = test["high"].to_numpy()
    low = test["low"].to_numpy()
    spread = test["spread"].to_numpy()
    atr_rel = test["atr_14"].to_numpy()
    equity_arr, _ = simulate_equity_barrier(close, high, low, positions, spread, atr_rel=atr_rel)
    results = prediction_results(test, predictions, positions, equity_arr)
    results.to_csv(run_dir / "predictions.csv", index=False)

    trades_df = extract_trades(results)
    trades_df.to_csv(run_dir / "trades.csv", index=False)

    if backtest_metrics:
        pd.Series(backtest_metrics).to_csv(run_dir / "backtest_metrics.csv")

    importance_df = save_feature_importance(model, features, run_dir / "feature_importance.csv")
    save_feature_importance_plot(importance_df, figures_dir / "feature_importance.png")

    save_oof_scores_plot(model, figures_dir / "oof_scores.png")
    save_equity_curve_plot(equity_arr, figures_dir / "equity_curve.png")

    artifact_files = _collect_artifact_files(run_dir, figures_dir) + ["run_data.json"]

    run_data = build_run_data(
        run_dir,
        model,
        config_payload,
        dataset,
        train,
        test_df,
        predictions,
        positions,
        results,
        features,
        backtest_metrics,
        artifact_files,
    )
    with open(run_dir / "run_data.json", "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nRun dir: {run_dir.resolve()}")
    print(f"Files: predictions.csv, trades.csv ({len(trades_df)} trades), backtest_metrics.csv, feature_importance.csv, run_data.json, figures/*.png")
