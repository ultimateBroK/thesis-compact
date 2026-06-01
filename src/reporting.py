"""Reporting: console output, trade extraction, feature importance, artifact persistence."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.config import LABELS
from src.models import HybridStackingSignalClassifier

if TYPE_CHECKING:
    from src.cli import PipelineOutputs


# ---------------------------------------------------------------------------
# Console printers
# ---------------------------------------------------------------------------


def determine_model_status(name: str, model: HybridStackingSignalClassifier) -> str:
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


def print_model_filtering_report(model: HybridStackingSignalClassifier) -> None:
    print("\n=== SMART FILTERING OOF F1 ===")
    for name, score in sorted(
        model.oof_scores_.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{name}: {score:.4f} [{determine_model_status(name, model)}]")


def print_classification_report(y_true: pl.Series, y_pred: np.ndarray | pl.Series) -> None:
    print("\n=== TEST CLASSIFICATION ===")
    y_np = y_true.to_numpy() if isinstance(y_true, pl.Series) else y_true
    print(f"Accuracy: {accuracy_score(y_np, y_pred):.4f}")
    print(f"F1 macro: {f1_score(y_np, y_pred, average='macro', zero_division=0):.4f}")


def print_backtest_metrics_report(metrics: dict[str, float]) -> None:
    print("\n=== COST-AWARE BACKTEST ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def print_device_acceleration_report(accelerator: Any) -> None:
    print("=== ACCELERATION ===")
    print(f"Device: {accelerator.device} | Processes: {accelerator.num_processes}")


def print_feature_importance_report(importance_df: pd.DataFrame) -> None:
    print("\n=== FEATURE IMPORTANCE (LightGBM) ===")
    for idx, row in importance_df.head(10).iterrows():
        bar = "#" * int(row["pct"] * 2)
        print(f"  {idx:>2d}. {row['feature']:<25s} {row['importance']:>6d}  {row['pct']:>5.1f}%  {bar}")


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
# Feature importance & plots
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


# ---------------------------------------------------------------------------
# Metadata dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DatasetMeta:
    total_rows: int
    train_rows: int
    test_rows: int
    feature_count: int
    features: list[str]
    fractional_d: float | None
    data_range: dict[str, str]
    train_date_range: dict[str, str]
    test_date_range: dict[str, str]
    label_distribution_total: dict[str, int]
    label_distribution_train: dict[str, int]
    label_distribution_test: dict[str, int]


@dataclass
class TrainingMeta:
    oof_scores: dict[str, float]
    per_class_oof_f1: dict[str, dict[str, float]]
    active_models: list[str]
    filtered_models: list[str]


@dataclass
class EvalMeta:
    accuracy: float
    f1_macro: float
    confusion_matrix: dict[str, Any]


@dataclass
class WinRateMeta:
    win_rate: float
    turnover: float


@dataclass
class RunMetadata:
    run_id: str
    timestamp: str
    config: dict[str, Any]
    dataset: DatasetMeta
    training: TrainingMeta
    evaluation: EvalMeta
    backtest: dict[str, Any]
    feature_importance: dict[str, float]
    trade_summary: dict[str, Any]
    artifacts: dict[str, Any]
    reproducibility: dict[str, Any]


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def collect_artifact_files(run_dir: Path, figures_dir: Path) -> list[str]:
    root_files = [f.name for f in run_dir.iterdir() if f.is_file()]
    fig_files = [f"figures/{f.name}" for f in figures_dir.iterdir() if f.is_file()]
    return sorted(root_files + fig_files)


def build_run_metadata(
    run_dir: Path,
    model: HybridStackingSignalClassifier,
    config_payload: dict[str, Any],
    dataset: pl.DataFrame,
    train: pl.DataFrame,
    test: pl.DataFrame,
    predictions: np.ndarray,
    positions: np.ndarray,
    results: pd.DataFrame,
    features: list[str],
    backtest_metrics: dict[str, float] | None,
    artifact_files: list[str],
    trades_df: pd.DataFrame,
    executed_trades: list[dict] | None = None,
    window_id: int | None = None,
    window_train_range: str | None = None,
    window_test_range: str | None = None,
) -> RunMetadata:
    import platform
    import subprocess
    import sys

    def get_git_value(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=Path.cwd(), text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    feature_imp = extract_lightgbm_feature_importance(model, features)

    return RunMetadata(
        run_id=run_dir.name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config=config_payload,
        dataset=build_dataset_metadata(dataset, train, test, features),
        training=build_training_metadata(model),
        evaluation=build_evaluation_metadata(test, predictions),
        backtest={
            **{k: round(float(v), 6) for k, v in (backtest_metrics or {}).items()},
            **asdict(build_win_rate_metadata(results, executed_trades)),
            "window_id": window_id,
            "window_train_range": window_train_range or "",
            "window_test_range": window_test_range or "",
        },
        feature_importance={
            row["feature"]: round(float(row["pct"]), 2)
            for _, row in feature_imp.iterrows()
        },
        trade_summary={
            "total_trades": len(trades_df),
            "wins": int(trades_df["win"].sum()) if len(trades_df) else 0,
            "losses": int((~trades_df["win"]).sum()) if len(trades_df) else 0,
            "avg_bars_held": round(float(trades_df["bars_held"].mean()), 1) if len(trades_df) else 0,
            "avg_pnl_usd": round(float(trades_df["trade_pnl_usd"].mean()), 2) if len(trades_df) else 0,
        },
        artifacts={"files": artifact_files, "figure_count": sum(".png" in n for n in artifact_files)},
        reproducibility={
            "python_version": sys.version.split()[0],
            "python_version_full": sys.version,
            "python_build": platform.python_build(),
            "platform": platform.platform(),
            "git_commit": get_git_value(["git", "rev-parse", "HEAD"]),
            "git_branch": get_git_value(["git", "branch", "--show-current"]),
            "git_dirty": bool(get_git_value(["git", "status", "--short"])),
            "run_entrypoint": "cli",
        },
    )


def build_date_range(frame: pl.DataFrame) -> dict[str, str]:
    ts = frame["timestamp"]
    return {"start": str(ts.min()), "end": str(ts.max())}


def build_label_counts(frame: pl.DataFrame) -> dict[str, int]:
    vc = frame["label"].value_counts()
    return {str(row["label"]): int(row["count"]) for row in vc.iter_rows(named=True)}


def build_dataset_metadata(
    dataset: pl.DataFrame,
    train: pl.DataFrame,
    test: pl.DataFrame,
    features: list[str],
) -> DatasetMeta:
    return DatasetMeta(
        total_rows=len(dataset),
        train_rows=len(train),
        test_rows=len(test),
        feature_count=len(features),
        features=features,
        fractional_d=None,
        data_range=build_date_range(dataset),
        train_date_range=build_date_range(train),
        test_date_range=build_date_range(test),
        label_distribution_total=build_label_counts(dataset),
        label_distribution_train=build_label_counts(train),
        label_distribution_test=build_label_counts(test),
    )


def build_training_metadata(model: HybridStackingSignalClassifier) -> TrainingMeta:
    return TrainingMeta(
        oof_scores=model.oof_scores_,
        per_class_oof_f1=getattr(model, "per_class_oof_", {}),
        active_models=getattr(model, "active_model_names_", []),
        filtered_models=[
            n for n in model.base_models
            if n not in getattr(model, "active_model_names_", [])
        ],
    )


def build_evaluation_metadata(test: pl.DataFrame, predictions: np.ndarray) -> EvalMeta:
    y_true = test["label"].to_numpy()
    labels = LABELS.tolist()
    return EvalMeta(
        accuracy=round(float(accuracy_score(y_true, predictions)), 6),
        f1_macro=round(float(f1_score(y_true, predictions, average="macro", zero_division=0)), 6),
        confusion_matrix={
            "labels": labels,
            "matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
        },
    )


def build_win_rate_metadata(results: pd.DataFrame, executed_trades: list[dict] | None = None) -> WinRateMeta:
    if executed_trades:
        wins = sum(1 for t in executed_trades if t.get("win", t.get("trade_pnl_usd", 0) > 0))
        total = len(executed_trades)
        win_rate = round(wins / total, 6) if total else 0.0
    else:
        pnl = results["bar_pnl_usd"]
        nonzero_pnl = pnl[pnl != 0]
        wins = float((nonzero_pnl[nonzero_pnl > 0]).sum())
        win_rate = round(wins / len(nonzero_pnl), 6) if len(nonzero_pnl) else 0.0
    trades_cnt = float(np.sum(np.diff(results["position"], prepend=0) != 0))
    return WinRateMeta(
        win_rate=win_rate,
        turnover=round(trades_cnt / len(results), 6) if len(results) else 0.0,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def publish_pipeline_results(
    accelerator: Any,
    config_payload: dict[str, Any],
    outputs: PipelineOutputs | dict[str, Any],
    window_id: int | None = None,
    window_train_range: str = "",
    window_test_range: str = "",
) -> None:
    if hasattr(outputs, "to_dict"):
        output_payload = outputs.to_dict(
            window_id=window_id,
            window_train_range=window_train_range,
            window_test_range=window_test_range,
        )
        artifact_outputs = outputs
    else:
        output_payload = dict(outputs)
        if window_id is not None:
            output_payload["window_id"] = window_id
            output_payload["window_train_range"] = window_train_range
            output_payload["window_test_range"] = window_test_range
        artifact_outputs = SimpleNamespace(
            train=output_payload["train"],
            test=output_payload["test"],
            features=output_payload["features"],
            model=output_payload["model"],
            predictions=output_payload["predictions"],
            positions=output_payload["positions"],
            backtest_metrics=output_payload.get("backtest_metrics"),
            equity=output_payload.get("equity", np.full(len(output_payload["test"]), 10_000.0)),
            executed_trades=output_payload.get("executed_trades"),
        )
    train = output_payload["train"]
    test = output_payload["test"]
    features = output_payload["features"]
    model = output_payload["model"]
    predictions = output_payload["predictions"]
    backtest_metrics = output_payload["backtest_metrics"]

    labeled_full = pl.concat([train, test])

    print_device_acceleration_report(accelerator)
    print_dataset_report(labeled_full, train, test, len(features))
    print_model_filtering_report(model)
    print_classification_report(test["label"], predictions)
    print_feature_importance_report(extract_lightgbm_feature_importance(model, features))
    print_backtest_metrics_report(backtest_metrics)

    save_run_artifacts(
        run_dir=config_payload.get("run_dir", Path("reports") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        outputs=artifact_outputs,
        config_payload=config_payload,
        window_id=window_id,
        window_train_range=window_train_range,
        window_test_range=window_test_range,
    )


def save_run_artifacts(
    run_dir: Path,
    outputs: PipelineOutputs,
    config_payload: dict[str, Any],
    window_id: int | None = None,
    window_train_range: str = "",
    window_test_range: str = "",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

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
    results.to_csv(run_dir / "predictions.csv", index=False)

    if executed_trades is not None:
        timestamps = test["timestamp"].to_numpy()
        trades_df = build_trades_dataframe(executed_trades, timestamps)
    else:
        trades_df = extract_trades_from_positions(results)
    trades_df.to_csv(run_dir / "trades.csv", index=False)

    if backtest_metrics:
        pd.Series(backtest_metrics).to_csv(run_dir / "backtest_metrics.csv")

    importance_df = save_feature_importance_csv(model, features, run_dir / "feature_importance.csv")
    save_feature_importance_bar_plot(importance_df, figures_dir / "feature_importance.png")
    save_oof_scores_bar_plot(model, figures_dir / "oof_scores.png")
    save_equity_curve_plot(equity_arr, figures_dir / "equity_curve.png")

    artifact_files = collect_artifact_files(run_dir, figures_dir) + ["run_data.json"]
    run_data = build_run_metadata(
        run_dir, model, config_payload, dataset, train, test,
        predictions, positions, results, features, backtest_metrics,
        artifact_files, trades_df, executed_trades=executed_trades,
        window_id=window_id,
        window_train_range=window_train_range,
        window_test_range=window_test_range,
    )
    with open(run_dir / "run_data.json", "w", encoding="utf-8") as f:
        json.dump(asdict(run_data), f, indent=2, ensure_ascii=False, default=str)

    print(f"\nRun dir: {run_dir.resolve()}")
    print(f"Files: predictions.csv, trades.csv ({len(trades_df)} trades), backtest_metrics.csv, feature_importance.csv, run_data.json, figures/*.png")
