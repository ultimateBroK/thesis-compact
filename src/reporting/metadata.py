"""Metadata: run metadata dataclasses and builders for JSON persistence."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.config import LABELS
from src.evaluation.metrics import compute_roc_auc
from src.models.stacking import HybridStackingSignalClassifier


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DatasetMeta:
    total_rows: int
    train_rows: int
    classification_test_rows: int
    backtest_test_rows: int
    feature_count: int
    features: list[str]
    data_range: dict[str, str]
    train_date_range: dict[str, str]
    test_date_range: dict[str, str]
    classification_test_date_range: dict[str, str]
    backtest_test_date_range: dict[str, str]
    label_distribution_total: dict[str, int]
    label_distribution_train: dict[str, int]
    label_distribution_test: dict[str, int]


@dataclass
class TrainingMeta:
    oof_scores: dict[str, float]
    per_class_oof_f1: dict[str, dict[str, float]]
    active_models: list[str]


@dataclass
class EvalMeta:
    accuracy: float
    f1_macro: float
    confusion_matrix: dict[str, Any]
    per_class_metrics: dict[str, Any]
    roc_auc: float | None = None


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


@dataclass(frozen=True)
class RunMetadataInputs:
    run_dir: Path
    model: HybridStackingSignalClassifier
    config_payload: dict[str, Any]
    dataset: pl.DataFrame
    train: pl.DataFrame
    test_labeled: pl.DataFrame
    test_continuous: pl.DataFrame
    predictions: np.ndarray
    positions: np.ndarray
    results: pd.DataFrame
    features: list[str]
    backtest_metrics: dict[str, float] | None
    artifact_files: list[str]
    trades_df: pd.DataFrame
    executed_trades: list[dict] | None = None
    pred_proba: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            args, cwd=Path.cwd(), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_date_range(frame: pl.DataFrame) -> dict[str, str]:
    ts = frame["timestamp"]
    return {"start": str(ts.min()), "end": str(ts.max())}


def build_label_counts(frame: pl.DataFrame) -> dict[str, int]:
    vc = frame["label"].value_counts()
    return {str(row["label"]): int(row["count"]) for row in vc.iter_rows(named=True)}


def build_dataset_metadata(
    dataset: pl.DataFrame,
    train: pl.DataFrame,
    test_labeled: pl.DataFrame,
    test_continuous: pl.DataFrame,
    features: list[str],
) -> DatasetMeta:
    return DatasetMeta(
        total_rows=len(dataset),
        train_rows=len(train),
        classification_test_rows=len(test_labeled),
        backtest_test_rows=len(test_continuous),
        feature_count=len(features),
        features=features,
        data_range=build_date_range(dataset),
        train_date_range=build_date_range(train),
        test_date_range=build_date_range(test_labeled),
        classification_test_date_range=build_date_range(test_labeled),
        backtest_test_date_range=build_date_range(test_continuous),
        label_distribution_total=build_label_counts(dataset),
        label_distribution_train=build_label_counts(train),
        label_distribution_test=build_label_counts(test_labeled),
    )


def build_training_metadata(model: HybridStackingSignalClassifier) -> TrainingMeta:
    return TrainingMeta(
        oof_scores=model.oof_scores_,
        per_class_oof_f1=getattr(model, "per_class_oof_", {}),
        active_models=getattr(model, "active_model_names_", []),
    )


def build_evaluation_metadata(
    test: pl.DataFrame,
    predictions: np.ndarray,
    pred_proba: np.ndarray | None = None,
) -> EvalMeta:
    from sklearn.metrics import classification_report

    y_true = test["label"].to_numpy()
    labels = list(LABELS)

    report = classification_report(
        y_true, predictions, labels=labels, output_dict=True, zero_division=0
    )
    per_class: dict[str, Any] = {}
    for lv in labels:
        k = str(int(lv))
        if k in report:
            per_class[str(lv)] = {
                "precision": round(report[k]["precision"], 6),
                "recall": round(report[k]["recall"], 6),
                "f1": round(report[k]["f1-score"], 6),
                "support": report[k]["support"],
            }

    roc_auc: float | None = None
    if pred_proba is not None and pred_proba.shape[1] >= 2:
        roc_auc = round(compute_roc_auc(y_true, pred_proba), 6)

    return EvalMeta(
        accuracy=round(float(accuracy_score(y_true, predictions)), 6),
        f1_macro=round(
            float(f1_score(y_true, predictions, average="macro", zero_division=0)), 6
        ),
        confusion_matrix={
            "labels": labels,
            "matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
        },
        per_class_metrics=per_class,
        roc_auc=roc_auc,
    )


def build_win_rate_metadata(
    results: pd.DataFrame,
    executed_trades: list[dict] | None = None,
) -> WinRateMeta:
    """Compute win rate and turnover from either trade records or bar PnL.

    Two branches: if ``executed_trades`` is provided, counts wins from trade
    records; otherwise falls back to counting positive bar-level PnL rows.
    Turnover is the fraction of bars where position changed.
    """
    if executed_trades:
        wins = sum(
            1 for t in executed_trades if t.get("win", t.get("trade_pnl_usd", 0) > 0)
        )
        total = len(executed_trades)
        win_rate = round(wins / total, 6) if total else 0.0
    else:
        pnl = results["bar_pnl_usd"]
        nonzero_pnl = pnl[pnl != 0]
        wins = int((nonzero_pnl > 0).sum())
        win_rate = round(wins / len(nonzero_pnl), 6) if len(nonzero_pnl) else 0.0
    position_col = "executed_position" if "executed_position" in results else "position"
    trades_cnt = float(np.sum(np.diff(results[position_col], prepend=0) != 0))
    return WinRateMeta(
        win_rate=win_rate,
        turnover=round(trades_cnt / len(results), 6) if len(results) else 0.0,
    )


def collect_artifact_files(
    run_dir: Path, figures_dir: Path, tables_dir: Path | None = None
) -> list[str]:
    root_files = [f.name for f in run_dir.iterdir() if f.is_file()]
    fig_files = [f"figures/{f.name}" for f in figures_dir.iterdir() if f.is_file()]
    tbl_files = (
        [f"tables/{f.name}" for f in tables_dir.iterdir() if f.is_file()]
        if tables_dir
        else []
    )
    return sorted(root_files + fig_files + tbl_files)


def build_trade_summary(
    trades_df: pd.DataFrame, positions: np.ndarray | None = None
) -> dict[str, Any]:
    """Build a summary dict from a trades DataFrame.

    Includes trade-level stats (win rate, avg PnL, avg bars held, long/short
    counts) and, when ``positions`` is provided, bar-level exposure stats
    (long/short bar counts and percentages). Bar-level exposure measures
    signal distribution across all bars; trade-level stats measure per-trade
    outcomes.
    """
    summary = {
        "total_trades": len(trades_df),
        "wins": int(trades_df["win"].sum()) if len(trades_df) else 0,
        "losses": int((~trades_df["win"]).sum()) if len(trades_df) else 0,
        "win_rate": round(float(trades_df["win"].mean()), 4) if len(trades_df) else 0.0,
        "avg_bars_held": round(float(trades_df["bars_held"].mean()), 1)
        if len(trades_df)
        else 0,
        "avg_pnl_usd": round(float(trades_df["trade_pnl_usd"].mean()), 2)
        if len(trades_df)
        else 0.0,
        "avg_win_pnl_usd": round(
            float(trades_df.loc[trades_df["win"], "trade_pnl_usd"].mean()), 2
        )
        if len(trades_df) and trades_df["win"].any()
        else 0.0,
        "avg_loss_pnl_usd": round(
            float(trades_df.loc[~trades_df["win"], "trade_pnl_usd"].mean()), 2
        )
        if len(trades_df) and (~trades_df["win"]).any()
        else 0.0,
        "max_win_usd": round(float(trades_df["trade_pnl_usd"].max()), 2)
        if len(trades_df)
        else 0.0,
        "max_loss_usd": round(float(trades_df["trade_pnl_usd"].min()), 2)
        if len(trades_df)
        else 0.0,
        "long_trades": int((trades_df["direction"] == "LONG").sum())
        if "direction" in trades_df.columns and len(trades_df)
        else 0,
        "short_trades": int((trades_df["direction"] == "SHORT").sum())
        if "direction" in trades_df.columns and len(trades_df)
        else 0,
    }
    # Bar-level exposure (not trade-level); Buy/Sell signal policy has no Flat.
    if positions is not None and len(positions) > 0:
        pos = np.asarray(positions, dtype=np.int64)
        total_bars = len(pos)
        long_bars = int((pos > 0).sum())
        short_bars = int((pos < 0).sum())
        summary["long_bar_count"] = long_bars
        summary["short_bar_count"] = short_bars
        summary["long_exposure_pct"] = (
            round(long_bars / total_bars * 100, 2) if total_bars else 0.0
        )
        summary["short_exposure_pct"] = (
            round(short_bars / total_bars * 100, 2) if total_bars else 0.0
        )
    return summary


def build_feature_importance_map(
    model: HybridStackingSignalClassifier,
    features: list[str],
) -> dict[str, float]:
    from src.evaluation.importance import extract_lightgbm_feature_importance

    df = extract_lightgbm_feature_importance(model, features)
    return {row["feature"]: round(float(row["pct"]), 2) for _, row in df.iterrows()}


def build_run_metadata_from_inputs(inputs: RunMetadataInputs) -> RunMetadata:
    return RunMetadata(
        run_id=inputs.run_dir.name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config=inputs.config_payload,
        dataset=build_dataset_metadata(
            inputs.dataset,
            inputs.train,
            inputs.test_labeled,
            inputs.test_continuous,
            inputs.features,
        ),
        training=build_training_metadata(inputs.model),
        evaluation=build_evaluation_metadata(
            inputs.test_labeled, inputs.predictions, inputs.pred_proba
        ),
        backtest={
            **{
                key: round(float(value), 6)
                for key, value in (inputs.backtest_metrics or {}).items()
            },
            **asdict(build_win_rate_metadata(inputs.results, inputs.executed_trades)),
        },
        feature_importance=build_feature_importance_map(inputs.model, inputs.features),
        trade_summary=build_trade_summary(inputs.trades_df, inputs.positions),
        artifacts={
            "files": inputs.artifact_files,
            "figure_count": sum(".png" in name for name in inputs.artifact_files),
        },
        reproducibility={
            "python_version": sys.version.split()[0],
            "python_version_full": sys.version,
            "python_build": platform.python_build(),
            "platform": platform.platform(),
            "git_commit": _get_git_value(["git", "rev-parse", "HEAD"]),
            "git_branch": _get_git_value(["git", "branch", "--show-current"]),
            "git_dirty": bool(_get_git_value(["git", "status", "--short"])),
            "run_entrypoint": "cli",
        },
    )
