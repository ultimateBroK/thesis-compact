"""Artifacts: CSV/JSON/PNG persistence for pipeline outputs."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from src.feature_importance import extract_lightgbm_feature_importance

from src.metadata import (
    RunMetadataInputs,
    build_run_metadata_from_inputs,
    collect_artifact_files,
)
from src.metrics import save_baseline_metrics_csv
from src.plotting import (
    save_baseline_comparison_figure,
    save_confusion_matrix_figure,
    save_equity_vs_buyhold_figure,
    save_feature_importance_bar_plot,
    save_label_distribution_figure,
    save_oof_scores_bar_plot,
    save_pipeline_overview_figure,
    save_position_exposure_figure,
    save_train_test_split_figure,
)
from src.trades import build_trades_dataframe, extract_trades_from_positions
from src.models import HybridStackingSignalClassifier




def save_feature_importance_csv(
    model: HybridStackingSignalClassifier,
    features: list[str],
    path: Path,
) -> pd.DataFrame:
    df = extract_lightgbm_feature_importance(model, features)
    df.to_csv(path)
    return df



# ---------------------------------------------------------------------------
# Main save function
# ---------------------------------------------------------------------------


def build_predictions_results(
    test_labeled: pl.DataFrame,
    test_continuous: pl.DataFrame,
    predictions: np.ndarray,
    raw_signals: np.ndarray,
    positions: np.ndarray,
    equity: np.ndarray,
) -> pd.DataFrame:
    """Build per-bar backtest results and align labeled predictions by timestamp."""
    labeled_predictions = test_labeled.select(["timestamp", "label"]).with_columns(
        pl.Series("prediction", predictions.astype(np.int64))
    )
    bar_pnl = np.diff(equity, prepend=equity[0])
    results = (
        test_continuous.select(["timestamp", "close", "spread"])
        .join(labeled_predictions, on="timestamp", how="left")
        .with_columns(
            [
                pl.Series("raw_signal", raw_signals.astype(np.int64)),
                pl.Series("executed_position", positions.astype(np.int64)),
                pl.Series("bar_pnl_usd", bar_pnl),
                pl.Series("equity_usd", equity),
            ]
        )
    )
    return results.to_pandas()


def _prepare_run_dirs(run_dir: Path) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)
    return figures_dir, tables_dir


def _write_predictions_table(outputs, tables_dir: Path) -> pd.DataFrame:
    results = build_predictions_results(
        outputs.test_labeled,
        outputs.test_continuous,
        outputs.predictions,
        outputs.raw_signals,
        outputs.positions,
        outputs.equity,
    )
    results.to_csv(tables_dir / "predictions.csv", index=False)
    return results


def _write_trades_table(outputs, results: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    if outputs.executed_trades is not None:
        timestamps = outputs.test_continuous["timestamp"].to_numpy()
        trades_df = build_trades_dataframe(outputs.executed_trades, timestamps)
    else:
        trades_df = extract_trades_from_positions(results)
    trades_df.to_csv(tables_dir / "trades.csv", index=False)
    return trades_df


def _write_backtest_metrics_table(outputs, tables_dir: Path) -> None:
    if outputs.backtest_metrics:
        pd.DataFrame([outputs.backtest_metrics]).to_csv(
            tables_dir / "backtest_metrics.csv", index=False
        )


def _write_baseline_metrics_table(outputs, tables_dir: Path) -> pd.DataFrame:
    baseline_metrics_df = save_baseline_metrics_csv(
        outputs.model,
        outputs.train,
        outputs.test_labeled,
        outputs.features,
        outputs.predictions,
        getattr(outputs, "pred_proba", None),
        tables_dir / "baseline_metrics.csv",
    )
    print("\n=== BASELINE TEST METRICS ===")
    print(baseline_metrics_df.to_string(index=False))
    return baseline_metrics_df


def _save_thesis_figures(
    outputs,
    baseline_metrics_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    figures_dir: Path,
    config_payload: dict[str, Any],
) -> None:
    save_pipeline_overview_figure(
        figures_dir / "fig1_pipeline.png",
        feature_count=len(outputs.features),
        labeling_horizon=int(config_payload["labeling_horizon"]),
        timeframe=str(config_payload["timeframe"]),
        base_model_names=list(outputs.model.active_model_names_),
        meta_model_name=outputs.model.meta_model.__class__.__name__,
    )
    save_train_test_split_figure(
        outputs.train,
        outputs.test_labeled,
        figures_dir / "fig2_split.png",
        purge_bars=int(config_payload["purge_bars"]),
    )
    save_label_distribution_figure(
        outputs.train,
        outputs.test_labeled,
        figures_dir / "fig3_labels.png",
        labels=outputs.model.labels,
    )
    save_baseline_comparison_figure(
        baseline_metrics_df, figures_dir / "fig4_baselines.png"
    )
    save_confusion_matrix_figure(
        outputs.test_labeled["label"].to_numpy(),
        outputs.predictions,
        figures_dir / "fig5_confusion.png",
        labels=outputs.model.labels,
    )
    save_equity_vs_buyhold_figure(
        outputs.equity,
        outputs.test_continuous["close"].to_numpy().astype(np.float64),
        figures_dir / "fig6_equity.png",
    )
    save_position_exposure_figure(outputs.positions, figures_dir / "fig7_exposure.png")
    save_feature_importance_bar_plot(importance_df, figures_dir / "fig8_importance.png")
    save_oof_scores_bar_plot(outputs.model, figures_dir / "fig9_oof_scores.png")


def _with_timing_payload(
    config_payload: dict[str, Any],
    timing: dict[str, float] | None,
    report_start: float | None,
    total_start: float | None,
) -> dict[str, Any]:
    if timing is None:
        return config_payload
    if report_start is not None:
        timing["reporting"] = time.perf_counter() - report_start
    if total_start is not None:
        timing["total"] = time.perf_counter() - total_start
    return {
        **config_payload,
        "timing": {key: value for key, value in timing.items() if value > 0.0},
    }


def _write_run_metadata(
    run_dir: Path,
    figures_dir: Path,
    tables_dir: Path,
    outputs,
    config_payload: dict[str, Any],
    results: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> None:
    artifact_files = collect_artifact_files(run_dir, figures_dir, tables_dir) + [
        "run_data.json"
    ]
    run_data = build_run_metadata_from_inputs(
        RunMetadataInputs(
            run_dir=run_dir,
            model=outputs.model,
            config_payload=config_payload,
            dataset=pl.concat([outputs.train, outputs.test_labeled]),
            train=outputs.train,
            test_labeled=outputs.test_labeled,
            test_continuous=outputs.test_continuous,
            predictions=outputs.predictions,
            positions=outputs.positions,
            results=results,
            features=outputs.features,
            backtest_metrics=outputs.backtest_metrics,
            artifact_files=artifact_files,
            trades_df=trades_df,
            executed_trades=outputs.executed_trades,
            pred_proba=getattr(outputs, "pred_proba", None),
        )
    )
    with open(run_dir / "run_data.json", "w", encoding="utf-8") as f:
        json.dump(asdict(run_data), f, indent=2, ensure_ascii=False, default=str)


def _print_artifact_summary(run_dir: Path, trades_df: pd.DataFrame) -> None:
    print(f"\nRun dir: {run_dir.resolve()}")
    print(
        f"Files: tables/predictions.csv, tables/trades.csv ({len(trades_df)} trades), "
        "tables/backtest_metrics.csv, tables/baseline_metrics.csv, "
        "tables/feature_importance.csv, run_data.json, figures/*.png"
    )


def save_run_artifacts(
    run_dir: Path,
    outputs,
    config_payload: dict[str, Any],
    timing: dict[str, float] | None = None,
    report_start: float | None = None,
    total_start: float | None = None,
) -> None:
    figures_dir, tables_dir = _prepare_run_dirs(run_dir)

    results = _write_predictions_table(outputs, tables_dir)
    trades_df = _write_trades_table(outputs, results, tables_dir)
    _write_backtest_metrics_table(outputs, tables_dir)

    baseline_metrics_df = _write_baseline_metrics_table(outputs, tables_dir)
    importance_df = save_feature_importance_csv(
        outputs.model, outputs.features, tables_dir / "feature_importance.csv"
    )
    _save_thesis_figures(
        outputs, baseline_metrics_df, importance_df, figures_dir, config_payload
    )

    config_payload = _with_timing_payload(
        config_payload,
        timing,
        report_start,
        total_start,
    )
    _write_run_metadata(
        run_dir,
        figures_dir,
        tables_dir,
        outputs,
        config_payload,
        results,
        trades_df,
    )
    _print_artifact_summary(run_dir, trades_df)
