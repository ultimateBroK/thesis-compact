"""Reporting: backward-compatible re-exports.

All reporting logic has been split into focused modules:
  - src/console.py    → console printers
  - src/metrics.py    → classification metrics & baseline comparison
  - src/artifacts.py  → CSV/JSON/PNG persistence & trade extraction
  - src/metadata.py   → run metadata dataclasses & builders

This file re-exports the public API so existing code continues to work.
"""

from src.console import (
    print_backtest_metrics_report,
    print_base_model_oof_report,
    print_classification_report,
    print_dataset_report,
    print_feature_importance_report,
)
from src.metrics import (
    build_baseline_metrics_dataframe,
    build_classification_metric_row,
    compute_roc_auc,
    save_baseline_metrics_csv,
)
from src.artifacts import (
    build_trades_dataframe,
    collect_artifact_files,
    extract_lightgbm_feature_importance,
    extract_trades_from_positions,
    save_equity_curve_plot,
    save_feature_importance_bar_plot,
    save_feature_importance_csv,
    save_oof_scores_bar_plot,
    save_run_artifacts,
)
from src.metadata import (
    DatasetMeta,
    EvalMeta,
    RunMetadata,
    TrainingMeta,
    WinRateMeta,
    build_date_range,
    build_dataset_metadata,
    build_evaluation_metadata,
    build_label_counts,
    build_run_metadata,
    build_training_metadata,
    build_win_rate_metadata,
)

__all__ = [
    # console
    "print_backtest_metrics_report",
    "print_base_model_oof_report",
    "print_classification_report",
    "print_dataset_report",
    "print_feature_importance_report",
    # metrics
    "build_baseline_metrics_dataframe",
    "build_classification_metric_row",
    "compute_roc_auc",
    "save_baseline_metrics_csv",
    # artifacts
    "build_trades_dataframe",
    "collect_artifact_files",
    "extract_lightgbm_feature_importance",
    "extract_trades_from_positions",
    "save_equity_curve_plot",
    "save_feature_importance_bar_plot",
    "save_feature_importance_csv",
    "save_oof_scores_bar_plot",
    "save_run_artifacts",
    # metadata
    "DatasetMeta",
    "EvalMeta",
    "RunMetadata",
    "TrainingMeta",
    "WinRateMeta",
    "build_date_range",
    "build_dataset_metadata",
    "build_evaluation_metadata",
    "build_label_counts",
    "build_run_metadata",
    "build_training_metadata",
    "build_win_rate_metadata",
]
