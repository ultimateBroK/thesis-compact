"""Reporting: pipeline result publishing (thin orchestrator).

All logic lives in focused modules:
  - src/console.py    → console printers
  - src/metadata.py   → run metadata dataclasses & builders
  - src/artifacts.py  → CSV/JSON/PNG persistence
"""

from __future__ import annotations

from datetime import datetime
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl

from src.config import REPORT_DIR
from src.console import (
    print_backtest_metrics_report,
    print_base_model_oof_report,
    print_classification_report,
    print_dataset_report,
    print_feature_importance_report,
)
from src.artifacts import (
    extract_lightgbm_feature_importance,
    save_run_artifacts,
)


def publish_pipeline_results(
    config_payload: dict[str, Any],
    outputs,
    timing: dict[str, float] | None = None,
    total_start: float | None = None,
) -> None:
    if hasattr(outputs, "as_dict"):
        output_payload = outputs.as_dict()
        artifact_outputs = outputs
    else:
        output_payload = dict(outputs)
        test_labeled = output_payload.get("test_labeled", output_payload.get("test"))
        test_continuous = output_payload.get("test_continuous", test_labeled)
        artifact_outputs = SimpleNamespace(
            train=output_payload["train"],
            test_labeled=test_labeled,
            test_continuous=test_continuous,
            features=output_payload["features"],
            model=output_payload["model"],
            predictions=output_payload["predictions"],
            raw_signals=output_payload.get("raw_signals", output_payload["positions"]),
            positions=output_payload["positions"],
            backtest_metrics=output_payload.get("backtest_metrics"),
            equity=output_payload.get(
                "equity", np.full(len(test_continuous), 10_000.0)
            ),
            executed_trades=output_payload.get("executed_trades"),
            pred_proba=output_payload.get("pred_proba"),
        )
    if "test_labeled" not in output_payload:
        output_payload["test_labeled"] = output_payload["test"]
    if "test_continuous" not in output_payload:
        output_payload["test_continuous"] = output_payload["test_labeled"]
    train = output_payload["train"]
    test_labeled = output_payload["test_labeled"]
    test_continuous = output_payload["test_continuous"]
    features = output_payload["features"]
    model = output_payload["model"]
    predictions = output_payload["predictions"]
    backtest_metrics = output_payload["backtest_metrics"]

    labeled_full = pl.concat([train, test_labeled])

    report_start = time.perf_counter()
    print_dataset_report(
        labeled_full, train, test_labeled, test_continuous, len(features)
    )
    print_base_model_oof_report(model)
    print_classification_report(test_labeled["label"], predictions)
    print_feature_importance_report(
        extract_lightgbm_feature_importance(model, features)
    )
    print_backtest_metrics_report(backtest_metrics)

    save_run_artifacts(
        run_dir=config_payload.get(
            "run_dir", REPORT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ),
        outputs=artifact_outputs,
        config_payload=config_payload,
        timing=timing,
        report_start=report_start,
        total_start=total_start,
    )
