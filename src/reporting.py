"""Reporting: pipeline result publishing (thin orchestrator).

All logic lives in focused modules:
  - src/console.py    → console printers
  - src/metadata.py   → run metadata dataclasses & builders
  - src/artifacts.py  → CSV/JSON/PNG persistence
"""

from __future__ import annotations

from datetime import datetime
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
) -> None:
    if hasattr(outputs, "as_dict"):
        output_payload = outputs.as_dict()
        artifact_outputs = outputs
    else:
        output_payload = dict(outputs)
        artifact_outputs = SimpleNamespace(
            train=output_payload["train"],
            test=output_payload["test"],
            features=output_payload["features"],
            model=output_payload["model"],
            predictions=output_payload["predictions"],
            positions=output_payload["positions"],
            backtest_metrics=output_payload.get("backtest_metrics"),
            equity=output_payload.get(
                "equity", np.full(len(output_payload["test"]), 10_000.0)
            ),
            executed_trades=output_payload.get("executed_trades"),
            pred_proba=output_payload.get("pred_proba"),
        )
    train = output_payload["train"]
    test = output_payload["test"]
    features = output_payload["features"]
    model = output_payload["model"]
    predictions = output_payload["predictions"]
    backtest_metrics = output_payload["backtest_metrics"]

    labeled_full = pl.concat([train, test])

    print_dataset_report(labeled_full, train, test, len(features))
    print_base_model_oof_report(model)
    print_classification_report(test["label"], predictions)
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
    )
