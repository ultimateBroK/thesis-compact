"""Reporting: pipeline result publishing."""

from __future__ import annotations

import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Protocol

import numpy as np
import polars as pl

from src.artifacts import save_run_artifacts
from src.config import REPORT_DIR
from src.console import (
    print_backtest_metrics_report,
    print_base_model_oof_report,
    print_classification_report,
    print_dataset_report,
    print_feature_importance_report,
)
from src.feature_importance import extract_lightgbm_feature_importance


class PipelineResultBundle(Protocol):
    train: pl.DataFrame
    test_labeled: pl.DataFrame
    test_continuous: pl.DataFrame
    features: list[str]
    model: Any
    predictions: np.ndarray
    raw_signals: np.ndarray
    positions: np.ndarray
    backtest_metrics: dict[str, float]
    equity: np.ndarray
    executed_trades: list[dict] | None
    pred_proba: np.ndarray | None


def _outputs_to_mapping(outputs: PipelineResultBundle | dict[str, Any]) -> dict[str, Any]:
    if hasattr(outputs, "as_mapping"):
        return outputs.as_mapping()
    if hasattr(outputs, "as_dict"):
        return outputs.as_dict()
    return dict(outputs)


def _normalize_artifact_outputs(
    outputs: PipelineResultBundle | dict[str, Any],
    output_payload: dict[str, Any],
) -> PipelineResultBundle:
    if not isinstance(outputs, dict):
        return outputs

    test_labeled = output_payload.get("test_labeled", output_payload.get("test"))
    test_continuous = output_payload.get("test_continuous", test_labeled)
    return SimpleNamespace(
        train=output_payload["train"],
        test_labeled=test_labeled,
        test_continuous=test_continuous,
        features=output_payload["features"],
        model=output_payload["model"],
        predictions=output_payload["predictions"],
        raw_signals=output_payload.get("raw_signals", output_payload["positions"]),
        positions=output_payload["positions"],
        backtest_metrics=output_payload.get("backtest_metrics"),
        equity=output_payload.get("equity", np.full(len(test_continuous), 10_000.0)),
        executed_trades=output_payload.get("executed_trades"),
        pred_proba=output_payload.get("pred_proba"),
    )


def _canonical_output_payload(output_payload: dict[str, Any]) -> dict[str, Any]:
    if "test_labeled" not in output_payload:
        output_payload["test_labeled"] = output_payload["test"]
    if "test_continuous" not in output_payload:
        output_payload["test_continuous"] = output_payload["test_labeled"]
    return output_payload


def _print_reports(output_payload: dict[str, Any]) -> None:
    train = output_payload["train"]
    test_labeled = output_payload["test_labeled"]
    test_continuous = output_payload["test_continuous"]
    features = output_payload["features"]
    model = output_payload["model"]
    predictions = output_payload["predictions"]

    labeled_full = pl.concat([train, test_labeled])
    print_dataset_report(
        labeled_full, train, test_labeled, test_continuous, len(features)
    )
    print_base_model_oof_report(model)
    print_classification_report(test_labeled["label"], predictions)
    print_feature_importance_report(
        extract_lightgbm_feature_importance(model, features)
    )
    print_backtest_metrics_report(output_payload["backtest_metrics"])


def _save_artifacts(
    config_payload: dict[str, Any],
    artifact_outputs: PipelineResultBundle,
    timing: dict[str, float] | None,
    report_start: float,
    total_start: float | None,
) -> None:
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


def publish_pipeline_results(
    config_payload: dict[str, Any],
    outputs: PipelineResultBundle | dict[str, Any],
    timing: dict[str, float] | None = None,
    total_start: float | None = None,
) -> None:
    output_payload = _canonical_output_payload(_outputs_to_mapping(outputs))
    artifact_outputs = _normalize_artifact_outputs(outputs, output_payload)

    report_start = time.perf_counter()
    _print_reports(output_payload)
    _save_artifacts(
        config_payload,
        artifact_outputs,
        timing,
        report_start,
        total_start,
    )