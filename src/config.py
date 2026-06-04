"""Configuration: paths, data split, model, label, and simple backtest parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── Path constants ────────────────────────────────────────────────
DATA_DIR = Path("data/XAUUSD")
REPORT_DIR = Path("reports")

# ── Pipeline parameters ──────────────────────────────────────────
TIMEFRAME = "1h"
CV_SPLITS = 5
EMBARGO_PCT = 0.02
TEST_SIZE = 0.20

# ── Labeling parameters ──────────────────────────────────────────
LABELING_HORIZON = 4
LABEL_RETURN_THRESHOLD = 0.0005  # drop samples with |return| <= 0.05%
MAX_LABEL_GAP_HOURS = LABELING_HORIZON + 1  # filter gaps in tick data

# ── Backtest position horizon (must come after LABELING_HORIZON) ─
BACKTEST_HOLD_BARS = LABELING_HORIZON  # hold each signal for the full label horizon

# ── Purge (must come after LABELING_HORIZON) ─────────────────────
PURGE_BARS = LABELING_HORIZON  # purge gap = labeling horizon to prevent label leakage

# ── Model parameters ─────────────────────────────────────────────
MIN_OOF_F1 = 0.0  # reporting only; all base models stay in stacking
SIGNAL_PROBABILITY_THRESHOLD = 0.50
SIGNAL_PROBABILITY_MARGIN = 0.02  # minimum P(Buy)-P(Sell) edge to open position
RANDOM_STATE = 42
LABELS = np.array([-1, 1])

# ── Backtest parameters ──────────────────────────────────────────
INITIAL_BALANCE = 10_000.0


# ── PipelineConfig dataclass ─────────────────────────────────────
@dataclass(frozen=True)
class PipelineConfig:
    months: int | None = 12
    timeframe: str = TIMEFRAME
    long_only: bool = False
