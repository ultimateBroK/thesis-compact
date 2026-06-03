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
PURGE_PCT = 0.02
TEST_SIZE = 0.20

# ── Labeling parameters ──────────────────────────────────────────
LABELING_HORIZON = 4

# ── Model parameters ─────────────────────────────────────────────
MIN_OOF_F1 = 0.0  # reporting only; all base models stay in stacking
SIGNAL_PROBABILITY_THRESHOLD = 0.55
RANDOM_STATE = 42
LABELS = np.array([-1, 1])

# ── Backtest parameters ──────────────────────────────────────────
INITIAL_BALANCE = 10_000.0


# ── PipelineConfig dataclass ─────────────────────────────────────
@dataclass(frozen=True)
class PipelineConfig:
    months: int | None = 12
    timeframe: str = TIMEFRAME
    walk_forward: bool = False
    long_only: bool = False
    n_windows: int = 3
