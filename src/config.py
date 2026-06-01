"""Configuration: paths, pipeline parameters, model parameters, backtest parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── Path constants ────────────────────────────────────────────────
DATA_DIR = Path("data/XAUUSD")
REPORT_DIR = Path("reports")

# ── Pipeline parameters ──────────────────────────────────────────
TIMEFRAME = "1h"
FRACTIONAL_D = 0.4
CV_SPLITS = 5
EMBARGO_PCT = 0.02
PURGE_PCT = 0.02
TEST_SIZE = 0.20

# ── Model parameters ─────────────────────────────────────────────
MIN_OOF_F1 = 0.36
CONFIDENCE_THRESHOLD = 0.35
RANDOM_STATE = 42
LABELS = np.array([-1, 1])

# ── Backtest parameters ──────────────────────────────────────────
BACKTEST_TP_ATR = 1.5
BACKTEST_SL_ATR = 1.0
MIN_POSITION_HOLD = 24
INITIAL_BALANCE = 10_000.0
CONTRACT_SIZE = 100.0
RISK_PER_TRADE = 0.01
TUNE_TP_RANGE_BT = (1.5, 1.5, 0.5)
TUNE_SL_RANGE_BT = (1.0, 1.0, 0.5)
TUNE_HOLD_VALUES = [24]

# ── Backtest tuning ──────────────────────────────────────────────
USE_BACKTEST_TUNING = False

# ── Labeling parameters ──────────────────────────────────────────
USE_META_LABELING = True
META_LABEL_THRESHOLD = 0.55
SHORT_META_LABEL_THRESHOLD = 0.60
BB_WIDTH_MIN_MULT = 1.2
SWING_WINDOW = 5
LABELING_HORIZON = 24
TUNE_TP_RANGE = (0.5, 4.0, 0.25)
TUNE_SL_RANGE = (0.5, 4.0, 0.25)
TUNE_TARGET_BALANCE = 0.35
ADX_THRESHOLD = 20.0
TREND_FILTER_ENABLED = True
TREND_EMA_PERIOD = 89


# ── PipelineConfig dataclass ─────────────────────────────────────
@dataclass(frozen=True)
class PipelineConfig:
    months: int | None = 12
    timeframe: str = "1h"
    walk_forward: bool = False
    long_only: bool = False
    n_windows: int = 3
    backtest_tp_atr: float = 1.5
    backtest_sl_atr: float = 1.0
    min_position_hold: int = 24
    tune_backtest: bool = True
