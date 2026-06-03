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
MIN_OOF_F1 = 0.50  # random baseline floor (binary, balanced)
CONFIDENCE_THRESHOLD = 0.45
RANDOM_STATE = 42
LABELS = np.array([-1, 1])

# ── Backtest parameters ──────────────────────────────────────────
INITIAL_BALANCE = 10_000.0
CONTRACT_SIZE = 100.0          # XAUUSD CFD: 1 lot = 100 oz
RISK_PER_TRADE = 0.02
SHORT_LOT_SCALE = 0.20        # SHORT positions sized at 20% of LONG (neutral bet sizing)
TUNE_TP_RANGE_BT = (3.0, 15.0, 1.0)   # grid search (start, stop, step) — wide barriers > tight
TUNE_SL_RANGE_BT = (3.0, 15.0, 1.0)
TUNE_HOLD_VALUES = [6, 8, 12, 16]
N_TUNING_TRIALS_APPROX = 700          # ≈ number of grid combos tried (for DSR deflation)

# ── CFD execution model ─────────────────────────────────────────
LEVERAGE = 100                 # 1:100 → 1% margin
LOT_STEP = 0.01                # broker minimum lot increment
LOT_MIN = 0.01
LOT_MAX = 5.0                  # soft cap; margin check enforces hard cap
COMMISSION_PER_LOT_SIDE = 0.0  # USD per lot per side (0 → spread-only broker)
SWAP_LONG_USD_PER_LOT = -2.50  # USD per lot per overnight (negative = charge)
SWAP_SHORT_USD_PER_LOT = -1.00 # USD per lot per overnight

# ── Labeling parameters ──────────────────────────────────────────
USE_META_LABELING = True
META_LABEL_THRESHOLD = 0.55
SHORT_META_LABEL_THRESHOLD = 0.55
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
    timeframe: str = TIMEFRAME
    walk_forward: bool = False
    long_only: bool = False
    n_windows: int = 3
