from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATA_DIR = Path("data/XAUUSD")
REPORT_DIR = Path("reports")
TIMEFRAME = "1h"
FRACTIONAL_D = 0.4
CV_SPLITS = 5
EMBARGO_PCT = 0.02
PURGE_PCT = 0.02
MIN_OOF_F1 = 0.36
CONFIDENCE_THRESHOLD = 0.15
USE_META_LABELING = False
META_LABEL_THRESHOLD = 0.35
RANDOM_STATE = 42
SWING_WINDOW = 5
LABELING_HORIZON = 12
FALLBACK_TP_ATR = 2.0
FALLBACK_SL_ATR = 1.5
LABELS = np.array([-1, 0, 1])
INITIAL_BALANCE = 10_000.0
CONTRACT_SIZE = 100.0
FIXED_LOTS = 0.03


@dataclass(frozen=True)
class TradingCosts:
    slippage_points: float = 0.03
    spread_multiplier: float = 1.0


@dataclass(frozen=True)
class PipelineConfig:
    months: int | None = 12
