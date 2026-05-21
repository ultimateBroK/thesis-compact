from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATA_DIR = Path("data/raw/XAUUSD")
TIMEFRAME = "1h"
CV_SPLITS = 5
EMBARGO_PCT = 0.02
MIN_OOF_F1 = 0.34
LABELS = np.array([-1, 0, 1])


@dataclass(frozen=True)
class TradingCosts:
    slippage_points: float = 0.03
    spread_multiplier: float = 1.0


@dataclass(frozen=True)
class PipelineConfig:
    months: int | None = 12
