"""Configuration: paths, data split, model, label, and backtest parameters."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

# ── Path constants ────────────────────────────────────────────────
DATA_DIR = Path("data/XAUUSD")
REPORT_DIR = Path("reports")

# ── Pipeline parameters ──────────────────────────────────────────
TIMEFRAME = "1h"
CV_SPLITS = 5
TEST_SIZE = 0.20

# ── Labeling parameters ──────────────────────────────────────────
SELL_LABEL = -1
BUY_LABEL = 1
LABELS = (SELL_LABEL, BUY_LABEL)
LABELING_METHOD = "fixed_horizon"
LABELING_HORIZON = 4
LABEL_RETURN_THRESHOLD = 0.0005  # drop samples with |return| <= 0.05%
MAX_LABEL_GAP_HOURS = LABELING_HORIZON + 1  # filter gaps in tick data

# ── Feature windows ──────────────────────────────────────────────
RETURN_SHORT_WINDOW = 4
RETURN_LONG_WINDOW = 12
EMA_FAST_WINDOW = 12
EMA_SLOW_WINDOW = 26
RSI_WINDOW = 14
ADX_WINDOW = 14
ATR_WINDOW = 14
BB_WINDOW = 20
VOL_SHORT_WINDOW = 6
VOL_LONG_WINDOW = 24
SPREAD_Z_WINDOW = 24
RANGE_WINDOW = 24
OBV_DELTA_WINDOW = 12
OBV_Z_WINDOW = 48
TICK_COUNT_Z_WINDOW = 24
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7

# ── Backtest position horizon ────────────────────────────────────
BACKTEST_HOLD_BARS = LABELING_HORIZON  # hold each signal for full label horizon

# ── Purge ────────────────────────────────────────────────────────
PURGE_BARS = LABELING_HORIZON  # purge gap = labeling horizon to prevent label leakage

# ── Model parameters ─────────────────────────────────────────────
RANDOM_STATE = 42

# ── Backtest parameters ──────────────────────────────────────────
INITIAL_BALANCE = 10_000.0
TRADING_DAYS_PER_YEAR = 252
ANNUALIZATION_BARS_PER_DAY = 24
ANNUALIZATION_FACTOR = sqrt(TRADING_DAYS_PER_YEAR * ANNUALIZATION_BARS_PER_DAY)


@dataclass(frozen=True)
class PipelineConfig:
    months: int | None = 12
    timeframe: str = TIMEFRAME
    test_size: float = TEST_SIZE
    cv_splits: int = CV_SPLITS
    purge_bars: int = PURGE_BARS
    random_state: int = RANDOM_STATE
    labels: tuple[int, int] = LABELS
    initial_balance: float = INITIAL_BALANCE
    labeling_horizon: int = LABELING_HORIZON
    label_return_threshold: float = LABEL_RETURN_THRESHOLD
    max_label_gap_hours: float = MAX_LABEL_GAP_HOURS
    backtest_hold_bars: int = BACKTEST_HOLD_BARS
