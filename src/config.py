"""Cấu hình: đường dẫn, chia dữ liệu, mô hình, nhãn và tham số kiểm thử lịch sử."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

# ── Hằng số đường dẫn ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/XAUUSD"
REPORT_DIR = PROJECT_ROOT / "reports"

# ── Tham số pipeline ─────────────────────────────────────────────
TIMEFRAME = "1h"
CV_SPLITS = 5
TEST_SIZE = 0.20

# ── Tham số gán nhãn ─────────────────────────────────────────────
SELL_LABEL = -1
BUY_LABEL = 1
LABELS = (SELL_LABEL, BUY_LABEL)
LABELING_METHOD = "fixed_horizon"
LABELING_HORIZON = 4
LABEL_RETURN_THRESHOLD = 0.0005  # bỏ mẫu có |return| <= 0.05%
MAX_LABEL_GAP_HOURS = LABELING_HORIZON + 1  # lọc gap bất thường trong dữ liệu tick

# ── Cửa sổ đặc trưng ─────────────────────────────────────────────
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

# ── Khoảng giữ vị thế trong backtest ────────────────────────────
BACKTEST_HOLD_BARS = LABELING_HORIZON  # giữ mỗi tín hiệu trọn label horizon

# ── Purge gap ───────────────────────────────────────────────────
PURGE_BARS = LABELING_HORIZON  # purge gap bằng label horizon để tránh rò rỉ nhãn

# ── Tham số mô hình ──────────────────────────────────────────────
RANDOM_STATE = 42

# ── Tham số backtest ─────────────────────────────────────────────
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
