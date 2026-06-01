from __future__ import annotations

from dataclasses import dataclass

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
