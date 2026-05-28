from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradingCosts:
    slippage_points: float = 0.03
    spread_multiplier: float = 1.0


@dataclass(frozen=True)
class PipelineConfig:
    months: int | None = 12
    timeframe: str = "1h"
