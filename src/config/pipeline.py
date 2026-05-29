from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class PipelineConfig:
    months: int | None = 12
    timeframe: str = "1h"
    walk_forward: bool = False
    n_windows: int = 3