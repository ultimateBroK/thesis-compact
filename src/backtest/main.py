"""
Backtesting pipeline: positions + OHLC → equity curve + trade metrics.

Orchestration: simulate_equity_barrier → aggregate_backtest_metrics.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from src.config import INITIAL_BALANCE

from .engine import simulate_equity_barrier
from .metrics import aggregate_backtest_metrics


def backtest_signal_positions(
    frame: pl.DataFrame,
    positions: np.ndarray,
    initial_balance: float = INITIAL_BALANCE,
) -> tuple[dict[str, float], list[dict]]:
    close = frame["close"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    spread = frame["spread"].to_numpy()
    atr_rel = frame["atr_14"].to_numpy()
    equity, num_trades, executed_trades = simulate_equity_barrier(
        close, high, low, positions, spread, atr_rel=atr_rel,
        initial_balance=initial_balance,
    )
    trade_signals = int(np.sum(np.diff(positions) != 0))
    metrics = aggregate_backtest_metrics(equity, initial_balance, num_trades, trade_signals)
    return metrics, executed_trades
