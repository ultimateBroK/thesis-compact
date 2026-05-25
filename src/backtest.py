from __future__ import annotations

import numpy as np
import polars as pl

from src.config import CONTRACT_SIZE, FIXED_LOTS, INITIAL_BALANCE

ANNUALIZATION_FACTOR = np.sqrt(24 * 252)
FALLBACK_SPREAD_PCT = 0.00015


def simulate_equity(
    close: np.ndarray,
    positions: np.ndarray,
    spread: np.ndarray | None = None,
    initial_balance: float = INITIAL_BALANCE,
    contract_size: float = CONTRACT_SIZE,
    lots: float = FIXED_LOTS,
) -> np.ndarray:
    if spread is None:
        spread = close * FALLBACK_SPREAD_PCT
    equity = np.full(len(close), initial_balance)
    balance = initial_balance
    position = 0.0
    active_lots = 0.0

    for i in range(len(close) - 1):
        new_pos = int(positions[i])
        if new_pos != position:
            cost = spread[i] * abs(new_pos - position) * max(active_lots, lots) * contract_size
            balance -= cost
            position = new_pos
        if position != 0:
            active_lots = lots
            balance += (close[i + 1] - close[i]) * active_lots * contract_size * position
        equity[i + 1] = max(balance, 0.0)
        if balance <= 0:
            balance = 0
            position = 0
            active_lots = 0

    return equity


def sharpe_ratio(equity: np.ndarray) -> float:
    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.std(returns)
    return float(ANNUALIZATION_FACTOR * np.mean(returns) / std) if std > 0 else 0.0


def max_drawdown(equity: np.ndarray) -> float:
    cummax = np.maximum.accumulate(equity)
    return float(np.min((equity - cummax) / cummax))


def profit_factor(equity: np.ndarray) -> float:
    pnl = np.diff(equity)
    gross_profit = np.sum(pnl[pnl > 0])
    gross_loss = abs(np.sum(pnl[pnl < 0]))
    return float(gross_profit / gross_loss) if gross_loss > 0 else np.inf


def backtest_signals(
    frame: pl.DataFrame,
    positions: np.ndarray,
    initial_balance: float = INITIAL_BALANCE,
) -> dict[str, float]:
    close = frame["close"].to_numpy()
    spread = frame["spread"].to_numpy() if "spread" in frame.columns else None
    equity = simulate_equity(close, positions, spread, initial_balance)
    final_balance = equity[-1]
    trades = int(np.sum(np.diff(positions) != 0))
    return {
        "total_return": float(final_balance / initial_balance - 1),
        "sharpe": sharpe_ratio(equity),
        "max_drawdown": max_drawdown(equity),
        "profit_factor": profit_factor(equity),
        "trades": trades,
    }
