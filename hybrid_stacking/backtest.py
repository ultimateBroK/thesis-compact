from __future__ import annotations

import numpy as np
import polars as pl

ANNUALIZATION_FACTOR = np.sqrt(24 * 252)
CONTRACT_SIZE = 100.0
FIXED_LOTS = 0.1
SPREAD_PCT = 0.0002


def simulate_equity(
    close: np.ndarray,
    positions: np.ndarray,
    initial_balance: float = 10_000.0,
    spread_pct: float = SPREAD_PCT,
    contract_size: float = CONTRACT_SIZE,
    lots: float = FIXED_LOTS,
) -> np.ndarray:
    equity = np.full(len(close), initial_balance)
    balance = initial_balance
    position = 0.0
    active_lots = 0.0

    for i in range(len(close) - 1):
        new_pos = int(positions[i])
        if new_pos != position:
            cost = spread_pct * close[i] * abs(new_pos - position) * max(active_lots, lots) * contract_size
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


def equity_returns(
    frame: pl.DataFrame,
    positions: np.ndarray,
    initial_balance: float = 10_000.0,
) -> np.ndarray:
    close = frame["close"].to_numpy()
    equity = simulate_equity(close, positions, initial_balance)
    returns = np.diff(equity) / equity[:-1]
    return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)


def backtest_signals(
    frame: pl.DataFrame,
    positions: np.ndarray,
    initial_balance: float = 10_000.0,
) -> dict[str, float]:
    close = frame["close"].to_numpy()
    equity = simulate_equity(close, positions, initial_balance)
    final_balance = equity[-1]
    return {
        "initial_balance": float(initial_balance),
        "final_balance": float(final_balance),
        "total_pnl_usd": float(final_balance - initial_balance),
        "trades": float(np.sum(np.diff(positions) != 0)),
        "total_return": float(final_balance / initial_balance - 1),
        "sharpe": sharpe_ratio(equity),
        "max_drawdown": max_drawdown(equity),
        "profit_factor": profit_factor(equity),
    }
