from __future__ import annotations

import numpy as np


ANNUALIZATION_FACTOR = np.sqrt(24 * 252)


def compute_sharpe_ratio(equity: np.ndarray) -> float:
    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.std(returns)
    return float(ANNUALIZATION_FACTOR * np.mean(returns) / std) if std > 0 else 0.0


def compute_max_drawdown(equity: np.ndarray) -> float:
    cummax = np.maximum.accumulate(equity)
    return float(np.min((equity - cummax) / cummax))


def compute_profit_factor(equity: np.ndarray) -> float:
    pnl = np.diff(equity)
    gross_profit = np.sum(pnl[pnl > 0])
    gross_loss = abs(np.sum(pnl[pnl < 0]))
    return float(gross_profit / gross_loss) if gross_loss > 0 else np.inf


def aggregate_backtest_metrics(equity: np.ndarray, initial_balance: float, num_trades: int, entry_signals: int, **extra) -> dict[str, float]:
    metrics = {
        "total_return": float(equity[-1] / initial_balance - 1),
        "sharpe": compute_sharpe_ratio(equity),
        "max_drawdown": compute_max_drawdown(equity),
        "profit_factor": compute_profit_factor(equity),
        "trades": num_trades,
        "entry_signals": entry_signals,
    }
    metrics.update(extra)
    return metrics
