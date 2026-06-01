from __future__ import annotations

import numpy as np


def compute_sharpe_ratio(equity: np.ndarray) -> float:
    """Annualized Sharpe from per-bar equity changes."""
    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.std(returns)
    if std > 0:
        return float(np.sqrt(252 * 24) * np.mean(returns) / std)
    return 0.0


def compute_max_drawdown(equity: np.ndarray) -> float:
    cummax = np.maximum.accumulate(equity)
    return float(np.min((equity - cummax) / cummax))


def compute_profit_factor(
    executed_trades: list[dict] | None = None,
    equity: np.ndarray | None = None,
) -> float:
    """Profit factor from realized trade PnL (preferred) or equity diffs (fallback)."""
    if executed_trades:
        pnl = np.array([t["pnl_usd"] for t in executed_trades])
    elif equity is not None:
        pnl = np.diff(equity)
    else:
        return 0.0
    gross_profit = np.sum(pnl[pnl > 0])
    gross_loss = abs(np.sum(pnl[pnl < 0]))
    return float(gross_profit / gross_loss) if gross_loss > 0 else np.inf


def aggregate_backtest_metrics(
    equity: np.ndarray,
    initial_balance: float,
    num_trades: int,
    trade_signals: int,
    executed_trades: list[dict] | None = None,
    **extra,
) -> dict[str, float]:
    metrics = {
        "total_return": float(equity[-1] / initial_balance - 1),
        "sharpe": compute_sharpe_ratio(equity),
        "max_drawdown": compute_max_drawdown(equity),
        "profit_factor": compute_profit_factor(executed_trades=executed_trades, equity=equity),
        "win_rate": compute_win_rate(executed_trades) if executed_trades else 0.0,
        "trades": num_trades,
        "trade_signals": trade_signals,
    }
    metrics.update(extra)
    return metrics


def compute_win_rate(executed_trades: list[dict]) -> float:
    if not executed_trades:
        return 0.0
    wins = sum(1 for t in executed_trades if t.get("win", False))
    return wins / len(executed_trades)
