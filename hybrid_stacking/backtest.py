from __future__ import annotations

import numpy as np
import pandas as pd

from hybrid_stacking.config import TradingCosts


def backtest_signals(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    costs: TradingCosts = TradingCosts(),
) -> dict[str, float]:
    strategy_returns = cost_adjusted_returns(frame, predictions, costs)
    equity = pd.Series(1 + strategy_returns, index=frame.index).cumprod()
    return {
        "trades": float((predictions != 0).sum()),
        "total_return": float(equity.iloc[-1] - 1),
        "sharpe": sharpe_ratio(strategy_returns),
        "max_drawdown": max_drawdown(equity),
        "profit_factor": profit_factor(strategy_returns),
    }


def cost_adjusted_returns(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    costs: TradingCosts,
) -> np.ndarray:
    returns = frame["close"].pct_change().shift(-1).fillna(0).to_numpy()
    spread_cost = (frame["spread"] / frame["close"]).fillna(0).to_numpy()
    slippage_cost = costs.slippage_points / frame["close"].to_numpy()
    trade_mask = predictions != 0
    strategy_returns = predictions * returns
    strategy_returns[trade_mask] -= costs.spread_multiplier * spread_cost[trade_mask]
    strategy_returns[trade_mask] -= slippage_cost[trade_mask]
    return strategy_returns


def sharpe_ratio(returns: np.ndarray) -> float:
    risk = np.std(returns)
    return 0.0 if risk == 0 else float(np.sqrt(24 * 252) * np.mean(returns) / risk)


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def profit_factor(returns: np.ndarray) -> float:
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    return float(gross_profit / gross_loss) if gross_loss else np.inf
