from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from hybrid_stacking.config import TradingCosts


def backtest_signals(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    costs: TradingCosts = TradingCosts(),
) -> dict[str, float]:
    strategy_returns = cost_adjusted_returns(frame, predictions, costs)
    equity = equity_curve(strategy_returns, frame.index)
    return {
        "trades": float((predictions != 0).sum()),
        "total_return": float(equity.iloc[-1] - 1),
        "sharpe": sharpe_ratio(strategy_returns),
        "max_drawdown": max_drawdown(equity),
        "profit_factor": profit_factor(strategy_returns),
    }


def equity_curve(returns: np.ndarray, index: pd.Index) -> pd.Series:
    return pd.Series(np.cumprod(1 + returns), index=index)


def cost_adjusted_returns(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    costs: TradingCosts = TradingCosts(),
) -> np.ndarray:
    returns = frame["close"].pct_change().shift(-1).fillna(0).to_numpy()
    spread_cost = (frame["spread"] / frame["close"]).fillna(0).to_numpy()
    slippage_cost = costs.slippage_points / frame["close"].to_numpy()
    return apply_trading_costs(
        predictions.astype(np.float64),
        returns,
        spread_cost,
        slippage_cost,
        costs.spread_multiplier,
    )


@njit(cache=True)
def apply_trading_costs(
    predictions: np.ndarray,
    returns: np.ndarray,
    spread_cost: np.ndarray,
    slippage_cost: np.ndarray,
    spread_multiplier: float,
) -> np.ndarray:
    strategy_returns = predictions * returns
    for i in range(len(strategy_returns)):
        if predictions[i] != 0:
            strategy_returns[i] -= spread_multiplier * spread_cost[i]
            strategy_returns[i] -= slippage_cost[i]
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
