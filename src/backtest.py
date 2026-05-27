from __future__ import annotations

import numpy as np
import polars as pl

from src.config import (
    CONTRACT_SIZE,
    FALLBACK_SL_ATR,
    FALLBACK_TP_ATR,
    FIXED_LOTS,
    INITIAL_BALANCE,
    LABELING_HORIZON,
    LEVERAGE,
)
from src.labeling import compute_swing_levels

ANNUALIZATION_FACTOR = np.sqrt(24 * 252)


def simulate_equity_barrier(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    positions: np.ndarray,
    spread: np.ndarray,
    atr_rel: np.ndarray | None = None,
    fallback_tp_atr: float = FALLBACK_TP_ATR,
    fallback_sl_atr: float = FALLBACK_SL_ATR,
    horizon: int = LABELING_HORIZON,
    initial_balance: float = INITIAL_BALANCE,
    contract_size: float = CONTRACT_SIZE,
    lots: float = FIXED_LOTS,
    leverage: float = LEVERAGE,
) -> tuple[np.ndarray, int]:
    n = len(close)
    equity = np.full(n, initial_balance)
    balance = initial_balance
    direction = 0.0
    tp_price = np.nan
    sl_price = np.nan
    deadline = 0
    entry_price = np.nan
    num_trades = 0

    if atr_rel is not None:
        atr_abs = atr_rel * close
    else:
        tr = np.maximum(high - low, np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1)),
        ))
        atr_abs = np.full(n, np.nan)
        for j in range(13, n):
            atr_abs[j] = tr[j] if j == 13 else (atr_abs[j - 1] * 13 + tr[j]) / 14

    swing_high, swing_low = compute_swing_levels(high, low, 5)

    for i in range(n):
        new_pos = int(positions[i])
        should_exit = False

        if direction != 0:
            exit_price = None

            if direction > 0:
                if high[i] >= tp_price:
                    exit_price = tp_price
                elif low[i] <= sl_price:
                    exit_price = sl_price
                elif i >= deadline:
                    exit_price = close[i]
            else:
                if low[i] <= tp_price:
                    exit_price = tp_price
                elif high[i] >= sl_price:
                    exit_price = sl_price
                elif i >= deadline:
                    exit_price = close[i]

            if new_pos == 0 or (direction > 0 and new_pos < 0) or (direction < 0 and new_pos > 0):
                should_exit = True
                if exit_price is None:
                    exit_price = close[i]

            if exit_price is not None:
                balance += (exit_price - entry_price) * lots * contract_size * direction
                if not should_exit:
                    balance -= 0.5 * spread[i] * abs(direction) * lots * contract_size
                num_trades += 1
                direction = 0

        if direction == 0 and new_pos != 0:
            notional = abs(new_pos) * close[i] * contract_size * lots
            required_margin = notional / leverage
            if balance >= required_margin:
                balance -= 0.5 * spread[i] * abs(new_pos) * lots * contract_size
                direction = new_pos
                entry_price = close[i]

                if np.isfinite(atr_abs[i]) and atr_abs[i] > 0:
                    if direction > 0:
                        tp_price = swing_high[i] if np.isfinite(swing_high[i]) and swing_high[i] > close[i] else close[i] + fallback_tp_atr * atr_abs[i]
                        sl_price = swing_low[i] if np.isfinite(swing_low[i]) and swing_low[i] < close[i] else close[i] - fallback_sl_atr * atr_abs[i]
                    else:
                        tp_price = swing_low[i] if np.isfinite(swing_low[i]) and swing_low[i] < close[i] else close[i] - fallback_tp_atr * atr_abs[i]
                        sl_price = swing_high[i] if np.isfinite(swing_high[i]) and swing_high[i] > close[i] else close[i] + fallback_sl_atr * atr_abs[i]
                else:
                    tp_price = np.inf if direction > 0 else -np.inf
                    sl_price = -np.inf if direction > 0 else np.inf

                deadline = i + horizon

        if direction != 0:
            equity[i] = max(balance + (close[i] - entry_price) * lots * contract_size * direction, 0.0)
        else:
            equity[i] = max(balance, 0.0)

        if balance <= 0:
            balance = 0
            direction = 0

    if direction != 0:
        balance += (close[-1] - entry_price) * lots * contract_size * direction
        num_trades += 1

    return equity, num_trades


def simulate_equity(
    close: np.ndarray,
    positions: np.ndarray,
    spread: np.ndarray,
    initial_balance: float = INITIAL_BALANCE,
    contract_size: float = CONTRACT_SIZE,
    lots: float = FIXED_LOTS,
    leverage: float = LEVERAGE,
) -> np.ndarray:
    equity = np.full(len(close), initial_balance)
    balance = initial_balance
    position = 0.0

    for i in range(len(close) - 1):
        new_pos = int(positions[i])
        if new_pos != position:
            notional = abs(new_pos) * close[i] * contract_size * lots
            required_margin = notional / leverage
            if balance >= required_margin:
                balance -= 0.5 * spread[i] * abs(new_pos - position) * lots * contract_size
                position = new_pos
        if position != 0:
            notional = abs(position) * close[i] * contract_size * lots
            maint_margin = notional / leverage / 2
            balance += (close[i + 1] - close[i]) * lots * contract_size * position
            if balance < maint_margin:
                balance = max(balance, 0.0)
                position = 0
        equity[i + 1] = max(balance, 0.0)
        if balance <= 0:
            balance = 0
            position = 0

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
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    spread = frame["spread"].to_numpy()
    atr_rel = frame["atr_14"].to_numpy()
    equity, num_trades = simulate_equity_barrier(
        close, high, low, positions, spread, atr_rel=atr_rel,
        initial_balance=initial_balance,
    )
    final_balance = equity[-1]
    trade_signals = int(np.sum(np.diff(positions) != 0))
    return {
        "total_return": float(final_balance / initial_balance - 1),
        "sharpe": sharpe_ratio(equity),
        "max_drawdown": max_drawdown(equity),
        "profit_factor": profit_factor(equity),
        "trades": num_trades,
        "entry_signals": trade_signals,
    }
