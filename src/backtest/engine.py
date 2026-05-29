from __future__ import annotations

import numpy as np
import polars as pl

from src.config import (
    CONTRACT_SIZE,
    FALLBACK_SL_ATR,
    FALLBACK_TP_ATR,
    INITIAL_BALANCE,
    LABELING_HORIZON,
    LEVERAGE,
    MAX_LOSS_ATR,
    RISK_PER_TRADE,
)
from src.labeling import derive_trailing_swing_levels

from .barriers import (
    compute_atr_from_raw_ohlc,
    derive_barrier_levels,
    detect_barrier_breach,
)
from .metrics import aggregate_backtest_metrics


def build_trade_record(entry_idx: int, exit_idx: int, direction: float, entry_price: float, exit_price: float, trade_pnl: float) -> dict:
    return {
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "direction": "LONG" if direction > 0 else "SHORT",
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "bars_held": exit_idx - entry_idx + 1,
        "pnl_usd": float(trade_pnl),
        "win": trade_pnl > 0,
    }


def _compute_position_size(
    equity: float,
    risk_per_trade: float,
    stop_pts: float,
    contract_size: float,
) -> float:
    """Compute lot size from risk budget and stop distance with safety bounds."""
    if stop_pts <= 0:
        return 0.01  # fallback minimum
    risk_usd = equity * risk_per_trade
    lots = risk_usd / (stop_pts * contract_size)
    return float(max(0.01, min(1.0, lots)))


def simulate_equity_barrier(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    positions: np.ndarray,
    spread: np.ndarray,
    atr_rel: np.ndarray | None = None,
    fallback_tp_atr: float = FALLBACK_TP_ATR,
    fallback_sl_atr: float = FALLBACK_SL_ATR,
    max_loss_atr: float = MAX_LOSS_ATR,
    horizon: int = LABELING_HORIZON,
    initial_balance: float = INITIAL_BALANCE,
    contract_size: float = CONTRACT_SIZE,
    risk_per_trade: float = RISK_PER_TRADE,
    leverage: float = LEVERAGE,
) -> tuple[np.ndarray, int, list[dict]]:
    n = len(close)
    equity = np.full(n, initial_balance)
    balance = initial_balance
    direction = 0.0
    tp_price = np.nan
    sl_price = np.nan
    deadline = 0
    entry_price = np.nan
    entry_idx = 0
    entry_lots = 0.01
    num_trades = 0
    executed_trades: list[dict] = []

    atr_abs = atr_rel * close if atr_rel is not None else compute_atr_from_raw_ohlc(n, close, high, low)
    swing_high, swing_low = derive_trailing_swing_levels(high, low, 5)

    for i in range(n):
        new_pos = int(positions[i])

        if direction != 0:
            exit_price = detect_barrier_breach(i, direction, high, low, close, tp_price, sl_price, deadline)

            if (direction > 0 and new_pos < 0) or (direction < 0 and new_pos > 0):
                if exit_price is None:
                    exit_price = close[i]

            if exit_price is not None:
                trade_pnl = (exit_price - entry_price) * entry_lots * contract_size * direction
                trade_pnl -= 0.5 * spread[i] * abs(direction) * entry_lots * contract_size
                balance += trade_pnl
                num_trades += 1
                executed_trades.append(build_trade_record(entry_idx, i, direction, entry_price, exit_price, trade_pnl))
                direction = 0

        if direction == 0 and new_pos != 0:
            direction = new_pos
            entry_price = close[i]
            entry_idx = i
            tp_price, sl_price = derive_barrier_levels(
                i, direction, close[i], entry_price, atr_abs,
                swing_high, swing_low, fallback_tp_atr, fallback_sl_atr, max_loss_atr,
            )
            # Size position by stop distance: risk 1% of equity per trade
            stop_pts = abs(entry_price - sl_price) if np.isfinite(sl_price) else close[i] * 0.02
            current_equity = max(balance, 0.01)
            entry_lots = _compute_position_size(current_equity, risk_per_trade, stop_pts, contract_size)
            notional = abs(new_pos) * close[i] * contract_size * entry_lots
            if balance >= notional / leverage:
                balance -= 0.5 * spread[i] * abs(new_pos) * entry_lots * contract_size
                deadline = i + horizon
            else:
                # Insufficient margin: skip trade
                direction = 0

        if direction != 0:
            equity[i] = max(balance + (close[i] - entry_price) * entry_lots * contract_size * direction, 0.0)
        else:
            equity[i] = max(balance, 0.0)

        if balance <= 0:
            balance = 0
            direction = 0

    if direction != 0:
        trade_pnl = (close[-1] - entry_price) * entry_lots * contract_size * direction
        balance += trade_pnl
        num_trades += 1
        executed_trades.append(build_trade_record(entry_idx, n - 1, direction, entry_price, float(close[-1]), trade_pnl))

    return equity, num_trades, executed_trades


def backtest_signal_positions(
    frame: pl.DataFrame,
    positions: np.ndarray,
    initial_balance: float = INITIAL_BALANCE,
    risk_per_trade: float = RISK_PER_TRADE,
) -> tuple[dict[str, float], list[dict]]:
    close = frame["close"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    spread = frame["spread"].to_numpy()
    atr_rel = frame["atr_14"].to_numpy()
    equity, num_trades, executed_trades = simulate_equity_barrier(
        close, high, low, positions, spread, atr_rel=atr_rel,
        initial_balance=initial_balance,
        risk_per_trade=risk_per_trade,
    )
    trade_signals = int(np.sum(np.diff(positions) != 0))
    timestamps = frame["timestamp"].to_numpy()
    metrics = aggregate_backtest_metrics(
        equity, initial_balance, num_trades, trade_signals,
        executed_trades=executed_trades,
        timestamps=timestamps,
    )
    return metrics, executed_trades
