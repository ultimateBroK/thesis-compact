from __future__ import annotations

import numpy as np
import polars as pl

from src.config import CONTRACT_SIZE, INITIAL_BALANCE, RISK_PER_TRADE

from .metrics import aggregate_backtest_metrics


def build_trade_record(
    entry_idx: int,
    exit_idx: int,
    direction: float,
    entry_price: float,
    exit_price: float,
    trade_pnl: float,
) -> dict:
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


def backtest_signal_positions(
    frame: pl.DataFrame,
    positions: np.ndarray,
    tp_atr: float = 2.0,
    sl_atr: float = 1.5,
    initial_balance: float = INITIAL_BALANCE,
    contract_size: float = CONTRACT_SIZE,
    risk_per_trade: float = RISK_PER_TRADE,
) -> tuple[dict[str, float], list[dict], np.ndarray]:
    """Barrier-exit backtest with risk-based position sizing.

    Uses pure ATR-derived TP/SL levels per trade.  Entry triggers
    only on signal CHANGE (positions[i] != positions[i-1]), so one
    signal block produces at most one trade.  Trades close on TP/SL
    breach, signal reversal, or end-of-series.
    """
    close = frame["close"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    atr_rel = frame["atr_14"].to_numpy()
    n = len(close)

    equity = np.full(n, initial_balance, dtype=np.float64)
    positions = positions.astype(np.float64)

    trades: list[dict] = []
    in_trade = False
    direction = 0.0
    entry_idx = 0
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    lots = 0.0

    for i in range(1, n):
        # Mark-to-market
        if in_trade and direction != 0:
            pnl = direction * (close[i] - close[i - 1]) * lots * contract_size
            equity[i] = equity[i - 1] + pnl
        else:
            equity[i] = equity[i - 1]

        # Barrier breach check
        if in_trade and direction != 0:
            exit_price: float | None = None
            if direction > 0:
                if high[i] >= tp_price:
                    exit_price = tp_price
                elif low[i] <= sl_price:
                    exit_price = sl_price
            else:  # direction < 0
                if low[i] <= tp_price:
                    exit_price = tp_price
                elif high[i] >= sl_price:
                    exit_price = sl_price

            # Signal change also exits
            if exit_price is None and positions[i] != direction:
                exit_price = close[i]

            if exit_price is not None:
                trade_pnl = (
                    direction * (exit_price - entry_price) * lots * contract_size
                )
                trades.append(
                    build_trade_record(
                        entry_idx,
                        i,
                        direction,
                        entry_price,
                        exit_price,
                        trade_pnl,
                    )
                )
                in_trade = False
                direction = 0.0
                lots = 0.0

        # Entry — only on signal CHANGE (not just any non-zero position)
        if not in_trade and positions[i] != 0 and positions[i] != positions[i - 1]:
            in_trade = True
            direction = positions[i]
            entry_idx = i
            entry_price = close[i]

            # Pure ATR-derived barriers
            atr_abs = atr_rel[i] * close[i]
            if direction > 0:
                tp_price = close[i] + tp_atr * atr_abs
                sl_price = close[i] - sl_atr * atr_abs
            else:
                tp_price = close[i] - tp_atr * atr_abs
                sl_price = close[i] + sl_atr * atr_abs

            # Position size by risk (1% of equity per trade)
            stop_dist = abs(entry_price - sl_price)
            if stop_dist > 0:
                lots = (equity[i] * risk_per_trade) / (stop_dist * contract_size)
                lots = float(max(0.01, min(lots, 1.0)))
            else:
                lots = 0.01

    # Close any open trade at end
    if in_trade and direction != 0:
        exit_price_val = float(close[-1])
        trade_pnl = direction * (exit_price_val - entry_price) * lots * contract_size
        trades.append(
            build_trade_record(
                entry_idx,
                n - 1,
                direction,
                entry_price,
                exit_price_val,
                trade_pnl,
            )
        )

    num_trades = len(trades)
    trade_signals = int(np.sum(np.diff(positions) != 0))
    metrics = aggregate_backtest_metrics(
        equity,
        initial_balance,
        num_trades,
        trade_signals,
        executed_trades=trades,
    )
    return metrics, trades, equity
