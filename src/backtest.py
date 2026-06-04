"""Backtest: simple vectorized signal demo for fixed-horizon predictions."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.config import INITIAL_BALANCE


# ---------------------------------------------------------------------------
# Returns and metrics
# ---------------------------------------------------------------------------


def compute_strategy_bar_returns(
    close: np.ndarray,
    spread: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Return per-bar strategy returns using position[t-1] for close[t-1:t].

    A position change pays one spread fraction. Reversals have turnover 2 and
    therefore pay two spread fractions: close old side, open new side.
    """
    n = len(close)
    bar_returns = np.zeros(n, dtype=np.float64)
    if n < 2:
        return bar_returns

    price_returns = np.zeros(n, dtype=np.float64)
    np.divide(close[1:], close[:-1], out=price_returns[1:], where=close[:-1] != 0)
    price_returns[1:] -= 1.0

    previous_positions = np.empty_like(positions)
    previous_positions[0] = 0
    previous_positions[1:] = positions[:-1]
    turnover = np.abs(positions - previous_positions).astype(np.float64)

    spread_fraction = np.zeros(n, dtype=np.float64)
    np.divide(spread, close, out=spread_fraction, where=close != 0)

    bar_returns[1:] = positions[:-1] * price_returns[1:]
    bar_returns[1:] -= turnover[:-1] * spread_fraction[:-1]
    return np.nan_to_num(bar_returns, nan=0.0, posinf=0.0, neginf=0.0)


def build_equity_curve(bar_returns: np.ndarray, initial_balance: float) -> np.ndarray:
    return initial_balance * np.cumprod(1.0 + bar_returns)


def compute_sharpe_ratio(equity: np.ndarray) -> float:
    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    std = float(np.std(returns))
    if std <= 0.0:
        return 0.0
    ann = np.sqrt(252 * 24)  # 1H XAU/USD bars
    return float(ann * np.mean(returns) / std)


def compute_max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.min((equity - peak) / peak))


def compute_profit_factor(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    pnl = np.array([trade["trade_pnl_usd"] for trade in trades], dtype=np.float64)
    gross_profit = float(np.sum(pnl[pnl > 0]))
    gross_loss = float(abs(np.sum(pnl[pnl < 0])))
    return gross_profit / gross_loss if gross_loss > 0.0 else np.inf


def compute_win_rate(trades: list[dict]) -> float:
    return (
        float(sum(1 for trade in trades if trade["win"]) / len(trades))
        if trades
        else 0.0
    )


# ---------------------------------------------------------------------------
# Trade extraction
# ---------------------------------------------------------------------------


def create_trade_record(
    entry_idx: int,
    exit_idx: int,
    direction: int,
    close: np.ndarray,
    equity: np.ndarray,
) -> dict:
    start_equity = float(equity[entry_idx])
    end_equity = float(equity[exit_idx])
    pnl = end_equity - start_equity
    return {
        "entry_idx": int(entry_idx),
        "exit_idx": int(exit_idx),
        "direction": "LONG" if direction > 0 else "SHORT",
        "entry_price": float(close[entry_idx]),
        "exit_price": float(close[exit_idx]),
        "bars_held": int(exit_idx - entry_idx),
        "trade_return": float(end_equity / start_equity - 1.0) if start_equity else 0.0,
        "trade_pnl_usd": float(pnl),
        "win": bool(pnl > 0.0),
    }


def extract_position_trades(
    close: np.ndarray, equity: np.ndarray, positions: np.ndarray
) -> list[dict]:
    trades: list[dict] = []
    active_position = 0
    entry_idx = 0

    for idx, position in enumerate(positions):
        position = int(position)
        if position == active_position:
            continue
        if active_position != 0:
            trades.append(
                create_trade_record(entry_idx, idx, active_position, close, equity)
            )
        if position != 0:
            entry_idx = idx
        active_position = position

    if active_position != 0 and len(positions) > 1:
        trades.append(
            create_trade_record(
                entry_idx, len(positions) - 1, active_position, close, equity
            )
        )
    return trades


def apply_fixed_horizon_positions(
    raw_positions: np.ndarray, hold_bars: int
) -> np.ndarray:
    """Hold each raw position decision for ``hold_bars`` consecutive bars.

    Aligns backtest execution with a fixed-horizon label: a signal emitted at
    bar ``t`` is held for the next ``hold_bars`` bars, then a new decision is
    taken. Reduces turnover and matches the semantics of labels defined over
    a multi-bar future return.
    """
    if hold_bars < 1:
        raise ValueError("hold_bars must be >= 1")
    n = len(raw_positions)
    held = np.zeros(n, dtype=raw_positions.dtype)
    for start in range(0, n, hold_bars):
        end = min(start + hold_bars, n)
        held[start:end] = raw_positions[start]
    return held


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_backtest_metrics(
    equity: np.ndarray,
    initial_balance: float,
    trades: list[dict],
    positions: np.ndarray,
) -> dict[str, float]:
    previous_positions = np.empty_like(positions)
    previous_positions[0] = 0
    previous_positions[1:] = positions[:-1]
    trade_signals = int(np.sum(positions != previous_positions))
    return {
        "total_return": float(equity[-1] / initial_balance - 1.0),
        "max_drawdown": compute_max_drawdown(equity),
        "sharpe": compute_sharpe_ratio(equity),
        "win_rate": compute_win_rate(trades),
        "trades": float(len(trades)),
        "trade_signals": float(trade_signals),
        "profit_factor": compute_profit_factor(trades),
    }


def run_signal_backtest(
    frame: pl.DataFrame,
    positions: np.ndarray,
    initial_balance: float = INITIAL_BALANCE,
) -> tuple[dict[str, float], list[dict], np.ndarray]:
    """Vectorized close-to-close Buy/Sell signal backtest.

    Positions must be {-1, +1}. This is intentionally a demo of signal quality,
    not a CFD execution engine: no leverage, margin, lots, swaps, TP/SL search,
    or forced risk sizing.
    """
    close = frame["close"].to_numpy().astype(np.float64)
    spread = (
        frame["spread"].to_numpy().astype(np.float64)
        if "spread" in frame.columns
        else np.zeros(len(close))
    )
    clean_positions = np.asarray(positions, dtype=np.int64)
    if len(clean_positions) != len(close):
        raise ValueError("positions length must match frame length")
    if np.any(~np.isin(clean_positions, (-1, 1))):
        raise ValueError("positions must contain only -1 (Sell) or +1 (Buy)")

    bar_returns = compute_strategy_bar_returns(close, spread, clean_positions)
    equity = build_equity_curve(bar_returns, initial_balance)
    trades = extract_position_trades(close, equity, clean_positions)
    metrics = compute_backtest_metrics(equity, initial_balance, trades, clean_positions)
    return metrics, trades, equity


__all__ = [
    "apply_fixed_horizon_positions",
    "build_equity_curve",
    "compute_backtest_metrics",
    "compute_max_drawdown",
    "compute_sharpe_ratio",
    "compute_strategy_bar_returns",
    "extract_position_trades",
    "run_signal_backtest",
]
