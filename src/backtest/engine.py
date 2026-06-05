"""Backtest: simple vectorized signal demo for fixed-horizon predictions."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.config import ANNUALIZATION_FACTOR, INITIAL_BALANCE, LABELS
from .trades import extract_position_trades


# ---------------------------------------------------------------------------
# Returns and metrics
# ---------------------------------------------------------------------------


def _lag_positions(positions: np.ndarray) -> np.ndarray:
    """Shift positions right by one bar; index 0 becomes 0 (flat).

    ``previous_positions[t]`` is the position that was active during bar
    ``[t-1, t]``, i.e. the position decided at or before bar ``t-1``.
    """
    previous_positions = np.empty_like(positions)
    if len(previous_positions) == 0:
        return previous_positions
    previous_positions[0] = 0
    previous_positions[1:] = positions[:-1]
    return previous_positions


def compute_strategy_bar_returns(
    close: np.ndarray,
    spread: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Return per-bar strategy returns using position[t-1] for close[t-1:t].

    A position change pays one spread fraction at the bar where the change is
    decided (bar t-1), amortized against the return of the next bar (bar t).
    Reversals have turnover 2 and therefore pay two spread fractions: close
    old side, open new side.
    """
    n = len(close)
    bar_returns = np.zeros(n, dtype=np.float64)
    if n < 2:
        return bar_returns

    price_returns = np.zeros(n, dtype=np.float64)
    np.divide(close[1:], close[:-1], out=price_returns[1:], where=close[:-1] != 0)
    price_returns[1:] -= 1.0

    previous_positions = _lag_positions(positions)
    turnover = np.abs(positions - previous_positions).astype(np.float64)

    spread_fraction = np.zeros(n, dtype=np.float64)
    np.divide(spread, close, out=spread_fraction, where=close != 0)

    bar_returns[1:] = positions[:-1] * price_returns[1:]
    # Spread cost amortized at bar t-1: decision at t-1 takes effect on bar t return
    bar_returns[1:] -= turnover[:-1] * spread_fraction[:-1]
    return np.nan_to_num(bar_returns, nan=0.0, posinf=0.0, neginf=0.0)


def compute_sharpe_ratio(equity: np.ndarray) -> float:
    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    std = float(np.std(returns))
    if std <= 0.0:
        return 0.0
    return float(ANNUALIZATION_FACTOR * np.mean(returns) / std)


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
    if gross_profit == 0.0:
        return 0.0
    if gross_loss == 0.0:
        return float("inf")
    return gross_profit / gross_loss


def compute_win_rate(trades: list[dict]) -> float:
    return (
        float(sum(1 for trade in trades if trade["win"]) / len(trades))
        if trades
        else 0.0
    )


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
    previous_positions = _lag_positions(positions)
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
    """Vectorized close-to-close Buy/Sell signal backtest on continuous candles.

    Positions must be {-1, +1} and aligned one-to-one with the continuous test
    frame. This is intentionally a demo of signal quality, not a CFD execution
    engine: no leverage, margin, lots, swaps, TP/SL search, or forced risk sizing.
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
    if np.any(~np.isin(clean_positions, LABELS)):
        raise ValueError("positions must contain only -1 (Sell) or +1 (Buy)")

    bar_returns = compute_strategy_bar_returns(close, spread, clean_positions)
    equity = initial_balance * np.cumprod(1.0 + bar_returns)
    trades = extract_position_trades(close, equity, clean_positions)
    metrics = compute_backtest_metrics(equity, initial_balance, trades, clean_positions)
    return metrics, trades, equity
