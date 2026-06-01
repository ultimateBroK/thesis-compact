"""Backtest: barrier-exit engine, metrics, tuning."""

from __future__ import annotations

import itertools
import warnings
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from src.config import CONTRACT_SIZE, INITIAL_BALANCE, RISK_PER_TRADE, USE_BACKTEST_TUNING

if TYPE_CHECKING:
    from src.models import HybridStackingSignalClassifier


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

def create_trade_record(
    entry_idx: int,
    exit_idx: int,
    direction: float,
    entry_price: float,
    exit_price: float,
    trade_pnl: float,
    cost: float = 0.0,
) -> dict:
    return {
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "direction": "LONG" if direction > 0 else "SHORT",
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "bars_held": exit_idx - entry_idx + 1,
        "trade_pnl_usd": float(trade_pnl),
        "cost_usd": float(cost),
        "win": trade_pnl > 0,
    }


# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------

def calculate_trade_cost(spread_at_bar: float, lots: float, contract_size: float) -> float:
    """Round-trip cost: spread paid at entry + exit.
    spread is in price units (USD/oz). Cost = spread * lots * contract_size * 2.
    """
    return spread_at_bar * lots * contract_size * 2


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def run_barrier_backtest(
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
    atr_relative = frame["atr_14"].to_numpy()
    spread = frame["spread"].to_numpy()
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
        # Carry forward flat while in trade; realized PnL applied on close
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
                cost = calculate_trade_cost(spread[entry_idx], lots, contract_size)
                trade_pnl -= cost
                trades.append(
                    create_trade_record(
                        entry_idx,
                        i,
                        direction,
                        entry_price,
                        exit_price,
                        trade_pnl,
                        cost,
                    )
                )
                in_trade = False
                direction = 0.0
                lots = 0.0
                # Override MTM equity with realized trade PnL
                equity[i] = equity[i - 1] + trade_pnl

        # Entry — only on signal CHANGE (not just any non-zero position)
        if not in_trade and positions[i] != 0 and positions[i] != positions[i - 1]:
            in_trade = True
            direction = positions[i]
            entry_idx = i
            entry_price = close[i]

            # Pure ATR-derived barriers
            atr_price = atr_relative[i] * close[i]
            if direction > 0:
                tp_price = close[i] + tp_atr * atr_price
                sl_price = close[i] - sl_atr * atr_price
            else:
                tp_price = close[i] - tp_atr * atr_price
                sl_price = close[i] + sl_atr * atr_price

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
        cost = calculate_trade_cost(spread[entry_idx], lots, contract_size)
        trade_pnl -= cost
        trades.append(
            create_trade_record(
                entry_idx,
                n - 1,
                direction,
                entry_price,
                exit_price_val,
                trade_pnl,
                cost,
            )
        )
        equity[-1] = equity[-2] + trade_pnl

    # Invariant: equity curve must match cumulative trade PnL
    total_trade_pnl = sum(t["trade_pnl_usd"] for t in trades)
    equity_drift = abs(equity[-1] - (initial_balance + total_trade_pnl))
    assert equity_drift < 0.01, f"Equity/PnL drift: {equity_drift:.4f}"

    num_trades = len(trades)
    trade_signals = int(np.sum(np.diff(positions) != 0))
    metrics = compute_backtest_metrics(
        equity,
        initial_balance,
        num_trades,
        trade_signals,
        executed_trades=trades,
    )
    return metrics, trades, equity


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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
        pnl = np.array([t["trade_pnl_usd"] for t in executed_trades])
    elif equity is not None:
        pnl = np.diff(equity)
    else:
        return 0.0
    gross_profit = np.sum(pnl[pnl > 0])
    gross_loss = abs(np.sum(pnl[pnl < 0]))
    return float(gross_profit / gross_loss) if gross_loss > 0 else np.inf


def compute_win_rate(executed_trades: list[dict]) -> float:
    if not executed_trades:
        return 0.0
    wins = sum(1 for t in executed_trades if t.get("win", False))
    return wins / len(executed_trades)


def compute_backtest_metrics(
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


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def search_backtest_parameters(
    model: HybridStackingSignalClassifier,
    train_data: pl.DataFrame,
    features: list[str],
    close_prices: np.ndarray,
    tp_range: tuple[float, float, float] = (1.5, 4.0, 0.5),
    sl_range: tuple[float, float, float] = (1.0, 3.0, 0.5),
    min_hold_values: list[int] | None = None,
) -> dict[str, float | int]:
    """Grid-search (min_hold, tp_atr, sl_atr) for best Sharpe on train data.

    Args:
        model: Trained HybridStackingSignalClassifier.
        train_data: Training DataFrame with OHLC and atr_14 columns.
        features: Feature column names.
        close_prices: Close prices for trend/EMA filtering in predict_positions.
        tp_range: (start, stop, step) in ATR multiples.
        sl_range: (start, stop, step) in ATR multiples.
        min_hold_values: List of min_hold values to try.

    Returns:
        Dict with keys: score, tp, sl, min_hold, trades, win_rate, profit_factor.
    """
    if min_hold_values is None:
        min_hold_values = [4, 6, 8, 12, 16, 24]

    if not USE_BACKTEST_TUNING:
        print(f"  Using fixed params: tp={tp_range[0]:.1f} sl={sl_range[0]:.1f} min_hold={min_hold_values[0]}")
        return {"score": 0.0, "tp": float(tp_range[0]), "sl": float(sl_range[0]),
                "min_hold": min_hold_values[0], "trades": 0, "win_rate": 0.0, "profit_factor": 0.0}

    from src.models import enforce_minimum_position_hold

    tp_start, tp_stop, tp_step = tp_range
    sl_start, sl_stop, sl_step = sl_range

    tp_values = np.arange(tp_start, tp_stop + tp_step / 2, tp_step)
    sl_values = np.arange(sl_start, sl_stop + sl_step / 2, sl_step)

    raw_positions = model.predict_positions(
        train_data[features], close_prices, skip_min_hold=True
    )

    best: dict[str, float | int] = {
        "score": -np.inf,
        "tp": 1.5,
        "sl": 1.0,
        "min_hold": 24,
        "trades": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
    }

    total_combos = len(min_hold_values) * len(tp_values) * len(sl_values)
    print(f"  Tuning backtest params: {total_combos} combos ...")

    for min_hold in min_hold_values:
        positions = enforce_minimum_position_hold(raw_positions.copy(), min_hold)

        for tp, sl in itertools.product(tp_values, sl_values):
            if sl >= tp:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics, trades, _ = run_barrier_backtest(
                    train_data, positions, tp_atr=float(tp), sl_atr=float(sl),
                )

            score = float(metrics.get("sharpe", -999))
            if score > best["score"]:
                best = {
                    "score": score,
                    "tp": float(tp),
                    "sl": float(sl),
                    "min_hold": min_hold,
                    "trades": len(trades),
                    "win_rate": float(metrics.get("win_rate", 0)),
                    "profit_factor": float(metrics.get("profit_factor", 0)),
                }

    print(
        f"  Best: tp={best['tp']:.1f} sl={best['sl']:.1f} "
        f"min_hold={best['min_hold']} "
        f"sharpe={best['score']:.3f} "
        f"trades={best['trades']} "
        f"pf={best['profit_factor']:.2f}"
    )
    return best
