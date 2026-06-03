"""Backtest: CFD-standard barrier-exit engine, metrics, tuning.

Cost model:
  • Half-spread at entry + half-spread at exit (one full round-trip spread).
  • Commission per side per lot.
  • Overnight swap per lot per night, separate long/short rates.
  • Lot granularity (LOT_STEP), leverage margin check.

Barrier triggers use bid for long / ask for short (via half-spread shift on mid OHLC).
"""

from __future__ import annotations

import itertools
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl

from src.config import (
    COMMISSION_PER_LOT_SIDE,
    CONTRACT_SIZE,
    INITIAL_BALANCE,
    LEVERAGE,
    LOT_MAX,
    LOT_MIN,
    LOT_STEP,
    RISK_PER_TRADE,
    SHORT_LOT_SCALE,
    SWAP_LONG_USD_PER_LOT,
    SWAP_SHORT_USD_PER_LOT,
)

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
    gross_pnl: float,
    spread_cost: float,
    commission: float,
    swap: float,
    lots: float,
    overnights: int,
) -> dict:
    net = gross_pnl - spread_cost - commission - swap
    return {
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "direction": "LONG" if direction > 0 else "SHORT",
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "lots": float(lots),
        "bars_held": exit_idx - entry_idx + 1,
        "overnights": int(overnights),
        "gross_pnl_usd": float(gross_pnl),
        "spread_cost_usd": float(spread_cost),
        "commission_usd": float(commission),
        "swap_usd": float(swap),
        "trade_pnl_usd": float(net),
        "cost_usd": float(spread_cost + commission + swap),
        "win": net > 0,
    }


# ---------------------------------------------------------------------------
# CFD money primitives
# ---------------------------------------------------------------------------

def round_lot(lots: float) -> float:
    """Snap lots to broker step, clamp to [LOT_MIN, LOT_MAX]."""
    stepped = round(lots / LOT_STEP) * LOT_STEP
    return float(max(LOT_MIN, min(stepped, LOT_MAX)))


def compute_lots_by_risk(equity: float, stop_dist: float) -> float:
    """Risk-based sizing: lots = (equity * risk) / (stop_dist * contract_size)."""
    if stop_dist <= 0:
        return LOT_MIN
    raw = (equity * RISK_PER_TRADE) / (stop_dist * CONTRACT_SIZE)
    return round_lot(raw)


def margin_required(lots: float, price: float) -> float:
    """Notional / leverage. 1 lot gold = price * 100 oz."""
    return (lots * CONTRACT_SIZE * price) / LEVERAGE


def compute_overnights(timestamps: np.ndarray, entry_idx: int, exit_idx: int) -> int:
    """Count date-boundaries crossed between entry and exit bar (UTC)."""
    if timestamps.dtype.kind not in ("M", "m"):
        return 0
    a = timestamps[entry_idx]
    b = timestamps[exit_idx]
    days = np.int64(np.asarray(b - a).astype("timedelta64[D]").astype(np.int64))
    return int(max(days, 0))


def compute_trade_costs(
    spread_entry: float,
    spread_exit: float,
    lots: float,
    direction: float,
    overnights: int,
) -> tuple[float, float, float]:
    """Return (spread_cost, commission, swap) in USD.

    spread_entry / spread_exit are full bid-ask spreads in price units.
    Round-trip cost = (s_entry + s_exit) / 2 × lots × contract_size.
    """
    spread_cost = (spread_entry + spread_exit) * 0.5 * lots * CONTRACT_SIZE
    commission = 2.0 * COMMISSION_PER_LOT_SIDE * lots
    swap_rate = SWAP_LONG_USD_PER_LOT if direction > 0 else SWAP_SHORT_USD_PER_LOT
    swap = overnights * swap_rate * lots
    return spread_cost, commission, swap


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def run_barrier_backtest(
    frame: pl.DataFrame,
    positions: np.ndarray,
    tp_atr: float = 2.0,
    sl_atr: float = 1.5,
    initial_balance: float = INITIAL_BALANCE,
    _tuning_trials: int = 0,
) -> tuple[dict[str, float], list[dict], np.ndarray]:
    """CFD-standard barrier-exit backtest.

    Entry triggers only on signal CHANGE.  Trades close on TP/SL breach
    (bid/ask-aware), signal reversal, or end-of-series.
    """
    close = frame["close"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    atr = frame["atr_14"].to_numpy()
    spread = frame["spread"].to_numpy()
    ts = frame["timestamp"].to_numpy() if "timestamp" in frame.columns else None
    n = len(close)

    equity = np.full(n, initial_balance, dtype=np.float64)
    positions = positions.astype(np.float64)

    trades: list[dict] = []
    in_trade = False
    direction = 0.0
    entry_idx = 0
    tp_price = 0.0
    sl_price = 0.0
    lots = 0.0

    for i in range(1, n):
        equity[i] = equity[i - 1]

        # ── Barrier breach (bid/ask-aware via half-spread shift) ─────
        if in_trade and direction != 0:
            exit_price: float | None = None
            s2 = spread[i] * 0.5  # half-spread at exit bar
            if direction > 0:
                # Long TP: bid >= tp  → mid_high >= tp + s2
                # Long SL: bid <= sl  → mid_low  <= sl + s2
                if high[i] >= tp_price + s2:
                    exit_price = tp_price
                elif low[i] <= sl_price + s2:
                    exit_price = sl_price
            else:
                # Short TP: ask <= tp → mid_low  <= tp - s2
                # Short SL: ask >= sl → mid_high >= sl - s2
                if low[i] <= tp_price - s2:
                    exit_price = tp_price
                elif high[i] >= sl_price - s2:
                    exit_price = sl_price

            if exit_price is None and positions[i] != direction:
                exit_price = close[i]

            if exit_price is not None:
                gross = direction * (exit_price - close[entry_idx]) * lots * CONTRACT_SIZE
                overnights = compute_overnights(ts, entry_idx, i) if ts is not None else 0
                sp_cost, comm, swap = compute_trade_costs(
                    spread[entry_idx], spread[i], lots, direction, overnights,
                )
                net = gross - sp_cost - comm - swap
                trades.append(create_trade_record(
                    entry_idx, i, direction, close[entry_idx], exit_price,
                    gross, sp_cost, comm, swap, lots, overnights,
                ))
                equity[i] = equity[i - 1] + net
                in_trade = False
                direction = 0.0
                lots = 0.0

        # ── Entry on signal change ──────────────────────────────────
        if not in_trade and positions[i] != 0 and positions[i] != positions[i - 1]:
            entry_price = close[i]
            atr_price = atr[i] * close[i]
            if positions[i] > 0:
                tp_price = close[i] + tp_atr * atr_price
                sl_price = close[i] - sl_atr * atr_price
            else:
                tp_price = close[i] - tp_atr * atr_price
                sl_price = close[i] + sl_atr * atr_price

            stop_dist = abs(entry_price - sl_price)
            lots = compute_lots_by_risk(equity[i], stop_dist)
            # Neutral bet sizing: scale SHORT positions down
            if positions[i] < 0:
                lots = round_lot(lots * SHORT_LOT_SCALE)

            # Margin guard: skip trade if margin would exceed available equity
            if margin_required(lots, entry_price) > equity[i]:
                lots = 0.0
                continue

            in_trade = True
            direction = positions[i]
            entry_idx = i

    # ── Force-close any open trade at end of series ─────────────────
    if in_trade and direction != 0:
        exit_price = float(close[-1])
        gross = direction * (exit_price - close[entry_idx]) * lots * CONTRACT_SIZE
        overnights = compute_overnights(ts, entry_idx, n - 1) if ts is not None else 0
        sp_cost, comm, swap = compute_trade_costs(
            spread[entry_idx], spread[-1], lots, direction, overnights,
        )
        net = gross - sp_cost - comm - swap
        trades.append(create_trade_record(
            entry_idx, n - 1, direction, close[entry_idx], exit_price,
            gross, sp_cost, comm, swap, lots, overnights,
        ))
        equity[-1] = equity[-2] + net

    # Invariant: equity = initial + sum of realized trade PnL
    total_pnl = sum(t["trade_pnl_usd"] for t in trades)
    drift = abs(equity[-1] - (initial_balance + total_pnl))
    assert drift < 1e-6, f"Equity/PnL drift: {drift:.6f}"

    num_trades = len(trades)
    trade_signals = int(np.sum(np.diff(positions) != 0))
    metrics = compute_backtest_metrics(
        equity, initial_balance, num_trades, trade_signals, executed_trades=trades,
        num_tuning_trials=_tuning_trials,
    )
    return metrics, trades, equity


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _annualization_factor() -> float:
    """Bars-per-hour aware annualization sqrt factor."""
    from src.config import TIMEFRAME
    tf = TIMEFRAME.lower().strip()
    if "h" in tf:
        bars_per_hour = 1.0 / float(tf.replace("h", ""))
    elif "m" in tf:
        bars_per_hour = 60.0 / float(tf.replace("m", ""))
    else:
        bars_per_hour = 1.0
    return np.sqrt(252 * 24 * bars_per_hour)


def compute_sharpe_ratio(equity: np.ndarray) -> float:
    """Annualised Sharpe. Bar frequency inferred from equity length vs
    approximate trading-year bars (252 days × 24h × bars_per_hour from TIMEFRAME).
    """
    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.std(returns)
    if std <= 0:
        return 0.0
    ann = _annualization_factor()
    return float(ann * np.mean(returns) / std)


def compute_sortino_ratio(equity: np.ndarray) -> float:
    """Annualised Sortino ratio using downside deviation only."""
    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    downside_std = np.sqrt(np.mean(downside ** 2))
    ann = _annualization_factor()
    return float(np.mean(returns) / downside_std * ann)


def compute_deflated_sharpe_ratio(
    equity: np.ndarray,
    num_trials: int = 100,
) -> tuple[float, float]:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Corrects for selection bias under multiple testing.
    Returns (DSR statistic, p-value).
    DSR > 0.95 with p < 0.05 ⇒ strategy likely not overfit.
    """
    from scipy import stats as ss

    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    T = len(returns)
    if T < 2 or num_trials < 1:
        return 0.0, 1.0

    # Annualise observed Sharpe (bar-frequency aware)
    ann_factor = _annualization_factor()
    std_ret = float(np.std(returns))
    if std_ret <= 0 or T < 2:
        return 0.0, 1.0
    SR_raw = float(np.mean(returns) / std_ret)
    SR_ann = float(ann_factor * SR_raw)

    # Return distribution moments (on raw returns, not annualised)
    skew = float(pd.Series(returns).skew()) if T > 2 else 0.0
    kurt = float(pd.Series(returns).kurtosis()) if T > 3 else 0.0  # excess kurtosis

    # Expected maximum Sharpe under null (true SR = 0) across num_trials
    emc = 0.5772156649  # Euler-Mascheroni constant
    max_Z = ((1 - emc) * ss.norm.ppf(1 - 1.0 / num_trials)
             + emc * ss.norm.ppf(1 - 1.0 / (num_trials * np.e)))
    expected_max_SR_ann = float(max_Z / np.sqrt(max(T - 1, 1)))  # annualised SE scale

    # Standard error of Sharpe (raw frequency, with non-normality correction)
    se_inner = (1 + (skew / 6) * SR_raw + (kurt / 24) * SR_raw**2 - 0.25 * SR_raw**2) / max(T - 1, 1)
    if se_inner <= 0:
        return 0.0, 1.0
    sigma_SR = np.sqrt(se_inner)
    sigma_SR_ann = float(ann_factor * sigma_SR)

    DSR = float((SR_ann - expected_max_SR_ann) / sigma_SR_ann)
    p_value = float(1 - ss.norm.cdf(DSR))
    return DSR, p_value


def compute_max_drawdown(equity: np.ndarray) -> float:
    cummax = np.maximum.accumulate(equity)
    return float(np.min((equity - cummax) / cummax))


def compute_profit_factor(
    executed_trades: list[dict] | None = None,
    equity: np.ndarray | None = None,
) -> float:
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
    num_tuning_trials: int = 0,
    **extra,
) -> dict[str, float]:
    metrics = {
        "total_return": float(equity[-1] / initial_balance - 1),
        "sharpe": compute_sharpe_ratio(equity),
        "sortino": compute_sortino_ratio(equity),
        "max_drawdown": compute_max_drawdown(equity),
        "profit_factor": compute_profit_factor(executed_trades=executed_trades, equity=equity),
        "win_rate": compute_win_rate(executed_trades) if executed_trades else 0.0,
        "trades": num_trades,
        "trade_signals": trade_signals,
    }
    if executed_trades:
        metrics["avg_cost_usd"] = float(np.mean([t["cost_usd"] for t in executed_trades]))
        metrics["avg_swap_usd"] = float(np.mean([t["swap_usd"] for t in executed_trades]))
    # Deflated Sharpe Ratio
    dsr_stat, dsr_p = compute_deflated_sharpe_ratio(equity, num_trials=max(num_tuning_trials, 1))
    metrics["dsr_statistic"] = dsr_stat
    metrics["dsr_p_value"] = dsr_p
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
    tp_range: tuple[float, float, float],
    sl_range: tuple[float, float, float],
    min_hold_values: list[int],
) -> dict[str, float | int]:
    """Grid-search (min_hold, tp_atr, sl_atr) for best Sharpe on train data.

    Required args — single source of truth là TUNE_TP_RANGE_BT / TUNE_SL_RANGE_BT /
    TUNE_HOLD_VALUES trong config.
    """
    from src.models import enforce_minimum_position_hold

    tp_start, tp_stop, tp_step = tp_range
    sl_start, sl_stop, sl_step = sl_range
    tp_values = np.arange(tp_start, tp_stop + tp_step / 2, tp_step)
    sl_values = np.arange(sl_start, sl_stop + sl_step / 2, sl_step)

    raw_positions = model.predict_positions(
        train_data[features], close_prices, skip_min_hold=True,
    )

    best: dict[str, float | int] = {
        "score": -np.inf, "tp": float(tp_values[0]), "sl": float(sl_values[0]),
        "min_hold": min_hold_values[0], "trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
    }

    total = len(min_hold_values) * len(tp_values) * len(sl_values)
    print(f"  Tuning backtest: {total} combos ...")

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
                    "score": score, "tp": float(tp), "sl": float(sl),
                    "min_hold": min_hold, "trades": len(trades),
                    "win_rate": float(metrics.get("win_rate", 0)),
                    "profit_factor": float(metrics.get("profit_factor", 0)),
                }

    print(
        f"  Best: tp={best['tp']:.2f} sl={best['sl']:.2f} "
        f"min_hold={best['min_hold']} sharpe={best['score']:.3f} "
        f"trades={best['trades']} pf={best['profit_factor']:.2f}"
    )
    return best
