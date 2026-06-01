from __future__ import annotations

import numpy as np
import pandas as pd


def extract_trades_from_results(results: pd.DataFrame) -> pd.DataFrame:
    ts = results["timestamp"].values
    close = results["close"].values
    pos = results["position"].values
    pnl = results["pnl_usd"].values

    trades = []
    in_trade = False
    entry_idx = 0
    entry_pos = 0

    for i in range(len(pos)):
        changed = (i == 0 and pos[i] != 0) or (i > 0 and pos[i] != pos[i - 1])
        if changed:
            if in_trade and entry_pos != 0:
                trade_pnl = float(np.sum(pnl[entry_idx : i + 1]))
                trades.append({
                    "entry_time": str(ts[entry_idx]),
                    "exit_time": str(ts[i]),
                    "direction": "LONG" if entry_pos > 0 else "SHORT",
                    "entry_price": float(close[entry_idx]),
                    "exit_price": float(close[i]),
                    "bars_held": i - entry_idx + 1,
                    "pnl_usd": trade_pnl,
                    "win": trade_pnl > 0,
                })
            if pos[i] == 0:
                in_trade = False
            else:
                in_trade = True
                entry_idx = i
                entry_pos = int(pos[i])

    if in_trade and entry_pos != 0:
        trade_pnl = float(np.sum(pnl[entry_idx:]))
        trades.append({
            "entry_time": str(ts[entry_idx]),
            "exit_time": str(ts[-1]),
            "direction": "LONG" if entry_pos > 0 else "SHORT",
            "entry_price": float(close[entry_idx]),
            "exit_price": float(close[-1]),
            "bars_held": len(pos) - entry_idx,
            "pnl_usd": trade_pnl,
            "win": trade_pnl > 0,
        })

    return pd.DataFrame(trades)


def convert_executed_trades_to_dataframe(
    executed_trades: list[dict],
    timestamps: np.ndarray,
) -> pd.DataFrame:
    cleaned = []
    for t in executed_trades:
        trade = t.copy()
        trade["entry_time"] = str(timestamps[t["entry_idx"]])
        trade["exit_time"] = str(timestamps[t["exit_idx"]])
        del trade["entry_idx"]
        del trade["exit_idx"]
        cleaned.append(trade)
    return pd.DataFrame(cleaned)
