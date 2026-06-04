"""Trade extraction and serialization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeRecord:
    entry_idx: int
    exit_idx: int
    direction: str
    entry_price: float
    exit_price: float
    bars_held: int
    trade_return: float
    trade_pnl_usd: float
    win: bool

    @classmethod
    def from_equity(
        cls,
        entry_idx: int,
        exit_idx: int,
        direction: int,
        close: np.ndarray,
        equity: np.ndarray,
    ) -> "TradeRecord":
        start_equity = float(equity[entry_idx])
        end_equity = float(equity[exit_idx])
        pnl = end_equity - start_equity
        return cls(
            entry_idx=int(entry_idx),
            exit_idx=int(exit_idx),
            direction="LONG" if direction > 0 else "SHORT",
            entry_price=float(close[entry_idx]),
            exit_price=float(close[exit_idx]),
            bars_held=int(exit_idx - entry_idx),
            trade_return=float(end_equity / start_equity - 1.0) if start_equity else 0.0,
            trade_pnl_usd=float(pnl),
            win=bool(pnl > 0.0),
        )

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return asdict(self)


def create_trade_record(
    entry_idx: int,
    exit_idx: int,
    direction: int,
    close: np.ndarray,
    equity: np.ndarray,
) -> dict[str, int | float | str | bool]:
    return TradeRecord.from_equity(
        entry_idx, exit_idx, direction, close, equity
    ).to_dict()


def extract_position_trades(
    close: np.ndarray, equity: np.ndarray, positions: np.ndarray
) -> list[dict[str, int | float | str | bool]]:
    trades: list[dict[str, int | float | str | bool]] = []
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


def extract_trades_from_positions(results: pd.DataFrame) -> pd.DataFrame:
    ts = results["timestamp"].values
    close = results["close"].values
    pos_col = "executed_position" if "executed_position" in results else "position"
    pos = results[pos_col].values
    pnl = results["bar_pnl_usd"].values

    trades = []
    in_trade = False
    entry_idx = 0
    entry_pos = 0

    for i in range(len(pos)):
        changed = (i == 0 and pos[i] != 0) or (i > 0 and pos[i] != pos[i - 1])
        if changed:
            if in_trade and entry_pos != 0:
                trade_pnl = float(np.sum(pnl[entry_idx : i + 1]))
                trades.append(
                    {
                        "entry_time": str(ts[entry_idx]),
                        "exit_time": str(ts[i]),
                        "direction": "LONG" if entry_pos > 0 else "SHORT",
                        "entry_price": float(close[entry_idx]),
                        "exit_price": float(close[i]),
                        "bars_held": i - entry_idx + 1,
                        "trade_pnl_usd": trade_pnl,
                        "win": trade_pnl > 0,
                    }
                )
            if pos[i] == 0:
                in_trade = False
            else:
                in_trade = True
                entry_idx = i
                entry_pos = int(pos[i])

    if in_trade and entry_pos != 0:
        trade_pnl = float(np.sum(pnl[entry_idx:]))
        trades.append(
            {
                "entry_time": str(ts[entry_idx]),
                "exit_time": str(ts[-1]),
                "direction": "LONG" if entry_pos > 0 else "SHORT",
                "entry_price": float(close[entry_idx]),
                "exit_price": float(close[-1]),
                "bars_held": len(pos) - entry_idx,
                "trade_pnl_usd": trade_pnl,
                "win": trade_pnl > 0,
            }
        )

    return pd.DataFrame(trades)


def build_trades_dataframe(
    executed_trades: list[dict],
    timestamps: np.ndarray,
) -> pd.DataFrame:
    cleaned = []
    for trade_payload in executed_trades:
        trade = trade_payload.copy()
        trade["entry_time"] = str(timestamps[trade_payload["entry_idx"]])
        trade["exit_time"] = str(timestamps[trade_payload["exit_idx"]])
        del trade["entry_idx"]
        del trade["exit_idx"]
        cleaned.append(trade)
    return pd.DataFrame(cleaned)


__all__ = [
    "TradeRecord",
    "build_trades_dataframe",
    "create_trade_record",
    "extract_position_trades",
    "extract_trades_from_positions",
]
