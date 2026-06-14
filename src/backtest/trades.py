"""Hàm hỗ trợ trích xuất và tuần tự hóa giao dịch."""

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
            trade_return=float(end_equity / start_equity - 1.0)
            if start_equity
            else 0.0,
            trade_pnl_usd=float(pnl),
            win=bool(pnl > 0.0),
        )

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return asdict(self)


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
                TradeRecord.from_equity(
                    entry_idx, idx, active_position, close, equity
                ).to_dict()
            )
        if position != 0:
            entry_idx = idx
        active_position = position

    if active_position != 0 and len(positions) > 1:
        trades.append(
            TradeRecord.from_equity(
                entry_idx, len(positions) - 1, active_position, close, equity
            ).to_dict()
        )
    return trades


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
