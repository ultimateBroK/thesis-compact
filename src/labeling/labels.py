from __future__ import annotations

import numpy as np
import polars as pl

from .barriers import scan_barriers_from_frame


def assign_triple_barrier_labels(
    frame: pl.DataFrame,
    horizon: int = 12,
    fallback_tp_atr: float = 2.0,
    fallback_sl_atr: float = 1.5,
    swing_window: int = 5,
) -> pl.DataFrame:
    labels, event_end = scan_barriers_from_frame(
        frame, horizon, fallback_tp_atr, fallback_sl_atr, swing_window,
    )
    labeled = frame.with_columns([
        pl.Series("label", labels),
        pl.Series("event_end", event_end),
    ])
    return labeled.head(-horizon)


def summarize_label_distribution(labels: np.ndarray) -> dict[str, int | float]:
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    mapping = {1: "TP (+1)", 0: "Time (0)", -1: "SL (-1)"}
    dist: dict[str, int | float] = {}
    for lbl, cnt in zip(unique, counts):
        dist[mapping.get(int(lbl), str(lbl))] = cnt
    dist["total"] = total
    dist["balance_ratio"] = float(round(min(counts) / max(counts), 4) if len(counts) > 1 else 0.0)
    return dist


def search_optimal_barrier_widths(
    frame: pl.DataFrame,
    horizon: int = 12,
    swing_window: int = 5,
    tp_range: tuple[float, float, float] = (0.5, 4.0, 0.25),
    sl_range: tuple[float, float, float] = (0.5, 4.0, 0.25),
    target_balance: float = 0.35,
) -> tuple[float, float, float, dict[str, int | float]]:
    best_balance = 0.0
    best_tp = 2.0
    best_sl = 1.5
    best_dist: dict[str, int | float] = {}

    tp_start, tp_end, tp_step = tp_range
    sl_start, sl_end, sl_step = sl_range
    tp_vals = np.arange(tp_start, tp_end + tp_step / 2, tp_step)
    sl_vals = np.arange(sl_start, sl_end + sl_step / 2, sl_step)

    for tp in tp_vals:
        for sl in sl_vals:
            if round(tp, 4) <= round(sl, 4):
                continue
            labels, _ = scan_barriers_from_frame(frame, horizon, round(tp, 4), round(sl, 4), swing_window)
            labels_clean = labels[: len(labels) - horizon]
            _, counts = np.unique(labels_clean, return_counts=True)
            if len(counts) < 3:
                balance = 0.0
            else:
                balance = float(min(counts) / max(counts))
            if balance > best_balance:
                best_balance = balance
                best_tp = round(tp, 4)
                best_sl = round(sl, 4)
                best_dist = summarize_label_distribution(labels_clean)
            if best_balance >= target_balance:
                break
        if best_balance >= target_balance:
            break

    return best_tp, best_sl, best_balance, best_dist
