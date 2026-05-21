from __future__ import annotations

import numpy as np
import pandas as pd

from hybrid_stacking.config import DATA_DIR, FRACTIONAL_D, TIMEFRAME, PipelineConfig
from hybrid_stacking.data import load_xauusd_candles
from hybrid_stacking.features import add_technical_features
from hybrid_stacking.labeling import triple_barrier_labels


def build_dataset(config: PipelineConfig) -> pd.DataFrame:
    candles = load_xauusd_candles(DATA_DIR, config.months, TIMEFRAME)
    featured = add_technical_features(candles, frac_d=FRACTIONAL_D)
    dataset = clean_labeled_frame(triple_barrier_labels(featured))
    dataset.attrs["fractional_d"] = featured.attrs.get("fractional_d")
    return dataset


def clean_labeled_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"label", "event_end", "open", "high", "low"}
    return [column for column in frame.columns if column not in excluded]


def train_test_time_split(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    purge_pct: float = 0.02,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = int(len(frame) * (1 - test_size))
    purge = int(np.ceil(len(frame) * purge_pct))
    return frame.iloc[:split], frame.iloc[split + purge:]
