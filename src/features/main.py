"""
Feature engineering pipeline: raw candles → enriched feature frame.

Orchestration: combine_market_features → enrich_with_technical_features.
"""
from __future__ import annotations

import polars as pl

from .builders import combine_market_features
from .fractional import derive_fractionally_differentiated_series


def enrich_with_technical_features(
    candles: pl.DataFrame,
    frac_d: float = 0.4,
) -> pl.DataFrame:
    close = candles["close"]
    frac = derive_fractionally_differentiated_series(close, frac_d).alias("close_fracdiff")
    return combine_market_features(candles).with_columns(frac.fill_nan(None).fill_null(strategy="forward"))
