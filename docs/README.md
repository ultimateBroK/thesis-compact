# Tài liệu Pipeline — Hybrid Stacking XAU/USD

Đọc theo thứ tự pipeline:

| # | File | Mô tả |
|---|---|---|
| 1 | [architecture-overview.md](architecture-overview.md) | Tổng quan kiến trúc, module, config, kết quả |
| 2 | [data-pipeline.md](data-pipeline.md) | Đọc Parquet → OHLC 1h (Polars streaming) |
| 3 | [feature-engineering.md](feature-engineering.md) | 20 features: frac diff, indicators, calendar |
| 4 | [labeling.md](labeling.md) | Triple-barrier labeling {-1, 0, +1} |
| 5 | [validation-split.md](validation-split.md) | Purged-embargo CV + train/test split |
| 6 | [model-training.md](model-training.md) | GRU + LightGBM + SVC → Stacking ensemble |
| 7 | [backtest-evaluation.md](backtest-evaluation.md) | Equity simulation + trading metrics |
