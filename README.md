# Thesis Compact

Hybrid stacking pipeline for forecasting XAU/USD CFD trading signals from tick-level market data.

The project follows the thesis design in `THEORY.md`: denoise price data, keep long-memory structure with fractional differencing, build technical features, label trades with the triple-barrier method, train a leakage-aware stacking classifier, and evaluate both classification and trading metrics.

## What It Does

- Reads XAU/USD parquet tick data from `data/raw/XAUUSD/`.
- Uses Polars lazy scan to aggregate tick data into `1h` OHLC candles before converting to pandas.
- Builds features from wavelet-denoised price, fractional differencing, EMA, MACD, RSI, Stochastic, AO, ATR, Bollinger Bands, volatility, and calendar fields.
- Creates labels `-1`, `0`, `1` with ATR-based triple barriers.
- Uses Numba JIT for numerical loops in fractional differencing, barrier scanning, and cost adjustment.
- Uses Accelerate for reproducible runtime setup and safe single-process reporting under `accelerate launch`.
- Uses purged and embargoed time-series validation to reduce information leakage.
- Trains a hybrid stacking model with XGBoost, LightGBM, RandomForest, ExtraTrees, SVC, and Logistic Regression meta-learner.
- Reports classification metrics, a simple cost-aware backtest with spread and slippage, and Matplotlib plots.

## Project Layout

```text
.
├── hybrid_stacking/
│   ├── backtest.py      # Trading metric calculations
│   ├── cli.py           # Command-line entrypoint
│   ├── config.py        # Hardcoded project constants
│   ├── data.py          # Polars parquet scan and OHLC aggregation
│   ├── dataset.py       # Dataset assembly
│   ├── features.py      # Wavelet, fractional differencing, indicators
│   ├── labeling.py      # Triple-barrier labels
│   ├── models.py        # Hybrid stacking classifier
│   ├── reporting.py     # Console reports
│   └── validation.py    # Purged embargo split
├── data/.gitkeep        # Keeps data directory structure
├── main.py              # Thin entrypoint
├── pixi.toml            # Environment and tasks
├── RESEARCH.md          # Research background
└── THEORY.md            # Thesis theory
```

## Setup

Install dependencies through Pixi:

```bash
pixi install
```

Raw market data is intentionally not committed. Put parquet files under:

```text
data/raw/XAUUSD/
```

Expected parquet columns:

```text
timestamp, ask, bid, ask_volume, bid_volume
```

## Run

Quick smoke test with the latest month:

```bash
pixi run smoke
```

Default run with the latest 12 months:

```bash
pixi run run
```

Run the full parquet dataset:

```bash
pixi run run-full
```

Custom month window:

```bash
pixi run python main.py --months 6
```

Compile check:

```bash
pixi run check
```

## CLI

Only data scope is configurable at runtime:

- `--months N`: use latest `N` monthly parquet files, default `12`.
- `--full`: use all parquet files.

Other parameters are fixed in `hybrid_stacking/config.py` because they are project assumptions, not routine runtime options:

- Data directory: `data/raw/XAUUSD`
- OHLC timeframe: `1h`
- CV splits: `5`
- Embargo: `0.02`
- Smart-filter OOF F1 threshold: `0.34`

## Output

The pipeline prints:

- Acceleration runtime details.
- Dataset size, train/test split, fractional differencing `d*`, feature count, label distribution.
- OOF macro F1 for each base model and whether it remains active after smart filtering.
- Accuracy, macro F1, and classification report on the holdout set.
- Cost-aware backtest metrics: trades, total return, Sharpe, max drawdown, profit factor.

The pipeline writes Matplotlib charts to `reports/`:

- `model_oof_f1.png`: base-model OOF macro F1 and active/filter status.
- `equity_curve.png`: cost-aware equity curve on the holdout set.

## Memory Notes

Full raw data is large. The current local dataset has 64 monthly parquet files and about 306 million ticks.

The memory-heavy step is tick loading and resampling. `hybrid_stacking/data.py` avoids pandas concatenation for that step by using `polars.scan_parquet(...)` and dynamic grouping before converting the much smaller OHLC result to pandas.
