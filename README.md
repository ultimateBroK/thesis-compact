# thesis-compact

Pipeline dự báo tín hiệu giao dịch **XAU/USD 1H** bằng **hybrid stacking ensemble**. Bản chính tập trung vào câu chuyện đồ án: dữ liệu → feature → label → baseline models → hybrid stacking → đánh giá ML → backtest tín hiệu đơn giản.

## Pipeline

```mermaid
flowchart LR
    A[Tick Parquet] --> B[OHLC 1H]
    B --> C[30 Technical Features]
    C --> D[4H Future-Return Labels]
    D --> E[Chronological Train/Test Split + Purge]
    E --> F[Logistic Regression + SVC + LightGBM]
    F --> G[Logistic Stacking Meta-Model]
    G --> H[ML Metrics + Baseline Comparison]
    H --> I[Signal Backtest + Reports]
```

## Chạy nhanh

```bash
# Cài dependencies (cần pixi)
pixi install

# Smoke test (1 tháng dữ liệu)
pixi run smoke

# Chạy 12 tháng
pixi run run

# Chạy toàn bộ dữ liệu
pixi run run-full
```

### Tham số CLI

```bash
python main.py [--full] [--months N]
```

| Flag | Mặc định | Mô tả |
|---|---:|---|
| `--full` | tắt | Dùng toàn bộ dữ liệu |
| `--months N` | 12 | Số file parquet mới nhất (mỗi tháng) |

## Cấu trúc thư mục

```
.
├── main.py                          # Entrypoint + CLI args
├── pixi.toml                        # Pixi workspace + tasks (run, smoke, run-full, check, test)
├── AGENTS.md                        # Hướng dẫn dùng `semble` cho code search
├── README.md
├── data/
│   └── XAUUSD/                      # Tick parquet đầu vào (không track)
├── reports/
│   └── run_<timestamp>/              # Artifacts đầu ra mỗi lần chạy
│       ├── run_data.json            # metadata + config + kết quả + timing
│       ├── figures/                 # PNG: equity, OOF, feature importance, baselines, splits
│       └── tables/                  # CSV: predictions, trades, metrics, baselines
├── src/
│   ├── __init__.py                  # Re-exports HybridStackingSignalClassifier, PipelineConfig
│   ├── config.py                    # Tham số cấu hình + PipelineConfig dataclass
│   ├── pipeline.py                  # Câu chuyện chính: load→features→labels→split→train→predict→backtest
│   ├── data/
│   │   ├── __init__.py              # Re-exports loader + labeling
│   │   ├── loader.py                # Parquet → OHLC, train/test split, dataset assembly
│   │   └── labeling.py              # Fixed-horizon future-return labels + distribution summary
│   ├── features/
│   │   ├── __init__.py              # Re-exports FEATURE_COLUMNS, combine_market_features, get_feature_columns
│   │   └── engineering.py           # Feature engineering (technical indicators, candle structure, microstructure)
│   ├── models/
│   │   ├── __init__.py              # Re-exports stacking, baselines, factories, CV
│   │   ├── stacking.py              # Hybrid stacking classifier + signal conversion
│   │   ├── baselines.py             # Naive baselines: majority, random prior, momentum, buy-only
│   │   ├── factories.py             # Base model constructors (LogisticRegression, SVC, LightGBM)
│   │   └── cross_validation.py      # Purged k-fold CV with embargo
│   ├── backtest/
│   │   ├── __init__.py              # Re-exports engine + trades
│   │   ├── engine.py                # Vectorized signal backtest + fixed-horizon positions
│   │   └── trades.py                # Trade extraction from position segments
│   ├── evaluation/
│   │   ├── __init__.py              # Re-exports metrics + importance
│   │   ├── metrics.py               # Accuracy, F1, baseline comparison, ROC-AUC
│   │   └── importance.py            # LightGBM feature importance extraction
│   └── reporting/
│       ├── __init__.py              # Re-exports console, artifacts, metadata, plotting, publisher
│       ├── publisher.py             # Thin orchestrator: console + artifacts
│       ├── console.py               # Console printers (dataset, OOF, classification, backtest, timing)
│       ├── artifacts.py             # CSV/JSON/PNG persistence
│       ├── metadata.py              # Run metadata dataclasses & builders for JSON
│       └── plotting.py              # Matplotlib chart generation
└── tests/
    └── test_simplified_pipeline.py  # unittest suite (labeling, loader, features, stacking, baselines, CV, metrics, backtest, trades, artifacts)
```

## Cấu hình chính (`src/config.py`)

| Tham số | Giá trị | Mô tả |
|---|---:|---|
| `TIMEFRAME` | `1h` | Khung thời gian OHLC |
| `LABELING_HORIZON` | `4` | Dự báo hướng giá 4 giờ tiếp theo |
| `LABEL_RETURN_THRESHOLD` | `0.0005` | Loại mẫu có \|return\| ≤ 0.05% |
| `MAX_LABEL_GAP_HOURS` | `5` | Lọc bars có gap thời gian > 5h |
| `TEST_SIZE` | `0.20` | Tỷ lệ test cuối chuỗi thời gian |
| `PURGE_BARS` | `4` | Purge gap = labeling horizon, ngăn label leakage |
| `CV_SPLITS` | `5` | Số fold purged CV cho OOF stacking |
| `BACKTEST_HOLD_BARS` | `4` | Mỗi tín hiệu giữ trong N bars = labeling horizon, khớp tầm nhìn nhãn |
| `RANDOM_STATE` | `42` | Seed cho purged split, base models, stacking reproducibility |
| `INITIAL_BALANCE` | `10000` | Vốn giả lập ban đầu cho backtest tín hiệu |

## Labeling

Đồ án sử dụng **thresholded fixed-horizon binary labeling**. Các mẫu có `|future_return| <= 0.05%` được loại bỏ thay vì gán nhãn Hold. Bài toán vẫn là binary Buy/Sell.

```text
future_return = close[t + 4] / close[t] - 1

future_return > 0.0005  → Buy / +1
future_return < -0.0005 → Sell / -1
|future_return| <= 0.0005 → loại mẫu (không gán nhãn)
```

Các bars có gap thời gian > 5 giờ giữa `t` và `t + horizon` cũng bị loại (phù hợp dữ liệu bid/ask tick). Các dòng cuối không đủ dữ liệu tương lai sẽ bị loại. `event_start = t` và `event_end = t + horizon` được giữ lại theo index gốc để purged CV tránh overlap nhãn.

## Model comparison

Pipeline chính luôn dùng cả 3 base models trong stacking:

```text
Base 1: Logistic Regression
Base 2: SVC (RBF, calibrated)
Base 3: LightGBM
Meta:   Logistic Regression
```

Mục tiêu đánh giá của đồ án là so sánh trực tiếp:

```text
Hybrid Stacking vs Logistic Regression
Hybrid Stacking vs SVC
Hybrid Stacking vs LightGBM
```

Mỗi lần chạy tạo `tables/baseline_metrics.csv` với naive baselines, 3 base models, và hybrid model:

```text
naive_majority
naive_random_prior
naive_momentum_return_4
naive_buy_only
logistic_regression
svc
lightgbm
hybrid_stacking
```

Các chỉ tiêu đánh giá gồm `accuracy`, `f1_macro`, `precision_sell`, `recall_sell`, `precision_buy`, `recall_buy`, `roc_auc`. Các metric classification được tính trên `test_labeled`: tập test chỉ gồm mẫu có biến động đủ lớn để gán nhãn Buy/Sell.

## Dự báo tín hiệu Buy/Sell

Phần ML classification và tín hiệu giao dịch đều dùng cùng một logic Buy/Sell:

```text
P(Buy) >= P(Sell) → Buy / +1
P(Buy) <  P(Sell) → Sell / -1
```

Không có lớp Hold/Flat trong bài toán dự báo. Các mẫu nhiễu nhỏ đã bị loại ở bước labeling (`|future_return| <= threshold`), nên đầu ra mô hình luôn là Buy hoặc Sell. Backtest dùng `raw_signal` từ mô hình cho chuỗi test 1H liên tục và chuyển thành `executed_position` giữ trong `LABELING_HORIZON` bars để khớp horizon nhãn.


## Signal backtest

Backtest chỉ kiểm tra hành vi của tín hiệu, không mô phỏng đầy đủ CFD:

```text
position[t] = model signal at bar t
strategy_return[t+1] = position[t] * close_return[t+1] - spread_cost
```

Metrics chính:

```text
Total return
Max drawdown
Sharpe
Win rate
Profit factor
Number of trades
```

Không có lot sizing, margin, leverage, swap, TP/SL grid search hay Deflated Sharpe trong pipeline chính.

## Giới hạn phương pháp

| Nội dung | Giới hạn |
|---|---|
| Bài toán nhãn | Binary Buy/Sell, không có Hold. Mẫu neutral (`|future_return| <= threshold`) bị loại khỏi tập classification. |
| Tập classification | `test_labeled` chỉ gồm mẫu có biến động đủ lớn. Dùng để đánh giá Accuracy, F1-macro, Precision/Recall, ROC-AUC. |
| Tập backtest | `test_continuous` là chuỗi 1H liên tục. Dùng để mô phỏng tín hiệu theo thời gian, nên số bar khác `test_labeled`. |
| Backtest | Chỉ là signal-level demo; không mô phỏng đầy đủ lot, leverage, margin, swap, slippage, TP/SL. |
| Diễn giải kết quả | Kết quả phục vụ đánh giá mô hình trong đồ án, không phải khuyến nghị giao dịch thực tế. |

## Kết quả đầu ra

Mỗi lần chạy tạo thư mục `reports/run_{timestamp}/`:

- `run_data.json` — metadata, config, kết quả
- `figures/` — equity curve, OOF scores, feature importance
- `tables/`
  - `baseline_metrics.csv` — so sánh Hybrid Stacking với Logistic Regression, SVC, LightGBM và naive baselines
  - `predictions.csv` — predictions + positions + equity/PnL
  - `trades.csv` — danh sách trades theo đoạn position
  - `feature_importance.csv`
  - `backtest_metrics.csv`

## Kiểm tra code

```bash
pixi run check       # ruff lint src/ tests/
pixi run test        # unit tests
```

## References

- Wolpert, D. H. (1992). Stacked Generalization. *Neural Networks*, 5(2), 241-259.
- Ke, G., Meng, Q., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.
  - Purged cross-validation background only; the main pipeline intentionally avoids research-grade labeling/backtest layers.
