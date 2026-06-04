# thesis-compact

Pipeline dự báo tín hiệu giao dịch **XAU/USD 1H** bằng **hybrid stacking ensemble**. Bản chính tập trung vào câu chuyện đồ án: dữ liệu → feature → label → baseline models → hybrid stacking → đánh giá ML → backtest tín hiệu đơn giản.

## Pipeline

```mermaid
flowchart LR
    A[Tick Parquet] --> B[OHLC 1H]
    B --> C[20+ Technical Features]
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
python main.py [--full] [--months N] [--long-only]
```

| Flag | Mặc định | Mô tả |
|---|---:|---|
| `--full` | tắt | Dùng toàn bộ dữ liệu |
| `--months N` | 12 | Số file parquet theo tháng |
| `--long-only` | tắt | Chỉ cho phép long positions |

## Cấu trúc thư mục

```
main.py                          # Entrypoint + CLI args
src/
  config.py                      # Tham số cấu hình
  pipeline.py                    # Câu chuyện chính: load→features→labels→split→train→predict→backtest
  data.py                        # Parquet → OHLC, train/test split
  features.py                    # Feature engineering (technical indicators, candle structure, microstructure)
  labeling.py                    # Fixed-horizon future-return labels
  models.py                      # Base models + stacking + signal conversion + purged CV
  backtest.py                    # Vectorized signal backtest
  metrics.py                     # Accuracy, F1, baseline comparison
  reporting.py                   # Thin orchestrator: console + artifacts
  console.py                     # Console printers (dataset, OOF, classification, backtest, timing)
  metadata.py                    # Run metadata dataclasses & builders for JSON
  artifacts.py                   # CSV/JSON/PNG persistence
data/XAUUSD/                     # Dữ liệu parquet đầu vào (không track)
reports/run_*/                   # Artifacts đầu ra mỗi lần chạy
  ├── run_data.json              # metadata + config + kết quả + timing
  ├── figures/                   # PNG: equity, OOF, feature importance
  └── tables/                    # CSV: predictions, trades, metrics, baseline comparison
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
| `EMBARGO_PCT` | `0.02` | Embargo mỗi fold |
| `MIN_OOF_F1` | `0.0` | Chỉ dùng để report; không loại base model |
| `SIGNAL_PROBABILITY_THRESHOLD` | `0.50` | Ngưỡng xác suất cơ bản |
| `SIGNAL_PROBABILITY_MARGIN` | `0.02` | Minimum P(Buy)-P(Sell) edge để mở position |
| `INITIAL_BALANCE` | `10000` | Vốn giả lập ban đầu cho backtest tín hiệu |

## Labeling

Đồ án sử dụng **thresholded fixed-horizon binary labeling**. Các mẫu có `|future_return| <= 0.05%` được loại bỏ thay vì gán nhãn Hold. Bài toán vẫn là binary Buy/Sell.

```text
future_return = close[t + 4] / close[t] - 1

future_return > 0.0005  → Buy / +1
future_return < -0.0005 → Sell / -1
|future_return| <= 0.0005 → loại mẫu (không gán nhãn)
```

Các bars có gap thời gian > 5 giờ giữa `t` và `t + horizon` cũng bị loại (phù hợp dữ liệu bid/ask tick). Các dòng cuối không đủ dữ liệu tương lai sẽ bị loại. `event_end = t + horizon` được giữ lại để purged CV tránh overlap nhãn.

## Model và baseline comparison

Pipeline chính luôn dùng cả 3 base models trong stacking:

```text
Base 1: Logistic Regression
Base 2: SVC (RBF, calibrated)
Base 3: LightGBM
Meta:   Logistic Regression
```

OOF macro-F1 của base models chỉ dùng để báo cáo, không dùng để loại model. Mỗi lần chạy tạo:

```text
tables/baseline_metrics.csv
```

Bảng này so sánh test-set metrics của:

```text
logistic_regression
svc
lightgbm
hybrid_stacking
```

Metrics gồm `accuracy`, `f1_macro`, `precision_sell`, `recall_sell`, `precision_buy`, `recall_buy`, `roc_auc`.

## Predict vs position

Phần ML classification đánh giá nhãn Buy/Sell bằng `argmax` xác suất. Backtest dùng thêm confidence margin:

```text
P(Buy) - P(Sell) >= 0.02  → Long
P(Buy) - P(Sell) <= -0.02 → Short
|P(Buy) - P(Sell)| < 0.02 → Flat
```

Model dự báo xác suất Buy/Sell. Position chỉ được mở khi xác suất đủ lệch theo confidence margin. Nếu không đủ confidence, position = Flat. Vì vậy classification report không có lớp Hold, còn backtest có position bằng 0.

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

## Kết quả đầu ra

Mỗi lần chạy tạo thư mục `reports/run_{timestamp}/`:

- `run_data.json` — metadata, config, kết quả
- `figures/` — equity curve, OOF scores, feature importance
- `tables/`
  - `baseline_metrics.csv` — so sánh baseline vs Hybrid Stacking trên test set
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
