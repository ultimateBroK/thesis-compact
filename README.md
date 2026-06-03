# thesis-compact

Pipeline dự báo tín hiệu giao dịch **XAU/USD 1H** bằng **hybrid stacking ensemble**. Bản chính được rút gọn để tập trung vào bài toán đồ án: dữ liệu → feature → label → stacking → đánh giá ML → backtest tín hiệu đơn giản.

## Pipeline

```mermaid
flowchart LR
    A[Tick Parquet] --> B[OHLC 1H]
    B --> C[15-25 Technical Features]
    C --> D[Fixed-Horizon Future-Return Labels]
    D --> E[Chronological Train/Test Split + Purge]
    E --> F[Logistic Regression + LightGBM + Random Forest]
    F --> G[Logistic Stacking Meta-Model]
    G --> H[Signal Backtest + Reports]
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
python main.py [--full] [--months N] [--long-only] [--walk-forward]
```

| Flag | Mặc định | Mô tả |
|---|---:|---|
| `--full` | tắt | Dùng toàn bộ dữ liệu |
| `--months N` | 12 | Số file parquet theo tháng |
| `--long-only` | tắt | Chỉ cho phép long positions |
| `--walk-forward` | tắt | Chạy walk-forward validation bổ sung |

## Cấu trúc thư mục

```
main.py                          # Entrypoint
src/
  cli.py                         # CLI + orchestration pipeline
  config.py                      # Hằng số cấu hình
  data.py                        # Parquet → OHLC (Polars streaming)
  dataset.py                     # Build dataset: features + labels + split
  features.py                    # Feature engineering (indicators, OBV, frac diff)
  labeling.py                    # Fixed-horizon future-return labels
  models.py                      # Logistic Regression, LightGBM, Random Forest + stacking
  backtest.py                    # Vectorized signal backtest demo
  reporting.py                   # Báo cáo + artifacts (JSON/CSV/PNG)
  validation.py                  # PurgedEmbargoTimeSeriesSplit
data/XAUUSD/                     # Dữ liệu parquet đầu vào (không track)
reports/run_*/                   # Artifacts đầu ra mỗi lần chạy
  ├── run_data.json              # metadata + config + kết quả
  ├── figures/                   # PNG: equity, OOF, confusion, importance, ...
  └── tables/                    # CSV: predictions, trades, metrics, ...
viz.ipynb                        # Notebook phân tích
```

## Cấu hình chính (`src/config.py`)

| Tham số | Giá trị | Mô tả |
|---|---:|---|
| `TIMEFRAME` | `1h` | Khung thời gian OHLC |
| `FRACTIONAL_D` | `0.4` | Bậc fractional differencing |
| `LABELING_HORIZON` | `24` | Số nến tương lai dùng để tạo nhãn |
| `TEST_SIZE` | `0.20` | Tỷ lệ test cuối chuỗi thời gian |
| `PURGE_PCT` | `0.02` | Purge gap giữa train/test |
| `CV_SPLITS` | `5` | Số fold purged CV cho OOF stacking |
| `EMBARGO_PCT` | `0.02` | Embargo mỗi fold |
| `MIN_OOF_F1` | `0.50` | Ngưỡng lọc base model theo OOF macro-F1 |
| `SIGNAL_PROBABILITY_THRESHOLD` | `0.55` | Ngưỡng xác suất để mở Buy/Sell; thấp hơn thì Hold |
| `INITIAL_BALANCE` | `10000` | Vốn giả lập ban đầu cho backtest tín hiệu |

## Labeling

Bản chính dùng nhãn nhị phân dễ giải thích:

```text
future_return = close[t + horizon] / close[t] - 1

future_return > 0  → Buy / +1
future_return <= 0 → Sell / -1
```

Các dòng cuối không đủ dữ liệu tương lai sẽ bị loại. `event_end = t + horizon` được giữ lại để purged CV tránh overlap nhãn.

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
- `figures/` — equity curve, OOF scores, confusion matrix, feature importance
- `tables/`
  - `predictions.csv` — predictions + positions + equity/PnL
  - `trades.csv` — danh sách trades theo đoạn position
  - `feature_importance.csv`
  - `backtest_metrics.csv`, `per_class_metrics.csv`, `trade_statistics.csv`

## Kiểm tra code

```bash
pixi run check       # ruff lint src/
python -m unittest   # unit tests
```

## References

- Wolpert, D. H. (1992). Stacked Generalization. *Neural Networks*, 5(2), 241-259.
- Ke, G., Meng, Q., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.
  - Purged cross-validation and fractional differencing background only; the main pipeline intentionally avoids research-grade labeling/backtest layers.
