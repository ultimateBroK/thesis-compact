# Kiến trúc đồ án — Hybrid Stacking cho tín hiệu XAU/USD CFD

## Tổng quan

Đồ án xây dựng pipeline dự báo tín hiệu giao dịch vàng (XAU/USD CFD) theo 3 nhãn `{Short: -1, Hold: 0, Long: 1}`.

Pipeline chạy tuyến tính: **Dữ liệu thô → Tổng hợp nến → Khử nhiễu → Kỹ thuật → Gán nhãn → Huấn luyện Stacking → Đánh giá → Báo cáo**.

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────────┐
│  data.py    │───>│  features.py │───>│  labeling.py  │───>│ dataset.py  │
│ Đọc Parquet │    │ Kỹ thuật +   │    │ Triple        │    │ Gộp + Clean │
│ → OHLC 1h   │    │ Khử nhiễu    │    │ Barrier       │    │ Train/Test  │
└─────────────┘    └──────────────┘    └───────────────┘    └──────┬──────┘
                                                                  │
                     ┌────────────────────────────────────────────┘
                     ▼
        ┌────────────────────────┐
        │    validation.py       │
        │ Purged-Embargo Time    │
        │ Series CV Split        │
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │      models.py         │
        │  Hybrid Stacking       │
        │  LSTM + LGBM + RF      │
        │  → LR Meta-Learner     │
        └───────────┬────────────┘
                    │
         ┌──────────▼──────────┐
         │   backtest.py       │
         │   Mô phỏng P&L      │
         │   Sharpe, MDD, PF   │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │   reporting.py      │
         │   Console + Charts  │
         │   CSV + JSON        │
         └─────────────────────┘
```

---

## Diễn giải từng module

### 1. `main.py` — Điểm khởi động

File mỏng, chỉ gọi `cli.main()`. Toàn bộ logic nằm trong package `hybrid_stacking/`.

```python
from hybrid_stacking.cli import main
if __name__ == "__main__":
    main()
```

### 2. `config.py` — Hằng số cấu hình

Chứa **tất cả siêu tham số cố định** của đồ án, không truyền qua CLI:

| Hằng số | Giá trị | Ý nghĩa |
|---|---|---|
| `TIMEFRAME` | `"1h"` | Chu kỳ nến OHLC |
| `FRACTIONAL_D` | `0.4` | Bậc phân sai phân số (fractional differencing) |
| `WAVELET` | `"sym4"` | Loại wavelet dùng khử nhiễu |
| `WAVELET_LEVEL` | `3` | Mức phân rã SWT |
| `CV_SPLITS` | `5` | Số fold cross-validation |
| `EMBARGO_PCT` | `0.02` | Phần trăm embargo giữa các fold |
| `PURGE_PCT` | `0.02` | Phần trăm purge giữa train/test |
| `MIN_OOF_F1` | `0.36` | Ngưỡng F1 macro để giữ base model |
| `LABELS` | `[-1, 0, 1]` | Short / Hold / Long |
| `INITIAL_BALANCE` | `$10,000` | Vốn ban đầu backtest |

`PipelineConfig` cho phép chọn số tháng dữ liệu hoặc chạy toàn bộ (`months=None`).

### 3. `data.py` — Đọc và tổng hợp nến

**Vai trò:** Đọc file Parquet tick-level → tổng hợp thành nến OHLC 1h bằng Polars lazy scan.

**Luồng:**
1. `parquet_files()`: Liệt kê file `.parquet` trong `data/raw/XAUUSD/`, giới hạn theo số tháng.
2. `load_xauusd_candles()`: Dùng `polars.scan_parquet` + `group_by_dynamic` để:
   - Tính **mid price** = `(ask + bid) / 2`
   - Tính **spread** = `ask - bid`
   - Tổng hợp thành OHLC (open, high, low, close) + volume + spread trung bình
   - Streaming engine để tiết kiệm RAM (~306 triệu ticks)

**Tại sao Polars thay pandas:** Tránh load toàn bộ tick data vào RAM. Lazy scan chỉ xử lý từng batch, kết quả OHLC nhỏ hơn rất nhiều.

### 4. `features.py` — Kỹ thuật đặc trưng

**Vai trò:** Tạo toàn bộ features từ nến OHLC, gồm 3 nhóm chính:

#### 4a. Khử nhiễu Wavelet (`wavelet_denoise`)
- Dùng **Stationary Wavelet Transform (SWT)** với `sym4`, level 3
- Phân rã giá close thành hệ số xấp xỉ + chi tiết
- Áp dụng **soft-threshold** universal (Median Absolute Deviation) để loại noise
- Tái tạo chuỗi `close_denoised`

#### 4b. Phân sai phân số (`fractional_diff`)
- Tính fractional differencing với bậc `d=0.4`
- Bảo toàn **bộ nhớ dài** (long memory) của chuỗi giá, khác với differencing bậc 1 (d=1) mất hoàn toàn memory
- Numba JIT để tăng tốc vòng lặp convolution

#### 4c. Chỉ báo kỹ thuật
- **Trend:** EMA(12), EMA(26), MACD, MACD Signal
- **Momentum:** RSI(14), Stochastic(14), Awesome Oscillator
- **Volatility:** ATR(14), Bollinger Band width + position, Rolling std(24)
- **Returns:** 1-bar, 4-bar, 12-bar returns
- **Calendar:** Hour, Day-of-week

### 5. `labeling.py` — Gán nhãn Triple Barrier

**Vai trò:** Tạo nhãn `{−1, 0, 1}` bằng phương pháp **Triple Barrier** của Marcos López de Prado.

**Cơ chế:**
1. Tại mỗi bar `t`, đặt 3 rào cản:
   - **Take Profit** = `close[t] + 3 × ATR[t]` → nhãn `+1` (Long)
   - **Stop Loss** = `close[t] − 1.5 × ATR[t]` → nhãn `−1` (Short)
   - **Vertical Barrier** = bar `t + 12` → nhãn `0` (Hold, hết horizon)
2. Quét từng bar tiếp theo, rào cản nào chạm trước quyết định nhãn
3. Numba JIT (`@njit`) tăng tốc vòng lặp quét rào cản

**Lợi ích:** Nhãn phản ánh **biên độ biến động thực tế** (ATR) thay vì ngưỡng cố định.

### 6. `dataset.py` — Lắp ráp dataset

**Vai trò:** Gọi lần lượt `data → features → labeling` và clean kết quả.

**Luồng trong `build_dataset()`:**
1. Load candles từ `data.py`
2. Thêm technical features từ `features.py`
3. Gán nhãn từ `labeling.py`
4. `clean_labeled_frame()`: Thay inf → NaN, drop nulls
5. `train_test_time_split()`: Chia 80/20 theo thời gian + purge gap 2%
6. `feature_columns()`: Trả về danh sách cột feature (loại label, OHLC, timestamp)

### 7. `validation.py` — Cross-validation chống rò rỉ

**Vai trò:** Chia fold CV cho chuỗi thời gian, ngăn information leakage.

**`PurgedEmbargoTimeSeriesSplit`:**
- Chia dữ liệu thành `n_splits` fold theo thời gian
- **Purge:** Loại bỏ các bar trong train set có `event_end` chồng lấn lên test set (vì triple barrier label có thể kéo dài nhiều bar)
- **Embargo:** Thêm khoảng trống `embargo_pct` (2%) sau mỗi test fold
- Đảm bảo train fold không "nhìn thấy" dữ liệu test fold qua nhãn chồng lấn

### 8. `models.py` — Hybrid Stacking Classifier

**Vai trò:** Core ML — mô hình stacking kết hợp LSTM, LightGBM, RandomForest + Logistic Regression meta-learner.

#### 8a. Base Models (Level 0)
| Model | Cấu hình chính |
|---|---|
| **LSTM** | Bidirectional, 2 layers, hidden=128, seq_len=8, dropout=0.25, 20 epochs |
| **LightGBM** | 120 trees, depth=5, lr=0.035, leaves=31 |
| **RandomForest** | 120 trees, depth=8, min_leaf=20 |

Mỗi base model được bọc trong pipeline `KNNImputer → StandardScaler → Estimator`.

#### 8b. Huấn luyện Stacking
```
                    ┌─────────────────────────────────┐
                    │        Training Phase            │
                    │                                   │
  Train Data ──────►│  5-fold Purged-Embargo CV        │
                    │  ┌─────────┐ ┌─────────┐ ┌────┐ │
                    │  │  LSTM   │ │ LightGBM │ │ RF │ │
                    │  └────┬────┘ └────┬─────┘ └─┬──┘ │
                    │       │           │          │    │
                    │       ▼           ▼          ▼    │
                    │    OOF probas (n × 9)             │
                    │       │                           │
                    │       ▼  Smart Filter (F1 ≥ 0.36) │
                    │    Selected OOF probas            │
                    │       │                           │
                    │       ▼                           │
                    │    Stack → hstack probas          │
                    │       │                           │
                    │       ▼                           │
                    │    Logistic Regression (Meta)     │
                    └───────┬───────────────────────────┘
                            │
  Test Data ────────────────►│
                            ▼
                    ┌───────────────────┐
                    │  Prediction Phase │
                    │                   │
                    │  Base models →    │
                    │  predict_proba    │
                    │       │           │
                    │       ▼           │
                    │  hstack probas    │
                    │       │           │
                    │       ▼           │
                    │  Meta model →     │
                    │  final signal     │
                    └───────────────────┘
```

**Smart Filtering:** Sau khi tính OOF F1 macro cho mỗi base model, chỉ giữ lại model đạt `≥ MIN_OOF_F1`. Nếu tất cả dưới ngưỡng, giữ model tốt nhất. Model bị lọc không tham gia prediction.

**Meta-Learner:** `LogisticRegression(C=2.0, class_weight="balanced")` — học cách kết hợp xác suất từ các base model.

#### 8c. `predict_positions()` — Chuyển signal thành position
- So sánh `prob_buy` vs `prob_hold + threshold` và `prob_sell` vs `prob_hold + threshold`
- Chỉ mở position Long/Short khi model **tự tin vượt trội** so với Hold
- Mặc định `confidence_threshold = 0.28`

#### 8d. LSTM internals
- `LSTMNet`: BiLSTM 2 layer → dropout → linear output
- `LSTMClassifier`: sklearn-compatible wrapper, tự tạo sliding window sequences
- Huấn luyện với `CrossEntropyLoss` có class weights để xử lý imbalance
- Hỗ trợ CUDA tự động

### 9. `backtest.py` — Mô phỏng giao dịch

**Vai trò:** Mô phỏng equity curve có tính chi phí giao dịch.

**Chi phí bao gồm:**
- **Spread:** 0.02% giá × kích thước position
- **Contract size:** 100 (1 lot XAU/USD = 100 oz)
- **Fixed lots:** 0.1

**Metrics tính toán:**
| Metric | Công thức |
|---|---|
| **Sharpe Ratio** | `√(24×252) × mean(returns) / std(returns)` |
| **Max Drawdown** | `min((equity - cummax) / cummax)` |
| **Profit Factor** | `gross_profit / gross_loss` |
| **Total Return** | `final_balance / initial_balance - 1` |
| **Trades** | Số lần chuyển position |

### 10. `reporting.py` — Báo cáo và lưu kết quả

**Vai trò:** In console report + lưu artifacts vào `reports/run_YYYYMMDD_HHMMSS/`.

**Console output:**
- Thông tin accelerator (device, processes)
- Dataset: rows, train/test split, features, label distribution
- OOF F1 mỗi base model + trạng thái ACTIVE/FILTERED
- Classification metrics: Accuracy, F1 macro
- Backtest metrics

**Files lưu:**
| File | Nội dung |
|---|---|
| `predictions.csv` | Close, spread, label, prediction, position, equity |
| `backtest_metrics.csv` | Tất cả metrics backtest |
| `run_data.json` | JSON đầy đủ: config, dataset info, OOF scores, evaluation, backtest |
| `model_oof_f1.png` | Bar chart F1 từng model |
| `equity_curve.png` | Equity curve trên holdout set |

### 11. `cli.py` — Điều phối pipeline

**Vai trò:** CLI entry point, gọi toàn bộ pipeline theo thứ tự.

**Luồng `run()`:**
1. `configure_accelerator()` — Thiết lập seed + Accelerate
2. `build_dataset()` — Load + feature + label
3. `train_test_time_split()` — Chia train/test
4. `train_model()` — Huấn luyện HybridStacking
5. `model.predict()` — Dự báo nhãn
6. `model.predict_positions()` — Dự báo position có confidence threshold
7. `backtest_signals()` — Mô phỏng backtest
8. In report + lưu artifacts

**CLI arguments:**
- `--months N`: Dùng N tháng đầu tiên (default: 12)
- `--full`: Dùng toàn bộ dữ liệu

---

## Stack công nghệ

| Thư viện | Vai trò |
|---|---|
| **Polars** | Lazy scan Parquet, tổng hợp OHLC |
| **PyWavelets** | SWT denoising |
| **Numba** | JIT cho fractional diff + barrier scanning |
| **PyTorch** | LSTM base model |
| **LightGBM** | Gradient boosting base model |
| **scikit-learn** | RF base model, LR meta-learner, preprocessing, metrics |
| **Accelerate** | Reproducible multi-process setup |
| **Matplotlib** | Charts |
| **Pixi** | Quản lý môi trường + dependencies |

---

## Luồng dữ liệu tổng

```
data/raw/XAUUSD/*.parquet
        │
        ▼
   [data.py] Tick → OHLC 1h (Polars lazy)
        │
        ▼
   [features.py]
   ├── Wavelet denoise (SWT sym4 L3)
   ├── Fractional diff (d=0.4, Numba)
   ├── Returns (1, 4, 12 bar)
   ├── Trend (EMA, MACD)
   ├── Momentum (RSI, Stoch, AO)
   ├── Volatility (ATR, BB, rolling std)
   └── Calendar (hour, dow)
        │
        ▼
   [labeling.py] Triple Barrier (ATR-based, Numba)
        │
        ▼
   [dataset.py] Clean + Split 80/20 + purge
        │
        ├── Train ──► [validation.py] Purged-Embargo CV
        │                    │
        │              [models.py] Stacking
        │              ├── LSTM (BiLSTM 2-layer)
        │              ├── LightGBM
        │              ├── RandomForest
        │              ├── Smart Filter (OOF F1 ≥ 0.36)
        │              └── LR Meta-Learner
        │                    │
        ├── Test ────────────►│
        │                    ▼
        │              predictions + positions
        │                    │
        │              [backtest.py]
        │              Equity simulation (spread, contract)
        │                    │
        │                    ▼
        └────────────► [reporting.py]
                       Console + reports/run_*/
                       ├── predictions.csv
                       ├── backtest_metrics.csv
                       ├── run_data.json
                       ├── model_oof_f1.png
                       └── equity_curve.png
```
