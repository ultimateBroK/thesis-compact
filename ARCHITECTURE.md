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
         │  BiLSTM-Att+LGBM+XGB  │
         │  → XGBoost Meta-Learner│
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
| `MIN_OOF_F1` | `0.34` | Ngưỡng F1 macro để giữ base model (class) |
| `MIN_OOF_F1` | `0.36` | Ngưỡng F1 macro config |
| `LABELS` | `[-1, 0, 1]` | Short / Hold / Long |
| `INITIAL_BALANCE` | `$10,000` | Vốn ban đầu backtest |
| `CONTRACT_SIZE` | `100` | 1 lot XAU/USD = 100 oz |
| `FIXED_LOTS` | `0.03` | Kích thước position cố định |

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

#### 4a. Khử nhiễu Wavelet (`rolling_wavelet_denoise`)
- Dùng **Stationary Wavelet Transform (SWT)** với `sym4`, level 3
- Áp dụng **rolling window** (min 256 bars) để xử lý edge effects
- Tại mỗi bar `i`, lấy window `[max(0, i-255), i]`, áp dụng SWT + soft-threshold (MAD)
- Tái tạo chuỗi `close_denoised`

#### 4b. Phân sai phân số (`fractional_diff`)
- Tính fractional differencing với bậc `d=0.4`
- Bảo toàn **bộ nhớ dài** (long memory) của chuỗi giá, khác với differencing bậc 1 (d=1) mất hoàn toàn memory
- Numba JIT để tăng tốc vòng lặp convolution

#### 4c. Chỉ báo kỹ thuật
- **Trend:** EMA(12), EMA(26) — dùng `ewm_mean` (exponential weighted), MACD, MACD Signal
- **Momentum:** RSI(14), Stochastic(14), Awesome Oscillator
- **Volatility:** ATR(14) × close → absolute dollar ATR, Bollinger Band width + position, Rolling std(24)
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
4. ATR được truyền dưới dạng `atr_14 × close` (absolute dollar value)

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

**Vai trò:** Core ML — mô hình stacking kết hợp BiLSTM-Attention, LightGBM, XGBoost + XGBoost meta-learner.

#### 8a. Base Models (Level 0)
| Model | Cấu hình chính |
|---|---|
| **BiLSTM-Attention** | Bidirectional, 2 layers, hidden=128, attention_heads=4, seq_len=8, dropout=0.25, lr=0.001, 20 epochs |
| **LightGBM** | 120 trees, depth=5, lr=0.035, leaves=31 |
| **XGBoost** | 120 trees, depth=5, lr=0.035, subsample=0.85, reg_alpha=0.5 |

Mỗi base model được bọc trong pipeline `KNNImputer → StandardScaler → Estimator`.

#### 8b. Huấn luyện Stacking
```
                    ┌─────────────────────────────────┐
                    │        Training Phase            │
                    │                                   │
  Train Data ──────►│  5-fold Purged-Embargo CV        │
                     │  ┌──────────┐ ┌─────────┐ ┌─────┐│
                     │  │BiLSTM-Att│ │ LightGBM │ │ XGB ││
                     │  └────┬─────┘ └────┬─────┘ └──┬──┘│
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
                     │    XGBoost (Meta, lr=0.1, depth=3, 100 trees)                 │
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

**Smart Filtering:** Sau khi tính OOF F1 macro cho mỗi base model, chỉ giữ lại model đạt `≥ MIN_OOF_F1` (0.34 trong class, 0.36 trong config). Nếu tất cả dưới ngưỡng, giữ model tốt nhất. Model bị lọc không tham gia prediction.

**Meta-Learner:** `XGBClassifier(n_estimators=100, max_depth=3, reg_alpha=1.0, reg_lambda=1.0)` — học cách kết hợp xác suất từ các base model bằng gradient boosting.

#### 8c. `predict_positions()` — Chuyển signal thành position
- So sánh `prob_buy` vs `prob_hold + threshold` và `prob_sell` vs `prob_hold + threshold`
- Chỉ mở position Long/Short khi model **tự tin vượt trội** so với Hold
- Mặc định `confidence_threshold = 0.28`

#### 8d. BiLSTM-Attention internals
- `LSTMNet`: BiLSTM 2 layer → multi-head self-attention (4 heads) → residual connection → LayerNorm → mean pooling → dropout → linear output
- `LSTMClassifier`: sklearn-compatible wrapper, tự tạo sliding window sequences
- Huấn luyện với `CrossEntropyLoss` có class weights để xử lý imbalance
- Hỗ trợ CUDA tự động

### 9. `backtest.py` — Mô phỏng giao dịch

**Vai trò:** Mô phỏng equity curve có tính chi phí giao dịch.

**Chi phí bao gồm:**
- **Spread:** `close × 0.00015` (fallback) hoặc từ dữ liệu parquet
- **Contract size:** 100 (1 lot XAU/USD = 100 oz)
- **Fixed lots:** 0.03

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
| `predictions.csv` | Close, spread, label, prediction, position, **pnl_usd**, equity |
| `backtest_metrics.csv` | Tất cả metrics backtest |
| `run_data.json` | JSON đầy đủ: config, dataset info, OOF scores, evaluation, backtest, **backtest_diagnostics**, **reproducibility** |
| `model_oof_f1.png` | Bar chart F1 từng model (xanh=ACTIVE, đỏ=FILTERED) |
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
8. In report + lưu artifacts + **timing breakdown**

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
| **PyTorch** | BiLSTM-Attention base model |
| **LightGBM** | Gradient boosting base model |
| **XGBoost** | Base model + Meta-learner (gradient boosting) |
| **scikit-learn** | Preprocessing, metrics |
| **Accelerate** | Reproducible multi-process setup |
| **Matplotlib** | Charts |
| **Pixi** | Quản lý môi trường + dependencies |
| **NumPy** | Array operations |

---

## Chi tiết tham số mô hình

---

## Chi tiết tham số mô hình

### BiLSTM-Attention
| Tham số | Giá trị |
|---|---|
| `sequence_length` | 8 |
| `hidden_size` | 128 |
| `num_layers` | 2 |
| `bidirectional` | True |
| `dropout` | 0.25 |
| `num_attention_heads` | 4 |
| `learning_rate` | 0.001 |
| `epochs` | 20 |
| `batch_size` | 64 |

### LightGBM
| Tham số | Giá trị |
|---|---|
| `n_estimators` | 120 |
| `max_depth` | 5 |
| `learning_rate` | 0.035 |
| `num_leaves` | 31 |
| `subsample` | 0.85 |
| `colsample_bytree` | 0.85 |
| `class_weight` | balanced |

### XGBoost (Base)
| Tham số | Giá trị |
|---|---|
| `n_estimators` | 120 |
| `max_depth` | 5 |
| `learning_rate` | 0.035 |
| `subsample` | 0.85 |
| `colsample_bytree` | 0.85 |
| `reg_alpha` | 0.5 |
| `reg_lambda` | 1.0 |

### XGBoost Meta-Learner
| Tham số | Giá trị |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 3 |
| `learning_rate` | 0.1 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_alpha` | 1.0 |
| `reg_lambda` | 1.0 |

### Stacking Pipeline
| Tham số | Giá trị |
|---|---|
| Base model pipeline | KNNImputer(k=5) → StandardScaler → Estimator |
| Smart Filter threshold | MIN_OOF_F1 = 0.34 (class), 0.36 (config) |
| Fallback khi all fail | Giữ model có F1 cao nhất |
| Meta features | OOF probas stacked (hstack), shape (n_samples, 3 × n_active_models) |
| Confidence threshold | 0.28 (trong `predict_positions`) |

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
         │              ├── BiLSTM-Attention (2-layer, 4-head)
         │              ├── LightGBM
         │              ├── XGBoost
         │              ├── Smart Filter (OOF F1 ≥ 0.36)
         │              └── XGBoost Meta-Learner
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
