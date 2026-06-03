---
doc: 03-labeling-triple-barrier
stage: labeling
thesis_chapter: 3
status: draft
last_updated: 2026-06-02
code_ref: null
---

# Labeling — Triple Barrier

> Cài đặt triple-barrier labeling với ba chế độ width: swing H/L, ATR fallback, auto-tune; tích hợp meta-labeling cho position sizing. Module `src/labeling.py` (244 dòng) — entrypoint `assign_triple_barrier_labels`.

## Tóm tắt

Module `src/labeling.py` triển khai phương pháp triple-barrier labeling \cite{de_prado_2018_afml} đã trình bày ở `11-methodology-triple-barrier.md`. Cài đặt chia thành ba lớp: (i) phát hiện swing high/low làm barrier động, (ii) quét mỗi nến để tìm barrier bị chạm đầu tiên trong horizon, (iii) auto-tune tham số width tối ưu trên out-of-fold. Module xuất ra vector nhãn $\in \{-1, +1\}$ và vector `event_end` đánh dấu thời điểm chạm barrier — cần thiết cho purged CV ở `13-methodology-purged-cv.md`.

## Cơ sở lý thuyết

Xem đầy đủ ở `11-methodology-triple-barrier.md`. Tóm tắt: tại mỗi $t$, đặt 3 barrier TP, SL, vertical; nhãn = barrier bị chạm đầu tiên. Cài đặt này sử dụng swing high/low làm mức chính, ATR làm fallback, và ánh xạ vertical hit $\to -1$ để bài toán trở thành binary classification.

## Công thức

Width động theo ba chế độ:

- **Swing H/L mode** (mặc định khi swing detect được):

$$
B^{\mathrm{TP}}_t = \mathrm{SH}_t, \qquad B^{\mathrm{SL}}_t = \mathrm{SL}_t,
$$

với $\mathrm{SH}_t, \mathrm{SL}_t$ là swing high/low được truy ngược (trailing) tại thời điểm $t$ — thực hiện bởi `derive_trailing_swing_levels(high, low, window)`.

- **ATR fallback** (khi swing không hợp lệ):

$$
B^{\mathrm{TP}}_t = P_t + c^{\mathrm{TP}} \cdot \mathrm{ATR}_t, \qquad B^{\mathrm{SL}}_t = P_t - c^{\mathrm{SL}} \cdot \mathrm{ATR}_t.
$$

- **Auto-tune**: search grid $(c^{\mathrm{TP}}, c^{\mathrm{SL}})$ tối đa hóa balance ratio:

$$
\mathrm{balance}(c^{\mathrm{TP}}, c^{\mathrm{SL}}) = \frac{\min(n_{+1}, n_{-1})}{\max(n_{+1}, n_{-1})},
$$

giới hạn $c^{\mathrm{TP}} > c^{\mathrm{SL}}$ (asymmetric risk-reward, ưu tiên take-profit rộng hơn).

Quy tắc gán nhãn thực tế (chuyển về binary):

$$
y_t = \begin{cases}
+1, & \exists s \in (t, t+h]: H_s \geq B^{\mathrm{TP}}_t \;\;\text{và SL chưa chạm}, \\
-1, & \text{còn lại (SL chạm trước hoặc vertical hit)}.
\end{cases}
$$

## Cài đặt

### Flow chính

```
assign_triple_barrier_labels(frame, horizon, fallback_tp_atr, fallback_sl_atr, swing_window)
├── scan_barriers_from_frame(frame, horizon, ...)
│   ├── derive_trailing_swing_levels(high, low, swing_window)   → swing H/L arrays
│   │   └── detect_swing_extremes(high, low, window)            → @njit, mark raw swings
│   └── scan_triple_barrier_arrays(...)                          → @njit, scan barriers
│       └── detect_first_barrier_breach(...)                     → @njit, per-bar scan
├── attach labels + event_end columns to frame
└── trim last `horizon` rows (vertical không tính được)
```

Toàn bộ vòng lặp barrier scan được đánh dấu `@njit(cache=True)` bằng Numba, đảm bảo throughput cao trên 44 000 nến. Hàm `detect_first_barrier_breach` quyết định width theo nguyên tắc: ưu tiên swing level nếu hợp lệ (ví dụ swing high $>$ close hiện tại), nếu không dùng ATR fallback.

### Ba chế độ width

| Chế độ | Khi nào dùng | Triển khai |
|---|---|---|
| Swing H/L | Mặc định, market có cấu trúc swing rõ | `derive_trailing_swing_levels(high, low, window=5)` |
| ATR fallback | Swing không detect được hoặc vượt close | `fallback_tp_atr`, `fallback_sl_atr` nhân với `atr_14 * close` |
| Auto-tune | Mặc định khi chạy pipeline (calibrate_barrier_params) | `search_optimal_barrier_widths(frame, ...)` quét grid `(c_TP, c_SL)` theo `TUNE_TP_RANGE`, `TUNE_SL_RANGE` |

### Auto-tune chi tiết

`search_optimal_barrier_widths(frame, horizon, swing_window, tp_range, sl_range, target_balance)`: quét $c^{\mathrm{TP}} \in [0.5, 4.0]$ step $0.25$, $c^{\mathrm{SL}} \in [0.5, 4.0]$ step $0.25$ (theo `TUNE_TP_RANGE`, `TUNE_SL_RANGE`). Bỏ qua cặp $c^{\mathrm{TP}} \leq c^{\mathrm{SL}}$. Với mỗi cặp, chạy `scan_barriers_from_frame`, tính balance ratio. Lưu cặp tốt nhất; dừng sớm khi balance $\geq$ `TUNE_TARGET_BALANCE = 0.35`.

### Meta-labeling

Hàm `train_meta_label_corrector` trong `src/models.py::HybridStackingSignalClassifier` huấn luyện một mô hình phụ (`CalibratedClassifierCV` trên `LogisticRegression`) nhận đầu vào là stack probability của stacking meta-learner + base learners, dự đoán xác suất primary model đúng. Position sizing theo ngưỡng:

- Position Long khi $P_{\mathrm{meta}}(\text{correct} \mid x) \geq$ `META_LABEL_THRESHOLD = 0.55`.
- Position Short khi $P_{\mathrm{meta}}(\text{correct} \mid x) \geq$ `SHORT_META_LABEL_THRESHOLD = 0.55` (bằng long threshold trong config hiện tại).

Chi tiết lý thuyết ở `12-methodology-meta-labeling.md`, cài đặt position strategy ở `05-models-stacking.md`.

### Quy ước chuyển binary

Mọi nhãn vertical (giá trị $0$ từ `scan_triple_barrier_arrays`) được ánh xạ sang $-1$ trong `scan_triple_barrier_arrays`. Quyết định thiết kế được comment trực tiếp trong `src/labeling.py::assign_triple_barrier_labels`:

> *"Time-expiry events (label=0) are mapped to -1 (failure). Rationale: This converts the problem to binary classification (-1, +1). Unresolved horizons are treated conservatively as failed signals — no trade would be generated for these cases."*

### Pseudocode flow

```
INPUT:  OHLC frame (Polars), horizon=24, tp_atr=1.5, sl_atr=1.5, swing_window=5
OUTPUT: frame với 2 cột mới: label ∈ {-1, +1}, event_end (int)

1. atr = (frame[atr_14] * frame[close]).to_numpy()   # ATR tuyệt đối
2. swing_H, swing_L = derive_trailing_swing_levels(high, low, 5)
3. FOR start = 0 .. n - horizon:
     upper = swing_H[start] nếu > close[start], else close + tp_atr * atr
     lower = swing_L[start] nếu < close[start], else close - sl_atr * atr
     FOR current = start+1 .. start+horizon:
         IF high[current] >= upper: label = +1; break
         IF low[current]  <= lower: label = -1; break
     IF không chạm: label = -1
4. RETURN frame.with_columns(label, event_end).head(-horizon)
```

## Tham số quan trọng

| Tham số | Giá trị | Vị trí | Lý do |
|---|---|---|---|
| `LABELING_HORIZON` | $24$ | `src/config.py`, `assign_triple_barrier_labels(horizon)` | 24 nến = 1 ngày trên timeframe 1h — đủ dài để TP/SL kích hoạt nhưng không quá xa để mất tín hiệu |
| `SWING_WINDOW` | $5$ | `src/config.py`, `detect_swing_extremes(window)` | 5 nến mỗi bên — cân bằng giữa phản ứng nhanh và ổn định |
| `TUNE_TP_RANGE_BT` | $(3.0, 15.0, 1.0)$ | `src/config.py`, `search_backtest_parameters` | Grid search TP ATR multiplier cho backtest engine — single source of truth |
| `TUNE_SL_RANGE_BT` | $(3.0, 15.0, 1.0)$ | `src/config.py` | Grid search SL ATR multiplier |
| `TUNE_TP_RANGE` | $(0.5, 4.0, 0.25)$ | `src/config.py`, `search_optimal_barrier_widths` | Grid auto-tune width TP cho labeling (riêng với backtest) |
| `TUNE_SL_RANGE` | $(0.5, 4.0, 0.25)$ | `src/config.py` | Grid auto-tune width SL cho labeling |
| `TUNE_TARGET_BALANCE` | $0.35$ | `src/config.py` | Dừng sớm khi balance $\geq 0.35$ — phản ánh long-bias của market |
| `USE_META_LABELING` | `True` | `src/config.py` | Bật meta-label filter |
| `META_LABEL_THRESHOLD` | $0.55$ | `src/config.py` | Ngưỡng long meta — take position khi $P_{\mathrm{meta}} \geq 0.55$ |
| `SHORT_META_LABEL_THRESHOLD` | $0.55$ | `src/config.py` | Ngưỡng short — bằng long threshold trong config hiện tại |
| | | | Note: default trong `HybridStackingSignalClassifier.__init__` là 0.60, nhưng config override thành 0.55 |
| `ADX_THRESHOLD` | $20.0$ | `src/config.py`, regime filter | Skip label khi ADX $< 20$ — sideway regime |
| `BB_WIDTH_MIN_MULT` | $1.2$ | `src/config.py`, regime filter | Skip label khi BB width $< 1.2 \cdot$ rolling mean |
| `TREND_FILTER_ENABLED` | `True` | `src/config.py` | Bật trend EMA filter |
| `TREND_EMA_PERIOD` | $89$ | `src/config.py` | Long-term EMA — Fibonacci standard |

Bảng đầy đủ ở `08-config.md`.

### Code refs

Hàm chính trong `src/labeling.py`: `detect_swing_extremes` (Numba-jit, phát hiện swing raw), `derive_trailing_swing_levels` (trailing forward-fill với lag chống lookahead), `detect_first_barrier_breach` (Numba-jit, scan per-bar), `scan_triple_barrier_arrays` (vector hóa toàn array), `scan_barriers_from_frame` (Polars → numpy → scan), `assign_triple_barrier_labels` (entrypoint chính), `search_optimal_barrier_widths` (grid search auto-tune), `summarize_label_distribution` (thống kê). Wrapper trong pipeline: `src/dataset.py::calibrate_barrier_params`, `src/dataset.py::apply_labels_to_frame`.

## Kết quả thực nghiệm

Kết quả auto-tune trên 12 tháng XAU/USD hourly (đọc từ `reports/run_*/run_data.json`):

| Cấu hình | $(c^{\mathrm{TP}}, c^{\mathrm{SL}})$ tối ưu | Balance ratio | Tỉ lệ $+1$ |
|---|---|---|---|
| Mặc định (không tune) | $(2.0, 1.5)$ | $0.43$ | $30\%$ |
| Auto-tune | $(1.75, 1.0)$ | $0.61$ | $38\%$ |

Auto-tune cải thiện balance $\approx 42\%$, trực tiếp giúp mô hình không học lệch về short class.

Phân bố nhãn cuối (có meta-labeling filter):

- Long ($+1$): $\approx 35\%$.
- Short ($-1$): $\approx 65\%$.
- Bị meta-label loại: $\approx 22\%$ mẫu primary dự đoán $+1$ bị filter, $\approx 31\%$ mẫu primary $-1$ bị filter.

## Tham khảo

- `\cite{de_prado_2018_afml}` — López de Prado, *Advances in Financial Machine Learning*, Wiley 2018, Ch. 3.
- `docs/11-methodology-triple-barrier.md` — lý thuyết triple-barrier.
- `docs/12-methodology-meta-labeling.md` — lý thuyết meta-labeling.
- `docs/05-models-stacking.md` — cài đặt meta-label model trong stacking.
- `docs/08-config.md` — bảng đầy đủ tham số `src/config.py`.
