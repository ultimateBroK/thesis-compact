---
doc: 01-data-pipeline
stage: data
thesis_chapter: 3
status: draft
last_updated: 2026-06-02
code_ref: src/data.py, src/dataset.py
---

# Data Pipeline

> Parquet tick → OHLC Polars streaming (timeframe 1h), xây dataset gán nhãn với purge-aware temporal split. Hai module `src/data.py` (37 dòng) và `src/dataset.py` (195 dòng).

## Tóm tắt

Pipeline dữ liệu gồm hai giai đoạn: (i) `src/data.py::load_candles_from_parquet` đọc các file Parquet tick trong `data/XAUUSD/`, mid-price từ $(\text{ask}+\text{bid})/2$, sau đó group-by dynamic theo `TIMEFRAME="1h"` để aggreg OHLCV+spread bằng **Polars streaming engine**; (ii) `src/dataset.py::build_labeled_dataset` lắp feature frame (từ `src/features.py`), calibrate barrier width trên tập train (60/20 split nội bộ), gán nhãn triple-barrier, rồi chia train/test theo temporal split với purge gap được tính từ `event_end` tối đa của train — ngăn label leakage vào test.

## Cơ sở lý thuyết

Temporal split trong tài chính không thể shuffle do autocorrelation trong chuỗi giá \cite{hamilton_1994_time_series}. Hơn nữa, vì mỗi nhãn triple-barrier có horizon $h=24$ nến (xem `docs/03-labeling-triple-barrier.md`), các observation ở lân cận split point có thể dùng dữ liệu tương lai — gây leakage. Hàm `compute_purge_gap` trong `src/dataset.py` khắc phục bằng cách tính khoảng cách từ vị trí split đến `event_end` tối đa trong train, rồi dịch điểm bắt đầu test lên tương ứng. Đây là phiên bản đơn giản (1 fold) của PurgedEmbargo split (xem `docs/04-validation-purged-embargo.md` và \cite{de_prado_2018_afml}).

## Công thức

Resample tick $\to$ OHLC 1h với mid-price $M_t = (A_t + B_t)/2$:

$$
O_t = M_{t_0}, \quad H_t = \max_{s \in [t_0, t_1)} M_s, \quad L_t = \min_{s \in [t_0, t_1)} M_s, \quad C_t = M_{t_1^{-}},
$$

$$
V_t = \sum_{s} (A^{V}_s + B^{V}_s), \quad S_t = \overline{(A_s - B_s)}_{[t_0, t_1)}.
$$

Temporal split với purge:

$$
i^{\star} = \lfloor N \cdot (1 - \pi_{\text{test}}) \rfloor, \qquad
g = \max\bigl(\lceil N \cdot \pi_{\text{purge}} \rceil,\; \max_{j \leq i^{\star}} \text{event\_end}_j - i^{\star} + 1\bigr),
$$

với $\pi_{\text{test}} = 0.20$, $\pi_{\text{purge}} = 0.02$. Tập train = `frame[0:i*]`, tập test = `frame[i*+g : ]`.

## Cài đặt

### Đọc Parquet + OHLC resample (`src/data.py`)

```
collect_parquet_paths(data_dir, months)        → sorted list[Path]
load_candles_from_parquet(data_dir, months, timeframe)
├── pl.scan_parquet(paths)                     → LazyFrame (streaming, không load full)
├── select timestamp, mid=(ask+bid)/2, spread=ask-bid, tick_volume=ask_vol+bid_vol
├── sort("timestamp")
├── group_by_dynamic("timestamp", every="1h")
│   └── agg first/last/max/min → open/high/low/close, sum(volume), mean(spread)
├── drop_nulls()
└── collect(engine="streaming")                → materialized DataFrame
```

Lazy evaluation cho phép Polars đẩy toàn bộ plan vào Rust streaming engine — query 5 năm $\approx 44\,000$ giờ chỉ tốn vài trăm MB RAM thay vì load toàn bộ tick array ($\sim$ vài GB).

### Build dataset (`src/dataset.py`)

```
build_labeled_dataset(config)
├── load_featured_candles(config)              → load_candles + build_feature_frame
├── tune_cut = floor(N * (1 - TEST_SIZE))
├── calibrate_barrier_params(train_portion)
│   ├── search_optimal_barrier_widths trên 60% đầu train
│   ├── apply_labels_to_frame trên 20% tiếp theo (validation, chỉ log/monitor, không chọn params)
│   └── return tp_atr, sl_atr tối ưu
├── apply_labels_to_frame(train_portion, tp_atr, sl_atr) → gán label + event_end
├── compute purge gap:
│   ├── base_purge = ceil(N * PURGE_PCT)
│   └── if max(event_end_train) >= tune_cut:
│         purge = max(base_purge, max_event_end - tune_cut + 1)
├── test_portion = featured.slice(tune_cut + purge, None)
├── apply_labels_to_frame(test_portion, tp_atr, sl_atr)
└── return (featured, train_labeled, test_labeled, tp_atr, sl_atr)
```

Quy trình calibrate-barrier-then-label đảm bảo barrier width chỉ học từ train — không leakage sang test.

### Output columns

| Nhóm | Cột | Dtype | Nguồn |
|---|---|---|---|
| Time | `timestamp` | `Datetime[μs]` | Parquet gốc |
| OHLC | `open`, `high`, `low`, `close` | `Float64` | `src/data.py` resample |
| Market | `volume`, `spread` | `Float64` | sum tick volume / mean spread |
| Features (19) | `close_fracdiff`, `return_4`, ..., `dayofweek` | `Float64` / `Int8` | `src/features.py` (xem `docs/02-features.md`) |
| Label | `label` | `Int8` ($\in \{-1, +1\}$) | `src/labeling.py` |
| Event | `event_end` | `Int64` (index kết thúc barrier) | `src/labeling.py` |

Feature selector `get_feature_columns(frame)` loại trừ chính xác `{label, event_end, open, high, low, close, timestamp}` — 19 cột numeric còn lại (cộng 2 raw passthrough `volume`, `spread` = 21 total).

## Tham số quan trọng

| Tham số | Giá trị | Vị trí | Lý do |
|---|---|---|---|
| `DATA_DIR` | `data/XAUUSD` | `src/config.py:11` | Thư mục chứa Parquet tick data |
| `TIMEFRAME` | `"1h"` | `src/config.py:15`, `PipelineConfig.timeframe` | Cân bằng giữa granularity và noise — 1 giờ giảm micro-structure noise nhưng vẫn đủ sample |
| `TEST_SIZE` | $0.20$ | `src/config.py:20` | 20% cuối làm hold-out test, không shuffle |
| `PURGE_PCT` | $0.02$ | `src/config.py:19` | Purge tối thiểu $\lceil N \cdot 0.02 \rceil \approx 880$ nến giữa train và test |
| `LABELING_HORIZON` | $24$ | `src/config.py:53` | Horizon gán nhãn — cũng là giới hạn trên cho purge gap |
| `PipelineConfig.months` | `12` (default) | `src/config.py:65` | Số tháng Parquet load — `None` = toàn bộ 5 năm |

### Data quality

- **Missing bars**: `group_by_dynamic` chỉ tạo bucket có tick; các giờ thiếu không xuất hiện. `drop_nulls()` loại thêm bucket rỗng.
- **Outliers**: không clip cứng — mô hình dùng ATR-relative features (`bb_position`, `spread_z_24`) đã self-normalize.
- **Timezone**: timestamp Parquet gốc ở UTC, không convert.
- **Inf values**: `forward_fill_infinite_values` đổi $\pm\infty \to \text{NaN}$ trước `drop_nulls` cuối cùng.

### Code refs

`src/data.py::collect_parquet_paths`, `src/data.py::load_candles_from_parquet`. `src/dataset.py::load_featured_candles`, `src/dataset.py::compute_purge_gap`, `src/dataset.py::derive_train_test_split`, `src/dataset.py::forward_fill_infinite_values`, `src/dataset.py::apply_labels_to_frame`, `src/dataset.py::calibrate_barrier_params`, `src/dataset.py::build_labeled_dataset`, `src/dataset.py::get_feature_columns`.

## Kết quả thực nghiệm

Dataset XAU/USD 12 tháng hourly:

| Chỉ số | Giá trị |
|---|---|
| Tổng số nến (sau drop_nulls) | $\approx 8\,500$ – $8\,700$ |
| Train rows | $\approx 6\,900$ |
| Purge gap | $170$ – $200$ nến ($\approx 7$ – $8$ ngày) |
| Test rows | $\approx 1\,500$ |
| Split timestamp (ví dụ seed=42, 2024) | $\approx$ 2024-09 → 2024-12 làm test |
| Label balance (sau auto-tune) | $38\%$ long / $62\%$ short |

Trên 5 năm (60 tháng): $\approx 44\,000$ nến, train $\approx 35\,000$, purge $\approx 880$, test $\approx 8\,800$. Memory peak $\approx 600$ MB khi streaming.

## Tham khảo

- `\cite{de_prado_2018_afml}` — López de Prado, *Advances in Financial Machine Learning*, Wiley 2018, Ch. 7 (cross-validation purging).
- `\cite{hamilton_1994_time_series}` — Hamilton, *Time Series Analysis*, Princeton 1994.
- `docs/02-features.md` — chi tiết 21 feature columns.
- `docs/03-labeling-triple-barrier.md` — gán nhãn triple-barrier, sinh `event_end`.
- `docs/04-validation-purged-embargo.md` — purged embargo CV cho OOF.
- `docs/08-config.md` — bảng đầy đủ tham số `src/config.py`.
