Được. Tôi nhận xét công tâm theo 3 lớp: **logic/tính toán**, **mismatch trong pipeline**, và **quy chuẩn đặt tên để ADHD đỡ rối**.

## 1. Các mismatch logic đáng sửa nhất

### 1. Backtest equity đang không khớp trade PnL

Đây là lỗi nặng nhất.

Trong code, khi đang giữ lệnh, equity được cộng/trừ theo biến động `close[i] - close[i-1]`. Nhưng khi ghi trade, PnL lại tính bằng `exit_price - entry_price`, trong đó `exit_price` có thể là TP/SL chạm trong nến. 

Hệ quả: equity curve và danh sách trade có thể lệch nhau.

Tôi kiểm tra file bạn gửi:

| Chỉ số                             |   Giá trị |
| ---------------------------------- | --------: |
| Equity cuối theo `predictions.csv` | 12,214.79 |
| Return theo equity                 |   +22.15% |
| Tổng PnL trong `trades.csv`        |   +778.51 |
| Equity đúng nếu cộng trade PnL     | 10,778.51 |
| Return theo trade PnL              |    +7.79% |

=> **Không được dùng return +22.15% trong báo cáo trước khi sửa.**

Cách sửa chuẩn hơn:

```python
def run_backtest(...):
    equity = np.full(n, initial_balance)

    for each bar:
        equity[i] = equity[i - 1]

        if exit_condition:
            trade_pnl = calculate_trade_pnl(...)
            equity[i] = equity[i - 1] + trade_pnl
            record_trade(...)
```

Sau đó thêm test/invariant:

```python
realized_pnl = sum(trade["pnl_usd"] for trade in trades)
assert abs(equity[-1] - (initial_balance + realized_pnl)) < 1e-6
```

Đây là ưu tiên số 1.

---

### 2. Backtest chưa tính spread/slippage thật sự

Trong `backtest_signal_positions`, engine lấy `close`, `high`, `low`, `atr_14`, nhưng chưa thấy chi phí giao dịch được trừ trực tiếp. 

Trong khi dataset và prediction có `spread`. Nếu không trừ spread/slippage, backtest XAU/USD dễ bị đẹp hơn thực tế.

Sửa tối giản:

```python
def calculate_trade_cost(spread, lots, contract_size):
    return spread * lots * contract_size
```

Rồi khi đóng lệnh:

```python
gross_pnl = direction * (exit_price - entry_price) * lots * contract_size
cost = calculate_trade_cost(entry_spread + exit_spread, lots, contract_size)
net_pnl = gross_pnl - cost
```

Nếu muốn đơn giản hơn cho đồ án:

```python
net_pnl = gross_pnl - spread[i] * lots * contract_size * 2
```

Trong báo cáo ghi rõ: “Chi phí giao dịch được mô phỏng bằng spread tại thời điểm vào/ra lệnh.”

---

### 3. `pnl_usd` trong `predictions.csv` dễ gây hiểu nhầm

`predictions.csv` có cột `pnl_usd`, nhưng nếu cột này là per-bar mark-to-market còn `trades.csv` là realized PnL thì tên `pnl_usd` không rõ nghĩa.

Nên đổi:

```text
pnl_usd        -> bar_pnl_usd
equity         -> equity_usd
```

Trong `trades.csv`:

```text
pnl_usd        -> trade_pnl_usd
```

Tên hiện tại làm bạn dễ tưởng mọi thứ cùng một loại PnL, trong khi không phải.

---

### 4. `tune_backtest_hyperparameters` hiện không còn đúng nghĩa “tune”

Config hiện tại:

```python
TUNE_TP_RANGE_BT = (1.5, 1.5, 0.5)
TUNE_SL_RANGE_BT = (1.0, 1.0, 0.5)
TUNE_HOLD_VALUES = [24]
```

Tức là chỉ có **một tổ hợp duy nhất**, không thật sự tune. 

Nhưng code vẫn gọi:

```python
tune_backtest_hyperparameters(...)
```

và in ra “Tuned”. 

Đây là mismatch về ý nghĩa.

Nên đổi một trong hai hướng:

**Hướng đơn giản, tôi khuyên dùng:**

```python
BACKTEST_TP_ATR = 1.5
BACKTEST_SL_ATR = 1.0
MIN_POSITION_HOLD = 24
USE_BACKTEST_TUNING = False
```

Và bỏ chữ “tune” khỏi pipeline chính.

**Hoặc nếu vẫn muốn tune thật:**

Cho nhiều giá trị:

```python
TUNE_TP_RANGE_BT = (1.0, 3.0, 0.5)
TUNE_SL_RANGE_BT = (0.5, 2.0, 0.5)
TUNE_HOLD_VALUES = [6, 12, 24]
```

Nhưng với deadline, tôi không khuyên mở rộng tuning.

---

### 5. `derive_train_test_split` có purge nhưng không được dùng trong pipeline chính

Trong `dataset/builder.py` có hàm `derive_train_test_split`, có tính purge gap. 

Nhưng `assemble_labeled_dataset` lại tự chia:

```python
tune_cut = int(len(featured) * (1 - TEST_SIZE))
train_portion = featured.head(tune_cut)
test_portion = featured.slice(tune_cut, None)
```

rồi labeling riêng train/test. 

Trong `run_data`, train kết thúc `2023-01-03`, test bắt đầu `2023-02-08`, tức là có gap thực tế.  Nhưng về code, logic purge trong `derive_train_test_split` đang bị bỏ ngoài pipeline chính, làm người đọc khó hiểu: “Có purge hay không?”

Cách sửa:

```python
def build_train_test_dataset(config):
    featured = load_featured_candles(config)
    train_raw, test_raw, purge_gap = split_train_test_with_purge(featured)
    tp_atr, sl_atr = calibrate_label_barriers(train_raw)
    train = label_dataset(train_raw, tp_atr, sl_atr)
    test = label_dataset(test_raw, tp_atr, sl_atr)
    return train, test, metadata
```

Tức là chỉ có **một đường split chính**, không có hàm phụ bị bỏ quên.

---

### 6. Labeling đang ép label `0` thành `-1`

Trong triple-barrier, nếu không chạm TP/SL trong horizon thì code map `0` thành `-1`:

```python
labels[labels == 0] = -1
```

Comment ghi: unresolved = assume failure. 

Về học thuật, cái này **không sai tuyệt đối**, nhưng cần nói rõ. Vì triple-barrier thường có thể có 3 class: `-1`, `0`, `1`. Bạn đang biến nó thành binary classification: `-1` và `1`.

Trong báo cáo phải viết:

> Đồ án quy đổi bài toán thành phân loại nhị phân. Các trường hợp không chạm take-profit trong horizon được xem là tín hiệu không thành công và gán nhãn -1.

Nếu không nói rõ, thầy có thể hỏi: “Tại sao triple-barrier mà không có nhãn 0?”

---

### 7. `atr_14` có thể đang là relative ATR, tên dễ gây nhầm

Trong backtest:

```python
atr_rel = frame["atr_14"].to_numpy()
atr_abs = atr_rel[i] * close[i]
```



Nhưng trong labeling:

```python
atr = (frame["atr_14"] * frame["close"]).to_numpy()
```



Tức là `atr_14` có vẻ đang là ATR tương đối, không phải ATR giá tuyệt đối.

Tên `atr_14` khiến người đọc nghĩ là ATR theo đơn vị giá. Nên đổi rõ hơn:

```text
atr_14        -> atr_14_rel
atr_abs       -> atr_14_price
```

Nếu không muốn đổi nhiều code, ít nhất comment trong `features.py`:

```python
# atr_14 is normalized by close; multiply by close to recover price-distance ATR.
```

---

### 8. `close_series` truyền vào `predict_positions` dễ sai nghĩa

Trong pipeline chính:

```python
outputs = run_prediction_and_backtest(
    model,
    (train, test),
    features,
    train["close"].to_numpy(),
    config,
)
```

Sau đó khi predict test:

```python
raw_positions = model.predict_positions(
    test[features],
    test["close"].to_numpy(),
    skip_min_hold=True,
)
```



Ở đây `close_series` lúc tune dùng train, lúc predict dùng test. Không sai, nhưng tên `close_series` quá chung. Vì nó dùng để tính EMA trend filter nội bộ. 

Nên đổi tên:

```text
close_series -> prices_for_trend_filter
```

Hoặc tốt hơn: đừng truyền riêng nếu DataFrame đã có `close`. Vì hiện `X` là `test[features]`, không chứa `close`, nên mới phải truyền riêng. Nhưng điều này làm logic rối.

Tôi khuyên thêm `close` vào input của position function:

```python
predict_positions(feature_frame, close_prices)
```

và đặt tên rõ.

---

## 2. Logic nên chuẩn hóa lại thành pipeline 7 bước

Để dễ bảo vệ, pipeline nên cố định như sau:

```text
1. load_market_data()
2. build_features()
3. calibrate_labeling_params()
4. assign_labels()
5. split_train_test()
6. train_signal_model()
7. run_backtest()
8. save_report()
```

Hiện tại tên trong code có cái đúng, cái hơi mơ hồ. Ví dụ `assemble_labeled_dataset` làm nhiều việc: load candles, enrich features, split, calibrate barrier, label train/test. 

Tên này không sai, nhưng với ADHD dễ bị mù vì không biết bên trong nó làm bao nhiêu thứ.

Nên đổi thành tên kể chuyện hơn:

```python
def build_labeled_train_test_dataset(config):
    featured = load_featured_candles(config)
    train_raw, test_raw = split_train_test(featured)
    label_params = calibrate_label_params(train_raw)
    train = assign_labels(train_raw, label_params)
    test = assign_labels(test_raw, label_params)
    return DatasetBundle(...)
```

---

## 3. Quy chuẩn đặt tên hàm tôi khuyên dùng

Dùng quy chuẩn này: **verb + object + purpose**.

Không cần quá academic. Không cần design pattern. Chỉ cần đọc tên là biết làm gì.

### Nhóm hàm load dữ liệu

Dùng prefix:

```text
load_      đọc dữ liệu từ disk/source
collect_   gom path/file/list
parse_     chuyển raw text/arg thành object
```

Ví dụ:

```python
load_candles_from_parquet()
collect_parquet_paths()
parse_cli_args()
```

Không nên dùng:

```python
get_data()
process_data()
handle_files()
```

Vì quá mơ hồ.

---

### Nhóm feature

Dùng prefix:

```text
add_       thêm cột mới vào DataFrame
compute_   tính một giá trị/series
build_     dựng một tập kết quả lớn
```

Ví dụ:

```python
add_return_features()
add_volatility_features()
compute_relative_atr()
build_feature_frame()
```

Không nên dùng:

```python
enrich_with_technical_features()
```

Tên này không sai, nhưng hơi rộng. Nếu giữ thì bên trong nên chia section rõ.

---

### Nhóm labeling

Dùng prefix:

```text
assign_       gán nhãn
scan_         quét barrier
calibrate_    tìm tham số bằng train
summarize_    thống kê
```

Ví dụ:

```python
scan_triple_barriers()
assign_triple_barrier_labels()
calibrate_barrier_params()
summarize_label_distribution()
```

Nên đổi:

```text
auto_calibrate_barrier_widths -> calibrate_barrier_params_on_train
```

Vì chữ `auto` không có nhiều ý nghĩa bảo vệ.

---

### Nhóm split/validation

Dùng prefix:

```text
split_      chia dữ liệu
purge_      loại overlap
validate_   kiểm tra điều kiện
```

Ví dụ:

```python
split_train_test_with_purge()
purge_overlapping_events()
validate_no_train_test_overlap()
```

Rất nên có hàm này:

```python
def validate_no_label_leakage(train, test):
    ...
```

Dù đơn giản, nó giúp bạn tự tin khi báo cáo.

---

### Nhóm model

Dùng prefix:

```text
create_     tạo model chưa train
train_      fit model
predict_    dự đoán label/proba
select_     chọn model/features
```

Ví dụ:

```python
create_lightgbm_model()
create_gru_model()
train_base_models()
train_meta_model()
predict_signal_labels()
predict_trade_positions()
```

Tên hiện tại `HybridStackingSignalClassifier` ổn. Nhưng các hàm position nên rõ hơn:

```text
derive_positions_by_confidence -> build_positions_from_class_probability
derive_positions_by_meta_label -> build_positions_from_meta_label
detect_market_regime_filters -> extract_market_regime_inputs
check_market_regime_pass -> passes_market_regime_filter
```

Hiện tại `derive_positions_by_meta_label` đọc hơi trừu tượng. 

---

### Nhóm backtest

Dùng prefix:

```text
open_       mở lệnh
close_      đóng lệnh
calculate_  tính tiền/risk/lot
run_        chạy mô phỏng lớn
record_     tạo log trade
```

Ví dụ:

```python
calculate_position_size()
calculate_trade_pnl()
calculate_trade_cost()
record_trade()
run_barrier_backtest()
calculate_backtest_metrics()
```

Nên đổi:

```text
backtest_signal_positions -> run_barrier_backtest
build_trade_record        -> create_trade_record
aggregate_backtest_metrics -> calculate_backtest_metrics
```

Lý do: `backtest_signal_positions` hơi khó hiểu. “Backtest positions” là gì? Trong khi `run_barrier_backtest` nói rõ đây là backtest có TP/SL barrier.

---

### Nhóm reporting

Dùng prefix:

```text
save_      ghi file
plot_      vẽ hình
build_     dựng dict/table
publish_   xuất toàn bộ report
```

Ví dụ:

```python
save_predictions_csv()
save_trades_csv()
plot_equity_curve()
build_run_summary()
publish_pipeline_report()
```

Tên hiện tại `publish_pipeline_results` khá ổn. Nhưng nếu chỉ ghi file local, `save_pipeline_report` dễ hiểu hơn.

---

## 4. Quy chuẩn tên theo “mức trừu tượng”

Đây là luật rất tốt cho ADHD:

### Hàm cấp cao dùng từ đời thường

```python
run_pipeline()
build_dataset()
train_model()
run_backtest()
save_report()
```

### Hàm cấp trung nói rõ object

```python
build_labeled_train_test_dataset()
train_hybrid_stacking_model()
predict_trade_positions()
calculate_backtest_metrics()
```

### Hàm cấp thấp nói rõ công thức

```python
calculate_trade_pnl()
calculate_position_size()
compute_max_drawdown()
scan_first_barrier_touch()
```

Không trộn cấp cao và cấp thấp trong cùng một hàm.

Ví dụ không nên:

```python
def run_prediction_and_backtest(...):
    tune params
    predict labels
    predict positions
    enforce hold
    run backtest
    return outputs
```

Hàm này đang làm hơi nhiều. 

Nên tách thành:

```python
def run_prediction_stage(...):
    predictions = predict_signal_labels(...)
    positions = predict_trade_positions(...)
    return predictions, positions

def run_backtest_stage(...):
    return run_barrier_backtest(...)

def evaluate_model_on_test_set(...):
    predictions, positions = run_prediction_stage(...)
    backtest = run_backtest_stage(...)
    return EvaluationBundle(...)
```

---

## 5. Quy chuẩn tên biến nên áp dụng

### Không dùng tên quá chung

Tránh:

```python
data
frame
result
output
best
payload
```

Chỉ dùng khi phạm vi rất nhỏ.

Nên dùng:

```python
train_frame
test_frame
feature_frame
labeled_frame
prediction_frame
trade_records
backtest_metrics
run_metadata
```

### Với giá và ATR

Nên rõ đơn vị:

```python
close_price
entry_price
exit_price
atr_rel
atr_price
spread_price
```

### Với PnL

Tách rõ:

```python
bar_pnl_usd
trade_pnl_usd
gross_pnl_usd
net_pnl_usd
realized_pnl_usd
equity_usd
```

### Với label/signal/position

Đây là 3 thứ khác nhau. Không trộn.

```text
label       = ground truth từ triple-barrier
prediction  = model dự đoán label
signal      = hướng model muốn giao dịch
position    = vị thế sau filter, threshold, hold rule
trade       = lệnh thật trong backtest
```

Nên đặt:

```python
true_labels
predicted_labels
raw_signals
filtered_positions
executed_trades
```

---

## 6. Đề xuất structure trong mỗi file

Ví dụ `backtest.py`:

```python
# ============================================================
# Public API
# ============================================================

def run_barrier_backtest(...):
    ...


# ============================================================
# Trade lifecycle
# ============================================================

def should_open_trade(...):
    ...

def should_close_trade(...):
    ...

def create_trade_record(...):
    ...


# ============================================================
# Money calculation
# ============================================================

def calculate_position_size(...):
    ...

def calculate_trade_pnl(...):
    ...

def calculate_trade_cost(...):
    ...


# ============================================================
# Metrics
# ============================================================

def calculate_backtest_metrics(...):
    ...

def calculate_max_drawdown(...):
    ...
```

Với ADHD, cách này tốt hơn package con quá nhiều, vì bạn cuộn trong một file vẫn biết mình đang ở đâu.

---

## 7. Bộ tên hàm tôi khuyên dùng cho toàn chương trình

```text
config.py
  build_pipeline_config()

data.py
  collect_parquet_paths()
  load_tick_data()
  resample_ticks_to_ohlc()
  load_candles_from_parquet()

features.py
  build_feature_frame()
  add_return_features()
  add_trend_features()
  add_momentum_features()
  add_volatility_features()
  add_volume_features()
  clean_feature_frame()

labeling.py
  derive_trailing_swing_levels()
  scan_triple_barriers()
  assign_triple_barrier_labels()
  calibrate_barrier_params_on_train()
  summarize_label_distribution()

dataset.py
  build_labeled_train_test_dataset()
  split_train_test_with_purge()
  validate_dataset_schema()
  validate_no_label_leakage()
  get_feature_columns()

validation.py
  split_purged_embargo_cv()
  purge_overlapping_events()
  walk_forward_split_by_year()

models.py
  create_lightgbm_model()
  create_svc_model()
  create_gru_model()
  create_meta_model()
  train_hybrid_stacking_model()
  predict_signal_labels()
  predict_trade_positions()
  build_positions_from_meta_label()
  apply_market_regime_filter()
  enforce_minimum_position_hold()

backtest.py
  run_barrier_backtest()
  calculate_position_size()
  calculate_trade_pnl()
  calculate_trade_cost()
  create_trade_record()
  calculate_backtest_metrics()

reporting.py
  build_run_summary()
  save_predictions_csv()
  save_trades_csv()
  save_backtest_metrics_csv()
  plot_equity_curve()
  save_pipeline_report()

cli.py
  parse_cli_args()
  run_pipeline()
```

Đây là bộ tên vừa đủ học thuật, vừa dễ đọc.

---

## Kết luận công tâm

Các logic chính của chương trình **không phải rác**. Ý tưởng tổng thể vẫn ổn. Nhưng có vài mismatch cần sửa trước khi báo cáo:

| Mức độ     | Vấn đề                                               | Cần làm                                |
| ---------- | ---------------------------------------------------- | -------------------------------------- |
| Rất nặng   | Equity curve lệch trade PnL                          | Sửa backtest ngay                      |
| Nặng       | Chưa trừ spread/slippage rõ ràng                     | Thêm cost tối giản                     |
| Trung bình | `tune_backtest` không thật sự tune                   | Đổi tên hoặc tắt                       |
| Trung bình | Split/purge có hàm nhưng pipeline chính chưa dùng rõ | Gom lại một đường split                |
| Trung bình | Label `0` bị ép thành `-1`                           | Giải thích rõ trong báo cáo            |
| Nhẹ        | Tên biến/hàm còn trừu tượng                          | Dùng quy chuẩn verb + object + purpose |

Ưu tiên thực tế: **sửa backtest → đổi tên PnL/equity → chuẩn hóa pipeline naming → rồi mới refactor file**. Không làm ngược lại. Refactor trước khi logic đúng sẽ chỉ làm bạn mệt hơn.
