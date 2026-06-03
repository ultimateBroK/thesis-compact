---
doc: 08-config
stage: config
thesis_chapter: 3
status: draft
last_updated: 2026-06-02
code_ref: null
---

# Configuration — Bảng tham số toàn diện

> Mọi hyperparameter trong `src/config.py` được liệt kê với giá trị, module sử dụng, và rationale — single source of truth, không hardcode ở module khác.

## Tóm tắt

Toàn bộ tham số pipeline được tập trung trong `src/config.py` (69 dòng). Module khác import trực tiếp — không hardcode giá trị lặp. Bảng dưới phân loại theo sáu nhóm: data, feature, labeling, model, training, backtest. Mọi tham số được log vào `reports/run_*/run_data.json` mỗi lần chạy để đảm bảo reproducibility.

## Cơ sở lý thuyết

Tập trung hóa cấu hình mang lại ba lợi ích:

1. **Reproducibility**: thay đổi giá trị tại một chỗ duy nhất, mọi module kế thừa.
2. **Ablation rõ ràng**: một tham số = một experiment dimension, dễ quay grid.
3. **Audit trail**: `run_data.json` snapshot toàn bộ config mỗi run.

Pattern `from src.config import X` được áp dụng đồng nhất trong `src/cli.py`, `src/dataset.py`, `src/labeling.py`, `src/models.py`, `src/backtest.py`, `src/validation.py`.

## Công thức

`PipelineConfig` là một `dataclass(frozen=True)` cho phép override qua CLI mà không phá immutability:

```python
@dataclass(frozen=True)
class PipelineConfig:
    months: int | None = 12
    timeframe: str = "1h"
    walk_forward: bool = False
    long_only: bool = False
    n_windows: int = 3
```

## Cài đặt

### Cách thay đổi tham số

1. **Sửa `src/config.py`** — cho thay đổi permanent, commit vào repo.
2. **CLI flag** — cho experiment ad-hoc: `python main.py --months 6 --seed 123`.
3. **KHÔNG hardcode** ở module khác. Nếu cần giá trị mới, thêm constant vào `src/config.py`.

### Logging

Mỗi lần chạy, `src/reporting.py` ghi snapshot config vào `run_data.json`:

```json
{
  "config": {
    "timeframe": "1h",
    "fractional_d": 0.4,
    "cv_splits": 5,
    "embargo_pct": 0.02,
    "min_oof_f1": 0.50,
    "use_meta_labeling": true,
    "...": "..."
  },
  "run_id": "run_20260602_153045",
  "git_sha": "abc1234"
}
```

## Tham số quan trọng

### Bảng đầy đủ

| Tên | Giá trị | Loại | Module sử dụng | Rationale |
|---|---|---|---|---|
| `DATA_DIR` | `Path("data/XAUUSD")` | data | `src/data.py`, `src/cli.py` | Parquet directory chuẩn |
| `REPORT_DIR` | `Path("reports")` | data | `src/reporting.py` | Output directory cho `run_*` |
| `TIMEFRAME` | `"1h"` | data | `src/data.py`, `src/dataset.py` | Cân bằng granularity vs noise |
| `FRACTIONAL_D` | $0.4$ | feature | `src/features.py` | ADF pass, retention $\approx 0.91$ (xem `10-methodology-fracdiff.md`) |
| `CV_SPLITS` | $5$ | training | `src/validation.py` | Standard financial CV \cite{de_prado_2018_cross_val} |
| `EMBARGO_PCT` | $0.02$ | training | `src/validation.py` | $\sim 24$ nến cho dataset $\sim 1200$, đủ triple-barrier horizon |
| `PURGE_PCT` | $0.02$ | training | `src/dataset.py` | Purge giữa tune/test split |
| `TEST_SIZE` | $0.20$ | training | `src/dataset.py` | 20% data cuối làm test set |
| `MIN_OOF_F1` | $0.50$ | model | `src/models.py` | Ngưỡng smart filtering — skip fold nếu OOF F1 dưới ngưỡng, tránh underfit. Lưu ý: 3 nguồn default khác nhau: config 0.50, HybridStackingSignalClassifier ctor 0.34, ablation doc sử dụng 0.36 |
| `CONFIDENCE_THRESHOLD` | $0.45$ | model | `src/models.py`, `src/backtest.py` | Ngưỡng confidence cho position sizing — chỉ take trade khi $|p - 0.5| > 0.15$ |
| `RANDOM_STATE` | $42$ | training | mọi module | Seed gốc, lan truyền xuống NumPy/Torch/LightGBM/sklearn |
| `LABELS` | `np.array([-1, 1])` | labeling | `src/models.py` | Binary long/short — bỏ class 0 do meta-labeling filter |
| `TUNE_TP_RANGE_BT` | $(3.0, 15.0, 1.0)$ | backtest | `src/backtest.py` | Grid search TP ATR multiplier — single source of truth cho TP barrier |
| `TUNE_SL_RANGE_BT` | $(3.0, 15.0, 1.0)$ | backtest | `src/backtest.py` | Grid search SL ATR multiplier — single source of truth cho SL barrier |
| `TUNE_HOLD_VALUES` | $[6, 8, 12, 16]$ | backtest | `src/backtest.py` | Grid search min hold values — single source of truth cho min hold |
| `INITIAL_BALANCE` | $\$10{,}000$ | backtest | `src/backtest.py` | Vốn giả lập chuẩn cho retail XAU/USD |
| `CONTRACT_SIZE` | $100.0$ | backtest | `src/backtest.py` | 1 lot XAU/USD = 100 oz |
| `RISK_PER_TRADE` | $0.02$ | backtest | `src/backtest.py` | 2% balance risk per trade — prudent money management |
| `LEVERAGE` | $100$ | backtest | `src/backtest.py` | 1:100 leverage — retail XAU/USD tiêu biểu, margin = notional / 100 |
| `LOT_STEP` | $0.01$ | backtest | `src/backtest.py` | Broker lot increment — lot snap về bội số gần nhất |
| `LOT_MIN` | $0.01$ | backtest | `src/backtest.py` | Minimum lot — 1 micro lot XAU |
| `LOT_MAX` | $5.0$ | backtest | `src/backtest.py` | Soft cap lot — margin guard áp hard cap thực tế |
| `COMMISSION_PER_LOT_SIDE` | $\$0$ | backtest | `src/backtest.py` | Commission USD/lot/side — 0 cho spread-only broker |
| `SWAP_LONG_USD_PER_LOT` | $-\$2.50$ | backtest | `src/backtest.py` | Swap long USD/lot/overnight — tiêu biểu XAU/USD retail |
| `SWAP_SHORT_USD_PER_LOT` | $-\$1.00$ | backtest | `src/backtest.py` | Swap short USD/lot/overnight — short thường rẻ hơn long |
| `SHORT_LOT_SCALE` | $0.20$ | backtest | `src/backtest.py` | SHORT position scale factor — giảm SHORT exposure do asymmetric risk |
| `N_TUNING_TRIALS_APPROX` | $700$ | backtest | `src/backtest.py` | Approximate grid search combinations |
| `USE_META_LABELING` | `True` | labeling | `src/models.py`, `src/labeling.py` | Position sizing thông qua binary filter secondary model \cite{kearns_2019_meta} |
| `META_LABEL_THRESHOLD` | $0.55$ | labeling | `src/models.py` | Ngưỡng long meta-label — take position khi meta prob $\geq 0.55$ |
| `SHORT_META_LABEL_THRESHOLD` | $0.55$ | labeling | `src/models.py` | Ngưỡng short rất cao — short chỉ khi very confident, hạn chế SHORT loss |
| `BB_WIDTH_MIN_MULT` | $1.2$ | labeling | `src/labeling.py` | Skip label khi BB width $< 1.2 \cdot$ rolling mean — sideway regime filter |
| `SWING_WINDOW` | $5$ | labeling | `src/labeling.py` | Window xác định swing high/low cho barrier placement |
| `LABELING_HORIZON` | $24$ | labeling | `src/labeling.py` | Vertical barrier = 24 nến (1 ngày) |
| `TUNE_TP_RANGE` | $(0.5, 4.0, 0.25)$ | labeling | `src/labeling.py` | Grid search TP barrier ATR multiplier — **labeling range nhỏ** (0.5–4.0), khác với `TUNE_TP_RANGE_BT` (3.0–15.0) dùng cho backtest barrier scaling |
| `TUNE_SL_RANGE` | $(0.5, 4.0, 0.25)$ | labeling | `src/labeling.py` | Grid search SL barrier ATR multiplier |
| `TUNE_TARGET_BALANCE` | $0.35$ | labeling | `src/labeling.py` | Target label balance $35/65$ cho long/short — reflect market upward bias |
| `ADX_THRESHOLD` | $20.0$ | labeling | `src/labeling.py` | Skip label khi ADX $< 20$ — weak trend, low signal-to-noise |
| `TREND_FILTER_ENABLED` | `True` | labeling | `src/labeling.py` | Bật trend filter EMA(89) |
| `TREND_EMA_PERIOD` | $89$ | labeling | `src/labeling.py` | Long-term trend EMA — chuẩn Fibonacci, multi-day trend |

### Phân loại theo nhóm

| Nhóm | Số tham số | Mục đích |
|---|---|---|
| data | 3 | Đường dẫn, timeframe |
| feature | 1 | Fractional differencing |
| labeling | 12 | Triple-barrier, meta-labeling, tuning |
| model | 3 | Smart filtering, confidence, label set |
| training | 5 | CV, embargo, purge, test size, seed |
| backtest | 18 | TP/SL/hold grid + CFD execution (leverage, lot, commission, swap) + balance/risk + SHORT_LOT_SCALE + N_TUNING_TRIALS_APPROX |

## Kết quả thực nghiệm

So sánh OOF F1 khi thay đổi `MIN_OOF_F1` (12 tháng):

| `MIN_OOF_F1` | Số fold pass | OOF F1 (deploy) | Nhận xét |
|---|---|---|---|
| $0.30$ | $5/5$ | $0.39$ | Tất cả fold — có fold underfit kéo xuống |
| $0.36$ (ablation doc) | $4/5$ | $0.42$ | Loại 1 fold yếu, F1 cải thiện |
| $0.50$ (mặc định config) | $4/5$ | $0.42$ | Config default — loại 1 fold yếu |
| $0.42$ | $2/5$ | $0.40$ | Quá khắt khe — mất diversity |

## Tham khảo

- `\cite{de_prado_2018_cross_val}` — purged CV, embargo.
- `\cite{kearns_2019_meta}` — meta-labeling cho position sizing.
