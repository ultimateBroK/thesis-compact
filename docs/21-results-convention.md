---
doc: 21-results-convention
stage: results
thesis_chapter: 4
status: draft
last_updated: 2026-06-02
code_ref: null
---

# Results Convention

> Cách đọc `reports/run_*/`, schema JSON, cột CSV, format bảng so sánh multi-run — quy ước chuẩn cho luận văn.

## Tóm tắt

Mỗi lần chạy pipeline (CLI hoặc `viz.ipynb`) sinh một thư mục `reports/run_<YYYYMMDD>_<HHMMSS>/` chứa `run_data.json` (metadata đầy đủ), CSV files trong `tables/`, và figure PNG trong `figures/`. Document này đặc tả schema từng file và format bảng chuẩn để so sánh nhiều run trong luận văn.

## Cơ sở lý thuyết

Artifact layout tuân theo nguyên tắc immutable + self-contained: mỗi run chứa đủ thông tin để reproduce (git commit, python version, config snapshot trong `run_data.json`). Không ghi đè run cũ. Tra cứu qua `run_id` (timestamp) — đơn điệu tăng theo thời gian chạy thực.

## Công thức

### Cấu trúc thư mục

```
reports/run_20260603_181525/
├── run_data.json              # metadata đầy đủ (schema dưới)
├── figures/                   # tất cả figure PNG
│   ├── equity_curve.png       # đường equity + buy & hold
│   ├── feature_importance.png # bar chart top-20
│   ├── oof_scores.png         # bar chart F1 OOF các base learner
│   ├── confusion_matrix.png
│   ├── label_distribution.png
│   ├── fracdiff_comparison.png
│   ├── technical_indicators.png
│   ├── feature_correlation.png
│   ├── feature_distributions_by_label.png
│   ├── cv_splits.png
│   ├── prediction_accuracy_map.png
│   ├── model_architecture.png
│   ├── equity_drawdown_positions.png
│   ├── pnl_analysis.png
│   ├── rolling_performance.png
│   ├── classification_diagnostics.png
│   ├── shap_bar.png
│   ├── shap_beeswarm.png
│   ├── model_comparison.png
│   ├── test_evaluation.png
│   ├── learning_curve.png
│   ├── walk_forward_quarterly.png
│   ├── adf_stationarity.png
│   ├── base_learner_diversity.png
│   ├── bootstrap_significance.png
│   └── ablation_study.png
└── tables/                    # tất cả CSV files
    ├── predictions.csv          # per-bar: timestamp, close, prediction, position, equity_usd
    ├── trades.csv               # per-trade: entry_time, exit_time, direction, prices, pnl
    ├── backtest_metrics.csv     # 1-row CSV: total_return, sharpe, max_drawdown, ...
    ├── feature_importance.csv   # rank, feature, importance, pct
    ├── data_statistics.csv     # dataset metadata
    ├── feature_description.csv  # feature documentation
    ├── hyperparameters.csv      # config snapshot
    ├── per_class_metrics.csv   # per-class precision/recall/f1
    ├── roc_auc.csv              # ROC-AUC score
    └── trade_statistics.csv    # trade-level aggregates
```

### `run_data.json` schema

Top-level keys (dataclass `RunMetadata` trong `src/reporting.py`):

| Key | Kiểu | Mô tả |
|---|---|---|
| `run_id` | str | `run_<YYYYMMDD>_<HHMMSS>` |
| `timestamp` | ISO 8601 UTC | Thời điểm chạy |
| `config` | dict | Snapshot toàn bộ `src/config.py` (months, cv_splits, fractional_d, tp_atr, sl_atr, …) |
| `dataset` | DatasetMeta | total/train/test rows, `fractional_d`, `feature_count`, `features` list, `data_range`, `train_date_range`, `test_date_range`, `label_distribution_train`, `label_distribution_test` |
| `training` | TrainingMeta | `oof_scores` (dict base → F1), `per_class_oof_f1`, `active_models`, `filtered_models` |
| `evaluation` | EvalMeta | `accuracy`, `f1_macro`, `confusion_matrix` |
| `backtest` | dict | `total_return`, `sharpe`, `max_drawdown`, `profit_factor`, `win_rate`, `trades`, `trade_signals`, `turnover` |
| `feature_importance` | dict feature → pct | LightGBM importance |
| `trade_summary` | dict | `total_trades`, `wins`, `losses`, `win_rate`, `avg_bars_held`, `avg_pnl_usd`, `avg_win_pnl_usd`, `avg_loss_pnl_usd`, `max_win_usd`, `max_loss_usd`, `long_trades`, `short_trades`, `avg_overnights` |
| `artifacts` | dict | `files` (list paths), `figure_count` |
| `reproducibility` | dict | `python_version`, `platform`, `git_commit`, `git_branch`, `git_dirty`, `run_entrypoint` |
| `timing` | dict | `data_loading`, `model_training`, `shap_analysis`, `ablation_study`, `baseline_training`, `bootstrap_significance`, `backtesting`, `total` (giây) |

### `predictions.csv` columns

| Cột | Kiểu | Mô tả |
|---|---|---|
| `timestamp` | str | ISO 8601, hourly bar |
| `close` | float | Close price XAU/USD |
| `spread` | float | Spread tại bar đó |
| `label` | int | Ground-truth $\{-1, +1\}$ |
| `prediction` | int | Stacking output $\{-1, +1\}$ |
| `position` | float | Vị thế thực tế sau meta-label + regime filter $\{-1, 0, +1\}$ |
| `bar_pnl_usd` | float | PnL tại bar đó |
| `equity_usd` | float | Equity running |

### `trades.csv` columns

| Cột | Kiểu | Mô tả |
|---|---|---|
| `entry_time` | str | Timestamp vào lệnh |
| `exit_time` | str | Timestamp ra |
| `direction` | str | `LONG` hoặc `SHORT` |
| `entry_price` | float | Mid price entry |
| `exit_price` | float | Mid price exit |
| `lots` | float | Lot thực tế |
| `bars_held` | int | Số bar giữ lệnh |
| `overnights` | int | Số overnight UTC crossed |
| `gross_pnl_usd` | float | PnL trước cost |
| `spread_cost_usd` | float | Round-trip spread cost |
| `commission_usd` | float | Commission hai chiều |
| `swap_usd` | float | Overnight swap |
| `trade_pnl_usd` | float | Net = gross $-$ spread $-$ commission $-$ swap |
| `cost_usd` | float | Tổng cost |
| `win` | bool | `trade_pnl_usd > 0` |

### `feature_importance.csv` columns

| Cột | Kiểu | Mô tả |
|---|---|---|
| `rank` | int | 1 = cao nhất |
| `feature` | str | Tên feature (e.g. `obv`, `volatility_24`) |
| `importance` | int | LightGBM split count |
| `pct` | float | `%` tổng importance |

### Figures (`figures/*.png`)

3 file cố định sinh bởi `src/reporting.py::save_run_artifacts`:

- `equity_curve.png` — đường equity, 9×4 inches (figsize), DPI 160.
- `feature_importance.png` — top-20 feature bar chart, 1000×800.
- `oof_scores.png` — OOF F1 các base learner, 800×400.

Nếu chạy qua `viz.ipynb` (`run_entrypoint: notebook`), thêm 24 figure khác (label_distribution, fracdiff_comparison, technical_indicators, feature_correlation, feature_distributions_by_label, cv_splits, confusion_matrix, prediction_accuracy_map, model_architecture, equity_drawdown_positions, pnl_analysis, rolling_performance, classification_diagnostics, shap_bar, shap_beeswarm, model_comparison, test_evaluation, learning_curve, walk_forward_quarterly, adf_stationarity, base_learner_diversity, bootstrap_significance, ablation_study) — notebook tự export vào `figures/`. Tổng: 27 figures.

## Cài đặt

### Naming convention

Mặc định: `run_<YYYYMMDD>_<HHMMSS>` (UTC, không tag). Nếu thêm tag system trong tương lai: `run_<YYYYMMDD>_<HHMMSS>_<tag>` với `tag` ∈ {`baseline`, `stacking`, `meta`, `noTune`, …}. Hiện chưa cài tag — coi `config` trong `run_data.json` là source of truth để phân loại experiment.

### Bảng so sánh multi-run (chuẩn luận văn)

| Run ID | Timestamp (UTC) | Months | Seed | F1 OOF (best base) | Sharpe | MDD | Win Rate | PF | Trades |
|---|---|---|---|---|---|---|---|---|---|
| `run_20260603_181525` | 2026-06-03T11:15 | 60 | 42 | 0.7168 | +0.689 | −0.051 | 0.443 | 1.26 | 106 |

Sắp xếp theo `Months` tăng dần rồi `Timestamp` — dễ theo dõi khi mở rộng sang seed sweep.

### Ablation comparison format

Bảng ablation (E0–E5, xem `20-experiments.md`):

| Experiment | F1 OOF | F1 test | Sharpe | MDD | PF | Win Rate |
|---|---|---|---|---|---|---|
| E0 baseline | — | — | — | — | — | — |
| E1 + TB | — | — | — | — | — | — |
| E2 + meta | — | — | — | — | — | — |
| E3 + purged | — | — | — | — | — | — |
| E4 full | 0.7168 | 0.7331 | +0.689 | −0.051 | 1.26 | 0.443 |
| E5 no-tune | — | — | — | — | — | — |

Mỗi giá trị = trung bình ± std trên 4 seed. Điền `_` cho cell chưa chạy.

### Snippet load multi-run

```python
import json
from pathlib import Path
import pandas as pd

runs = []
for path in sorted(Path("reports").glob("run_*/run_data.json")):
    data = json.loads(path.read_text())
    runs.append({
        "run_id": data["run_id"],
        "timestamp": data["timestamp"],
        "months": data["config"].get("months"),
        "seed": data["config"].get("random_state"),
        "f1_oof_best": max(data["training"]["oof_scores"].values()),
        "f1_macro_test": data["evaluation"]["f1_macro"],
        "sharpe": data["backtest"]["sharpe"],
        "max_drawdown": data["backtest"]["max_drawdown"],
        "win_rate": data["backtest"]["win_rate"],
        "profit_factor": data["backtest"]["profit_factor"],
        "trades": data["backtest"]["trades"],
    })
df = pd.DataFrame(runs)
print(df.to_markdown(index=False))
```

Lưu vào `reports/comparison.csv` để tra cứu nhanh.

## Tham số quan trọng

- **Timezone**: tất cả timestamp trong `run_data.json` và CSV là **UTC** (data Dukascopy chuẩn UTC). Khi trích xuất ra bảng luận văn, có thể ghi thêm cột local time (VN = UTC+7) nhưng không override raw data.
- **Figure DPI**: 160 cho PNG export (config trong `src/reporting.py::save_*_plot`). Đủ chất lượng cho thesis PDF (300 DPI yêu cầu in ấn — convert sang SVG nếu cần).
- **JSON encoding**: UTF-8, indent=2, `ensure_ascii=False` — hỗ trợ ghi chú tiếng Việt trong config payload nếu cần.

## Kết quả thực nghiệm

Hiện có 1 run trong `reports/` (`run_20260603_181525`). Run tham chiếu chính: `run_20260603_181525` (E4 full, 60 tháng, seed 42, notebook entrypoint). Bảng multi-run đầy đủ sẽ bổ sung sau khi hoàn tất ablation E0–E5 × seed {21, 42, 63, 84}.

## Tham khảo

- \cite{de_prado_2018_backtest} — López de Prado, quy ước artifact và reproducibility cho backtest.
- \cite{pedregosa_2011_sklearn} — scikit-learn, schema `classification_report` và `confusion_matrix`.
