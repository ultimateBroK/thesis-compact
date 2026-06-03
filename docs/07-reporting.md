---
doc: 07-reporting
stage: reporting
thesis_chapter: 3
status: draft
last_updated: 2026-06-02
code_ref: src/reporting.py
---

# Reporting

> Lưu artifact mỗi lần chạy pipeline: CSV predictions/trades/feature importance, JSON metadata, PNG figures. Module `src/reporting.py` (601 dòng) — entrypoint `publish_pipeline_results`.

## Tóm tắt

Mỗi lần chạy pipeline sinh một directory `reports/run_{timestamp}/` chứa đầy đủ artifact để reproduce và phân tích sau này: file `run_data.json` chứa metadata + config snapshot + OOF scores + backtest metrics + reproducibility (Python version, git commit), các CSV chi tiết (`predictions.csv`, `trades.csv`, `feature_importance.csv`, `backtest_metrics.csv`), và figures PNG (`equity_curve.png`, `oof_scores.png`, `feature_importance.png`). Module cũng in báo cáo tóm tắt ra console trong khi chạy. Walk-forward windows chia artifact theo sub-run riêng (mỗi window = một `run_data.json`).

## Cơ sở lý thuyết

Nguyên tắc *reproducible research*: mỗi experiment run phải tự lưu đủ metadata để tái lập chính xác kết quả \cite{de_prado_2018_afml}. Module `src/reporting.py` triển khai ba lớp: (i) **console output** cho real-time monitoring trong quá trình chạy; (ii) **CSV artifact** cho post-hoc analysis bằng pandas/polars; (iii) **JSON metadata** (`run_data.json`) tổng hợp toàn bộ ngữ cảnh run — config, dataset stats, model scores, backtest metrics, git commit — để so sánh giữa các run. Cách đọc và quy ước đặt tên report ở `docs/21-results-convention.md` (sẽ viết Phase 4).

## Công thức

Không có công thức toán học — module thuần I/O + serialization. Cấu trúc serialize:

$$
\text{run\_data.json} = \text{RunMetadata}(\text{run\_id}, \text{timestamp}, \text{config}, \text{dataset}, \text{training}, \text{evaluation}, \text{backtest}, \text{feature\_importance}, \text{trade\_summary}, \text{artifacts}, \text{reproducibility}).
$$

Mỗi field là `@dataclass` (`DatasetMeta`, `TrainingMeta`, `EvalMeta`, `WinRateMeta`), serialized qua `dataclasses.asdict` + `json.dump`. Git info được capture qua `subprocess.check_output(["git", ...])`.

## Cài đặt

### Directory layout

```
reports/
└── run_20260602_153045/                  # timestamp UTC
    ├── run_data.json                      # master metadata
    ├── figures/
    │   ├── equity_curve.png               # matplotlib Figure(9, 4) dpi=160
    │   ├── oof_scores.png                 # Figure(8, 4) dpi=160
    │   └── feature_importance.png         # Figure(10, 8) dpi=160
    └── tables/
        ├── predictions.csv                # per-bar prediction + position + pnl
        ├── trades.csv                     # per-trade record
        ├── backtest_metrics.csv           # metric dict
        └── feature_importance.csv         # LightGBM gain
```

Walk-forward mode: mỗi window sinh một sub-directory hoặc một `run_data.json` riêng với `window_id`, `window_train_range`, `window_test_range` field — tích hợp vào `RunMetadata.backtest`.

### File schemas

#### `run_data.json`

```jsonc
{
  "run_id": "run_20260602_153045",
  "timestamp": "2026-06-02T15:30:45+00:00",
  "config": { ... },                       // PipelineConfig.to_dict() snapshot
  "dataset": {                             // DatasetMeta
    "total_rows": 8500, "train_rows": 6900, "test_rows": 1500,
    "feature_count": 21, "features": [...],
    "fractional_d": 0.4,
    "data_range": {"start": "...", "end": "..."},
    "train_date_range": {...}, "test_date_range": {...},
    "label_distribution_total": {"-1": 5300, "1": 3200},
    "label_distribution_train": {...}, "label_distribution_test": {...}
  },
  "training": {                            // TrainingMeta
    "oof_scores": {"gru": 0.40, "lightgbm": 0.42, "svc": 0.34},
    "per_class_oof_f1": {"gru": {"-1": 0.43, "1": 0.37}, ...},
    "active_models": ["gru", "lightgbm"],
    "filtered_models": ["svc"]
  },
  "evaluation": {                          // EvalMeta
    "accuracy": 0.65, "f1_macro": 0.43,
    "confusion_matrix": {"labels": [-1, 1], "matrix": [[...], [...]]}
  },
  "backtest": {                            // backtest_metrics + WinRateMeta + window
    "total_return": 0.184, "sharpe": 1.52, "max_drawdown": -0.073,
    "profit_factor": 2.14, "win_rate": {"value": 0.47, "turnover": 0.073},
    "trades": 62, "trade_signals": 84,
    "sortino": 2.01, "dsr_statistic": 1.87, "dsr_p_value": 0.031,
    "window_id": null, "window_train_range": "", "window_test_range": ""
  },
  "feature_importance": {"close_fracdiff": 18.5, "bb_position": 13.7, ...},
  "trade_summary": {"total_trades": 62, "wins": 29, "losses": 33,
                    "avg_bars_held": 8.4, "avg_pnl_usd": 29.6,
                    "avg_win_pnl_usd": 185.2, "avg_loss_pnl_usd": -112.3,
                    "max_win_usd": 542.1, "max_loss_usd": -198.7,
                    "long_trades": 35, "short_trades": 27, "avg_overnights": 1.2},
  "artifacts": {"files": [...], "figure_count": 3},
  "reproducibility": {
    "python_version": "3.12.x",
    "python_version_full": "3.12.4",
    "python_build": "main",
    "platform": "Linux ...",
    "git_commit": "abc1234",
    "git_branch": "main",
    "git_dirty": false,
    "run_entrypoint": "cli"
  }
}
```

#### `predictions.csv`

| Cột | Dtype | Mô tả |
|---|---|---|
| `timestamp` | str | ISO datetime |
| `close` | float | Close price |
| `spread` | float | Spread tại bar |
| `label` | int | Ground truth $\in \{-1, +1\}$ |
| `prediction` | int | Predicted label |
| `position` | int | Position sau meta-label + min_hold $\in \{-1, 0, +1\}$ |
| `bar_pnl_usd` | float | PnL bar-to-bar `diff(equity, prepend=equity[0])` |
| `equity_usd` | float | Equity cumulative |

#### `trades.csv`

Sinh từ `build_trades_dataframe(executed_trades, timestamps)` — chuyển `entry_idx/exit_idx` sang timestamp string:

| Cột | Dtype | Mô tả |
|---|---|---|
| `entry_time` / `exit_time` | str | ISO datetime |
| `direction` | str | `"LONG"` / `"SHORT"` |
| `entry_price` / `exit_price` | float | Mid price tại entry/exit |
| `lots` | float | Lot thực tế (sau round + clamp) |
| `bars_held` | int | Số bar giữ |
| `overnights` | int | Số overnight UTC crossed |
| `gross_pnl_usd` | float | PnL trước cost |
| `spread_cost_usd` | float | Half-spread entry + half-spread exit |
| `commission_usd` | float | Commission hai chiều |
| `swap_usd` | float | Overnight swap |
| `trade_pnl_usd` | float | Net = gross $-$ spread $-$ commission $-$ swap |
| `cost_usd` | float | Tổng cost = spread + commission + swap |
| `win` | bool | `trade_pnl_usd > 0` |

#### `feature_importance.csv`

Sinh từ `extract_lightgbm_feature_importance(model, features)`:

| Cột | Dtype | Mô tả |
|---|---|---|
| `rank` (index) | int | 1-based |
| `feature` | str | Tên feature |
| `importance` | int | LightGBM gain (split count weighted) |
| `pct` | float | `%` của tổng importance |

#### Figures

Tất cả figure sinh bằng `matplotlib.figure.Figure` (không cần pyplot), dpi=160, `tight_layout()`:

- **`equity_curve.png`** — `save_equity_curve_plot(equity, path)`: line plot equity, màu `#1f77b4`, figsize $(9, 4)$.
- **`oof_scores.png`** — `save_oof_scores_bar_plot(model, path)`: horizontal bar chart F1 per base learner, màu xanh (`#2ca02c`) cho active / đỏ (`#d62728`) cho filtered, figsize $(8, 4)$.
- **`feature_importance.png`** — `save_feature_importance_bar_plot(importance_df, path)`: top 20 feature, màu đậm (`#1f77b4`) cho pct $\geq 5\%$, nhạt (`#aec7e8`) cho pct $< 5\%$, vertical line tại $5\%$, figsize $(10, 8)$.

### Reproducibility field

`build_run_metadata` capture:

- `python_version`, `python_version_full`, `python_build` từ `sys.version` + `platform.python_build()`.
- `platform` từ `platform.platform()`.
- `git_commit` = `git rev-parse HEAD`.
- `git_branch` = `git branch --show-current`.
- `git_dirty` = `bool(git status --short)` — True nếu có uncommitted change.
- `run_entrypoint` = `"cli"` (hardcode — single entrypoint).

Lệnh git fail silently (`subprocess.DEVNULL` stderr) — field = `None` nếu không phải git repo.

### Pipeline chính

```
publish_pipeline_results(accelerator, config_payload, outputs, window_id, ...)
├── In console:
│   ├── print_device_acceleration_report
│   ├── print_dataset_report(labeled_full, train, test, n_features)
│   ├── print_model_filtering_report(model)
│   ├── print_classification_report(test[label], predictions)
│   ├── print_feature_importance_report(...)
│   └── print_backtest_metrics_report(metrics)
└── save_run_artifacts(run_dir, outputs, config_payload, window_id, ...)
    ├── mkdir run_dir/figures
    ├── build results DataFrame (test + prediction + position + pnl + equity)
    ├── results.to_csv(predictions.csv)
    ├── build_trades_dataframe(executed_trades, timestamps).to_csv(trades.csv)
    │   └── fallback: extract_trades_from_positions(results) nếu executed_trades=None
    ├── backtest_metrics → backtest_metrics.csv
    ├── save_feature_importance_csv → feature_importance.csv
    ├── save_feature_importance_bar_plot → figures/feature_importance.png
    ├── save_oof_scores_bar_plot → figures/oof_scores.png
    ├── save_equity_curve_plot → figures/equity_curve.png
    ├── collect_artifact_files(run_dir, figures_dir) + ["run_data.json"]
    └── build_run_metadata(...).asdict() → json.dump → run_data.json
```

### Code refs

Console: `src/reporting.py::print_dataset_report`, `::print_model_filtering_report`, `::print_classification_report`, `::print_backtest_metrics_report`, `::print_device_acceleration_report`, `::print_feature_importance_report`, `::determine_model_status`. Trade extraction: `::extract_trades_from_positions`, `::build_trades_dataframe`. Feature importance: `::extract_lightgbm_feature_importance`, `::save_feature_importance_csv`. Plots: `::save_oof_scores_bar_plot`, `::save_equity_curve_plot`, `::save_feature_importance_bar_plot`. Metadata dataclasses: `::DatasetMeta`, `::TrainingMeta`, `::EvalMeta`, `::WinRateMeta`, `::RunMetadata`. Metadata builder: `::build_run_metadata`, `::build_dataset_metadata`, `::build_training_metadata`, `::build_evaluation_metadata`, `::build_win_rate_metadata`, `::build_date_range`, `::build_label_counts`, `::collect_artifact_files`. Public: `::publish_pipeline_results`, `::save_run_artifacts`.

## Tham số quan trọng

| Tham số | Giá trị | Vị trí | Lý do |
|---|---|---|---|
| `REPORT_DIR` | `Path("reports")` | `src/config.py:12` | Root thư mục report |
| Run dir name format | `run_{YYYYMMDD_HHMMSS}` | `publish_pipeline_results` (default) | UTC timestamp |
| Figure DPI | $160$ | `save_*_plot(dpi=160)` | Cân bằng giữa file size và chất lượng in |
| Top feature plot | $20$ | `save_feature_importance_bar_plot(head(20))` | Đủ thấy đa số important features |

## Kết quả thực nghiệm

Một run 12 tháng điển hình sinh:

| Artifact | Kích thước | Số dòng |
|---|---|---|
| `predictions.csv` | $\approx 90$ KB | $\approx 1\,500$ (test rows) |
| `trades.csv` | $\approx 6$ KB | $\approx 60$ |
| `feature_importance.csv` | $\approx 1$ KB | $21$ |
| `backtest_metrics.csv` | $< 1$ KB | $7$ dòng |
| `run_data.json` | $\approx 25$ KB | — |
| `figures/*.png` | $\approx 100$ KB / file | $3$ files |

Tổng dung tích mỗi run $\approx 0.5$ MB — lưu trữ 100 run gần như miễn phí. Walk-forward 3 window $\approx 1.5$ MB.

## Tham khảo

- `\cite{de_prado_2018_afml}` — López de Prado, *Advances in Financial Machine Learning*, Wiley 2018, Ch. 11 (backtest reporting).
- `docs/21-results-convention.md` — quy ước đọc và so sánh reports (Phase 4).
- `docs/22-evaluation-metrics.md` — định nghĩa metrics trong `run_data.json` (Phase 4).
- `docs/05-models-stacking.md` — `oof_scores_`, `active_model_names_`.
- `docs/06-backtest.md` — `backtest_metrics`, `executed_trades`.
- `docs/08-config.md` — bảng đầy đủ tham số.
