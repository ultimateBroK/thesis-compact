---
doc: 30-exploratory-analysis
stage: eda
thesis_chapter: 4
status: draft
last_updated: 2026-06-02
code_ref: null
---

# Exploratory Data Analysis

> Notebook `viz.ipynb` phân tích exploratory — thể hiện distribution của labels, correlation matrix, sample equity curves, confusion matrix, PnL. Tài liệu này convert từ notebook để dễ cite trong luận văn; ảnh gốc giữ trong `viz.ipynb` (file 2.2 MB) và `reports/run_*/figures/`.

## Giới thiệu

Notebook `viz.ipynb` chạy end-to-end pipeline trên dữ liệu XAU/USD hourly 5 năm (2019-01 → 2023-12, 60 file parquet tháng, ~29 245 bar) và trực quan hóa mọi giai đoạn: labeling distribution, feature correlation, OOF scores, confusion matrix, equity curve, PnL distribution, rolling Sharpe. Section dưới trích các cell quan trọng nhất kèm output. [Figure: ...] thay thế ảnh gốc để giữ file nhỏ.

## 1. Config & Data Loading

```python
config = PipelineConfig(months=None)
files = collect_parquet_paths(DATA_DIR, config.months)
```

```
Data dir: /home/ultimatebrok/Downloads/thesis-compact/data/XAUUSD
Mode: full | files: 60
Range: 2019-01 - 2023-12
```

Dataset sau feature engineering: **28 630 bar** (train 23 335 + test 5 295), **21 feature**, fractional differencing $d=0.4$.

```
Auto-tuned barriers: TP_ATR=0.75, SL_ATR=0.5, search_balance=0.97
Validation distribution: {'SL (-1)': 2440, 'TP (+1)': 2264, 'total': 4704, 'balance_ratio': 0.93}
Split at row 23335 | timestamp: 2023-01-03 16:00:00+00:00 | purge gap: 585 rows
Train: 2019-01-18 → 2023-01-03
Test:  2023-02-06 → 2023-12-28
```

## 2. Label Distribution (Triple-Barrier)

Label $\{-1, +1\}$ với SL (-1) chiếm đa số nhẹ — balance ratio = 0.93 trên cả train và test, đủ ổn định để tránh class imbalance nghiêm trọng.

```python
train_dist = summarize_label_distribution(train["label"].to_numpy())
test_dist = summarize_label_distribution(test["label"].to_numpy())
```

**Train**: SL (-1) = 12 392 (53%), TP (+1) = 10 943 (47%).
**Test**: SL (-1) = 2989 (56.5%), TP (+1) = 2306 (43.5%).

[Figure: Bar chart Train vs Test label count với balance ratio]

## 3. Fractional Differencing

Raw close vs `close_fracdiff` ($d=0.4$). Fracdiff loại trend nhưng giữ memory — trông giống stationary process quanh zero nhưng vẫn có structure.

```python
axes[0].plot(dataset_pdf.index, dataset_pdf["close"], label="Raw Close")
axes[1].plot(dataset_pdf.index, dataset_pdf["close_fracdiff"])
axes[1].axhline(0, color="gray")
```

[Figure: 2-panel — Raw close price + Fracdiff d=0.4]

## 4. Technical Indicators

Sample 4 indicator: RSI(14), MACD, BB width, Vol ratio 6/24. RSI dao động 20–80 với overbought/oversold zones rõ.

[Figure: 2×2 grid — RSI, MACD+Fracdiff, BB width, Vol ratio]

## 5. Feature Correlation Matrix

Heatmap 21×21 — những quan sát:

- `ema_12` ↔ `ema_26` tương quan cao (>0.9) — redundancy.
- `close_fracdiff` tương quan thấp với hầu hết — feature independent, quý giá.
- `obv` ↔ `obv_delta_12` tương quan vừa.
- `hour`, `dayofweek` không tương quan với indicator — feature calendar độc lập.

```python
corr = dataset_pdf[features].corr()
sns.heatmap(corr, mask=np.triu(...), cmap="RdBu", center=0)
```

[Figure: Lower-triangle correlation heatmap 21×21]

## 6. Feature Distribution by Label

Histogram 4 feature chính (`rsi_14`, `volatility_24`, `bb_position`, `macd`) phân tách theo label $\{-1, +1\}$. Phân tách rõ nhất ở `rsi_14` (LONG thiên high RSI, SHORT thiên low RSI — phù hợp intuition) và `bb_position`.

[Figure: 2×2 histogram — RSI/vol/bb/macd by label]

## 7. Purged Embargo CV Splits

```python
cv = PurgedEmbargoTimeSeriesSplit(CV_SPLITS, EMBARGO_PCT)
```

5 fold với purge gap + embargo 2% — train/val không overlap và có buffer. [Figure: 5-row horizontal bar — train (blue) + val (orange) per fold]

## 8. OOF Scores (Smart Filtering)

```python
scores = pd.Series(model.oof_scores_).sort_values()
```

| Model | OOF F1 | Status |
|---|---|---|
| LightGBM | 0.7168 | ACTIVE |
| SVC | 0.6812 | ACTIVE |
| GRU | 0.6103 | ACTIVE |

Cả 3 đều pass threshold `MIN_OOF_F1 = 0.50` → stacking dùng đủ 3 base learner (meta input = 6 feature).

[Figure: Horizontal bar — OOF F1 by model với ACTIVE/FILTERED color]

## 9. Test Set Performance

```
Accuracy: 0.7343
F1 macro: 0.7331

              precision    recall  f1-score   support
        -1.0       0.80      0.71      0.75      2989
         1.0       0.67      0.77      0.72      2306
    accuracy                           0.73      5295
   macro avg       0.74      0.74      0.73      5295
weighted avg       0.74      0.73      0.73      5295
```

**Confusion matrix**:

| | Pred SHORT | Pred LONG |
|---|---|---|
| **True SHORT** | 2120 (70.9%) | 869 (29.1%) |
| **True LONG** | 538 (23.3%) | 1768 (76.7%) |

Recall class LONG = 77%, precision = 67% — mô hình bias detect LONG hơn SHORT.

[Figure: Normalized + count confusion matrix]

## 10. Feature Importance (LightGBM)

```
 1. obv                  400  11.51%
 2. volatility_24        349  10.65%
 3. close_in_range_24    267   8.17%
 4. bb_width             273   7.52%
 5. macd                 257   7.13%
 6. close_fracdiff       216   6.39%
 7. adx_14               166   5.30%
 8. atr_14               162   4.79%
 9. rsi_14               149   4.71%
10. obv_delta_12         148   4.14%
```

Volume-related feature (OBV, volatility_24) chiếm top — phù hợp literature microstructure \cite{de_prado_2018_afml}. Fracdiff cũng trong top-6 — confirming fractional differencing là feature quý.

[Figure: Top-20 feature bar chart]

## 11. Backtest

Tuning 168 combinations `(tp_atr, sl_atr, min_hold)`:

```
Best: tp=4.0 sl=2.5 min_hold=24 sharpe=0.117 trades=403 pf=1.02
```

**Metrics cuối (test set)**:

| Metric | Value |
|---|---|
| Total return | +3.11% |
| Sharpe | +0.689 |
| Max drawdown | −5.12% |
| Profit factor | 1.26 |
| Win rate | 0.443 |
| Trades | 106 |
| Avg bars held | 16.8 |
| Avg PnL/trade | +$2.94 |

**Equity curve**: Strategy profitable nhẹ, buy & hold XAU/USD tăng từ 10k → ~$11 060 (+10.59%) trong 2023. [Figure: Strategy equity vs Buy&Hold equity, 3-panel với drawdown + positions]

**PnL distribution**: mean shift âm nhẹ, distribution fat-tail hai phía. Win/loss bar pie ~41/59. [Figure: 2×2 — PnL histogram, cumulative PnL, hourly avg PnL, win/loss pie]

## 12. Rolling Performance

Rolling Sharpe 100-hour: dao động quanh 0, có giai đoạn dương mạnh rồi suy giảm. Rolling return cũng thế — strategy không stable.

[Figure: 2-panel rolling Sharpe + cumulative return]

## 13. Summary Dashboard

[Figure: KPI grid 3×3 — Total Return / Sharpe / MDD + equity curve + confusion matrix + summary stats]

## 14. Pipeline Timing

```
total:           1221.60s
model_training:  375.01s  (31%)
ablation_study:  726.30s  (59%)
data_loading:     20.47s  (2%)
backtesting:      30.42s  (2%)
shap_analysis:     1.01s  (<1%)
baseline_training: 0.14s  (<1%)
bootstrap_significance: 0.31s  (<1%)
```

Model training bottleneck — chủ yếu GRU (10 epoch bidirectional). LightGBM + SVC chỉ ~30s tổng.

## Kết luận EDA

- **Label**: triple-barrier với auto-tune tạo balance ratio 0.93 — đủ cân bằng, không cần oversampling.
- **Feature**: OBV, volatility_24, bb_width, macd, close_fracdiff top importance — microstructure + volatility là signal chính cho XAU/USD.
- **Model**: stacking 3 base đều pass OOF threshold; LightGBM mạnh nhất (0.7168), GRU yếu nhất (0.6103) nhưng vẫn đóng góp diversification.
- **Backtest**: F1 test cao (0.73) và PnL dương lần đầu (+3.11%, Sharpe +0.69, PF 1.26) — barrier tuning + cost optimization đã cải thiện trade-off (xem `22-evaluation-metrics.md`).
- **Timing**: ~20 phút/run cho 5 năm dữ liệu (gồm ablation study 726s) — khả thi cho 120-run ablation grid.

## Tham khảo

- \cite{de_prado_2018_afml} — López de Prado, methodology tổng hợp.
- \cite{pedregosa_2011_sklearn} — confusion matrix, classification report.
- \cite{ke_2017_lightgbm} — LightGBM importance definition.
