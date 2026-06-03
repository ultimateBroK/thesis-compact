---
doc: 22-evaluation-metrics
stage: metrics
thesis_chapter: 4
status: draft
last_updated: 2026-06-02
code_ref: null
---

# Evaluation Metrics

> Định nghĩa, công thức và tham chiếu code cho F1, Sharpe, Sortino, max-drawdown, win rate, profit factor, PnL, Calmar — các chỉ số dùng đánh giá mô hình và backtest.

## Tóm tắt

Luận văn đánh giá hệ thống trên hai khía cạnh độc lập: (i) **chất lượng phân loại** của stacking ensemble — thông qua F1 macro OOF và trên tập test, (ii) **hiệu quả giao dịch** — thông qua Sharpe annualized, max drawdown, win rate, profit factor, PnL tổng. Bảng dưới tổng kết nhanh:

| Metric | Công thức rút gọn | Code function | Đơn vị |
|---|---|---|---|
| F1 macro | $2PR/(P+R)$ trung bình các class | `src/models.py::evaluate_oof_predictions`, `src/reporting.py::build_evaluation_metadata` | [0, 1] |
| Sharpe (annualized) | $\sqrt{N}\mu_R/\sigma_R$ | `src/backtest.py::compute_sharpe_ratio` | dimensionless |
| Sortino | $\mu_R/\sigma_D$ | `src/backtest.py::compute_sortino_ratio` | dimensionless |
| Max drawdown | $\max_t (P_t - E_t)/P_t$ | `src/backtest.py::compute_max_drawdown` | [0, 1] |
| Win rate | $\#\text{wins}/\#\text{trades}$ | `src/backtest.py::compute_win_rate` | [0, 1] |
| Profit factor | $\sum^+ / |\sum^-|$ | `src/backtest.py::compute_profit_factor` | $\geq 0$ |
| Total return | $E_T/E_0 - 1$ | `src/backtest.py::compute_backtest_metrics` | fraction |
| Calmar | $CAGR / |MDD|$ | (chưa cài) | dimensionless |
| DSR | (Sharpe $- \hat{\sigma}_{SR})/\hat{\sigma}_{SR}$ | `src/backtest.py::compute_backtest_metrics` | dimensionless |

## Cơ sở lý thuyết

Phân loại signal giao dịch là bài toán mất cân bằng nhẹ giữa class LONG và SHORT do triple-barrier không đối xứng (TP/SL ratio khác nhau). Do đó F1 macro — trung bình harmonique precision và recall trên từng class — phù hợp hơn accuracy \cite{pedregosa_2011_sklearn}. Các chỉ số backtest (Sharpe, drawdown, profit factor) đo lường hiệu quả trên equity curve, tức là có tính đến position sizing và cost — đây là metric quyết định khi so sánh chiến lược \cite{de_prado_2018_backtest}.

## Công thức

### F1 score

Cho nhãn thật $y_i$ và dự báo $\hat{y}_i$ trên $N$ mẫu, với $\mathrm{TP}, \mathrm{FP}, \mathrm{FN}$ theo class:

$$
P = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}, \quad R = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}, \quad F_1 = 2 \cdot \frac{P \cdot R}{P + R}.
$$

**Macro F1** (dùng trong code, `average='macro'`):

$$
F_1^{\mathrm{macro}} = \frac{1}{K}\sum_{k=1}^{K} F_1^{(k)}.
$$

Variants khác trong scikit-learn: **micro** (tính TP/FP/FN tổng cộng trên tất cả class, hợp khi class imbalance mạnh), **weighted** (trung bình có trọng số theo support) \cite{pedregosa_2011_sklearn}. Module `src/models.py` tính macro F1 trên OOF predictions của từng base learner (lệnh `f1_score(y_np[valid], pred, average='macro', zero_division=0)` ở dòng 115) — đây là tiêu chí smart filtering.

### Sharpe ratio (annualized)

Từ chuỗi equity $E_t$ trên $T$ bar, tính per-bar returns $r_t = (E_t - E_{t-1})/E_{t-1}$:

$$
S = \frac{\sqrt{N} \cdot \mu_R}{\sigma_R}, \quad \mu_R = \frac{1}{T}\sum_{t=1}^{T} r_t, \quad \sigma_R = \sqrt{\frac{1}{T}\sum_t (r_t - \mu_R)^2}.
$$

Với dữ liệu XAU/USD 1-hour, $N = 252 \times 24 = 6048$ periods/năm. Risk-free rate giả định bằng 0 — phù hợp cho tài sản 24/7 không có benchmark risk-free chuẩn như T-bill cho equity market \cite{de_prado_2018_backtest}. Code: `src/backtest.py::compute_sharpe_ratio` dùng `np.sqrt(252 * 24)`.

### Sortino ratio

Variation của Sharpe chỉ xét downside deviation:

$$
S_D = \frac{\mu_R}{\sigma_D}, \quad \sigma_D = \sqrt{\frac{1}{|\{t: r_t < 0\}|}\sum_{t: r_t < 0} r_t^2}.
$$

Sortino công bằng hơn với chiến lược có upside vol lớn (pop profitable). Sortino đã cài trong `src/backtest.py::compute_sortino_ratio`.

### Maximum drawdown

Từ equity $E_t$, tính running peak $P_t = \max_{s \leq t} E_s$:

$$
\mathrm{MDD} = \min_t \frac{E_t - P_t}{P_t} \in (-\infty, 0].
$$

Code: `src/backtest.py::compute_max_drawdown` dùng `np.maximum.accumulate` + `(equity - cummax) / cummax`. MDD không có window cố định — đo lường worst peak-to-trough trên toàn kỳ.

### Win rate

$$
W = \frac{|\{i: \mathrm{PnL}_i > 0\}|}{|\{i: \mathrm{PnL}_i \neq 0\}|}.
$$

Code: `src/backtest.py::compute_win_rate` đếm `t['win'] == True` trên `executed_trades`. Cách tính chính xác hơn `reporting.py::build_win_rate_metadata` (fallback dựa trên `bar_pnl_usd`).

### Profit factor

$$
\mathrm{PF} = \frac{\sum_{i: \mathrm{PnL}_i > 0} \mathrm{PnL}_i}{\left|\sum_{i: \mathrm{PnL}_i < 0} \mathrm{PnL}_i\right|}.
$$

$\mathrm{PF} > 1$ ⇔ chiến lược có lãi; $\mathrm{PF} = \infty$ nếu không có trade thua. Code: `src/backtest.py::compute_profit_factor` xử lý cả fallback từ equity diffs khi thiếu `executed_trades`.

### PnL

Tổng PnL thực hiện (realized) từ trades:

$$
\mathrm{PnL}_{\mathrm{total}} = \sum_{i=1}^{N_{\mathrm{trades}}} \mathrm{trade\_pnl\_usd}_i = E_T - E_0.
$$

Invariant: `equity[-1] == initial_balance + total_trade_pnl` (assert ở `src/backtest.py::run_barrier_backtest`). PnL đã trừ đầy đủ CFD cost: spread (round-trip), commission, overnight swap — xem `src/backtest.py::compute_trade_costs`.

### Calmar ratio

$$
C = \frac{\mathrm{CAGR}}{|\mathrm{MDD}|}, \quad \mathrm{CAGR} = \left(\frac{E_T}{E_0}\right)^{1/Y} - 1,
$$

với $Y$ = số năm. Calmar đo trade-off annual return / worst loss. Hiện chưa cài.

## Cài đặt

Tham chiếu trực tiếp trong codebase:

- **F1 macro OOF**: `src/models.py::HybridStackingSignalClassifier.fit` — tính cho mỗi base learner, lưu vào `self.oof_scores_`, in ra bởi `src/reporting.py::print_model_filtering_report`.
- **F1 macro test**: `src/reporting.py::build_evaluation_metadata` (dòng 421), lưu vào `run_data.json` tại `evaluation.f1_macro`.
- **Sharpe / MDD / PF / Win rate / total_return**: `src/backtest.py::compute_backtest_metrics` (dòng 379), lưu vào `run_data.json` tại `backtest.*`.
- **Trade-level PnL**: `src/backtest.py::create_trade_record` — mỗi trade có `gross_pnl_usd`, `spread_cost_usd`, `commission_usd`, `swap_usd`, `trade_pnl_usd`, `cost_usd`, `lots`, `overnights`, `win`.
- **Equity curve**: mảng numpy `equity` từ `run_barrier_backtest`, được plot bởi `src/reporting.py::save_equity_curve_plot`.

## Tham số quan trọng

- **Risk-free rate**: giả định 0 (crypto/commodity 24/7, không có benchmark chuẩn). Nếu cần so sánh với S&P 500 thì dùng $r_f \approx 4\%$/năm.
- **Annualization factor**: $N = 252 \times 24 = 6048$ cho timeframe 1h. Nếu đổi sang timeframe 4h thì $N = 252 \times 6 = 1512$.
- **Min OOF F1**: `MIN_OOF_F1 = 0.50` — ngưỡng smart filtering base learner (xem `08-config.md`).
- **Initial balance**: `INITIAL_BALANCE = 10\,000` USD — ảnh hưởng tới PnL tuyệt đối nhưng không ảnh hưởng Sharpe/MDD (relative).

## Kết quả thực nghiệm

**Trade-off quan trọng**: F1 cao không đảm bảo PnL cao. Nguyên nhân: (i) label imbalance khiến macro F1 quá lạc quan với class majority, (ii) position sizing phụ thuộc confidence threshold, không chỉ label đúng, (iii) transaction cost (spread) ăn mòn PnL mà F1 không thấy, (iv) TP/SL barrier exit tạo PnL phân phối heavily-skewed không khớp với correct/wrong ratio.

Ví dụ từ `reports/run_20260603_181525`:

| Metric | Value |
|---|---|
| F1 macro (test) | 0.7331 |
| Accuracy | 0.7343 |
| Sharpe | +0.689 |
| Max drawdown | −5.12% |
| Total return | +3.11% |
| Profit factor | 1.26 |
| Win rate | 0.443 |
| Trades | 106 |
| DSR statistic | 0.596 |
| DSR p-value | 0.276 |

F1 = 0.73 và Sharpe dương (+0.69) — mô hình đúng direction VÀ barrier exit + cost không chiếm hết edge. Profit factor > 1 lần đầu.

## Tham khảo

- \cite{pedregosa_2011_sklearn} — scikit-learn, định nghĩa F1 macro/micro/weighted.
- \cite{de_prado_2018_backtest} — López de Prado, ch. 11–14, Sharpe và drawdown cho backtest tài chính.
- \cite{de_prado_2018_afml} — López de Prado, ch. 3, labeling và metrics cho financial ML.
- \cite{hamilton_1994_time_series} — Hamilton, thời gian row và annualization conventions.
