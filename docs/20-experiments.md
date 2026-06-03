---
doc: 20-experiments
stage: experiments
thesis_chapter: 4
status: draft
last_updated: 2026-06-02
code_ref: null
---

# Experiments

> Thiết kế thực nghiệm: hypothesis, ablation matrix, hyperparameter sweep, thống kê so sánh — phục vụ Chương 4 luận văn.

## Tóm tắt

Luận văn đề xuất 4 hypothesis chính cần kiểm chứng bằng thực nghiệm. Thiết kế ablation 6 cell (E0–E5) bật/tắt từng thành phần cốt lõi (stacking, triple-barrier, meta-labeling, purged CV) để đo lường marginal contribution. Bổ sung seed sweep 4 giá trị và months sweep 5 mức dữ liệu (3/6/12/24/60 tháng) để đánh giá ổn định.

## Cơ sở lý thuyết

Methodology cốt lõi được tham chiếu từ López de Prado \cite{de_prado_2018_afml}: triple-barrier labeling (ch. 3), purged k-fold với embargo (ch. 7), meta-labeling (ch. 3), và backtesting risk (ch. 11–14). Stacking ensemble dựa trên Wolpert \cite{wolpert_1992_stacking}. Đánh giá so sánh forecast sử dụng Diebold-Mariano test khi có đủ mẫu dự báo, hoặc bootstrap confidence interval trên chênh lệch Sharpe — tránh kết luận dựa trên 1 lần chạy.

## Công thức

### Hypothesis

Bốn phát ngôn cần kiểm chứng:

- **H1 — Stacking > base**: Stacking ensemble (GRU + LightGBM + SVC + Logistic meta) có F1 macro OOF cao hơn từng base learner riêng lẻ cùng điều kiện dữ liệu.
- **H2 — Triple-barrier > fixed-horizon**: Label theo triple-barrier (TP/SL + vertical barrier) tạo ra metric backtest (Sharpe, PF) cao hơn fixed-horizon forward return labeling.
- **H3 — Meta-labeling tăng precision**: Thêm meta-label corrector (confidence gate $P(\mathrm{correct}|x) \geq 0.55$) làm tăng win rate và profit factor, chấp nhận giảm số trades.
- **H4 — Purged-embargo CV > standard KFold**: Cross-validation chuẩn bị leakage trong time series; purged k-fold với embargo giảm overfitting OOF — chứng minh qua chênh lệch F1 OOF vs test.

### Ablation matrix

Bảng dưới là **master design** cho 6 cell thực nghiệm (E0 = baseline tối giản, E5 = full pipeline). Mỗi cell giữ nguyên seed, dữ liệu, hyperparam khác — chỉ bật/tắt cờ ghi trong cột:

| Experiment | Stacking | Triple-barrier | Meta-label | Purged CV | Auto-tune barriers | Mục tiêu kiểm chứng |
|---|---|---|---|---|---|---|
| **E0** baseline | ✗ (LightGBM only) | ✗ (fixed-horizon) | ✗ | ✗ (KFold) | ✗ | Mô hình đơn + labeling cổ điển |
| **E1** | ✗ | ✓ | ✗ | ✗ (KFold) | ✗ | Cộng triple-barrier |
| **E2** | ✗ | ✓ | ✓ | ✗ (KFold) | ✗ | Cộng meta-label |
| **E3** | ✗ | ✓ | ✓ | ✓ | ✗ | Cộng purged CV |
| **E4** full | ✓ | ✓ | ✓ | ✓ | ✓ | Đầy đủ pipeline (contribution) |
| **E5** no-auto-tune | ✓ | ✓ | ✓ | ✓ | ✗ | Ablation auto-tune barriers |

Quan sát: chênh lệch $E_k - E_{k-1}$ đo marginal contribution của thành phần vừa bật. E4 − E0 = contribution tổng hợp của toàn bộ pipeline.

### Hyperparameter grid

Seed sweep 4 giá trị (đánh giá variance theo initialization):

$$
\mathrm{seed} \in \{21, 42, 63, 84\}.
$$

Months sweep 5 mức (đánh giá ổn định theo kích thước dữ liệu):

$$
\mathrm{months} \in \{3, 6, 12, 24, 60\}.
$$

Tổng kombinasi: $6 \times 4 \times 5 = 120$ runs (khả thi nếu mỗi run $\approx 8$ phút — tổng $\approx 16$ giờ). Nếu giới hạn thời gian, ưu tiên seed sweep trên E0 + E4 (so sánh baseline vs full), months = 12 và 60.

### Statistical test

So sánh hai forecast $e^{(A)}_t, e^{(B)}_t$ (loss function $L$ = squared error hoặc 0-1 loss) bằng **Diebold-Mariano**:

$$
\mathrm{DM} = \frac{\bar{d}}{\hat{\sigma}_d / \sqrt{T}}, \quad d_t = L(e^{(A)}_t, y_t) - L(e^{(B)}_t, y_t), \quad \bar{d} = \frac{1}{T}\sum_t d_t.
$$

DM chuẩn N(0,1) dưới $H_0$: hai forecast equal predictive accuracy. Cách thay thế đơn giản hơn: bootstrap 1000 lần resample trên chênh lệch Sharpe, tính 95% CI — nếu CI không chứa 0 thì bác bỏ $H_0$ về Sharpe bằng nhau.

## Cài đặt

Không có code ablation riêng — thiết kế chạy thủ công bằng cách override `src/config.py`:

```python
# E0 baseline
USE_META_LABELING = False
LABELING_HORIZON = 24  # fixed-horizon (bypass triple-barrier)
CV_SPLITS = 5  # standard KFold
TUNE_TP_RANGE = (0.5, 4.0, 0.25)  # auto-tune disabled for baseline
TUNE_SL_RANGE = (0.5, 4.0, 0.25)
# stacking disable: dùng LightGBMClassifier trực tiếp thay vì HybridStackingSignalClassifier
```

Lệnh chạy: `pixi run smoke` (1 tháng, smoke test) → `pixi run run` (12 tháng) → `pixi run run-full` (5 năm). Mỗi lần chạy sinh `reports/run_<timestamp>/` (xem `21-results-convention.md`).

## Tham số quan trọng

- **Seed**: `RANDOM_STATE = 42` hardcoded trong `src/config.py` — đổi trực tiếp trong config.
- **Months**: CLI arg `--months 12` (hoặc `--full` cho 60 tháng).
- **CV splits**: `CV_SPLITS = 5` — cố định cho mọi experiment để fold count comparable.
- **Min OOF F1**: `MIN_OOF_F1 = 0.50` — giữ cố định cho mọi experiment có stacking.

## Kết quả thực nghiệm

### Run matrix

Bảng run thực tế (placeholder — user fill sau khi chạy):

| Run ID | Experiment | Seed | Months | F1 (OOF) | Sharpe | MDD | Win Rate | PF |
|---|---|---|---|---|---|---|---|---|
| `run_20260603_181525` | E4 (full) | 42 | 60 | 0.7168 | +0.689 | −0.051 | 0.443 | 1.26 |
| _chưa chạy_ | E0 | 42 | 60 | — | — | — | — | — |
| _chưa chạy_ | E0–E5 × seeds {21,63,84} | — | 60 | — | — | — | — | — |

Sau khi hoàn tất 120 runs, tổng hợp vào bảng ablation chính trong luận văn: mỗi cell = trung bình ± std trên 4 seed.

### Quan sát sơ bộ từ E4 full

Từ `reports/run_20260603_181525`:

- F1 OOF LightGBM = 0.7168 (base cao nhất), SVC = 0.6812, GRU = 0.6103 — cả 3 đều pass threshold `MIN_OOF_F1 = 0.50`.
- Test F1 macro = 0.7331 — cao hơn OOF LightGBM, gợi ý stacking có lợi (hoặc may mắn do seed).
- Backtest Sharpe = +0.69, total return = +3.11% — lần đầu F1 chuyển thành PnL dương (xem `22-evaluation-metrics.md`).
- Top feature: OBV (11.51%), volatility_24 (10.65%), close_in_range_24 (8.17%), bb_width (7.52%) — phù hợp literature tài chính \cite{de_prado_2018_afml}.

## Tham khảo

- \cite{de_prado_2018_afml} — López de Prado, methodology tổng hợp.
- \cite{wolpert_1992_stacking} — stacking generalization.
- \cite{de_prado_2018_cross_val} — purged k-fold + embargo.
- \cite{de_prado_2018_backtest} — backtesting risk, Sharpe, evaluation.
