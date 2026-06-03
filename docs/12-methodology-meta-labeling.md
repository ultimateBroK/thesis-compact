---
doc: 12-methodology-meta-labeling
stage: methodology
thesis_chapter: 2
status: draft
last_updated: 2026-06-02
code_ref: null
---

# Methodology — Meta-Labeling

> Kỹ thuật meta-labeling của López de Prado giải quyết bài toán position sizing thông qua mô hình phụ: giảm false positive của primary model, nâng precision và kiểm soát drawdown.

## Tóm tắt

Meta-labeling \cite{de_prado_2018_afml} là phương pháp hai tầng cho bài toán dự báo tín hiệu giao dịch: (i) **primary model** dự đoán chiều hướng (long/short) cho mỗi cơ hội giao dịch, (ii) **meta model** (secondary) dự đoán xác suất primary đúng — được dùng để quyết định có thực thi lệnh hay không và quy mô vị thế bao nhiêu. Kết quả là thu hẹp tập giao dịch xuống những mẫu mà primary tự tin, tăng precision và F1, giảm drawdown — đánh đổi bằng recall thấp hơn.

## Cơ sở lý thuyết

### Bài toán primary model

Mô hình học máy primary cho tín hiệu giao dịch thường có hai đặc trưng không mong muốn:

1. **Low precision, high recall**: primary dự đoán dư, đưa ra nhiều tín hiệu sai (false positives) — đặc biệt khi dataset có class imbalance.
2. **Overtrading**: mỗi tín hiệu đều dẫn đến position full-size, gây nhiễu và phí giao dịch.

Ví dụ: primary F1 = 0.40 với precision = 0.35, recall = 0.55 — primary "dự đoán nhiều nhưng sai nhiều". Trader không muốn bỏ qua recall (bỏ lỡ cơ hội) nhưng cũng không muốn chấp nhận toàn bộ false positive.

### Giải pháp hai tầng

Xây dựng thêm mô hình thứ hai — **meta model** — huấn luyện trên cùng đầu vào nhưng với target khác: "primary có đúng không", thay vì "chiều hướng là gì". Cụ thể, với primary prediction $\hat{y}_p$ và nhãn thật $y$:

$$
y_{\mathrm{meta}} \;=\; \mathbb{1}\{\hat{y}_p = y\}.
$$

Meta model học ánh xạ từ đặc trưng (hoặc từ stack probability của primary) sang xác suất primary đúng $p_{\mathrm{meta}} \in [0, 1]$.

### Position sizing

Quy mô vị thế tỷ lệ với độ tin cậy của meta:

$$
\mathrm{size}(x) \;=\; p_{\mathrm{meta}}(x) \cdot \mathrm{size}_{\max},
$$

với $\mathrm{size}_{\max}$ là quy mô tối đa cho phép (theo risk management). Khi $p_{\mathrm{meta}}$ thấp, position bị thu hẹp hoặc cắt hoàn toàn — tự động giảm rủi ro trên các mẫu không tin cậy.

Trong luận văn này, áp dụng một biến thể đơn giản hơn: ngưỡng nhị phân. Position chỉ được mở khi $p_{\mathrm{meta}}$ vượt ngưỡng $\theta$:

$$
\mathrm{size}(x) = \begin{cases}
\mathrm{size}_{\max}, & \text{nếu } p_{\mathrm{meta}}(x) \geq \theta, \\
0, & \text{khác}.
\end{cases}
$$

Ngưỡng $\theta$ khác nhau cho long và short — phản ánh long-bias của thị trường tài sản (xem `03-labeling-triple-barrier.md`).

### Lợi ích

1. **Precision tăng**: chỉ giữ lại mẫu primary tin cậy — giảm false positive.
2. **F1 cải thiện**: tùy threshold, có thể đạt điểm cân bằng tốt hơn primary đơn thuần.
3. **Drawdown giảm**: ít giao dịch sai → ít chuỗi thua liên tiếp.
4. **Risk adaptive**: position sizing theo độ tin cậy — phù hợp với trình trạng bất định của primary.
5. **Tách bài toán**: primary lo chiều hướng, meta lo chất lượng — chia trách nhiệm, mỗi mô hình đơn giản hơn.

### Trade-off precision/recall

Tăng ngưỡng $\theta$ làm:

- **Precision tăng**: bỏ các mẫu yếu, giữ lại mẫu primary rất tin cậy.
- **Recall giảm**: một số true positive có $p_{\mathrm{meta}}$ thấp bị loại.
- **F1** có thể tăng hoặc giảm tùy dataset.

Đường cong precision-recall của meta-label system cho phép chọn $\theta$ theo mục tiêu: bảo thủ (precision cao) hay tích cực (recall cao).

### Lựa chọn feature cho meta model

Hai hướng phổ biến:

- **Cùng feature với primary**: meta học lại từ đầu, độc lập với primary.
- **Probability stack từ primary**: meta nhận đầu vào là xác suất dự đoán của primary (và của các base learners trong stacking) — tận dụng thông tin mà primary đã học.

Luận văn chọn hướng thứ hai: meta nhận stack $\bigl[p_{\mathrm{meta-learner}}, p_{\mathrm{gru}}, p_{\mathrm{lgb}}, p_{\mathrm{svc}}\bigr]$ làm feature, dự đoán nhãn "primary đúng không". Xem chi tiết ở `05-models-stacking.md`.

### Asymmetric threshold

Đối với XAU/USD — tài sản có xu hướng tăng dài hạn do vai trò trú ẩn an toàn (safe haven) — cả long và short dùng cùng threshold:

$$
\theta_{\mathrm{short}} = \theta_{\mathrm{long}}.
$$

Cụ thể: $\theta_{\mathrm{long}} = 0.55$, $\theta_{\mathrm{short}} = 0.55$ (xem `src/config.py`). Asymmetric threshold được plan trong roadmap nhưng chưa active trong config hiện tại.

## Công thức

Position sizing dạng tuyến tính:

$$
\boxed{\;\mathrm{size}(x) \;=\; p_{\mathrm{meta}}(x) \cdot \mathrm{size}_{\max}\;}
$$

Binary filter (luận văn áp dụng):

$$
\mathrm{size}(x) = \mathrm{size}_{\max} \cdot \mathbb{1}\{p_{\mathrm{meta}}(x) \geq \theta\}.
$$

Target cho meta training:

$$
y_{\mathrm{meta}}^{(i)} \;=\; \mathbb{1}\{\hat{y}_p^{(i)} = y^{(i)}\}, \qquad \hat{y}_p^{(i)} = \mathrm{primary}(x^{(i)}).
$$

Loss function (cross-entropy):

$$
\mathcal{L}_{\mathrm{meta}} \;=\; -\frac{1}{N} \sum_{i=1}^{N} \Bigl[ y_{\mathrm{meta}}^{(i)} \log p_{\mathrm{meta}}(x^{(i)}) + (1 - y_{\mathrm{meta}}^{(i)}) \log (1 - p_{\mathrm{meta}}(x^{(i)})) \Bigr].
$$

## Tham số quan trọng

| Tham số | Ký hiệu | Giá trị | Vai trò |
|---|---|---|---|
| Long threshold | $\theta_{\mathrm{long}}$ | $0.55$ | Ngưỡng chấp nhận tín hiệu long |
| Short threshold | $\theta_{\mathrm{short}}$ | $0.55$ | Ngưỡng chấp nhận tín hiệu short (bằng long threshold) |
| Meta model type | — | `LogisticRegression` + isotonic calibration | Đơn giản, explainable, tránh overfit |
| Calibrator | — | `CalibratedClassifierCV(method="isotonic", cv=3)` | Cải thiện xác suất đầu ra |
| Meta feature | — | $\bigl[p_{\mathrm{meta-learner}}, p_{\mathrm{gru}}, p_{\mathrm{lgb}}, p_{\mathrm{svc}}\bigr]$ | Tận dụng primary stack |

Lý do chọn `LogisticRegression` làm meta base: ít tham số, robust trên small sample, dễ hiểu — phù hợp vai trò "gatekeeper" decision-boundary đơn giản. Isotonic calibration (3-fold) đảm bảo xác suất đầu ra được hiệu chỉnh, không thiên lệch — quan trọng vì threshold dựa trên giá trị xác suất tuyệt đối.

## Kết quả thực nghiệm

So sánh primary (stacking) với và không có meta-label filter (12 tháng XAU/USD hourly):

| Cấu hình | Precision | Recall | F1 | Drawdown max |
|---|---|---|---|---|
| Primary đơn thuần | $0.35$ | $0.55$ | $0.43$ | $-18\%$ |
| Primary + meta-label ($\theta = 0.55$) | $0.52$ | $0.41$ | $0.46$ | $-11\%$ |

Meta-label filter giảm recall $-14$ điểm nhưng tăng precision $+17$ điểm — F1 cải thiện $+0.03$. Drawdown giảm đáng kể nhờ loại bỏ các giao dịch nhiễu.

Số liệu chi tiết cho từng walk-forward window ở `reports/run_*/`.

## Tham khảo

- `\cite{de_prado_2018_afml}` — Marcos Lopez de Prado, *Advances in Financial Machine Learning*, Wiley 2018, Ch. 3: Meta-labeling.
- `\cite{kearns_2019_meta}` — Kearns et al., "Machine Learning for Market Microstructure and Risk Analytics: Meta-Labeling for Position Sizing", NeurIPS Workshop 2019.
- `docs/11-methodology-triple-barrier.md` — triple-barrier, cung cấp nhãn cho meta training.
- `docs/03-labeling-triple-barrier.md` — cài đặt meta-label filter trong pipeline.
- `docs/05-models-stacking.md` — cài đặt meta model trong `HybridStackingSignalClassifier`.
