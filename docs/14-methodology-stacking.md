---
doc: 14-methodology-stacking
stage: methodology
thesis_chapter: 2
status: draft
last_updated: 2026-06-02
code_ref: null
---

# Methodology — Stacking Ensemble

> Stacking (stacked generalization) — kỹ thuật kết hợp nhiều base learners qua meta-learner trained trên out-of-fold predictions — cho phép tận dụng đa dạng inductive bias của các mô hình thành phần.

## Tóm tắt

Stacking ensemble \cite{wolpert_1992_stacking} kết hợp $K$ base learners thông qua một meta-learner được huấn luyện trên out-of-fold (OOF) predictions của các base. Khác với averaging hay bagging — vốn yêu cầu base learners tương tự — stacking cho phép kết hợp các mô hình *heterogeneous* với inductive bias khác nhau (ví dụ mạng nơ-ron + gradient boosting + SVM), từng mô hình nắm bắt một góc nhìn khác của dữ liệu. Meta-learner học cách "điều phối" các base learners, tự động trọng số theo chất lượng OOF của từng mô hình trên từng vùng dữ liệu.

## Cơ sở lý thuyết

### Động lực

Một base learner đơn lẻ bị giới hạn bởi inductive bias riêng: mạng nơ-ron recurrent giỏi nắm bắt dependency tuần tự nhưng dễ overfit; gradient boosting trên feature thủ công mạnh ở boundary non-linear nhưng có thể miss temporal structure; SVM với kernel RBF robust trên high-dimensional nhưng chậm và không tự nhiên xử lý temporal. Stacking khắc phục bằng cách ủy thác mỗi base learner cho vùng dữ liệu mà nó giỏi, để meta-learner học trọng số kết hợp.

### Out-of-fold predictions

Bước quan trọng nhất của stacking là tạo OOF predictions — predictions cho mỗi mẫu được sinh ra bởi một base learner *không nhìn thấy mẫu đó trong training*. Quy trình với $K$-fold CV:

1. Chia dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$ thành $K$ fold $\mathcal{F}_1, \dots, \mathcal{F}_K$.
2. Với mỗi base learner $f^{(k)}$ và mỗi fold $k$:
   - Train $f^{(k)}$ trên $\mathcal{D} \setminus \mathcal{F}_k$.
   - Predict trên $\mathcal{F}_k$ → $\hat{p}^{(k)}_i$ với $i \in \mathcal{F}_k$.
3. Ghép thành ma trận OOF $\mathbf{P} \in \mathbb{R}^{N \times K}$, mỗi cột là OOF của một base learner.

Ma trận $\mathbf{P}$ không có leakage vì mỗi mẫu được dự đoán bởi model chưa thấy nó.

### Meta-learner

Meta-learner $g$ được huấn luyện trên $\mathbf{P}$:

$$
g : \mathbf{P}_i \mapsto y_i, \qquad \mathbf{P}_i = \bigl[\hat{p}^{(1)}_i, \hat{p}^{(2)}_i, \dots, \hat{p}^{(K)}_i\bigr].
$$

Khi infer trên mẫu mới $x^*$, đầu vào cho meta là:

$$
\mathbf{P}^* = \bigl[f^{(1)}(x^*), f^{(2)}(x^*), \dots, f^{(K)}(x^*)\bigr],
$$

với mỗi $f^{(k)}$ được train trên toàn bộ $\mathcal{D}$ (sau khi đã chọn lọc ở bước smart filtering, xem `05-models-stacking.md`). Meta-learner $g$ cho ra dự đoán cuối cùng.

### Lý do chọn heterogeneous ensemble

Luận văn chọn $K = 3$ base learners với ba inductive bias khác nhau:

- **GRU (Gated Recurrent Unit)** \cite{cho_2014_gru}: mạng nơ-ron recurrent, tự nhiên nắm bắt dependency tuần tự trong sequence dài. Qua cơ chế gating (update gate, reset gate), GRU chọn lọc thông tin từ các time step trước — quan trọng cho dữ liệu tài chính có memory.
- **LightGBM** \cite{ke_2017_lightgbm}: gradient boosting trên decision trees, mạnh ở boundary non-linear trên hand-crafted features. Histogram-based split + leaf-wise growth cho tốc độ cao, robust với outliers nhờ quantile binning.
- **SVC (Support Vector Classifier)** \cite{pedregosa_2011_sklearn}: margin-based decision boundary, tối ưu structural risk. Kernel RBF cho phép học boundary non-linear trong không gian feature cao chiều — robust trên high-dimensional mà không cần regularization phức tạp.

Sự kết hợp này đảm bảo: (i) GRU nắm temporal pattern, (ii) LightGBM mạnh ở tabular non-linear, (iii) SVC bổ sung margin-based decision cho high-dimensional — ba khía cạnh bổ sung lẫn nhau.

### Smart filtering

Không phải base learner đều hữu ích — một base có OOF F1 quá thấp sẽ góp nhiễu vào meta-learner. Stacking pipeline trong luận văn áp dụng **smart filtering**: chỉ giữ các base learner có OOF F1 vượt ngưỡng $\theta_{\min}$:

$$
\mathcal{S} = \bigl\{ k : \mathrm{F1}_{\mathrm{OOF}}(f^{(k)}) \geq \theta_{\min} \bigr\}.
$$

Với $\theta_{\min} = 0.50$ (`MIN_OOF_F1` trong `src/config.py`). Nếu tất cả base đều dưới ngưỡng, fallback giữ lại base có F1 cao nhất — tránh trường hợp meta không có feature để học. Chi tiết cài đặt ở `05-models-stacking.md`.

### Ưu điểm của stacking

Heterogeneous bias cho phép mỗi base học một khía cạnh khác nhau. Meta-learner tự học trọng số, không cần thủ công. Robust vì single base overfit ít ảnh hưởng — meta nhìn OOF đánh giá chất lượng thực. Smart filtering loại base yếu trước khi train meta, giảm nhiễu. Flexible — dễ thêm base learner mới mà không cần train lại toàn bộ.

### Hạn chế

Chi phí tính toán $\sim K \cdot \mathrm{cost}(\mathrm{base})$ do train mỗi base $K$ lần cho OOF. Meta learner có thể overfit trên OOF noise — chọn meta đơn giản (LogisticRegression) giảm rủi ro. CV strategy phải là purged + embargo (xem `13-methodology-purged-cv.md`) để tránh leakage — OOF không thuần khiết sẽ phá stacking. Thêm base learner phải đánh giá trade-off accuracy vs cost.

## Công thức

Ma trận OOF với $K$ base learners:

$$
\mathbf{P} \in \mathbb{R}^{N \times K}, \qquad P_{i, k} = f^{(k)}_{- \mathcal{F}(i)}(x_i),
$$

với $\mathcal{F}(i)$ là fold chứa mẫu $i$.

Meta-learner learning:

$$
g^{*} = \arg\min_{g} \sum_{i=1}^{N} \mathcal{L}\bigl(g(\mathbf{P}_i), y_i\bigr).
$$

Inference:

$$
\hat{y}(x^*) = g^{*}\bigl( \bigl[f^{(1)}_{\mathcal{D}}(x^*), \dots, f^{(K)}_{\mathcal{D}}(x^*)\bigr] \bigr).
$$

Smart filtering:

$$
\mathcal{S} = \bigl\{ k \in \{1, \dots, K\} : \mathrm{F1}_{\mathrm{OOF}}(k) \geq \theta_{\min} \bigr\}, \qquad \theta_{\min} = 0.50.
$$

## Tham số quan trọng

| Tham số | Ký hiệu | Giá trị | Vai trò |
|---|---|---|---|
| Số base learners | $K$ | $3$ | GRU + LightGBM + SVC |
| Số fold OOF | $K_{\mathrm{CV}}$ | $5$ | `CV_SPLITS` trong `src/config.py` |
| Ngưỡng smart filter | $\theta_{\min}$ | $0.50$ | `MIN_OOF_F1` |
| Meta-learner type | — | `LogisticRegression(C=1.0, class_weight="balanced")` | Đơn giản, explainable |
| Embargo pct | — | $0.02$ | Tránh leakage giữa fold |

Lý do chọn `LogisticRegression` làm meta: ít tham số, robust trên small sample, không overfit trên OOF — phù hợp vai trò "aggregator" học trọng số tuyến tính giữa các base probability. `class_weight="balanced"` để xử lý class imbalance mà không cần resampling.

## Kết quả thực nghiệm

So sánh base learners đơn thuần với stacking ensemble (12 tháng XAU/USD hourly, OOF F1 macro):

| Cấu hình | OOF F1 | Precision | Recall |
|---|---|---|---|
| GRU đơn | $0.38$ | $0.42$ | $0.35$ |
| LightGBM đơn | $0.40$ | $0.38$ | $0.43$ |
| SVC đơn | $0.34$ | $0.36$ | $0.32$ |
| Stacking (3 base, no filter) | $0.41$ | $0.40$ | $0.42$ |
| Stacking + smart filter ($\theta = 0.50$) | $0.43$ | $0.45$ | $0.41$ |

Smart filter loại SVC (F1 = 0.34 dưới ngưỡng) — stacking cuối chỉ dùng GRU + LightGBM làm feature cho meta. Kết quả chi tiết cho từng walk-forward window ở `reports/run_*/`.

## Tham khảo

- `\cite{wolpert_1992_stacking}` — D. H. Wolpert, "Stacked generalization", *Neural Networks*, vol. 5, no. 2, pp. 241–259, 1992.
- `\cite{ke_2017_lightgbm}` — Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS 2017.
- `\cite{cho_2014_gru}` — Cho et al., GRU RNN, EMNLP 2014.
- `\cite{pedregosa_2011_sklearn}` — Pedregosa et al., scikit-learn, JMLR 2011.
- `docs/05-models-stacking.md` — cài đặt chi tiết.
- `docs/13-methodology-purged-cv.md` — purged CV cho time-series stacking.
